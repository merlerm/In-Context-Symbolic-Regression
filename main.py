import os
import sys
import json
import copy
import time
import datetime
import warnings
import signal
import cProfile

import hydra
import torch
import matplotlib.pyplot as plt
import numpy as np
import utils

from transformers import set_seed
from omegaconf import OmegaConf, DictConfig, listconfig
from sklearn.metrics import r2_score

from plotter import Plotter
from optimizer import Optimizer
from current_functions import CurrentFunctions
from scorers import BasicScorer, MinMaxScorer, ComplexityScorer
from mloggers import ConsoleLogger, FileLogger, MultiLogger, LogLevel

from typing import Dict, Tuple, List, Any
from collections.abc import Callable


class Workspace(object):
    """
    Workspace class for running the symbolic regression experiment.
    """
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        
        # Output setup
        self.root_dir = cfg.get("root", os.getcwd())
        self.output_dir = cfg.get("output_dir", "output")
        model_folder_name = cfg.model.name.strip()
        if "/" in model_folder_name:
            model_folder_name = model_folder_name.split("/")[-1]
        experiment_folder_name = os.path.join(cfg.experiment.function.group, cfg.experiment.function.name) if hasattr(cfg.experiment.function, "group") else cfg.experiment.function.name
        self.output_path = os.path.join(self.root_dir, self.output_dir, experiment_folder_name, model_folder_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/")
        while os.path.exists(self.output_path):
            self.output_path = os.path.join(self.root_dir, self.output_dir, experiment_folder_name, model_folder_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + str(np.random.randint(0, 1000)) + "/")
        os.makedirs(self.output_path)

        # Logger setup
        cfg.logger.run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        loggers_list = cfg.logger.loggers
        log_level = LogLevel[cfg.logger.get("level", "INFO")]
        loggers = []
        for logger in loggers_list:
            if logger == "console":
                loggers.append(ConsoleLogger(default_priority=log_level))
            elif logger == "file":
                loggers.append(FileLogger(os.path.join(self.output_path, 'log.json'), default_priority=log_level))
            elif logger == "":
                pass
            else:
                print(f'[WARNING] Logger "{logger}" is not supported')
        self.logger = MultiLogger(loggers, default_priority=log_level)
        self.logger.info(f"Project root: {self.root_dir}.")
        self.logger.info(f"Logging to {self.output_path}.")
        job_id = utils.get_job_id()
        self.logger.info(f"Slurm job ID: {job_id}.") if job_id is not None else None

        # Redirect warnings to logger
        warnings.filterwarnings("default")
        warnings.showwarning = lambda *args, **kwargs: self.logger.warning(str(args[0]))
        
        # RNG setup
        if not hasattr(cfg, "seed") or cfg.seed is None or cfg.seed == -1:
            self.cfg.seed = np.random.randint(0, np.iinfo(np.int32).max)
            self.logger.info(f"Seed not specified, using random seed: {self.cfg.seed}.")
        else:
            self.logger.info(f"Using seed: {self.cfg.seed}.")

        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed_all(self.cfg.seed) if torch.cuda.is_available() else None
        set_seed(self.cfg.seed)

        if torch.cuda.is_available():
            torch.cuda.init()

        if cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device)

        if cfg.get("use_bfloat16", False):
            self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        else:
            self.dtype = torch.float16

        self.logger.info(f"Using device: {self.device} with dtype: {self.dtype}.")
        if torch.cuda.is_available() and ('cuda' in cfg.device or 'auto' in cfg.device):
            self.logger.info(f"Device name: {torch.cuda.get_device_name()}.")
        
        self.cache_dir = self.cfg.model.get("cache_dir", os.environ.get("HF_HOME", None))
        if self.cache_dir == "":
            self.cache_dir = os.environ.get("HF_HOME", None)
        
        if self.cache_dir is not None:
            os.environ['HF_HOME'] = self.cache_dir 
            os.environ['TRANSFORMERS_CACHE'] = os.environ['HF_HOME']
        self.logger.info(f"Cache dir: {os.environ.get('HF_HOME', None)}.")
        
        # Experiment settings
        self.data_folder = cfg.experiment.function.train_points.get("data_folder", None)
        self.data_folder = os.path.join(self.root_dir, self.data_folder) if self.data_folder is not None else None

        if cfg.experiment.function.train_points.generate_points:
            self.min_train_points = cfg.experiment.function.train_points.min_points
            self.max_train_points = cfg.experiment.function.train_points.max_points
            self.num_train_points = cfg.experiment.function.train_points.num_points
            self.xs_noise_std = cfg.experiment.function.train_points.xs_noise_std
            self.ys_noise_std = cfg.experiment.function.train_points.ys_noise_std
            self.random_train_points = self.cfg.experiment.function.train_points.random_points \
                if hasattr(self.cfg.experiment.function.train_points, "random_points") \
                and self.cfg.experiment.function.train_points.random_points else False
        else:
            assert self.data_folder is not None, "No data folder specified."
            assert os.path.exists(self.data_folder), f"Data folder {self.data_folder} does not exist."
            assert os.path.exists(os.path.join(self.data_folder, 'train_points.npy')), f"Train points file {os.path.join(self.data_folder, 'train_points.npy')} does not exist."
            assert os.path.exists(os.path.join(self.data_folder, 'test_points.npy')), f"Test points file {os.path.join(self.data_folder, 'test_points.npy')} does not exist."
            
            train_points_file = os.path.join(self.data_folder, 'train_points.npy')
            self.train_points = utils.load_points(train_points_file)
            self.num_train_points = len(self.train_points)
            self.min_train_points = np.min(self.train_points)
            self.max_train_points = np.max(self.train_points)
            self.logger.info(f"Loaded train points from {train_points_file}.")
            
            test_points_file = os.path.join(self.data_folder, 'test_points.npy')
            self.test_points = utils.load_points(test_points_file)
            self.num_test_points = len(self.test_points)
            self.min_test_points = np.min(self.test_points)
            self.max_test_points = np.max(self.test_points)
            self.logger.info(f"Loaded test points from {test_points_file}.")

        self.tolerance = cfg.experiment.function.tolerance
        self.num_variables = cfg.experiment.function.num_variables
        if self.num_variables > 2 and self.visual_model:
            self.logger.error("Visual models only support up to 2 variables.")
            exit(1)
        
        self.iterations = cfg.experiment.function.iterations
        self.max_retries = cfg.max_retries
        self.force_valid = cfg.force_valid
        self.force_unique = cfg.force_unique
        self.checkpoints = cfg.checkpoints

        if "test_function" not in cfg.experiment.function:
            self.logger.info("Test function is not known.")
            self.test_function = None
        else:
            self.test_function_name = cfg.experiment.function.test_function
            self.test_function = utils.string_to_function(self.test_function_name, self.num_variables)

        # Points setup
        if cfg.experiment.function.train_points.generate_points:
            add_extremes = cfg.experiment.function.train_points.add_extremes if hasattr(cfg.experiment.function.train_points, "add_extremes") else False
            self.train_points = self.generate_points(self.test_function, self.min_train_points, self.max_train_points, self.num_train_points,
                                                                                xs_noise_std=self.xs_noise_std, ys_noise_std=self.ys_noise_std, 
                                                                                random_points=self.random_train_points, save_fig=True, add_extremes=add_extremes)
            np.save(os.path.join(self.output_path, "train_points.npy"), self.train_points)

            self.min_test_points = cfg.experiment.function.test_points.min_points if hasattr(cfg.experiment.function.test_points, "min_points") else self.min_train_points
            self.max_test_points = cfg.experiment.function.test_points.max_points if hasattr(cfg.experiment.function.test_points, "max_points") else self.max_train_points
            self.num_test_points = cfg.experiment.function.test_points.num_points
            self.test_points = self.generate_points(self.test_function, self.min_test_points, self.max_test_points, self.num_test_points, random_points=False, save_fig=False)
            np.save(os.path.join(self.output_path, "test_points.npy"), self.test_points)
        
        if cfg.experiment.get("normalize_points", False):
            self.train_points = utils.normalize_points(self.train_points, cfg.experiment.normalize_method, cfg.experiment.normalize_percentile)
            self.logger.info(f"Normalized train points: {self.train_points}.")
        
        self.logger.info(f"Train points: {utils.array_to_string(self.train_points)}.")
        if self.num_test_points > 100:
            self.logger.info(f"Not logging test points as there are more than 100 ({self.num_test_points}).")
        else:
            self.logger.info(f"Test points: {utils.array_to_string(self.test_points)}.")

        # Optimizer settings
        self.optimizer = Optimizer(cfg, self.train_points, self.logger)

        # Plotter setup
        if self.num_variables > 2:
            self.logger.warning("Plotter will not plot points and animation as there are more than 2 variables.")
        self.save_frames = cfg.plotter.save_frames if hasattr(cfg.plotter, "save_frames") else False
        if self.save_frames:
            os.makedirs(self.output_path + "frames/")
        
        self.save_video = cfg.plotter.save_video if hasattr(cfg.plotter, "save_video") else True
        self.save_video = False if self.num_variables > 2 else self.save_video
        if self.save_video:
            plt.rcParams.update({'figure.max_open_warning': cfg.experiment.function.iterations + 5})
        self.plotter = Plotter(cfg, self.train_points, self.test_points, self.output_path)
        self.plotter.plot_points(save_fig=True, plot_test=False)

        # Base prompt
        self.prompts_path = os.path.join(self.root_dir, cfg.prompts_path)
        self.prompt_size = cfg.model.base_prompt.prompt_size
        with open(os.path.join(self.prompts_path, "OPRO", cfg.model.base_prompt.prompt), "r") as f:
            self.base_prompt = f.read()
            
        self.prompt_points = utils.decimate_points(self.train_points, cfg.max_points_in_prompt)
        self.prompt_points = utils.array_to_string(self.prompt_points)
        if self.num_train_points > self.cfg.max_points_in_prompt:
            self.logger.info(f"Found {self.num_train_points} train points, decimated to {cfg.max_points_in_prompt} for the prompt.")
            self.logger.info(f"Prompt points: {self.prompt_points}.")
            
        self.base_prompt = self.base_prompt.format(points=self.prompt_points, num_variables=self.num_variables, 
                                                    variables_list=[f"x{i+1}" for i in range(self.num_variables)], functions="{functions}")

        # Model settings
        self.model_name = cfg.model.name
        self.model = None

        self.visual_model = cfg.model.visual
        if hasattr(cfg.model.base_prompt, "input_image"):
            self.input_img = cfg.model.base_prompt.input_image
            valid_inputs = ["points", "previous_guess", "best_guess", "all_guesses"]
            if self.input_img not in valid_inputs:
                self.logger.error(f"Input image {self.input_img} not supported. Valid inputs are {valid_inputs}.")
                exit(1)
        else:
            self.input_img = None

        self.logger.info(f"Base Prompt: {self.base_prompt} with input image {self.input_img}.")
        self.tokenizer_pad = cfg.model.tokenizer_pad
        self.tokenizer_padding_side = cfg.model.tokenizer_padding_side

        self.max_new_tokens = cfg.model.max_new_tokens
        self.top_p = cfg.model.top_p
        self.top_k = cfg.model.top_k
        self.num_beams = cfg.model.num_beams

        self.temperature = cfg.model.temperature
        if cfg.model.temperature_schedule:
            self.temperature_scheduler = torch.optim.lr_scheduler.ExponentialLR(torch.optim.Adam([torch.tensor(self.temperature)], lr=1), gamma=cfg.model.temperature_schedule_gamma)
        else:
            self.temperature_scheduler = None
            
        model_args = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "num_beams": self.num_beams,
            "max_length": self.max_new_tokens,
            "min_length": 0,
            "tokenizer_pad": self.tokenizer_pad,
            "tokenizer_padding_side": self.tokenizer_padding_side,
            "seed": self.cfg.seed,
            "api_key_path": os.path.join(self.root_dir, cfg.model.api_key_path) if hasattr(cfg.model, "api_key_path") else None,
            "organization_id_path": os.path.join(self.root_dir, cfg.model.organization_id_path) if hasattr(cfg.model, "organization_id_path") else None,
        }
        if torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name(0):
            model_args['attn_implementation'] = 'flash_attention_2'
            model_args['use_flash_attn'] = True
            self.logger.info("Using Flash Attention 2")
        
        self.model = utils.load_model(self.model_name, self.device, self.dtype, self.cache_dir, model_args)
        self.logger.info("Model loaded - {model_name}.".format(model_name=self.model_name))

        # Scorer settings
        if "basic" in cfg.experiment.scorer.name.lower():
            self.scorer = BasicScorer(self.train_points, rounding=cfg.experiment.scorer.rounding, scientific=cfg.experiment.scorer.scientific)
            self.test_scorer = BasicScorer(self.test_points, rounding=cfg.experiment.scorer.rounding, scientific=cfg.experiment.scorer.scientific)
        elif "minmax" in cfg.experiment.scorer.name.lower():
            min_score = cfg.experiment.scorer.min_score
            max_score = cfg.experiment.scorer.max_score
            self.scorer = MinMaxScorer(self.train_points, min_score=min_score, max_score=max_score, rounding=cfg.experiment.scorer.rounding, scientific=cfg.experiment.scorer.scientific)
            self.test_scorer = MinMaxScorer(self.test_points, min_score=min_score, max_score=max_score, rounding=cfg.experiment.scorer.rounding, scientific=cfg.experiment.scorer.scientific)
        elif "complexity" in cfg.experiment.scorer.name.lower():
            self.logger.info(f"Complexity scorer with lambda {cfg['experiment']['scorer']['lambda']} and max nodes {cfg.experiment.scorer.max_nodes}.")
            alternative = False
            if hasattr(cfg.experiment.scorer, "alternative") and cfg.experiment.scorer.alternative:
                alternative = True
                self.logger.info("Using alternative complexity scorer.")
            self.scorer = ComplexityScorer(self.train_points, rounding=cfg.experiment.scorer.rounding, scientific=cfg.experiment.scorer.scientific, lam=cfg['experiment']['scorer']['lambda'], max_nodes=cfg.experiment.scorer.max_nodes, alternative=alternative)
            self.test_scorer = ComplexityScorer(self.test_points, rounding=cfg.experiment.scorer.rounding, scientific=cfg.experiment.scorer.scientific, lam=cfg['experiment']['scorer']['lambda'], max_nodes=cfg.experiment.scorer.max_nodes, alternative=alternative)
        else:
            self.logger.error(f"Scorer {cfg.experiment.scorer.name} not supported.")
            exit(1)

        # Seed functions
        self.seed_functions = {}
        min_seed_functions = max(5, self.prompt_size) # If the prompt size is small (e.g. 1) we still want to generate a few seed functions to avoid getting stuck
        gen_time = 0
        if cfg.experiment.generate_seed_functions:
            self.seed_functions, gen_time = self.generate_seed_functions()
            assert len(self.seed_functions) >= min_seed_functions, f"Could not generate {min_seed_functions} seed functions. Generated {len(self.seed_functions)} seed functions."
        else:
            self.seed_functions = {name: utils.string_to_function(name, self.num_variables) for name in cfg.experiment.seed_functions.functions}
            self.logger.info(f"Loaded seed functions: {self.seed_functions}.")
        self.current_functions = CurrentFunctions(self.seed_functions, self.scorer, self.optimizer, self.prompt_size, self.logger, self.num_variables)
        self.logger.info(f"Current functions: {self.current_functions.functions}.")
        self.logger.info(f"Current scores: {self.current_functions.scores}.")

        if len(self.current_functions.functions) < self.prompt_size:
            self.logger.warning(f"Could not generate {self.prompt_size} seed functions. Generated {len(self.current_functions.functions)} seed functions.")
            if len(self.current_functions.functions) == 0:
                self.logger.error("No seed functions generated. Exiting.")
                exit(1)
        else:
            self.logger.info(f"Succesfully generated {self.prompt_size} seed functions in {gen_time} seconds.")

        # Results json
        self.results = {
            "experiment_name": self.cfg.experiment.function.name,
            "seed": self.cfg.seed,
            "train_points": utils.array_to_string(self.train_points),
            "test_points": utils.array_to_string(self.test_points),
            "best_expr": "",
            "best_function": "",
            "scores": [],
            "R2_trains": [],
            "R2_tests": [],
            "R2_alls": [],
            "best_scores": [],
            "best_scores_normalized": [],
            "iterations": 0,
            "tries_per_iteration": [],
            "generations_per_iteration": [],
            "num_unique": len(self.current_functions.functions),
            "best_found_at": 0,
            "sympy_equivalent": False,
            "temperatures": [],
            "times": {
                "iteration": [],
                "seed_function_generation": gen_time,
                "generation_per_iteration": [],
                "optimization_per_iteration": [],
            }
        }
        if "test_function" in self.cfg.experiment.function:
            self.results["test_function"] = self.cfg.experiment.function.test_function

        # Save config
        with open(self.output_path + "config.yaml", "w") as f:
            OmegaConf.save(self.cfg, f)

    def generate_points(self, function: Callable, min_points: float, max_points: float, num: int, xs_noise_std: float = 0, ys_noise_std: float = 0, 
                        random_points: bool = False, add_extremes: bool = True) -> str:
        """ 
        Generates points from a given function, with optional noise.

        Parameters
        ----------
        function -> the function to generate points from.
        min_points -> the minimum value of the points to generate.
        max_points -> the maximum value of the points to generate.
        num -> the number of points to generate.
        xs_noise_std -> the standard deviation of the noise to add to the xs.
        ys_noise_std -> the standard deviation of the noise to add to the ys.
        random_points -> whether to generate random points instead of a grid/meshgrid.
        add_extremes -> whether to add points at the extreme values of the interval manually to ensure they are included.

        Returns
        -------
        points -> the points as a string.
        """
        min_value = copy.deepcopy(min_points)
        max_value = copy.deepcopy(max_points)
        if type(min_points) != list and type(min_points) != listconfig.ListConfig:
            min_points = [min_points] * self.num_variables
        if type(max_points) != list and type(max_points) != listconfig.ListConfig:
            max_points = [max_points] * self.num_variables
        min_points = np.array(min_points, dtype=np.float32)
        max_points = np.array(max_points, dtype=np.float32)
        
        points_per_dim = int(np.floor(num**(1/self.num_variables)))
        self.logger.info(f"Generating {points_per_dim} points per dimension for a total of {points_per_dim**self.num_variables} points.")
        
        if random_points:
            # Add points at the extreme values of the interval manually to ensure they are included
            # This depends on the number of dimensions
            # For example, in 1D if the interval is [0, 1] we need to add points at 0 and 1
            # In 2D, if the interval is [(0, 0), (1, 1)] we need to add points at (0, 0), (0, 1), (1, 0), (1, 1)
            if add_extremes:
                variable_ranges = np.array([[min_points[i], max_points[i]] for i in range(self.num_variables)])
                extreme_points = np.array(np.meshgrid(*variable_ranges)).T.reshape(-1, self.num_variables)
                self.logger.info(f"Adding {len(extreme_points)} extreme points ({extreme_points}).")
            
            # Reshape min and max points to match the random shape. Currently min and max are of shape (num_variables,), so we need to add n dimensions of size points_per_dim by copying the min and max values
            # For example, if min is [0, 1] and num_variables is 2 and points_per_dim is 3, we need to reshape min to an array of shape (2, 3, 3) with all values being 0 and 1 across the last dimension
            random_shape = tuple([self.num_variables, *([points_per_dim] * self.num_variables)])
            min_points = np.expand_dims(min_points, axis=tuple(range(1, self.num_variables + 1)))
            max_points = np.expand_dims(max_points, axis=tuple(range(1, self.num_variables + 1)))
            max_points += 1e-10 # Add small eps to max_points as the rightmost value is not included in np.random.uniform
            for i in range(1, self.num_variables+1):
                min_points = np.repeat(min_points, points_per_dim, axis=i)
                max_points = np.repeat(max_points, points_per_dim, axis=i)
            Xs = np.random.uniform(min_points, max_points, random_shape)
            
        else:
            Xs = np.meshgrid(*[np.linspace(min_points[i], max_points[i], points_per_dim) for i in range(self.num_variables)])
        Xs = np.array(Xs)
        if xs_noise_std:
            Xs += np.random.normal(0, xs_noise_std, Xs.shape)
        pts = np.array(list(zip(*[x.flat for x in Xs])))

        ys = utils.eval_function(function, pts, self.num_variables).T
            
        if ys_noise_std:
            ys += np.random.normal(0, ys_noise_std, ys.shape)
        
        if random_points and add_extremes:
            pts = np.concatenate((extreme_points, pts))
            extreme_ys = utils.eval_function(function, extreme_points, self.num_variables).T
            ys = np.concatenate((extreme_ys, ys))
        
        points = np.concatenate((pts, ys.reshape(-1, 1)), axis=1)
        if add_extremes and len(points) > num:
            # Remove random points to account for the extra extremes
            # The points are sampled randomly so removing from the end is the same as removing random indices
            self.logger.info(f"Removing {len(points)-num} randomly generated points: {points[num:]}")
            points = points[:num]
        while any(np.isinf(points[:, -1])):
            # Remove points where the function is infinite
            inf_indices = np.where(np.isinf(points))
            self.logger.info(f"Removing {len(inf_indices)} points where the function is infinite.")
            points = np.delete(points, inf_indices[0], axis=0)
            
            if len(points) < num and random_points:
                # Generate new points to replace the infinite ones
                self.logger.info(f"Recursively generating {num-len(points)} new points.")
                new_points = self.generate_points(function, min_value, max_value, num-len(points), xs_noise_std, ys_noise_std, random_points, add_extremes=False)
                points = np.concatenate((points, new_points))

        return points
    
    def generate_seed_functions(self) -> Tuple[Dict[str, Any], float]:
        """
        Generates initial seed functions for the experiment.

        Parameters
        ----------

        Returns
        -------
        seed_functions -> the generated seed functions.
        gen_time -> the time it took to generate the seed functions.
        """
        generation_tokens = self.cfg.experiment.seed_functions.generation_tokens if hasattr(self.cfg.experiment.seed_functions, "generation_tokens") else 512
        max_tries = self.cfg.experiment.seed_functions.max_tries if hasattr(self.cfg.experiment.seed_functions, "max_tries") else 10
        seed_functions = {}
        
        seed_prompt = self.cfg.get("model").get("seed_function_prompt", None)
        assert seed_prompt is not None, "Seed function prompt not specified."
        seed_prompt = os.path.join(self.prompts_path, seed_prompt)
        
        with open(seed_prompt, "r") as f:
            prompt = f.read()
        img_path = os.path.join(self.output_path, "points.png") if self.input_img else None

        prompt = prompt.format(points=self.prompt_points, num_variables=self.num_variables, variables_list=[f"x{i+1}" for i in range(self.num_variables)])
        self.logger.info("Prompt for seed functions generation:")
        self.logger.info(prompt)
        
        start_time = time.perf_counter()
        with torch.inference_mode():
            for i in range(max_tries):
                # Generate seed functions using the model
                self.logger.info(f"Attempt {i+1} of {max_tries} to generate seed functions.")
                seeds = self.model.generate(prompt, return_prompt=False, image_files=img_path, temperature=self.temperature, max_new_tokens=generation_tokens)
                self.logger.info("Model output for seed functions: " + seeds)

                # Parse model output
                for seed in seeds.split("\n"):
                    if "x" not in seed:
                        self.logger.info(f"Skipping line {seed} as it does not contain 'x' and is likely not a function.")  
                        continue
                    if "Error" in seed:
                        self.logger.info(f"Skipping line {seed} as it contains 'Error'.")
                        continue
                    seed = utils.clean_function(seed)
                    self.logger.info(f"Seed function: {seed}.")
                    if seed == "":
                        continue
                    try:
                        valid, reason = utils.is_valid_function(seed, None, self.num_variables)
                        self.logger.info(f"Function {seed}. Valid: {valid}. Reason: {reason}.")
                        if valid:
                            function = utils.string_to_function(seed, self.num_variables)
                            seed_functions[seed] = function
                    except Exception as e:
                        self.logger.warning(f"Could not parse line {seed}.")
                        self.logger.warning(str(e))
                        pass
                # Here we continue even if we already have enough seed functions, as we might not have enough valid seed functions after optimization
                # Perhaps a better approach should be optimizing here directly and exiting if we have enough valid seed functions
        end_time = time.perf_counter()

        self.logger.info(f"Generated seed functions: {seed_functions}.")
        return seed_functions, end_time - start_time


    def get_new_function(self, prompt: str) -> Tuple[List, bool]:
        """
        Generates a new function from the model, given a prompt.

        Parameters
        ----------
        prompt -> the prompt to use for the model.

        Returns
        -------
        functions -> the new functions generated by the model as a string.
        found_valid -> whether a valid function was found.
        """
        img = None
        if self.visual_model:
            if self.input_img == "points":
                img = os.path.join(self.output_path, "points.png")
            elif self.input_img == "previous_guess":
                img = os.path.join(self.output_path, "frames", f"{self.results['iterations']-1}.png")
            elif self.input_img == "best_guess":
                fig, ax = self.plotter.plot_results(self.current_functions.get_best_function(return_coeff=False), self.test_function, plot_true=False)
                fig.savefig(self.output_path + "best_guess.png")
                plt.close(fig)
                img = os.path.join(self.output_path, "best_guess.png")
            elif self.input_img == "all_guesses":
                os.makedirs(self.output_path + "prompt_input/")
                path = os.path.join(self.output_path, "prompt_input/")
                img = []
                functions = self.current_functions.get_prompt_functions()
                for expr, _ in functions:
                    try:
                        function = self.current_functions.functions[expr]
                        fig, ax = self.plotter.plot_results(function, self.test_function, plot_true=False, label="Function: " + str(function))
                        function_string = str(expr)
                        fig.suptitle("Plot of " + function_string)
                        fig.text(0.5, 0.90, "Error: " + str(self.current_functions.norm_scores[expr]), ha='center')
                        file_name = function_string.replace(" ", "_").replace("/", "div")
                        fig.savefig(os.path.join(path, f"{file_name}.png"))
                        plt.close(fig)
                        img.append(os.path.join(path, f"{file_name}.png"))
                    except Exception as e:
                        self.logger.warning(f"Could not plot function {function}.")
                        self.logger.warning(str(e))
                        pass

        new_output = self.model.generate(prompt, return_prompt=False, image_files=img, temperature=self.temperature)
        self.logger.info("Prompt: " + prompt)
        self.logger.info("Model output: " + new_output)

        # Clean up images
        if self.visual_model and self.input_img == "best_guess":
            os.remove(self.output_path + "best_guess.png")
        elif self.visual_model and self.input_img == "all_guesses":
            for file in os.listdir(path):
                os.remove(os.path.join(path, file))
            os.rmdir(path)

        functions = []
        lines = new_output.split("\n")
        for line in lines:
            if "x" not in line:
                    self.logger.info(f"Skipping line {line} as it does not contain 'x' and is likely not a function.")  
                    continue
            if "Error" in line:
                self.logger.info(f"Skipping line {line} as it contains 'Error'.")
                continue
            line = utils.clean_function(line)
            if line == "":
                continue
            self.logger.info("Cleaned line: " + line + ".")
            try:
                valid, reason = utils.is_valid_function(line, self.current_functions, self.num_variables)
                self.logger.info(f"Valid: {valid}. Reason: {reason}.")
                if valid:
                    functions.append(line)
                elif not valid and reason == "Function already in prompt." and not self.force_unique:
                    functions.append(line)
            except Exception as e:
                self.logger.warning(f"Could not parse line {line}.")
                self.logger.warning(str(e))
                pass
        
        found_valid = False
        if len(functions) == 0:
            self.logger.warning("Could not find a valid function in the output. Using the last function in the output.")
            functions = [self.current_functions.get_best_function()]
        else:
            found_valid = True
            functions = [utils.string_to_function(function, self.num_variables) for function in functions]
            self.logger.info(f"Found functions: {functions}.")

        return functions, found_valid

    def get_new_function_and_score(self, prompt: str) -> Tuple[Any, Any, float]:
        """
        Generates a new function from the model, given a prompt, and scores it if it is valid.

        Parameters
        ----------
        prompt -> the prompt to use for the model.

        Returns
        -------
        expression -> the coefficient form of the function.
        function -> function with the optimized coefficients.
        score -> the score of the function.
        """
        valid = False
        start_time = time.perf_counter()
        for i in range(self.max_retries):
            self.logger.info(f"Attempt {i+1} of {self.max_retries} to find a valid function.")
            functions, valid = self.get_new_function(prompt)
            if valid and len(functions) > 1:
                self.logger.info(f"Found {len(functions)} functions in the output.")
                break
            else:
                self.logger.info(f"Could not find a valid function in the output. Trying again.")

        self.results["tries_per_iteration"].append(i+1)
        self.results["times"]["generation_per_iteration"].append(time.perf_counter() - start_time)

        if not valid:
            if self.force_valid:
                self.logger.error(f"Could not find a valid function after {self.max_retries} tries. Exiting.")
                exit(1)
            else:
                best_expr = self.current_functions.get_best_function(return_coeff=True)
                best_function = self.current_functions.functions[best_expr]
                best_score = self.current_functions.scores[best_expr]
                self.logger.warning(f"Could not find a valid function after {self.max_retries} tries. Using {best_function}.")
        else:
            best_score = np.inf
            start_time = time.perf_counter()
            for function in functions:
                if not self.current_functions.func_in_list(function):
                    self.results["num_unique"] += 1
                self.results["generations_per_iteration"].append(len(functions))
                try:
                    opt_function, exp = self.optimizer.optimize(function, return_coeff=True)
                    score = self.scorer.score(opt_function)
                    self.logger.info(f"New function: {str(opt_function)}. Score: {score}.")
                    if score < best_score:
                        best_score = score
                        best_function = opt_function
                        best_expr = exp
                except Exception as e:
                    self.logger.warning(f"Could not optimize function {function}. {e}")
                    pass

            self.results["times"]["optimization_per_iteration"].append(time.perf_counter() - start_time)
            self.logger.info(f"Optimizer time: {time.perf_counter() - start_time}.")
            if best_score == np.inf:
                self.logger.warning(f"No functions scored below inf. Using the best function in the prompt.")
                best_expr = self.current_functions.get_best_function(return_coeff=True)
                best_function = self.current_functions.functions[best_expr]
                best_score = self.current_functions.scores[best_expr]
            self.logger.info(f"Best function: {best_function}. Score: {best_score}.")

        self.logger.info(f"Finished get new function and score. Best function: {best_function}. Score: {best_score}.")
        return best_expr, best_function, best_score
    
    def get_R2_scores(self, function: Any) -> Tuple[float, float, float]:
        """
        Computes the R2 scores for the train and test sets given a function, removing the 5% worst predictions.
        
        Parameters
        ----------
        function -> the function to evaluate.
        
        Returns
        -------
        r2_train -> the R2 score for the train set.
        r2_test -> the R2 score for the test set.
        r2_all -> the R2 score for all points.
        """
        
        y_true_train = self.train_points[:, -1]
        y_true_test = self.test_points[:, -1]
        
        def compute_predictions(function, points, num_variables):
            y_pred = utils.eval_function(function, points[:, 0:-1], num_variables)
            
            # Compute a boolean mask of the 5% worst predictions
            worst_indices = np.argsort(np.abs(y_pred - points[:, -1]))[-int(len(points) * 0.05):]
            mask = np.zeros(len(points), dtype=bool)
            mask[worst_indices] = True
            
            return y_pred[~mask], mask
        
        try:
            y_pred_train, y_train_mask = compute_predictions(function, self.train_points, self.num_variables)
            y_pred_test, y_test_mask = compute_predictions(function, self.test_points, self.num_variables)
            
            y_true_train = y_true_train[~y_train_mask]
            y_true_test = y_true_test[~y_test_mask]
            
            assert len(y_true_train) == len(y_pred_train), f"Length of true train points ({len(y_true_train)}) does not match length of predicted train points ({len(y_pred_train)})."
            assert len(y_true_test) == len(y_pred_test), f"Length of true test points ({len(y_true_test)}) does not match length of predicted test points ({len(y_pred_test)})."
            
        except Exception as e:
            self.logger.warning(f"Could not evaluate function {function}. {e}")
            return np.nan, np.nan, np.nan
        try:
            r2_train = r2_score(y_true_train, y_pred_train)
        except Exception as e:
            self.logger.warning(f"Could not calculate R2 score for train set. {e}")
            r2_train = np.nan
        try:
            r2_test = r2_score(y_true_test, y_pred_test)
        except Exception as e: 
            self.logger.warning(f"Could not calculate R2 score for test set. {e}")
            r2_test = np.nan
        try:
            r2_all = r2_score(np.concatenate((y_true_train, y_true_test)), np.concatenate((y_pred_train, y_pred_test)))
        except Exception as e:
            self.logger.warning(f"Could not calculate R2 score for all points. {e}")
            r2_all = np.nan

        return r2_train, r2_test, r2_all

    def run(self) -> None:
        """
        Runs the main experiment, iterating and generating new functions until the tolerance is reached.
        """
        main_timer_start = time.perf_counter()
        if self.save_video:
            frames = []

        # Check if one of the generated seed functions is already below the tolerance
        best_expr = self.current_functions.get_best_function(return_coeff=True)
        best_function = self.current_functions.get_best_function(return_coeff=False)
        score = self.current_functions.scores[best_expr]
        r2_train, r2_test, r2_all = self.get_R2_scores(best_function)
        
        if r2_train >= self.tolerance:
            self.logger.info(f"The seed function {best_expr} is already above the R2 tolerance {self.tolerance}.")
            self.logger.info(f"Best function: {best_function}. R2 (train): {r2_train}.")

            self.results["scores"].append(score) if score != np.inf else self.results["scores"].append("inf")
            self.results["best_scores"].append(self.current_functions.scores[best_expr])
            self.results["best_scores_normalized"].append(self.current_functions.norm_scores[best_expr])
            self.results["best_found_at"] = 0
            self.results["temperatures"].append(self.temperature)
            
            if self.save_video:
                frame, ax = self.plotter.record_frame(best_function, best_function, r2_test, self.test_function, -1, plot_true=True)
                if self.save_frames:
                    frame.savefig(self.output_path + "frames/" + f"{i}.png")
                frames.append(frame)
        else:
            # Start the main loop
            prompt = self.current_functions.get_prompt(self.base_prompt)
            for i in range(self.iterations):
                start_time = time.perf_counter()
                self.logger.info(f"Round {i+1}.")
                self.logger.info(f"Scores: {self.current_functions.scores}.")

                # Handle temperature schedule
                if self.temperature_scheduler is not None:
                    self.temperature = self.temperature_scheduler.get_last_lr()[0]
                    self.logger.info(f"Temperature: {self.temperature}.")
                    self.results["temperatures"].append(self.temperature)
                    self.temperature_scheduler.step()

                # Get new function and score
                expr, function, score = self.get_new_function_and_score(prompt)
                self.current_functions.add_function(expr, function)
                best_expr = self.current_functions.get_best_function(return_coeff=True)
                best_function = self.current_functions.functions[best_expr]

                # Update results
                if expr == best_expr:
                    self.results["best_found_at"] = i+1
                self.results["iterations"] = i+1
                self.results["scores"].append(score) if score != np.inf else self.results["scores"].append("inf")
                self.results["best_scores"].append(self.current_functions.scores[best_expr])
                self.results["best_scores_normalized"].append(self.current_functions.norm_scores[best_expr])
                self.results["times"]["iteration"].append(time.perf_counter() - start_time)
                
                r2_train, r2_test, r2_all = self.get_R2_scores(function)
                self.results["R2_trains"].append(r2_train)
                self.results["R2_tests"].append(r2_test)
                self.results["R2_alls"].append(r2_all)

                # Update video
                if self.save_video:
                    if not score == np.inf:
                        frame, ax = self.plotter.record_frame(best_function, function, r2_test, self.test_function, i, plot_true=True)
                        if self.save_frames:
                            frame.savefig(self.output_path + "frames/" + f"{i}.png")
                        frames.append(frame)
                    else:
                        self.logger.warning(f"Skipping frame {i} as the score is inf.")
                
                # Check if the tolerance is reached
                if self.test_function is not None:
                    if utils.func_equals(best_function, self.test_function, self.num_variables):
                        self.logger.info(f"Function is equivalent to the true function.")
                        self.results["equivalent"] = True
                        break
                
                if r2_train >= self.tolerance:
                    self.logger.info(f"Found a function with R2 (train) = {r2_train} above the tolerance {self.tolerance}.")
                    break
                prompt = self.current_functions.get_prompt(self.base_prompt)
                
                if i in self.checkpoints:
                    self.logger.info(f"Checkpoint {i}. Saving results.")
                    results_checkpoint = copy.deepcopy(self.results)
                    checkpoint_timer_end = time.perf_counter()
                    results_checkpoint["times"]["total"] = checkpoint_timer_end - main_timer_start
                    
                    best_expr = self.current_functions.get_best_function(return_coeff=True)
                    best_function = self.current_functions.get_best_function(return_coeff=False)
                    test_score = self.test_scorer.score(best_function)
                    results_checkpoint["test_score"] = test_score
                    results_checkpoint["best_function"] = str(best_function)
                    results_checkpoint["best_expr"] = str(best_expr)
                    r2_train, r2_test, r2_all = self.get_R2_scores(best_function)
                    results_checkpoint["r2_train"] = r2_train
                    results_checkpoint["r2_test"] = r2_test
                    results_checkpoint["r2_all"] = r2_all
                    results_checkpoint["final_complexity"] = utils.count_nodes(best_function)
                    with open(self.output_path + f"results_checkpoint_{i}.json", "w") as f:
                        json.dump(results_checkpoint, f)
                    self.logger.info(f"Checkpoint {i} saved.")

        # Save final results
        main_timer_end = time.perf_counter()
        best_expr = self.current_functions.get_best_function(return_coeff=True)
        best_function = self.current_functions.get_best_function(return_coeff=False)
        test_score = self.test_scorer.score(best_function)
        self.logger.info(f"Test score: {test_score}.")

        self.logger.info(f"Best function: {best_function}. Score: {self.current_functions.scores[best_expr]} ({self.current_functions.norm_scores[best_expr]}).")
        
        if hasattr(self, "test_function_name") and self.test_function_name is not None:
            self.logger.info(f"True function: {self.test_function_name}")
        fig, ax = self.plotter.plot_results(best_function, self.test_function)
        fig.savefig(self.output_path + "final.png")

        self.results["best_function"] = str(best_function)
        self.results["best_expr"] = str(best_expr)
        self.results["test_score"] = test_score
        self.results["times"]["total"] = main_timer_end - main_timer_start + self.results["times"]["seed_function_generation"]
        self.results["times"]["avg_generation"] = np.mean(self.results["times"]["generation_per_iteration"]) if len(self.results["times"]["generation_per_iteration"]) > 0 else 0
        self.results["times"]["avg_optimization"] = np.mean(self.results["times"]["optimization_per_iteration"]) if len(self.results["times"]["optimization_per_iteration"]) > 0 else 0

        r2_train, r2_test, r2_all = self.get_R2_scores(best_function)
        self.results["r2_train"] = r2_train
        self.results["r2_test"] = r2_test
        self.results["r2_all"] = r2_all
        self.logger.info(f"R2 train: {np.round(r2_train, 6)}. R2 test: {np.round(r2_test, 6)}. R2 all: {np.round(r2_all, 6)}.")
        
        final_complexity = utils.count_nodes(best_function)
        self.results["final_complexity"] = final_complexity
        self.logger.info(f"Number of nodes in final expression tree: {final_complexity}.")

        with open(self.output_path + "results.json", "w") as f:
            json.dump(self.results, f)

        if self.save_video and len(frames) > 0:
            self.plotter.record_video(frames)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    workspace = Workspace(cfg)
    workspace.run()

def dump_profile():
    profiler.disable()
    job_id = utils.get_job_id()
    print(f"Dumping profile to {os.path.join(os.getcwd(), 'profiles', 'profile')}_{job_id if job_id is not None else 'local'}")
    if not os.path.exists("./profiles"):
            os.makedirs("./profiles")
    profiler.dump_stats(f"./profiles/profile_{job_id if job_id is not None else 'local'}")

def signal_handler(sig, frame):
    # Ignore warnings, as otherwise we break the logger
    warnings.filterwarnings("ignore")
    dump_profile()
    print(f"Detecting signal {sig}. Dumping profile to {os.path.join(os.getcwd(), 'profiles', 'profile')}_{job_id if job_id is not None else 'local'}")
    sys.stdout.flush()
    if sig == signal.SIGTERM or sig == signal.SIGINT:
        sys.exit(1)

if __name__ == "__main__":
    # Run full profiler if env variable PROFILE is set
    do_profile = os.environ.get("PROFILE", False)
    print("Initializing profiler.")
    print("Profile will only be created if the code fails or is terminated.") if not do_profile else print("Profile will be created.")
    job_id = utils.get_job_id()

    # Set termination signal handlers to dump profile when terminated by SLURM
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGCONT, signal_handler)
    
    # Setup profiler
    global profiler
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        main()
    except Exception as e:
        # Catch exceptions and dump profile
        print("Caught exception in main.")
        print(e)
        dump_profile()
        sys.exit(2)

    if do_profile:
        dump_profile()