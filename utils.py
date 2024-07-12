import os
import re
import numpy as np
import pandas as pd

from torch import dtype, device
from torch.cuda import get_device_name, is_available
from collections.abc import Callable
from typing import Tuple, List, Dict, Any

from sklearn.metrics import mean_squared_error, r2_score
from models import LLaVaModelHF, HuggingFaceModel, OpenAIModel

import sympy
from sympy.parsing.sympy_parser import parse_expr

def get_job_id() -> str:
    """
    Gets the SLURM job id from the environment variables if available.
    
    Returns
    -------
    job_id -> The SLURM job id (or None if not available).
    """
    
    job_id = os.environ.get("SLURM_JOB_ID", None)
    if job_id is not None:
        job_id += "_" + os.environ.get("SLURM_ARRAY_TASK_ID", None) if "SLURM_ARRAY_TASK_ID" in os.environ else ""
        
    return job_id

def load_model(model_name: str, device: device, dtype: dtype, cache_dir: str = None, model_args = None) -> Any:
    """
    Utility to load a model from the HuggingFace model hub.
    Mostly needed to deal with LLaVA models, that are not available on the model hub yet.

    Parameters
    ----------
    model_name -> the name of the model to load.
    device -> the device to load the model on.
    dtype -> the dtype to load the model with.
    cache_dir -> the cache directory to use for the model.

    Returns
    -------
    model -> the loaded model.
    """ 
    if 'llava' in model_name:
        model = LLaVaModelHF(model_name, device, dtype, cache_dir, **model_args)
    elif 'gpt' in model_name:
        model = OpenAIModel(model_name, device, dtype, cache_dir, **model_args)
    else:
        model = HuggingFaceModel(model_name, device, dtype, cache_dir, **model_args)

    return model

def get_messages(prompt: str, splits: List[str] = ["system", "user"]) -> List[Dict[str, str]]:
    """
    Converts a prompt string into a list of messages for each split.
    
    Parameters:
        prompt (str): The prompt string.
        splits (list[str]): A list of the splits to parse. Defaults to ["system", "user"].
        
    Returns:
        list[dict[str, str]]: A dictionary of the messages for each split.
    """
    
    messages = []
    for split in splits:
        start_tag = f"<{split}>"
        end_tag = f"</{split}>"

        start_idx = prompt.find(start_tag)
        end_idx = prompt.find(end_tag)
        
        # Skip if the split is not in the prompt (e.g. no system prompt)
        if start_idx == -1 or end_idx == -1:
            continue
        messages.append({
            "role": split,
            "content": prompt[start_idx + len(start_tag):end_idx].strip()
        })
    
    # If no splits at all, assume the whole prompt is a user message
    if len(messages) == 0:
        messages.append({
            "role": "user",
            "content": prompt
        })

    return messages

def load_points(file_path: str) -> np.ndarray:
    """
    Loads a set of points from a file.

    Parameters
    ----------
    file_path -> the path to the file containing the points.

    Returns
    -------
    points -> the points.
    """
    if file_path.endswith(".npy"):
        points = np.load(file_path)
    elif file_path.endswith(".txt"):
        points = np.loadtxt(file_path)
    elif file_path.endswith(".csv"):
        points = pd.read_csv(file_path).values
    elif file_path.endswith(".tsv"):
        points = pd.read_csv(file_path, sep="\t").values
    else:
        raise ValueError("Invalid file format. (only .npy, .txt, .csv, and .tsv are supported)")
    return points

def normalize_points(points: np.ndarray, method: str = "minmax", percentile: int = None) -> np.ndarray:
    """
    Normalizes a set of points.

    Parameters
    ----------
    points -> the points to normalize.
    method -> the normalization method to use. (minmax, zscore, percentile)
    percentile -> the percentile to use for percentile normalization (if applicable).

    Returns
    -------
    points -> the normalized points.
    """
    if method == "percentile" and percentile is None:
        raise ValueError("Percentile normalization requires a percentile value.")
    
    ys = np.array([point[-1] for point in points])
    if method == "minmax":
        points = np.array([np.concatenate([point[:-1], [(y - ys.min()) / (ys.max() - ys.min())]]) for point, y in zip(points, ys)])
    elif method == "zscore":
        points = np.array([np.concatenate([point[:-1], [(y - ys.mean()) / ys.std()]]) for point, y in zip(points, ys)])
    elif method == "percentile":
        points = np.array([np.concatenate([point[:-1], [y /np.percentile(ys, percentile)]]) for point, y in zip(points, ys)])
    else:
        raise ValueError("Invalid normalization method.")

    points = np.round(points, 4)
    return points

def decimate_points(points: np.ndarray, max_points: int) -> np.ndarray:
    """
    Reduces the number of points to a maximum number to be used in the prompt.
    
    Parameters
    ----------
    points -> the points to decimate.
    max_points -> the maximum number of points to keep.
    
    Returns
    -------
    points -> the decimated points.
    """
    
    if points.shape[0] <= max_points:
        return points
    
    # Find an evenly spaced subset of points
    indices = np.linspace(0, points.shape[0] - 1, max_points, dtype=int)
    points = points[indices]
    return points
    
def split_points(points: np.ndarray, test_fraction: float, split_strategy: str = "random", seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits a set of points into train and test sets.

    Parameters
    ----------
    points -> the points to split.
    test_fraction -> the fraction of points to use for the test set.
    split_strategy -> the strategy to use for splitting the points. (random, middle, end)
    seed -> the seed to use for the random split.

    Returns
    -------
    train_points -> the train points.
    test_points -> the test points.
    """
    num_points = points.shape[0]
    num_test_points = int(num_points * test_fraction)
    points = points[points[:, 0].argsort()]

    if seed is not None:
        np.random.seed(seed)

    #! Middle and end are not working properly with n_variables > 1, not fixed as unused in final version
    if split_strategy == "random":
        indices = np.random.choice(num_points, num_test_points, replace=False)
        mask = np.ones(num_points, dtype=bool)
        mask[indices] = False
        train_points = points[mask]
        test_points = points[~mask]
    elif split_strategy == "middle":
        start = (num_points - num_test_points) // 2
        end = start + num_test_points
        train_points = np.concatenate([points[:start], points[end:]])
        test_points = points[start:end]
    elif split_strategy == "end":
        train_points = points[:-num_test_points]
        test_points = points[-num_test_points:]
    else:
        raise ValueError("Invalid split strategy.")

    return train_points, test_points

def array_to_string(points: np.ndarray) -> str:
    """
    Converts a numpy array of points to a string.

    Parameters
    ----------
    points -> the numpy array of points to convert.

    Returns
    -------
    points -> the string of points.
    """
    points = points.tolist()
    points_str = ""
    for point in points:
        point_str = ", ".join([str(np.round(x, 2)) for x in point])
        point_str = f"({point_str})"
        points_str += point_str + ", "

    return points_str[:-2]

def string_to_array(points: str) -> np.ndarray:
    """
    Converts a string of points to a numpy array.

    Parameters
    ----------
    points -> the string of points to convert.

    Returns
    -------
    points -> the numpy array of points.
    """
    points = points.replace("(", "").split("), ")
    points = [point.replace(")", "") for point in points]
    points = [point.split(", ") for point in points]
    points = [[float(coordinate) for coordinate in point] for point in points]
    return np.array(points)

def eval_function(function: sympy.core.function.Function, Xs: np.ndarray, num_variables: int) -> float:
    """
    Evaluates a sympy function at a point.

    Parameters
    ----------
    function -> the function to evaluate.
    Xs -> the points to evaluate the function at. (Variables have to be sorted alphabetically)
    num_variables -> the number of variables the function takes.
    
    Returns
    -------
    ys -> the value of the function at x.
    """
    symbols = function.free_symbols
    symbols = sorted(symbols, key=lambda x: str(x))
    if Xs.shape[-1] != num_variables:
        Xs = np.array(list(zip(*[x.flat for x in Xs])))

    ys = []
    for point in Xs:
        if type(point) == np.ndarray:
            subs = {symbol: value for symbol, value in zip(symbols, point)}
        else:
            subs = {symbols[0]: point}
        try :
            y = function.evalf(subs=subs)
            y = float(y)
        except Exception as e:
            print(f"Error evaluating function: {function} at point {point}. {e}")
            y = np.inf
        ys.append(y)

    ys = np.array(ys)
    ys = ys.astype(np.float32)
    return ys

def clean_function(function: str) -> str:
    """
    Cleans a function string to be evaluable.
    """
    function = function.strip(".")
    function = function.replace(" ", "")

    if "=" in function:
        function = function.split("=")[1]
    elif ":" in function:
        function = function.split(":")[1]

    # Remove characters that are not allowed in a function
    removals = ["'", '"', "\\", "\n", "\t", "\r", " ", "_"]
    for removal in removals:
        function = function.replace(removal, "")

    # Remove trailing operators
    while len(function) > 1 and function[-1] in ["+", "-", "*", "/", "**"]:
        if len(function) == 1:
            return lambda x: 0
        function = function[:-1]

    # Remove leading operators
    while len(function) > 1 and function[0] in ["+", "*", "/", "**"]:
        if len(function) == 1:
            return lambda x: 0
        function = function[1:]

    # Remove leading indicators of a function definition
    removals = ["Function", "Newfunction", "Thefunctionis", ":"]

    for removal in removals:
        if removal.lower() in function.lower():
            function = function.replace(removal, "")
            function = function.strip()

    return function

def string_to_function(function: str, num_variables: int = 1) -> Callable[[float], float]:
    """
    Converts a string to a callable function using eval.

    Parameters
    ----------
    function -> the string to convert.
    num_variables -> the number of variables the function should take.

    Returns
    -------
    f -> the callable function.
    """
    function = clean_function(function)

    np_func = ["sin", "cos", "tan", "exp", "log", "sqrt"]
    function = function.replace("^", "**")
    #! This only works for variables in x (x1, x2, x3, ...)
    #! This only works with coefficients that end with numbers (e.g. c0, c1, c2, ...)
    function = re.sub(r"(\d)x", r"\1*x", function)
    regex = r"(\d)(" + "|".join(np_func) + ")"
    function = re.sub(regex, r"\1*\2", function)
    f = parse_expr(function)
    return f

def is_valid_function(function: str, current_functions: Any, num_variables: int = 1) -> Tuple[bool, str]:
    """
    Checks if a function is valid.

    Parameters
    ----------
    function -> the function to check.
    current_functions -> the current functions in the prompt.
    num_variables -> the number of variables the function should take.

    Returns
    -------
    valid -> whether the function is valid.
    reason -> the reason the function is invalid (if applicable).
    """ 
    valid = True
    reason = ""
    if type(function) == str:
        f = string_to_function(function, num_variables)
    else:
        f = function
    symbols = f.free_symbols
    variables = [str(symbol) for symbol in symbols if str(symbol).startswith("x")]

    if len(variables) > num_variables:
        valid = False
        reason = "Too many variables in function."
        return valid, reason

    if current_functions is not None and current_functions.func_in_list(f):
        valid = False
        reason = "Function already in prompt."
        return valid, reason

    return valid, reason

def format_exp(x: float, d: int = 6) -> str:
    """
    Formats a number in scientific notation with custom precision. (used in Scorers)

    Parameters
    ----------
    x -> the number to format.
    d -> the number of decimal places to round to.

    Returns
    -------
    x -> the formatted number.
    """
    n = int(np.floor(np.log10(abs(x))))
    significand = x / 10 ** n
    exp_sign = '+' if n >= 0 else '-'
    return f'{significand:.{d}f}e{exp_sign}{n:02d}'

def func_equals(f1: Any, f2: Any, num_variables: int) -> bool:
    """
    Checks if two functions are equal. Used in place of sympy.equals as the latter can become very slow for certain functions.
    https://stackoverflow.com/questions/37112738/sympy-comparing-expressions

    Parameters
    ----------
    f1 -> the first function.
    f2 -> the second function.
    num_variables -> the number of variables the functions should take.

    Returns
    -------
    equal -> whether the functions are equal.
    """
    if f1 == f2:
        return True
    if f1 is None or f2 is None:
        return False
    if f1.free_symbols != f2.free_symbols:
        return False
    if f1.free_symbols != set([sympy.Symbol(f"x{i + 1}") for i in range(num_variables)]):
        return False
    return False

def count_nodes(formula: Any) -> int:
    """
    Gets the complexity of a sympy formula, represented by the number of nodes in its expression tree.
    
    Parameters
    ----------
    formula -> the formula to get the complexity of.
    
    Returns
    -------
    complexity -> the complexity of the formula.
    """
    return formula.count_ops()

def replace_zero_coefficients(expr: Any, formula: Any, threshold: float = 1e-2) -> Any:
    """
    Replaces coefficients that are close to zero in a formula with zero.
    
    Parameters
    ----------
    expr -> the expression to replace coefficients in (with coefficients c0, c1...)
    formula -> the formula to replace coefficients in (with numerical coefficients)
    threshold -> the threshold to consider a coefficient zero.
    
    Returns
    -------
    expr -> the expression with zero coefficients replaced.
    formula -> the formula with zero coefficients replaced.
    """
    coeffs_dict = formula.as_coefficients_dict()
    expr_dict = expr.as_coefficients_dict()
    
    for key, value in coeffs_dict.items():
        if abs(value) < threshold:
            expr_dict[key] = 0
            formula = formula.subs(key, 0)
            
    expr = expr.subs(expr_dict)
    
    print(expr, formula)