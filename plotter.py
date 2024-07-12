import PIL
import utils
import tempfile

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d
from omegaconf import DictConfig
from collections.abc import Callable
from typing import List, Any

class Plotter(object):
    def __init__(self, cfg: DictConfig, train_points: np.ndarray, test_points: np.ndarray, output_path: str) -> None:
        """
        Initializes the plotter.

        Parameters
        ----------
        cfg : DictConfig -> The configuration file.
        train_points : np.ndarray -> The train points.
        test_points : np.ndarray -> The test points.
        Xs : np.ndarray -> The Xs of the points, as produced by meshgrid (necessary for contour plots).
        ys : np.ndarray -> The ys of the points (necessary for contour plots).
        output_path : str -> The path to save the plots to.
        """
        self.cfg = cfg
        self.num_variables = cfg.experiment.function.num_variables
        self.output_path = output_path

        self.train_points = train_points
        self.test_points = test_points
        assert self.test_points.shape[1] == self.num_variables + 1
        Xs = self.train_points[:, :-1]
        Xs = np.concatenate((Xs, self.test_points[:, :-1]))
        self.min_points = [np.min(self.test_points[:, i]) for i in range(self.num_variables)]
        self.max_points = [np.max(self.test_points[:, i]) for i in range(self.num_variables)]

        num_test = self.cfg.plotter.plotter_resolution if hasattr(self.cfg, "plotter") and hasattr(self.cfg.plotter, "plotter_resolution") else 1000
        if self.num_variables == 1:
            self.Xs_test = np.linspace(self.min_points[0], self.max_points[0], num_test).reshape(-1, 1)
        elif self.num_variables == 2:
            num_test = np.floor(np.sqrt(num_test)).astype(int)
            self.Xs_test = np.meshgrid(*[np.linspace(self.min_points[i], self.max_points[i], num_test) for i in range(self.num_variables)])

        self.gif_duration = self.cfg.plotter.gif_duration if hasattr(self.cfg, "plotter") and hasattr(self.cfg.plotter, "gif_duration") else 1000
        self.fig_size = (self.cfg.plotter.plotter_fig_size, self.cfg.plotter.plotter_fig_size) if hasattr(self.cfg, "plotter") and hasattr(self.cfg.plotter, "plotter_fig_size") else (10, 10)
        self.function_cache = {}

    def _eval_function(self, function: Any, Xs: np.ndarray, num_variables: int) -> np.ndarray:
        if function in self.function_cache:
            return self.function_cache[function]
        else:
            ys = utils.eval_function(function, Xs, num_variables)
            self.function_cache[function] = ys
            return ys

    def plot_points(self, save_fig: bool = False, save_path: str = "points.png", plot_test=False) -> None:
        """
        Plots a set of points. Used to feed to visual models.

        Parameters
        ----------
        save_fig : bool -> Whether to save the figure or not.
        save_path : str -> The path to save the figure to.
        """
        if self.num_variables > 2:
            return
        
        save_path = self.output_path + save_path
        if self.num_variables == 1:
            plt.figure(figsize=self.fig_size)
            plt.scatter(self.train_points[:, 0], self.train_points[:, 1], color="blue", label="Train points", zorder=1)
            if plot_test:
                plt.scatter(self.test_points[:, 0], self.test_points[:, 1], color="red", label="Test points", alpha=.25, zorder=1)
            # plt.grid(alpha=.4,linestyle='--')
            plt.xlabel("x")
            plt.ylabel("f(x)")
            plt.xlim(self.min_points[0] - 0.1 * (self.max_points[0] - self.min_points[0]), self.max_points[0] + 0.1 * (self.max_points[0] - self.min_points[0]))
            plt.ylim(np.min(self.train_points[:, 1]) - 0.1 * (np.max(self.train_points[:, 1]) - np.min(self.train_points[:, 1])), np.max(self.train_points[:, 1]) + 0.1 * (np.max(self.train_points[:, 1]) - np.min(self.train_points[:, 1])))
            plt.legend()
        elif self.num_variables == 2:
            ax = plt.figure(figsize=self.fig_size).add_subplot(projection='3d')
            ax.scatter(self.train_points[:, 0], self.train_points[:, 1], self.train_points[:, 2], c='b', label='Train points')
            if plot_test:
                ax.scatter(self.test_points[:, 0], self.test_points[:, 1], self.test_points[:, 2], c='r', label='Test points', alpha=.25)
            plt.xlabel("x1")
            plt.ylabel("x2")
            ax.legend(loc="upper right")
        else:
            raise ValueError("Invalid number of variables.")
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(save_path)
        else:
            return plt.gcf(), plt.gca()

    def plot_results(self, function: Any, test_function: Any = None, plot_true: bool = True, label: str = "Model's best guess") -> plt.Figure:
        """
        Plots the results of the experiment, showing the test points, the true function, and the model's best guess.

        Parameters
        ----------
        function : str -> The model's best guess.
        test_function : Callable[[float], float] -> The true function.
        plot_true : bool -> Whether to plot the true function or not.
        label : str -> The label of the model's guess.

        Returns
        -------
        plt.Figure -> The figure of the plot.
        """
        fig, ax = self.plot_points(plot_test=False)
        if test_function is None:
            plot_true = False

        if self.num_variables == 1:
            if plot_true:
                ys_test = self._eval_function(test_function, self.Xs_test, self.num_variables)
                plt.plot(self.Xs_test, ys_test, color="red", label="True function", zorder=0, linestyle="--")

            ys = self._eval_function(function, self.Xs_test, self.num_variables)
            plt.plot(self.Xs_test, ys, color="green", label=label, zorder=0)
            plt.legend(loc="lower right")
            
        elif self.num_variables == 2:
            X1, X2 = self.Xs_test
            if plot_true:
                Z_test = self._eval_function(test_function, np.array([X1, X2]), self.num_variables).reshape(X1.shape)
                ax.plot_surface(X1, X2, Z_test, edgecolor='orangered', lw=0.25, alpha=0.1, label="True function", color="red")
            
            Z = self._eval_function(function, np.array([X1, X2]), self.num_variables).reshape(X1.shape)
            ax.plot_surface(X1, X2, Z, edgecolor='mediumseagreen', lw=0.5, alpha=0.3, label=label, color="green")
            ax.legend(loc="upper right")

        return plt.gcf(), plt.gca()

    def record_frame(self, best_function: str, last_function: str, score: float, test_function: Callable[[float], float], round: int, plot_true : bool = True) -> plt.Figure:
        """
        Records a frame of the animation.

        Parameters
        ----------
        best_function : str -> The model's best guess.
        last_function : str -> The model's last guess.
        score : float -> The score of the best function.
        test_function : Callable[[float], float] -> The true function.
        round : int -> The round number.
        plot_true : bool -> Whether to plot the true function or not.

        Returns
        -------
        plt.Figure -> The figure of the frame.
        """
        fig, ax = self.plot_results(best_function, test_function, plot_true=plot_true)
        if self.num_variables == 1:
            ys_last = self._eval_function(last_function, self.Xs_test, self.num_variables)
            plt.plot(self.Xs_test, ys_last, color="orange", label="Last guess")
            plt.legend()
        elif self.num_variables == 2:
            X1, X2 = self.Xs_test
            Z_last = self._eval_function(last_function, np.array([X1, X2]), self.num_variables).reshape(X1.shape)
            ax.plot_surface(X1, X2, Z_last, edgecolor='orange', lw=0.5, alpha=0.3, label="Last guess", color="orange")
            ax.text2D(0.05, 0.95, f"Score: {score:.3f}", transform=ax.transAxes, fontsize=10, verticalalignment='top')
            ax.legend(loc="upper right")

        ax.set_title(f"Round {round+1}, R2: {score:.5f}")
        fig.tight_layout()
        return plt.gcf(), plt.gca()

    def record_video(self, frames: List[plt.Figure]) -> None:
        """
        Records the animation from the frames buffer.

        Parameters
        ----------
        frames : List[plt.Figure] -> The frames buffer.
        """
        images = []
        with tempfile.TemporaryDirectory() as tmp_path:
            for i, frame in enumerate(frames):
                frame.savefig(tmp_path + f"{i}.png")
            for i in range(len(frames)):
                images.append(PIL.Image.open(tmp_path + f"{i}.png"))
            # Extend the last frame to make the last (best) result more visible
            for _ in range(5):
                images.append(images[-1])

        images[0].save(self.output_path + "animation.gif", save_all=True, append_images=images[1:], duration=self.gif_duration, loop=0)