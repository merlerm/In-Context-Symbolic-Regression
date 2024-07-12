from .scorer import Scorer
from utils import format_exp
from sklearn.metrics import mean_squared_error
import numpy as np
import utils
import sys

from typing import Dict, Tuple

class BasicScorer(Scorer):
    """
    Basic scorer for symbolic regression.
    Scores the function using unnormalized MSE.
    """

    def __init__(self, points: np.ndarray, rounding: int = 2, scientific: bool = False):
        """
        Initialize the scorer.

        Parameters
        ----------
        points -> the points to evaluate the function on.
        rounding -> number of decimal places to round the score to (-1 for no rounding)
        scientific -> whether to use scientific notation for the score.

        Returns
        -------
        None.
        """

        super().__init__(points)
        self.round = rounding
        self.scientific = scientific

    def score(self, function: callable) -> float:
        """
        Scores a function on a given set of points.

        Parameters
        ----------
        function -> the function to score.
        round -> whether to round the score.
        scientific -> whether to use scientific notation for the score.

        Returns
        -------
        score -> the score of the function.
        """
        xs = self.points[:, 0:-1]
        ys = self.points[:, -1]
        num_variables = xs.shape[1]

        try:
            ys_pred = utils.eval_function(function, xs, num_variables)
            fit = mean_squared_error(ys, ys_pred)
        except Exception as e:
            # If the function is invalid for some points (e.g. division by 0), return inf
            fit = np.inf

        if self.round > 0:
            fit = np.round(fit, self.round)
        fit = fit.astype(np.float64)
        return fit
    
    def score_current_functions(self, current_functions: Dict[str, callable]) -> Tuple[Dict, Dict]:
        """
        Scores the current functions in the prompt.

        Parameters
        ----------
        current_functions -> the current functions in the prompt.
        round -> whether to round the score.
        scientific -> whether to use scientific notation for the score.

        Returns
        -------
        scores -> the score of the current functions.
        normalized_scores -> the normalized score of the current functions.
        """
        scores = { function: self.score(current_functions[function]) for function in current_functions }
        normalized_scores = scores.copy()
        if self.scientific:
            normalized_scores = { name: format_exp(score, self.round) for name, score in normalized_scores.items()}

        return scores, normalized_scores