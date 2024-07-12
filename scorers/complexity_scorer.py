from .scorer import Scorer
from utils import format_exp
from sklearn.metrics import mean_squared_error
import numpy as np
import utils
import sys

from typing import Dict, Tuple

class ComplexityScorer(Scorer):
    """
    Complexity scorer for symbolic regression.
    Scores the function using normalized MSE combined with a measure for the complexity of the function.
    The score measure was defined in https://arxiv.org/abs/2303.06833
    """

    def __init__(self, points: np.ndarray, rounding: int = 2, scientific: bool = False, lam: float = 0.5, max_nodes: int = 30, alternative: bool = False):
        """
        Initialize the scorer.

        Parameters
        ----------
        points -> the points to evaluate the function on.
        rounding -> number of decimal places to round the score to (-1 for no rounding)
        scientific -> whether to use scientific notation for the score.
        lam -> the lambda parameter for the complexity term.
        max_nodes -> the maximum number of nodes in the expression tree (used for normalization).
        alternative -> whether to use the alternative scoring function.

        Returns
        -------
        None.
        """

        super().__init__(points)
        self.round = rounding
        self.scientific = scientific
        self.eps = 1e-6
        self.lam = lam
        self.max_nodes = max_nodes
        self.alternative = alternative

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
        n = xs.shape[0]
        num_variables = xs.shape[1]

        try:
            ys_pred = utils.eval_function(function, xs, num_variables)
            # Calculate normalized MSE
            fit = 1/n * np.linalg.norm(ys - ys_pred)**2 / (1/n * np.linalg.norm(ys)**2 + self.eps)
        except Exception as e:
            # If the function is invalid for some points (e.g. division by 0), return inf
            fit = np.inf
        fit = fit.astype(np.float64)
        
        complexity = utils.count_nodes(function)
        complexity_term = np.exp(-complexity/self.max_nodes)
        
        if self.alternative:
            error = fit + self.lam * (1-complexity_term)
        else:
            fit_term = 1/(1 + fit)
            error = 1/(fit_term + self.lam * complexity_term + self.eps)
            
        if self.round > 0:
            error = np.round(error, self.round)
            
        return error 
    
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