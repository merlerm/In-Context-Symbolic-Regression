from .scorer import Scorer
from utils import format_exp
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import utils

from typing import Dict, Tuple

class MinMaxScorer(Scorer):
    """
    Basic scorer for symbolic regression.
    Scores the function using normalized MSE.
    The MSE is normalized by the MinMaxScaler.
    """

    def __init__(self, points: np.ndarray, min_score: float, max_score: float, rounding: int = 2, scientific: bool = False):
        """
        Initialize the scorer.

        Parameters
        ----------
        points -> the points to evaluate the function on.
        min_score -> the minimum value for the interval to scale the scores to.
        max_score -> the maximum value for the interval to scale the scores to.
        rounding -> number of decimal places to round the score to (-1 for no rounding)
        scientific -> whether to use scientific notation for the score.

        Returns
        -------
        None.
        """

        super().__init__(points)
        self.round = rounding
        self.scientific = scientific
        self.min_score = min_score
        self.max_score = max_score
        self.scaler = MinMaxScaler(feature_range=(min_score, max_score))
        
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
        ys = self.points[:, 1]
        num_variables = xs.shape[1]

        try:
            ys_pred = utils.eval_function(function, xs, num_variables)
            fit = mean_squared_error(ys, ys_pred)
        except:
            # If the function is invalid for some points (e.g. division by 0), return inf
            fit = np.inf
            
        if self.round > 0:
            fit = np.round(fit, self.round)
        return fit.astype(np.float64)
    
    def score_current_functions(self, current_functions: list) -> Tuple[Dict, Dict]:
        """
        Scores the current functions in the prompt.

        Parameters
        ----------
        current_functions -> the current functions in the prompt.

        Returns
        -------
        scores -> the scores of the current functions.
        """
        scores = { function: self.score(current_functions[function]) for function in current_functions }
        inf_scores = { name: score for name, score in scores.items() if score == np.inf }
        # Remove inf scores from the list of scores to normalize
        # Can't remove them entirely otherwise this would cause inconsistencies (current_functions would be missing some scores). We handle those in the main loop
        if len(inf_scores) > 0:
            for name in inf_scores:
                scores.pop(name)
        scores_array = np.array(list(scores.values())).reshape(-1, 1)
        if len(scores_array) != 0:
            self.scaler.fit(scores_array)
            normalized_scores = { name: self.scaler.transform(np.array(score).reshape(-1, 1))[0][0] for name, score in scores.items() }
        else:
            normalized_scores = scores.copy()

        # Add inf scores back to the list of scores
        for name in inf_scores:
            scores[name] = np.inf
            normalized_scores[name] = np.inf
        if self.round > 0 and not self.scientific:
            normalized_scores = { name: np.round(score, self.round) for name, score in normalized_scores.items() }
        elif self.scientific:
            normalized_scores = { name: format_exp(score, self.round) for name, score in normalized_scores.items() }
        return scores, normalized_scores