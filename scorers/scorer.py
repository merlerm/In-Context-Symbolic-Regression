from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np

class Scorer(ABC):
    """
    Abstract class for different scoring methods to evaluate the goodness of a function on a given set of points.
    """

    def __init__(self, points: np.ndarray, round: int = 2, scientific: bool = False):
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

        self.points = points
        # self.tolerance = 1e-6 # Add tolerance to avoid division by 0
        # self.points[:, 0:-1] = self.points[:, 0:-1] + self.tolerance

    @abstractmethod
    def score(self, function: callable) -> float:
        """
        Scores a function on a given set of points.

        Parameters
        ----------
        function -> the function to score.

        Returns
        -------
        score -> the score of the function.
        """

        pass

    @abstractmethod
    def score_current_functions(self, current_functions: Dict[str, callable]) -> Tuple[Dict, Dict]:
        """
        Scores the current functions in the prompt.

        Parameters
        ----------
        current_functions -> the current functions in the prompt.

        Returns
        -------
        scores -> the scores of the current functions.
        """

        pass
