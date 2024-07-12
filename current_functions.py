import re
import sys
import sympy
import utils
import numpy as np

from typing import Any, Dict

class CurrentFunctions(object):
    """
    Helper class to manage the current functions in the prompt.
    """

    def __init__(self, seed_functions, scorer, optimizer, context_len, logger, num_variables) -> None:
        """
        Initialize the class.

        Parameters
        ----------
        seed_functions -> the seed functions to use.
        scorer -> the scorer to use.
        context_len -> the length of the context.
        logger -> the logger to use.
        num_variables -> the number of variables.
        """
        
        self.seed_functions = seed_functions
        self.scorer = scorer
        self.optimizer = optimizer
        self.context_len = context_len
        self.logger = logger
        self.num_variables = num_variables
        functions = [utils.string_to_function(name, self.num_variables) for name in self.seed_functions.keys()]
        self.logger.info(f"Seed functions: {functions}.")
        self.functions = {}
        self.scores = {}
        self.norm_scores = {}
        self.screen_names = {}

        # Optimize seed function coefficients
        for function in functions:
            try:
                optimized_function, coeff_function = self.optimizer.optimize(function, return_coeff=True, quiet=False)
                if self.func_in_list(coeff_function):
                    self.logger.warning(f"Function {coeff_function} already in prompt.")
                    continue
                self.functions[coeff_function] = optimized_function
                self.logger.info(f"Optimized seed function: {str(coeff_function)}.")
            except Exception as e:
                self.logger.warning(f"Could not optimize function {function}. {e}")
                pass
        self.logger.info(f"Optimized seed functions: {self.functions}.")
        if len(self.functions) == 0:
            self.logger.warning("Failed to optimize all seed functions. Function list will be empty.")
        else:
            self.scores, self.norm_scores = self.scorer.score_current_functions(self.functions)
            self.clean_scores()
            self.screen_names = {function: re.sub(r'c\d+', 'c', str(function)) for function in self.functions}

        self.logger.info(f"Current scores: {self.scores}.")
        self.logger.info(f"Current normalized scores: {self.norm_scores}.")

    def func_in_list(self, function: Any) -> bool:
        """
        Checks if a function is already in the prompt by assigning the same symbol to all coefficients.
        
        Parameters
        ----------
        function -> the function to check.
        
        Returns
        -------
        bool -> whether the function is already in the prompt or not.
        """
        symbols = set(function.free_symbols)
        for f in self.functions:
            symbols = symbols | set(f.free_symbols)
        coeffs = [s for s in symbols if str(s).startswith("c")]
        subs = {c: sympy.Symbol('c') for c in coeffs}
        function = function.subs(subs)
        for f in self.functions:
            f = f.subs(subs)
            if utils.func_equals(f, function, self.num_variables):
                return True
        return False

    def clean_scores(self) -> None:
        """
        Remove eventual inf scores from the scores.
        """
        print(f"Started cleaning scores {self.scores}.")
        removals = []
        removals = [function for function in self.scores if self.scores[function] == np.inf]
        removals += [function for function in self.norm_scores if self.norm_scores[function] == np.inf and function not in removals]

        for function in removals:
            self.logger.warning(f"Removing function {function} with score {self.scores[function]} ({self.norm_scores[function]}) from the prompt.")
            self.functions.pop(function)
            self.scores.pop(function)
            self.norm_scores.pop(function)
        
        print(f"Finished cleaning scores {self.scores}.")

    def add_function(self, expr: Any, function: Any) -> None:
        """
        Adds a function to the current functions.

        Parameters
        ----------
        expr -> the coefficient form of the function.
        function -> the function to add.
        """
        self.logger.info(f"Adding function {expr}.")
        
        # Check if the function is already in the prompt, necessary if force_unique is False
        if self.func_in_list(expr):
            self.logger.info(f"Function {expr} already in prompt.")
            return
        
        if len(self.scores) >= self.context_len and self.scorer.score(function) > np.max(list(self.scores.values())):
            self.logger.info(f"Function {expr} has score {self.scorer.score(function)}, which is higher than the current worst score {np.max(list(self.scores.values()))}.")
            return
        
        self.functions[expr] = function
        self.screen_names[expr] = re.sub(r'c\d+', 'c', str(expr))
        self.scores, self.norm_scores = self.scorer.score_current_functions(self.functions)
        self.clean_scores()

        # Remove the worst function if the context is full
        if len(self.functions) > self.context_len:
            worst_function = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)[0][0]
            self.logger.info(f"Removing function {worst_function}.")
            self.functions.pop(worst_function)
            self.screen_names.pop(worst_function)
            self.scores.pop(worst_function)
            self.norm_scores.pop(worst_function)

        self.logger.info(f"Current scores: {self.scores}.")
        self.logger.info(f"Current normalized scores: {self.norm_scores}.")
    
    def get_best_function(self, return_coeff: bool = True) -> str:
        """
        Gets the best function in the current functions.

        Returns
        -------
        best_function -> the best function in the current functions.
        return_coeff -> whether to return the function in coefficient form.
        """
        best_function = sorted(self.scores.items(), key=lambda x: x[1])[0][0]
        if return_coeff:
            return best_function
        else:
            return self.functions[best_function]
    
    def get_prompt_functions(self) -> Dict[str, float]:
        """
        Gets the prompt functions (from the normalized scores)

        Returns
        -------
        prompt_functions -> the current functions.
        """
        top_functions = sorted(self.norm_scores.items(), key=lambda x: x[1])
        top_functions = top_functions[:self.context_len]
        top_functions = sorted(top_functions, key=lambda x: x[1], reverse=True)
        return top_functions
    
    def get_prompt(self, base_prompt: str) -> str:
        """
        Generates a prompt for the model, by appending the current functions and their scores to a base prompt.

        Parameters
        ----------
        base_prompt -> the base prompt to append to.

        Returns
        -------
        prompt -> the prompt to use for the model.
        """
        top_functions = self.get_prompt_functions()
        functions = "\n".join([f'Function: {self.screen_names[function_name]}\nError: {fit}\n' for function_name, fit in top_functions])
        functions += "\nNew Functions: "
        prompt = base_prompt.format(functions=functions)
        return prompt