import re
import sympy
import threading
import warnings
import numpy as np

from omegaconf import DictConfig
from scipy.optimize import curve_fit
from mloggers import MultiLogger
from typing import Any
from sympy import Mul, Add, Dummy, sift, numbered_symbols

class Optimizer(object):
    """
    Optimizer class used to fit a function to a set of points, given a base shape.
    For example, takes as input something like "ax^2 + bx + c" and fits (a, b, c) to a set of points.
    """
    def __init__(self, cfg: DictConfig, points: np.ndarray, logger: MultiLogger) -> None:
        """
        Initializes the optimizer.

        Parameters
        ----------
        cfg : DictConfig -> The configuration file.
        points : np.ndarray -> The points to fit to.
        logger : MultiLogger -> The logger to log to.
        """
        self.cfg = cfg
        self.logger = logger
        self.points = points
        self.num_variables = cfg.experiment.function.num_variables
        self.invalid_coefficients = ["x", "y", "e"]

        self.coeff_rounding = cfg.experiment.optimizer.coeff_rounding if hasattr(cfg.experiment, "optimizer") and hasattr(cfg.experiment.optimizer, "coeff_rounding") else 2
        self.tol = cfg.experiment.optimizer.tol if hasattr(cfg.experiment, "optimizer") and hasattr(cfg.experiment.optimizer, "tol") else 1e-3 # Tolerance used to zero out coefficients that are close to 0
        self.num_threads = cfg.experiment.optimizer.optimizer_threads if hasattr(cfg.experiment, "optimizer") and hasattr(cfg.experiment.optimizer, "optimizer_threads") else 5
        self.timeout = cfg.experiment.optimizer.timeout if hasattr(cfg.experiment, "optimizer") and hasattr(cfg.experiment.optimizer, "timeout") else 10
        self.p0_min = cfg.experiment.optimizer.p0_min if hasattr(cfg.experiment, "optimizer") and hasattr(cfg.experiment.optimizer, "p0_min") else -10
        self.p0_max = cfg.experiment.optimizer.p0_max if hasattr(cfg.experiment, "optimizer") and hasattr(cfg.experiment.optimizer, "p0_max") else 10

    def replace_coefficients(self, exp: sympy.core.add.Add) -> sympy.core.add.Add:
        """
        Replaces the number coefficients of a function with symbols.

        Parameters
        ----------
        exp : sympy.core.add.Add -> The function to replace coefficients of.

        Returns
        -------
        exp : sympy.core.add.Add -> The function with coefficients replaced.
        """
        def is_coefficient(symbol: Any) -> bool:
            if len(symbol.args) > 0:
                for arg in symbol.args:
                    if not is_coefficient(arg):
                        return False
            
            if re.match(r"c\d+", str(symbol)):
                return True
            elif symbol.is_Dummy:
                return True
            elif symbol.is_number:
                return True
            
            return False

        # Adapted from https://stackoverflow.com/questions/59686990/replacing-numbers-with-parameters-in-sympy
        def nfact2dum(m):
            assert m.is_Mul or m.is_Add or m.is_Function
            if m.is_Mul:
                if not any([is_coefficient(i) for i in m.args]):
                    return m
                nonnum = sift(m.args, lambda i:is_coefficient(i), binary=True)[1]
                return Mul(*([Dummy()] + nonnum))
            elif m.is_Add:
                if not any([is_coefficient(i) for i in m.args]):
                    return m
                nonnum = sift(m.args, lambda i:is_coefficient(i), binary=True)[1]
                return Add(*([Dummy()] + nonnum))
            elif m.is_Function:
                args = []
                for arg in m.args:
                    if arg.is_Mul or arg.is_Add or arg.is_Function:
                        args.append(nfact2dum(arg))
                    else:
                        args.append(arg)
                return Dummy() * m.func(*args)

        # Add +1 at the end of the expression to make sure that a constant is included
        exp = exp + 1

        # Replace all symbols beginning with c with a dummy 
        # (as they are coefficients, otherwise we could generate a symbol that is already in the expression)
        exp = exp.replace(lambda x: re.match(r"c\d+", str(x)) or str(x).lower() == "c", lambda x: Dummy())
        
        # Replace all coefficients with dummies
        exp = exp.replace(
            lambda x:x.is_Mul or x.is_Add or x.is_Function,
            lambda x: nfact2dum(x))
        # Replace all exponents of dummy variables with 1
        exp = exp.replace(lambda x: x.is_Pow and x.base.is_Dummy, lambda x: x.base)

        # Replace all dummies with symbols
        exp = exp.subs(list(zip(exp.atoms(Dummy),numbered_symbols('c'))))

        return exp
    
    def get_optimizable_sympy_exp(self, exp: sympy.core.add.Add, quiet: bool = False) -> Any:
        """
        Returns a sympy expression that can be optimized by scipy.

        Parameters
        ----------
        exp : sympy.core.add.Add -> The expression to make optimizable.
        quiet : bool -> Whether to log the results.

        Returns
        -------
        exp : Any -> The optimizable expression.
        """
        exp = self.replace_coefficients(exp)
        self.logger.info("Optimizing function: " + str(exp)) if not quiet else None
        symbols = list(exp.free_symbols)

        # Sort symbols so that all x's come first (to find the variables that aren't coefficients)
        symbols.sort(key=lambda x: str(x).replace("x", " "))

        # Safety check to make sure that the number of variables is correct
        num_variables = len(re.findall(r"x\d*", str(symbols)))
        assert num_variables == self.num_variables, f"Number of variables ({num_variables}) does not match number of variables in config ({self.num_variables})"

        symbols = [symbols[:num_variables], *symbols[num_variables:]]
        return sympy.lambdify(symbols, exp, "numpy"), exp
    
    def _run_curve_fit(self, f: Any, num_parameters: int, results: Any, done_event: Any, quiet: bool = True, random_p0: bool = True) -> Any:
        """
        Runs the curve fit function with a timeout.

        Parameters
        ----------
        f : Any -> The function to fit.
        num_parameters : int -> The number of parameters to fit.
        results : Any -> The results list to append to.
        done_event : Any -> The event to set when done.
        quiet : bool -> Whether to log the results.
        random_p0 : bool -> Whether to use random starting points.

        Returns
        -------
        popt : np.ndarray -> The optimized parameters.
        pcov : np.ndarray -> The covariance matrix.
        """
        p0 = np.random.uniform(self.p0_min, self.p0_max, num_parameters) if random_p0 else np.ones(num_parameters)
        popt = None
        try:
            popt, pcov = curve_fit(f, self.points[:, :-1].T, self.points[:, -1].T, p0=p0)
            results.append((popt, pcov))
            done_event.set()
            return True
        except Exception as e:
            print(f"Optimization failed: {e}")
            pass
        
        return False
    
    def optimize(self, exp: sympy.core.add.Add, return_coeff: bool = False, quiet: bool = False) -> sympy.core.add.Add:
        """
        Optimizes a function to a set of points.

        Parameters
        ----------
        exp : sympy.core.add.Add -> The base shape to optimize.
        return_coeff : bool -> Whether to return the expression in coefficient form.
        quiet : bool -> Whether to log the results.

        Returns
        -------
        exp : sympy.core.add.Add -> The optimized function.
        coeff_exp : sympy.core.add.Add -> The optimized function in coefficient form. (Only if return_coeff is True)
        """
        f, exp = self.get_optimizable_sympy_exp(exp, quiet=quiet)
        symbols = exp.free_symbols
        symbols = sorted(symbols, key=lambda x: str(x).replace("x", " "))
        coefficients = symbols[self.num_variables:]
        coeff_exp = exp if return_coeff else None
        Xs = self.points[:, :-1].T
        ys = self.points[:, -1].T

        # Direct warnings only to console logger as file logger breaks with threading
        warnings.filterwarnings("default")
        warnings.showwarning = lambda *args, **kwargs: self.logger.warning(str(args[0]), mask=["file"])
        self.logger.info("Redirecting warnings to console logger only.")

        # Run curve fit with a timeout with num_threads random starting points
        results = []
        if self.num_threads == 1:
            self.logger.info("Running optimization with 1 attempt.") if not quiet else None
            self._run_curve_fit(f, len(coefficients), results=results, done_event=threading.Event(), quiet=quiet, random_p0=False)
            popt, pcov = results[0]
        else:
            done_event = threading.Event()
            threads = []
            for i in range(self.num_threads):
                threads.append(threading.Thread(target=lambda: self._run_curve_fit(f, len(coefficients), results=results, done_event=done_event, quiet=quiet, random_p0=i!=0)))
                threads[-1].start()
            
            done_event.wait(self.timeout)
            for thread in threads:
                thread.join()
                if thread.is_alive(): 
                    self.logger.warning(f"Thread {thread} did not finish in time.")
                    thread._stop()
            self.logger.info("All threads finished.") if not quiet else None
            if not done_event.is_set():
                raise ValueError("Optimization failed: timeout reached")
            
            # Direct warnings back to normal
            warnings.filterwarnings("default")
            warnings.showwarning = lambda *args, **kwargs: self.logger.warning(str(args[0]))
            self.logger.info("Redirecting warnings back to normal (both file and console).")

            # Get the best parameters
            best_popt = None
            best_pcov = None
            best_error = np.inf
            for popt, pcov in results:
                error = np.sum((f(Xs, *popt) - ys) ** 2)
                if error < best_error:
                    best_error = error
                    best_popt = popt
                    best_pcov = pcov

            popt = best_popt
            pcov = best_pcov

        if pcov is None or np.isinf(pcov).any() or np.isnan(pcov).any():
            raise ValueError("Optimization failed: covariance matrix is invalid")
        popt = [np.round(x, self.coeff_rounding) for x in popt]
        self.logger.info("Optimized parameters: " + str(popt)) if not quiet else None
        
        assert len(coefficients) == len(popt), f"Number of found coefficients {coefficients} does not match number of parameters {len(popt)})"
        zero_subs = {}
        for i, coefficient in enumerate(coefficients):
            if popt[i] < self.tol and popt[i] > -self.tol:
                zero_subs[coefficient] = 0
        return exp.subs(list(zip(coefficients, popt))), coeff_exp.subs(zero_subs) if return_coeff else None
