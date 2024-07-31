"""
Fittign the average potential data to a polynomial function
"""


import typing
import inspect

import numpy as np

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

from sklearn.metrics import r2_score  # type: ignore
from sklearn.metrics import mean_squared_error  # type: ignore
from sklearn.metrics import mean_absolute_error  # type: ignore


class FitPotential:
    """Fitting the decay of the potential"""
    info_msg: str = 'Message from FitPotential:\n'
    config: "AllConfig"
    fitted_pot: np.ndarray
    popt: np.ndarray
    evaluate_fit: tuple[float, float, float]

    def __init__(self,
                 radii: np.ndarray,
                 radial_average: np.ndarray,
                 r_np: float,
                 psi_inf: float,
                 config: "AllConfig",
                 ) -> None:
        # pylint: disable=too-many-arguments
        self.config = config
        self.config.psi_infty_init_guess = psi_inf
        fitted_func: typing.Callable[..., np.ndarray | float]

        interpolate_data: tuple[np.ndarray, np.ndarray] = \
            self.interpolate_radial_average(radii, radial_average)

        fitted_func, self.popt = \
            self.fit_potential(interpolate_data[0], interpolate_data[1], r_np)
        self.fitted_pot: np.ndarray | float = fitted_func(radii, *self.popt)
        surface_pot: np.ndarray | float = \
            fitted_func(radii[0], *self.popt)
        self.popt[1] = surface_pot

        self.evaluate_fit = self.analyze_fit_quality(radial_average,
                                                     self.fitted_pot)

    def interpolate_radial_average(self,
                                   radii: np.ndarray,
                                   radial_average: np.ndarray
                                   ) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate the radii and radial average"""
        func = interp1d(radii,
                        radial_average,
                        kind=self.config.fit_interpolate_method)
        radii_new = np.linspace(radii[0],
                                radii[-1],
                                self.config.fit_interpolate_points)
        radial_average_new = func(radii_new)
        return radii_new, radial_average_new

    def fit_potential(self,
                      radii: np.ndarray,
                      shifted_pot: np.ndarray,
                      r_np: float
                      ) -> tuple[typing.Callable[..., np.ndarray | float],
                                 np.ndarray]:
        """Fit the potential to the planar surface approximation"""
        # Define the exponential decay function

        # Initial guess for the parameters [psi_0, lambda_d]

        # Use curve_fit to find the best fitting parameters
        # popt contains the optimal values for psi_0 and lambda_d
        fit_fun: typing.Callable[..., np.ndarray | float] = \
            self.get_fit_function()

        initial_guess: list[float] = self.get_initial_guess(
            shifted_pot[0],
            self.config.debye_intial_guess,
            r_np,
            self.config.psi_infty_init_guess)

        popt, *_ = curve_fit(f=fit_fun,
                             xdata=radii,
                             ydata=shifted_pot,
                             p0=initial_guess,
                             maxfev=5000)
        return fit_fun, popt

    def get_fit_function(self) -> typing.Callable[...,
                                                  np.ndarray | float]:
        """Get the fit function"""
        fit_fun_type: str = self.validate_fit_function()
        return {
            'exponential_decay': self.exp_decay,
            'linear_sphere': self.linear_sphere,
            'non_linear_sphere': self.non_linear_sphere,
        }[fit_fun_type]

    def get_initial_guess(self,
                          phi_0: float,
                          lambda_d: float,
                          r_np: float,
                          psi_infty: float
                          ) -> list[float]:
        """Get the initial guess for the Debye length"""
        fit_fun_type = self.validate_fit_function()
        return {
            'exponential_decay': [phi_0, lambda_d, psi_infty],
            'linear_sphere': [phi_0, lambda_d, r_np],
            'non_linear_sphere': [phi_0, lambda_d, r_np],
        }[fit_fun_type]

    @staticmethod
    def get_function_args(func: typing.Callable[..., np.ndarray | float]
                          ) -> list[str]:
        """
        Get the list of argument names for a given function, excluding 'self'.
        """
        signature = inspect.signature(func)
        return [
            name for name, _ in signature.parameters.items() if name != 'self']

    @staticmethod
    def exp_decay(radius: np.ndarray,
                  lambda_d: float,
                  phi_0: float,
                  psi_infty: float
                  ) -> np.ndarray | float:
        """Exponential decay function"""
        return phi_0 * np.exp(-radius / lambda_d) + psi_infty

    @staticmethod
    def linear_sphere(radius: np.ndarray,
                      lambda_d: float,
                      psi_0: float,
                      r_np: float
                      ) -> np.ndarray:
        """Linear approximation of the potential"""
        return psi_0 * np.exp(-(radius - r_np) / lambda_d) * r_np / radius

    @staticmethod
    def non_linear_sphere(radius: np.ndarray,
                          lambda_d: float,
                          psi_0: float,
                          r_np: float
                          ) -> np.ndarray:
        """Non-linear approximation of the potential"""
        parameters: dict[str, float] = {
            'e_charge': 1.602176634e-19,
            'epsilon_0': 8.854187817e-12,
            'k_b': 1.380649e-23,
            'T': 298.15,
            'n_avogadro': 6.022e23,
            }
        alpha: float = np.arctanh(parameters['e_charge'] * psi_0 /
                                  (4 * parameters['k_b'] * parameters['T']))
        radial_term: np.ndarray = r_np / radius
        power_term: np.ndarray = (radius - r_np) / lambda_d
        alpha_term: np.ndarray = alpha * np.exp(-power_term) * radial_term
        co_factor: float = parameters['k_b'] * parameters['T'] / \
            parameters['e_charge']
        return co_factor * np.log((1 + alpha_term) / (1 - alpha_term))

    def validate_fit_function(self) -> str:
        """Validate and return the fit function type from config"""
        fit_fun_type = self.config.fit_function
        valid_functions = [
            'exponential_decay', 'linear_sphere', 'non_linear_sphere']
        if fit_fun_type not in valid_functions:
            raise ValueError(
                f'\n\tThe fit function: `{fit_fun_type}` is not valid!\n'
                f'\tThe valid options are: \n'
                f'\t{" ,".join(valid_functions)}\n')
        return fit_fun_type

    def analyze_fit_quality(self,
                            y_true: np.ndarray,
                            y_fitted: np.ndarray
                            ) -> tuple[float, float, float]:
        """Compute the fit metrics
        r2_score: Coefficient of Determination:
            Measures how well the observed outcomes are replicated by
            the model.
            (R^2 = 1 - \\frac{SS_{res}}{SS_{tot}}), where (SS_{res}) is
            the sum of squares of residuals and (SS_{tot}) is the total
            sum of squares.
            An (R^2) value close to 1 indicates a good fit.

        Root Mean Square Error (RMSE):
            Measures the square root of the average of the squares of
            the errors.
            (RMSE = \\sqrt{\\frac{1}{n}\\sum_{i=1}^{n}(observed_i -
                                                    predicted_i)^2}).
            Lower RMSE values indicate a better fit.

        Mean Absolute Error (MAE):
            Measures the average magnitude of the errors in a set of
            predictions, without considering their direction.
            (MAE = \\frac{1}{n}\\sum_{i=1}^{n}|observed_i - predicted_i|).
            Like RMSE, lower MAE values indicate a better fit.
        """
        r2_scored: float = r2_score(y_true, y_fitted)
        mean_squre_err: float = mean_squared_error(y_true, y_fitted)
        mean_absolute_err: float = mean_absolute_error(y_true, y_fitted)
        return r2_scored, mean_squre_err, mean_absolute_err
