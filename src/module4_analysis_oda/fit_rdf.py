"""
smoothing and fitting rdf:
Smoothing:
    - Moving average
Fitting:
    - 5PL2S

The Five-Parameters Logistic Function with Double Slopes (5PL2S) is a
modified version of the logistic function, designed to provide a better
fit for asymmetric sigmoid curves. In this model, five parameters:
(a, b, c, µ, η) are used to define the curve. The 'c' parameter
represents the halfway point between the upper and lower plateaus of
the curve, while 'µ' and 'η' are parameters related to the slopes at
different stages of the curve. The 5PL2S function is particularly
useful for describing curves that exhibit a sigmoid shape but with
asymmetry, offering a more accurate and flexible fitting approach
for such data.

"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from common import logger
from common.colors_text import TextColor as bcolors


@dataclass
class FitConfigur:
    """parameters and sets for the fitting class"""
    maxfev: int = 5000
    response_zero: float = 0.0
    response_infinite: float = 1.0


class FitRdf2dTo5PL2S:
    """
    Fitting the rdf in 2d, to five-parameters logistic function with
    double slopes.
    """

    info_msg: str = 'Messeges from FitRdf2dTo5PL2S:\n'

    config: "FitConfigur"
    fitted_rdf: dict[float, float]
    midpoind: float  # Midpoint of fitting: c
    first_turn: float  # First turnning point
    second_turn: float  # Second turnning point

    def __init__(self,
                 rdf_2d: dict[float, float],
                 log: logger.logging.Logger,
                 fit_config: "FitConfigur" = FitConfigur()
                 ) -> None:
        self.config = fit_config
        self._initiate(rdf_2d, log)
        self.write_msg(log)

    def _initiate(self,
                  rdf_2d: dict[float, float],
                  log: logger.logging.Logger
                  ) -> None:
        """set the data and do the fit"""
        radii: np.ndarray
        rdf_values: np.ndarray
        radii_interpolated: np.ndarray
        rdf_interpolated: np.ndarray
        first_derivative: np.ndarray
        second_derivative: np.ndarray
        fitted_data: np.ndarray

        radii, rdf_values = self.initiate_data(rdf_2d)
        radii_interpolated, rdf_interpolated = \
            self.interpolation_smoothing(radii, rdf_values)
        first_derivative, second_derivative = \
            self.initiate_derivative(radii_interpolated, rdf_interpolated)
        curvature: np.ndarray = \
            self.initiate_curvature(first_derivative, second_derivative)
        initial_guesses: list[float] = \
            self.set_initial_guess(rdf_values,
                                   radii_interpolated,
                                   second_derivative,
                                   curvature)
        fitted_data, _ = self.fit_data(radii_interpolated,
                                       rdf_interpolated,
                                       initial_guesses)
        self.fitted_rdf = dict(zip(radii_interpolated, fitted_data))
        self._comput_and_set_turn_points()
        FitStatistics(rdf_2d=rdf_2d, fitted_rdf=self.fitted_rdf, log=log)

    def _comput_and_set_turn_points(self) -> None:
        """find the set the turns points on the fitted rdf"""
        first_derivative: np.ndarray
        second_derivative: np.ndarray

        radii: np.ndarray = np.array(list(self.fitted_rdf.keys()))
        fitted_rdf: np.ndarray = np.array(list(self.fitted_rdf.values()))

        first_derivative, second_derivative = \
            self.initiate_derivative(radii, fitted_rdf)
        curvature: np.ndarray = \
            self.initiate_curvature(first_derivative, second_derivative)
        self.first_turn = radii[np.argmax(curvature)]
        self.second_turn = radii[np.argmin(curvature)]
        self.midpoind = radii[np.argmax(first_derivative)]

    def initiate_data(self,
                      rdf_2d: dict[float, float]
                      ) -> tuple[np.ndarray, np.ndarray]:
        """get the radii and rdf"""
        radii = np.array(list(rdf_2d.keys()))
        rdf_values = np.array(list(rdf_2d.values()))
        return radii, rdf_values

    def set_initial_guess(self,
                          rdf_values: np.ndarray,
                          radii_interpolated: np.ndarray,
                          second_derivative: np.ndarray,
                          curvature: np.ndarray
                          ) -> list[float]:
        """set the initial guesses for the fitting"""

        inflection_point_index: np.int64 = np.argmin(second_derivative)
        c_initial_guess: float = \
            float(radii_interpolated[inflection_point_index])

        g_initial_guess: float = \
            float(np.abs(np.min(curvature)/np.max(curvature)))

        b_initial_guess: float = 5.0
        # response_infinite = max(rdf_values)
        # Assuming rdf_values is a Pandas Series or a NumPy array
        rdf_values = np.array(rdf_values)
        # Get the 4 largest values and calculate their mean
        response_infinite = np.mean(np.sort(rdf_values)[-4:])
        initial_guesses = [c_initial_guess,
                           b_initial_guess,
                           g_initial_guess,
                           response_infinite]
        self.info_msg += ('\tinitial guesses:\n'
                          f'\t\td {response_infinite:.3f}\n'
                          f'\t\tc {c_initial_guess:.3f}\n'
                          f'\t\tb {b_initial_guess:.3f}\n'
                          f'\t\tg {g_initial_guess:.3f}\n'
                          )
        return initial_guesses

    def fit_data(self,
                 radii_interpolated,
                 rdf_interpolated,
                 initial_guesses
                 ) -> tuple[np.ndarray, np.ndarray]:
        """do the fit"""
        popt, _ = curve_fit(self.logistic_5pl2s,
                            radii_interpolated,
                            rdf_interpolated,
                            p0=initial_guesses,
                            maxfev=self.config.maxfev)
        self.info_msg += ('\tfitted constants:\n'
                          f'\t\tc {popt[0]:.3f}\n'
                          f'\t\tb {popt[1]:.3f}\n'
                          f'\t\tg {popt[2]:.3f}\n'
                          f'\t\td {popt[3]:.3f}\n'
                          )
        return self.logistic_5pl2s(radii_interpolated, *popt), popt

    @staticmethod
    def interpolation_smoothing(radii: np.ndarray,
                                rdf_values: np.ndarray
                                ) -> tuple[np.ndarray, ...]:
        """calculatet the interploations and smoothing the rdf"""
        radii_interpolated = np.linspace(radii.min(), radii.max(), 1000)
        itp = interp1d(radii, rdf_values, kind='cubic')
        window_size, poly_order = 100, 3
        yy_sg = savgol_filter(itp(radii_interpolated), window_size, poly_order)
        return radii_interpolated, yy_sg

    @staticmethod
    def initiate_derivative(xdata,
                            ydate
                            ) -> tuple[np.ndarray, ...]:
        """return the dervetive of the data"""
        first_derivative = np.gradient(ydate, xdata)
        second_derivative = np.gradient(first_derivative, xdata)
        return first_derivative, second_derivative

    @staticmethod
    def initiate_curvature(first_derivative,
                           second_derivative
                           ) -> np.ndarray:
        """calculate the cuvature"""
        return second_derivative / (1 + first_derivative**2)**(1.5)

    def logistic_5pl2s(self,
                       x_data: np.ndarray,
                       c_init_guess: float,
                       b_init_guess: float,
                       g_init_guess: float,
                       response_infinite: float
                       ) -> np.ndarray:
        """Five-parameters logistic function with double slopes"""
        g_modified_guess: float = \
            2 * abs(b_init_guess) * g_init_guess / (1 + g_init_guess)
        return response_infinite + \
            (self.config.response_zero - response_infinite) /\
            (1+(x_data/c_init_guess)**b_init_guess) ** g_modified_guess

    def write_msg(self,
                  log: logger.logging.Logger  # To log
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKGREEN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


class FitStatistics:
    """calculate the fit statistics"""

    info_msg: str = 'Messege from FitStatistics:\n'
    rdf_2d: dict[float, float]
    fitted_rdf: dict[float, float]

    wsse: float

    def __init__(self,
                 rdf_2d: dict[float, float],
                 fitted_rdf: dict[float, float],
                 log: logger.logging.Logger
                 ) -> None:
        self.rdf_2d = rdf_2d
        self.fitted_rdf = fitted_rdf
        self.calculate_statistics()
        self.write_msg(log)

    def calculate_statistics(self) -> None:
        """Calculate the fit statistics."""
        # Implementation for calculating wSSE, degrees of freedom, etc.
        radii = np.array(list(self.rdf_2d.keys()))
        actual_rdf_values = np.array(list(self.rdf_2d.values()))

        fitted_radii = np.array(list(self.fitted_rdf.keys()))
        fitted_values = np.array(list(self.fitted_rdf.values()))

        predicted_rdf_values = \
            self._get_corrsponded_radii_in_fitted(radii,
                                                  fitted_radii,
                                                  fitted_values)
        variances: np.ndarray
        residuals: np.ndarray
        residuals, variances = self._calculate_residuals_variances(
            actual_rdf_values, predicted_rdf_values)

        self.wsse = self._calculate_wsse(variances, residuals)
        self.info_msg += f'\tThe wSSE is `{self.wsse:.3f}`\n'

    @staticmethod
    def _calculate_wsse(variances: np.ndarray,
                        residuals: np.ndarray
                        ) -> float:
        """Calculate the weighted sum of squared errors."""
        # Filter out zero variance points
        nonzero_variance_mask = variances != 0
        filtered_variances = variances[nonzero_variance_mask]
        filtered_residuals = residuals[nonzero_variance_mask]
        weights = 1 / filtered_variances
        return np.sum(weights * filtered_residuals**2)

    def _calculate_degrees_of_freedom(self):
        """Calculate the degrees of freedom."""
        # Implementation

    def _calculate_fit_probability(self):
        """Calculate the fit probability."""
        # Implementation

    @staticmethod
    def _get_corrsponded_radii_in_fitted(radii: np.ndarray,
                                         fitted_radii: np.ndarray,
                                         fitted_values: np.ndarray
                                         ) -> np.ndarray:
        """calculate the corrsponding radii in fitted data"""
        fitted_data_interpolator = interp1d(fitted_radii,
                                            fitted_values,
                                            kind='linear',
                                            fill_value="extrapolate")
        return fitted_data_interpolator(radii)

    @staticmethod
    def _calculate_residuals_variances(actual_rdf_values: np.ndarray,
                                       predicted_rdf_values: np.ndarray
                                       ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the variances of the residuals.
        """
        # Calculate the residuals
        residuals: np.ndarray = actual_rdf_values - predicted_rdf_values
        # Calculate the variances of the residuals
        variances: np.ndarray = np.abs(residuals - np.mean(residuals))**2
        return residuals, variances

    def write_msg(self,
                  log: logger.logging.Logger  # To log
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKGREEN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    pass
