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
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

from common import logger
from common.colors_text import TextColor as bcolors


@dataclass
class FitConfigur:
    """parameters and sets for the fitting class"""
    maxfev: int = 1500


class FitRdf2dTo5PL2S:
    """
    Fitting the rdf in 2d, to five-parameters logistic function with
    double slopes.
    """

    info_msg: str = 'Messeges from FitRdf2dTo5PL2S:\n'

    config: "FitConfigur"
    fitted_rdf: dict[float, float]

    def __init__(self,
                 rdf_2d: dict[float, float],
                 log: logger.logging.Logger,
                 fit_config: "FitConfigur" = FitConfigur()
                 ) -> None:
        self.config = fit_config
        self._initiate(rdf_2d)
        self.write_msg(log)

    def _initiate(self,
                  rdf_2d: dict[float, float]
                  ) -> None:
        """set the data and do the fit"""
        radii: np.ndarray
        rdf_values: np.ndarray
        radii_interpolated: np.ndarray
        rdf_interpolated: np.ndarray
        first_derivative: np.ndarray
        second_derivative: np.ndarray

        radii, rdf_values = self.initiate_data(rdf_2d)
        radii_interpolated, rdf_interpolated = \
            self.initiate_interpolation(radii, rdf_values)
        first_derivative, second_derivative = \
            self.initiate_derivative(radii_interpolated, rdf_interpolated)
        curvature: np.ndarray = \
            self.initiate_curvature(first_derivative, second_derivative)
        initial_guesses: list[float] = \
            self.set_initial_guess(radii_interpolated,
                                   rdf_values,
                                   second_derivative,
                                   curvature)
        fitted_data: np.ndarray = self.fit_data(radii_interpolated,
                                                rdf_interpolated,
                                                initial_guesses)
        plt.plot(radii, rdf_values, 'o', label='Original Data')
        plt.plot(radii_interpolated, fitted_data, '-', label='Fitted Curve')
        plt.xlabel('Radius')
        plt.ylabel('RDF')
        plt.legend()
        plt.show()
        self.fitted_rdf = dict(zip(radii_interpolated, fitted_data))

    def initiate_data(self,
                      rdf_2d: dict[float, float]
                      ) -> tuple[np.ndarray, np.ndarray]:
        """get the radii and rdf"""
        radii = np.array(list(rdf_2d.keys()))
        rdf_values = np.array(list(rdf_2d.values()))
        return radii, rdf_values

    def set_initial_guess(self,
                          radii_interpolated,
                          rdf_values,
                          second_derivative,
                          curvature
                          ) -> list[float]:
        """set the initial guesses for the fitting"""
        a_initial_guess: float = max(rdf_values)

        inflection_point_index: np.int64 = np.argmin(second_derivative)
        c_initial_guess: float = \
            float(radii_interpolated[inflection_point_index])

        eta_initial_guess: float = \
            float(np.abs(np.min(curvature)/np.max(curvature)))

        d_initial_guess: float = 0
        mu_initial_guess: float = 1.0
        initial_guesses = [a_initial_guess,
                           d_initial_guess,
                           c_initial_guess,
                           mu_initial_guess,
                           eta_initial_guess]
        self.info_msg += ('\tinitial guesses:\n'
                          f'\t\ta {a_initial_guess:.3f}\n'
                          f'\t\td {d_initial_guess:.3f}\n'
                          f'\t\tc {c_initial_guess:.3f}\n'
                          f'\t\tmu {mu_initial_guess:.3f}\n'
                          f'\t\teta {eta_initial_guess:.3f}\n'
                          )
        return initial_guesses

    def fit_data(self,
                 radii_interpolated,
                 rdf_interpolated,
                 initial_guesses
                 ) -> np.ndarray:
        """do the fit"""
        popt, _ = curve_fit(self.logistic_5pl2s,
                            radii_interpolated,
                            rdf_interpolated,
                            p0=initial_guesses,
                            maxfev=self.config.maxfev)
        return self.logistic_5pl2s(radii_interpolated, *popt)

    @staticmethod
    def initiate_interpolation(radii: np.ndarray,
                               rdf_values: np.ndarray
                               ) -> tuple[np.ndarray, ...]:
        """calculatet the interploations"""
        radii_interpolated = np.linspace(radii.min(), radii.max(), 500)
        interpolation = interp1d(radii, rdf_values, kind='cubic')
        rdf_interpolated = interpolation(radii_interpolated)
        return radii_interpolated, rdf_interpolated

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

    @staticmethod
    def logistic_5pl2s(y_data,
                       a_init_guess: float,
                       b_init_guess: float,
                       c_init_guess: float,
                       mu_init_guess: float,
                       eta_init_guess: float
                       ):
        """ Five-parameters logistic function with double slopes """
        mu_bar = \
            2 * abs(mu_init_guess) * eta_init_guess / (1 + eta_init_guess)
        f_x = 1 + (y_data/c_init_guess)**mu_bar - 1
        return (a_init_guess + (a_init_guess - b_init_guess) /
                (1 + (y_data/c_init_guess)**(-mu_init_guess) + (1 - f_x) *
                (y_data/c_init_guess)**(-mu_init_guess*eta_init_guess)))

    def write_msg(self,
                  log: logger.logging.Logger  # To log
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKGREEN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    pass
