"""
Computing the DLVO potential using the non-linear approximation
The approximation is based on the following equation:
exp(y/2) = (1 + \\alpha exp(-\\kappa(r-a)) (a/r)) /
           (1 - \\alpha exp(-\\kappa(r-a)) (a/r))
    where:
    - y =  e\\psi / kT (dimmensionless potential coordinate)
    - \\psi = potential
    - \\alpha = e\\psi_0 / 4kT
    - \\kappa = 1/\\lambda_D
    - \\lambda_D = debye length
    - a = particle radius
    - r = distance from Center of the particle
    - e = electron charge
    - k = Boltzmann constant
    - T = temperature
The computation is based on the book by:
Hans-Jürgen Butt,  and Michael Kappl
"Surface and Interfacial Forces"
pp. 98-102
Saeed
11 June 2024
"""

import typing
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from common import logger, elsevier_plot_tools
from common.colors_text import TextColor as bcolors
from module9_electrostatic_analysis.dlvo_potential_configs import \
    AllConfig


@dataclass
class AnalyticConfig:
    """set the parameters for the analytic approximation plots"""
    alpha: np.ndarray = np.array([1.0, 1.0])
    r_np: float = 3.0  # [nm]
    debye_l: float = 1.0  # [nm]
    phi_0: np.ndarray = np.array([1.0, 1.0])
    charge: np.ndarray = np.array([1.0, 1.0])
    beta: float = 1.0
    colors: list[str] = field(default_factory=lambda: [
        '#d9d9d9', '#bdbdbd', '#969696', '#737373', '#525252', '#252525',
        '#000000'])


class NonLinearPotential:
    """
    Compute the DLVO potential using the non-linear approximation
    """
    # pylint: disable=too-many-arguments

    info_msg: str = 'Message from NoneLinearPotential:\n'
    radii: np.ndarray
    phi_r: np.ndarray

    def __init__(self,
                 debye_l: float,
                 phi_0: np.ndarray,
                 log: logger.logging.Logger,
                 charge: np.ndarray,
                 configs: AllConfig = AllConfig()
                 ) -> None:
        """write and log messages"""
        self.configs = configs
        self.radii, self.phi_r = \
            self.compute_potential(debye_l, phi_0, charge, log)
        self._write_msg(log)

    def compute_potential(self,
                          debye_l: float,
                          phi_0: np.ndarray,
                          charge: np.ndarray,
                          log: logger.logging.Logger
                          ) -> tuple[np.ndarray, np.ndarray]:
        """compute the DLVO potential"""
        box_lim: float = self.configs.phi_parameters['box_xlim'] / 2.0
        r_np: float = self.configs.np_radius / 10.0  # [A] -> [nm]
        radii: np.ndarray = np.linspace(r_np+0.01, box_lim, len(charge))
        phi_r: np.ndarray = np.zeros(radii.shape)

        alpha: np.ndarray = phi_0 * self.configs.phi_parameters['e_charge'] / \
            (4.0 * self.configs.phi_parameters['k_boltzman_JK'] *
             self.configs.phi_parameters['T'])

        kappa: float = 1.0 / debye_l
        phi_r = self.compute_phi_r(radii, phi_r, alpha, kappa, r_np)
        self.test_equation(radii, log)
        return radii, phi_r

    def compute_phi_r(self,
                      radii: np.ndarray,
                      phi_r: np.ndarray,
                      alpha: np.ndarray,
                      kappa: float,
                      r_np: float
                      ) -> np.ndarray:
        """compute the potential"""
        # pylint: disable=too-many-arguments
        a_r: np.ndarray = r_np / radii
        beta: float = self.configs.phi_parameters['k_boltzman_JK'] * \
            self.configs.phi_parameters['T']
        co_factor: float = 2.0 * beta / self.configs.phi_parameters['e_charge']

        for alpha_i in alpha:
            alpha_exp: np.ndarray = alpha_i * np.exp(-kappa * (radii - r_np))
            radial_term: np.ndarray = alpha_exp * a_r
            phi_r += \
                co_factor * np.log((1.0 + radial_term) / (1.0 - radial_term))

        phi_r = phi_r / len(alpha)
        self.info_msg += \
            ('\tComputing the potential in nonlinear approximation of the'
             'Boltzmann-Poisson equation\n')
        return phi_r

    def test_equation(self,
                      radii: np.ndarray,
                      log: logger.logging.Logger
                      ) -> None:
        """test the equation"""
        AnalyticAnalysis(log, radii)

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        now = datetime.now()
        self.info_msg += \
            f'\tTime: {now.strftime("%Y-%m-%d %H:%M:%S")}\n'
        print(f'{bcolors.OKCYAN}{NonLinearPotential.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        # log.info(msg=self.info_msg)


class AnalyticAnalysis:
    """PLoting the analytic approximation of the Poisson-Boltzmann
    equation. The equation is based on the book by:
    Hans-Jürgen Butt,  and Michael Kappl
    "Surface and Interfacial Forces"
    pp. 101, eq. 4.28
    """

    info_msg: str = 'Message from AnalyticAnalysis:\n'

    def __init__(self,
                 log: logger.logging.Logger,
                 radii: np.ndarray,
                 configs: AnalyticConfig = AnalyticConfig()
                 ) -> None:
        self.configs = configs
        param: dict[typing.Any, typing.Any] = self.inialize_data(radii)
        phi_r_list: list[np.ndarray] = self.test_equation(**param)
        self.plot(radii, phi_r_list, param)
        self._write_msg(log)

    def inialize_data(self,
                      radii: np.ndarray
                      ) -> dict[typing.Any, typing.Any]:
        """initialize the data"""
        return {'radii': radii,
                'alpha': self.configs.alpha,
                'r_np': self.configs.r_np,
                'debye_l': self.configs.debye_l,
                'phi_0': self.configs.phi_0,
                'charge': self.configs.charge,
                'beta': self.configs.beta}

    def test_equation(self,
                      radii: np.ndarray,
                      alpha: np.ndarray,
                      r_np: float,
                      debye_l: float,
                      phi_0: np.ndarray,
                      charge: np.ndarray,
                      beta: float
                      ) -> list[np.ndarray]:
        """test the equation"""
        # pylint: disable=unused-argument
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-arguments

        phi_r = np.zeros(radii.shape)
        kappa = 1.0 / debye_l
        a_r = r_np / radii
        co_factor = 2.0 * beta
        phi_r_list: list[np.ndarray] = []
        for strength in range(1, 6):
            for alpha_i in alpha:
                alpha_exp = \
                    strength * alpha_i * np.exp(-kappa * (radii - r_np))
                radial_term = alpha_exp * a_r
                phi_r += co_factor * np.log((1.0 + radial_term) /
                                            (1.0 - radial_term))
            phi_r_list.append(phi_r / len(alpha))
        return phi_r_list

    def plot(self,
             radii: np.ndarray,
             phi_r_list: list[np.ndarray],
             param: dict[typing.Any, typing.Any]
             ) -> None:
        """plot the data"""

        fig_i, ax_i = elsevier_plot_tools.mk_canvas('single_column')
        for strength, phi_r in enumerate(phi_r_list, 1):
            ax_i.plot(radii,
                      phi_r,
                      color=self.configs.colors[strength],
                      label=rf'$\alpha$={strength:.1f}')
        ax_i.set_xlabel('r')
        ax_i.set_ylabel('y')
        y_lo: float = ax_i.get_ylim()[0]
        plt.vlines(param['r_np'],
                   y_lo,
                   15,
                   colors='r',
                   linestyles='dashed',
                   lw=0.7)
        plt.xlim(2, 10)
        ax_i.set_ylim(y_lo, 15)
        ax_i.set_yticks([])
        plt.legend()
        elsevier_plot_tools.save_close_fig(fig_i, 'analytic_approximation.png')

    def _write_msg(self,
                     log: logger.logging.Logger  # To log
                     ) -> None:
        """write and log messages"""
        now = datetime.now()
        self.info_msg += \
              f'\tTime: {now.strftime("%Y-%m-%d %H:%M:%S")}\n'
        print(f'{bcolors.OKCYAN}{AnalyticAnalysis.__name__}:\n'
                f'\t{self.info_msg}{bcolors.ENDC}')
        # log.info(msg=self.info_msg)


if __name__ == '__main__':
    print(f'{bcolors.CAUTION}This module is not meant to be run '
          f'independently.{bcolors.ENDC}\n')
