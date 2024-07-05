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

import numpy as np
import matplotlib.pyplot as plt

from common import logger, elsevier_plot_tools
from common.colors_text import TextColor as bcolors
from module9_electrostatic_analysis.dlvo_potential_configs import AllConfig


@dataclass
class AnalyticConfig:
    """set the parameters for the analytic approximation plots"""
    alpha: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0]))
    r_np: float = 3.0  # [nm]
    debye_l: float = 1.0  # [nm]
    phi_0: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0]))
    charge: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0]))
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
                 configs: AllConfig = AllConfig()
                 ) -> None:
        """write and log messages"""
        self.configs = configs
        self.radii, self.phi_r = \
            self.compute_potential(debye_l, phi_0, log)
        self._write_msg(log)

    def compute_potential(self,
                          debye_l: float,
                          phi_0: np.ndarray,
                          log: logger.logging.Logger
                          ) -> tuple[np.ndarray, np.ndarray]:
        """compute the DLVO potential"""
        # pylint: disable=unused-argument
        r_np: float = self.configs.computation_radius / 10.0  # [A] -> [nm]
        radii: np.ndarray = self.get_radii(r_np)

        phi_r: np.ndarray

        alpha: np.ndarray = self.compute_alpha(phi_0)

        kappa: float = 1.0 / debye_l
        phi_r = self.compute_phi_r(radii, alpha, kappa, r_np)
        self.test_equation(radii, log)
        self._get_potential_to_log(radii, phi_r)
        return radii, phi_r

    def compute_alpha(self,
                      phi_0: np.ndarray
                      ) -> np.ndarray:
        """compute the alpha based on the equation 4.23
        from the book by Hans-Jürgen Butt,  and Michael Kappl
        alpha = tanh(e\\psi_0 / 4kT)
        """
        return np.tanh(phi_0 * self.configs.phi_parameters['e_charge'] /
                       (4.0 * self.configs.phi_parameters['k_boltzman_JK'] *
                        self.configs.phi_parameters['T']))

    def compute_phi_r(self,
                      radii: np.ndarray,
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
        phi_r: np.ndarray = np.zeros(radii.shape)
        for alpha_i in alpha:
            alpha_exp: np.ndarray = alpha_i * np.exp(-kappa * (radii - r_np))
            radial_term: np.ndarray = alpha_exp * a_r
            # Handle division carefully to avoid division by zero
            valid_indices = np.abs(radial_term) < 1
            phi_r[valid_indices] += co_factor * \
                np.log(
                    (1.0 + radial_term[valid_indices]) /
                    (1.0 - radial_term[valid_indices])
                    )

        phi_r = phi_r / len(alpha)
        self.info_msg += \
            ('\tComputing the potential in nonlinear approximation of the'
             'Boltzmann-Poisson equation\n')
        return phi_r

    def get_radii(self,
                  r_np: float
                  ) -> np.ndarray:
        """create the radii based on the grids
        The box is divided into grids
        Box lims, grids, z_grid_up_limit are defined in the configs
        the center of the box is the center of the NP, the interface is
        above the NP's center.
        The z indices are from 0 to z_grid_up_limit
        The x and y indices are from 0 to limit-1 of the grids
        Basically, the radii are created based on the grids, covers only
        the part of the sphere which is under the interface
        """
        box_lims: list[float] = [self.configs.phi_parameters['box_xlim'],
                                 self.configs.phi_parameters['box_ylim'],
                                 self.configs.phi_parameters['box_zlim']]
        grids: list[int] = self.configs.grids
        z_grid_up_limit: int = self.configs.z_gird_up_limit
        x_grid: np.ndarray = np.linspace(0, box_lims[0], grids[0])
        y_grid: np.ndarray = np.linspace(0, box_lims[1], grids[1])
        z_grid: np.ndarray = np.linspace(0, box_lims[2], grids[2])
        z_limit_index = np.abs(z_grid - z_grid_up_limit).argmin()
        z_grid = z_grid[:int(z_limit_index//2)]
        radii = np.sqrt(x_grid[:, None, None] ** 2 +
                        y_grid[None, :, None] ** 2 +
                        z_grid[None, None, :] ** 2)
        radii = radii.flatten()
        radii = radii[(radii >= r_np) & (radii <= np.min(box_lims)/2)]
        self.info_msg += '\tThe radii are created based on the grids\n'
        radii = np.sort(radii)  # Sort the radii array
        return radii

    def test_equation(self,
                      radii: np.ndarray,
                      log: logger.logging.Logger
                      ) -> None:
        """test the equation"""
        AnalyticAnalysis(log, radii)

    def _get_potential_to_log(self,
                              radii: np.ndarray,
                              phi_r: np.ndarray
                              ) -> None:
        """get the potential to log"""
        phi_r0: np.float64 = phi_r[0]
        stern_loc: float = self.configs.stern_layer
        # Get the potential at the stern layer, or colsest value to it
        phi_stern: np.float64 = phi_r[np.abs(radii - stern_loc).argmin()]
        self.info_msg += ('\tThe potential from the anaytical results:\n'
                          f'\t\t{phi_r0 = :.5f} [V]\n'
                          f'\t\t{phi_stern = :.5f} [V]\n')

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{NonLinearPotential.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


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
        self.test_equation(**param)
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
                      ) -> None:
        """test the equation"""
        # pylint: disable=unused-argument
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-arguments

        kappa = 1.0 / debye_l
        a_r = r_np / radii
        co_factor = 2.0 * beta
        self._plot_different_alpha(radii, alpha, r_np, kappa, a_r, co_factor)
        self._plot_different_np_r(radii, alpha, kappa, a_r, co_factor)

    def _plot_different_alpha(self,
                              radii: np.ndarray,
                              alpha: np.ndarray,
                              r_np: float,
                              kappa: float,
                              a_r: np.ndarray,
                              co_factor: float,
                              ) -> None:
        """plot the data for different alpha"""
        # pylint: disable=too-many-arguments
        phi_r_list: list[np.ndarray] = []
        strength: float = 1.0
        for alpha_i in range(1, 6):
            phi_r = np.zeros(radii.shape)
            alpha_exp = \
                strength * alpha_i * np.exp(-kappa * (radii - r_np))
            radial_term = alpha_exp * a_r
            log_argument: np.ndarray = \
                (1.0 + radial_term) / (1.0 - radial_term)
            # Check if log_argument is valid for np.log
            if np.any(log_argument <= 0):
                # Handle invalid log_argument here, e.g., skip \
                #  or set phi_r to a default value
                pass
            phi_r += co_factor * np.log(log_argument)
            phi_r_list.append(phi_r / len(alpha))
        self.plot_diff_alpha_graphs(radii, phi_r_list, r_np)

    def plot_diff_alpha_graphs(self,
                               radii: np.ndarray,
                               phi_r_list: list[np.ndarray],
                               r_np: float
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
        plt.vlines(r_np,
                   y_lo,
                   15,
                   colors='r',
                   linestyles='dashed',
                   lw=0.7)
        plt.xlim(2, 10)
        ax_i.set_ylim(y_lo, 15)
        self._add_text(ax_i, text=f'$r_{{np}}$={r_np:.1f}')
        elsevier_plot_tools.save_close_fig(
            fig_i, 'analytic_approximation_alpha.jpg')

    def _plot_different_np_r(self,
                             radii: np.ndarray,
                             alpha: np.ndarray,
                             kappa: float,
                             a_r: np.ndarray,
                             co_factor: float,
                             ) -> None:
        """plot the data for different NP radius"""
        # pylint: disable=too-many-arguments
        phi_r_list: list[np.ndarray] = []
        for r_np in range(1, 6):
            phi_r = np.zeros(radii.shape)
            for alpha_i in alpha:
                alpha_exp = alpha_i * np.exp(-kappa * (radii - r_np))
                radial_term = alpha_exp * a_r
                log_argument = (1.0 + radial_term) / (1.0 - radial_term)
                # Check if log_argument is valid for np.log
                valid_indices = log_argument > 0
                # Only update phi_r for valid indices
                phi_r[valid_indices] += \
                    co_factor * np.log(log_argument[valid_indices])
            phi_r_list.append(phi_r / len(alpha))
        self.plot_diff_np_r_graphs(radii, phi_r_list)

    def plot_diff_np_r_graphs(self,
                              radii: np.ndarray,
                              phi_r_list: list[np.ndarray]
                              ) -> None:
        """plot the data for different NP radius"""
        fig_i, ax_i = elsevier_plot_tools.mk_canvas('single_column')
        for r_np, phi_r in enumerate(phi_r_list, 1):
            ax_i.plot(radii,
                      phi_r,
                      color=self.configs.colors[r_np],
                      label=rf'$r_{{np}}$={r_np:.1f}')
        ax_i.set_xlabel('r')
        ax_i.set_ylabel('y')
        y_lo: float = ax_i.get_ylim()[0]
        plt.xlim(2, 10)
        ax_i.set_ylim(y_lo, 15)
        self._add_text(ax_i, text=f'$\\alpha$={1.0:.1f}')
        elsevier_plot_tools.save_close_fig(
            fig_i, 'analytic_approximation_r_np.jpg')

    def _add_text(self,
                  ax_i: plt.Axes,
                  text: str
                  ) -> None:
        """add text to the plot"""
        ax_i.text(7, 13.5, text,
                  horizontalalignment='center',
                  verticalalignment='center',
                  fontsize=elsevier_plot_tools.LABEL_FONT_SIZE_PT,
                  )

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{AnalyticAnalysis.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    print(f'{bcolors.CAUTION}This module is not meant to be run '
          f'independently.{bcolors.ENDC}\n')
