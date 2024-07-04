"""
Compute and compare the DLVO potential
The comparison are:
  1. The value of psi_0 for different computation radius, also compare
        the value of psi_0 from different PBE approximations
  2. The value of psi_r for different computation radius
  3. Comparison of psi_r for different cut radius in the APBS calculation

The experimental data also should be plotted!
"""

import numpy as np

import matplotlib.pyplot as plt

from common import logger, elsevier_plot_tools
from common.colors_text import TextColor as bcolors

from module9_electrostatic_analysis.dlvo_potential_charge_density import \
    ChargeDensity
from module9_electrostatic_analysis.dlvo_potential_non_linear_aprox import \
    NonLinearPotential
from module9_electrostatic_analysis.dlvo_potential_phi_zero import \
    DLVOPotentialPhiZero
from module9_electrostatic_analysis.dlvo_potential_phi_0_sigma import \
    PhiZeroSigma
from module9_electrostatic_analysis.dlvo_potential_configs import AllConfig, \
    ComparisonConfigs


class CompareChrages:
    """Comapre charge density of the systms in different cut off radius"""

    info_msg: str = 'Message from CompareChrages:\n'
    configs: AllConfig
    charge_density_dict: dict[str, np.ndarray]
    charge_dict: dict[str, np.ndarray]
    r_cuts: list[float]
    densities_ave: list[np.float64]

    def __init__(self,
                 log: logger.logging.Logger,
                 configs: AllConfig = AllConfig()
                 ) -> None:
        self.configs = configs
        self.compare_systems(log)
        self._write_msg(log)

    def compare_systems(self,
                        log: logger.logging.Logger
                        ) -> None:
        """compare the charge density of the systems"""
        s_configs: "ComparisonConfigs" = self.configs.comparison_configs
        self.charge_density_dict, self.charge_dict = \
            self.get_density_charge(s_configs, log)
        self.r_cuts, self.density_ave = self.get_average_charge_density(log)
        self.plot_charge_density()

    def get_density_charge(self,
                           s_configs: "ComparisonConfigs",
                           log: logger.logging.Logger
                           ) -> tuple[dict[str, np.ndarray],
                                      dict[str, np.ndarray]]:
        """get the charge density"""
        charge_density_dict: dict[str, np.ndarray] = {}
        charge_dict: dict[str, np.ndarray] = {}

        for key, charges_fname in s_configs.charge_files.items():
            _configs: AllConfig = self.configs
            _configs.charge_fname = charges_fname
            density_charge = ChargeDensity(
                log=log, configs=_configs)
            charge_density_dict[key] = density_charge.density
            charge_dict[key] = density_charge.charge

        return charge_density_dict, charge_dict

    def get_average_charge_density(self
                                   ) -> tuple[list[float], list[np.float64]]:
        """get the average charge density"""
        r_cuts: list[float] = \
            [float(i) for i in list(self.charge_density_dict.keys())]
        denisties: list[np.float64] = \
            [np.mean(i) for i in list(self.charge_density_dict.values())]
        return r_cuts, denisties

    def plot_charge_density(self,
                            ) -> None:
        """plot the charge density"""
        fig_i: plt.Figure
        ax_i: plt.Axes
        fig_i, ax_i = elsevier_plot_tools.mk_canvas(size_type='single_column')
        ax_i.plot(self.r_cuts,
                  self.densities_ave,
                  'o:',
                  label='Charge Density',
                  color='black',
                  lw=0.5,
                  markersize=3)
        ax_i.set_xlabel('Cut off radius [nm]')
        ax_i.set_ylabel(r'$\sigma$ [C/m$^2$]')
        elsevier_plot_tools.save_close_fig(
            fig_i, fname := 'charge_density_comparison.jpg')
        self.info_msg += (
            f'\tThe charge density is plotted and saved as `{fname}`\n')

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{CompareChrages.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


class ComparePhi_0:
    """Compare the phi_0 for different cut off radius"""

    info_msg: str = 'Message from ComparePhi_0:\n'
    configs: AllConfig
    phi_0_dict: dict[str, np.ndarray]

    def __init__(self,
                 r_cuts: list[float],
                 densities_ave: list[np.float64],
                 log: logger.logging.Logger,
                 configs: AllConfig = AllConfig()
                 ) -> None:
        self.configs = configs
        self.compare_systems(r_cuts, densities_ave, log)
        self._write_msg(log)

    def compare_systems(self,
                        log: logger.logging.Logger
                        ) -> None:
        """compare the phi_0 of the systems"""

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{ComparePhi_0.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    R_CUTS: list[float]
    DENISTY_AVG: list[np.float64]
    R_CUTS, DENISTY_AVG = \
        CompareChrages(log=logger.setup_logger('compare_charges.log'))
    ComparePhi_0(R_CUTS,
                 DENISTY_AVG,
                 log=logger.setup_logger('compare_phi_0.log'))
