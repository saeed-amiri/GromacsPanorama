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

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{CompareChrages.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    CompareChrages(log=logger.setup_logger('compare_charges.log'))
