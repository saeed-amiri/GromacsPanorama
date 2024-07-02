"""
Plot phi_0 with respect to sigma
=================================
This script is used to plot the DLVO potential for different values of
phi_0 and sigma.
The aim is the plot values of phi_0 with respect to sigma, for different
vlaues of lambda_D.
Lambda_D is for two different values: from simulation and from the
experimental data.
To determine the rationality of the DLVO modleing
"""

import os
import sys
import typing
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from common import logger, elsevier_plot_tools
from common.colors_text import TextColor as bcolors
from module9_electrostatic_analysis.dlvo_potential_phi_zero import \
    DLVOPotentialPhiZero
from module9_electrostatic_analysis.dlvo_potential_configs import AllConfig


class PhiZeroSigma:
    """compute the phi_0 with respect to sigma"""
    info_msg: str = 'Message from PhiZeroSigma:\n'
    configs: AllConfig
    lambda_md: float
    lambda_exp: list[float]

    def __init__(self,
                 lambda_md: float,  # Debye length from the MD simulation
                 log: logger.logging.Logger,
                 configs: AllConfig
                 ) -> None:
        self.configs = configs
        self.lambda_md = lambda_md
        self.compute_experiments_values(log)
        self.write_msg(log)

    def compute_experiments_values(self,
                                   log: logger.logging.Logger
                                   ) -> None:
        """compute the phi_0 with respect to sigma"""
        configs = self.configs.experiment_config
        self.lambda_exp = self._compute_lambda_exp(configs)

    def _compute_lambda_exp(self,
                            configs: AllConfig
                            ) -> list[float]:
        """compute the Debye length from the experimental data
        from concentrations of salts
        """
        lambda_exp: list[float] = []
        for ionic_strength in configs.salt_concentration:
            lambda_exp.append(self._get_debye(ionic_strength))
        return lambda_exp

    def _get_debye(self,
                   ionic_strength: float
                   ) -> float:
        """computing the debye length based on Poisson-Boltzmann apprx.
        See:
        pp. 96, Surface and Interfacial Forces, H-J Burr and M.Kappl
        """
        param: dict[str, float] = self.configs.phi_parameters

        # ionnic strength in mol/m^3
        ionic_str_mol_m3: float = ionic_strength * 1e3

        # Getting debye length
        debye_l: np.float64 = np.sqrt(
            param['T'] * param['k_boltzman_JK'] *
            param['epsilon'] * param['epsilon_0'] /
            (2 * ionic_str_mol_m3 * param['n_avogadro'] * param['e_charge']**2
             ))

        # convert to nm
        debye_l_nm = debye_l * 1e9

        self.info_msg += (
            f'\tSalt C is: `{ionic_strength}` -> `{debye_l_nm = :.4f}` [nm]\n')
        return float(debye_l_nm)

    def write_msg(self,
                  log: logger.logging.Logger  # To log
                  ) -> None:
        """write and log messages"""
        now = datetime.now()
        self.info_msg += \
            f'\tTime: {now.strftime("%Y-%m-%d %H:%M:%S")}\n'
        print(f'{bcolors.OKCYAN}{DLVOPotentialPhiZero.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    PhiZeroSigma(0.1, logger.logging.Logger('phi_0_sigma.log'), AllConfig())
