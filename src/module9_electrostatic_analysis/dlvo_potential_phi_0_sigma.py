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

import typing
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from common import logger, elsevier_plot_tools
from common.colors_text import TextColor as bcolors
from module9_electrostatic_analysis.dlvo_potential_phi_zero import \
    DLVOPotentialPhiZero
from module9_electrostatic_analysis.dlvo_potential_configs import \
    AllConfig, PhiZeroSigmaConfig


# Helper functions
# ----------------
def check_no_kwargs(kwargs: typing.Any,
                    log: logger.logging.Logger
                    ) -> None:
    """raise an error if the kwargs are not given"""
    if not kwargs:
        msg: str = '\nNo kwargs are given!\n'
        log.error(msg)
        raise ValueError(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')


def check_all_kwargs_exist(kwargs: typing.Any,
                           expected_kwargs: list[str],
                           log: logger.logging.Logger
                           ) -> None:
    """check if all the expected kwargs are given"""
    for key in expected_kwargs:
        if key not in kwargs:
            msg: str = f'\n{key} is not given!\n'
            log.error(msg)
            raise ValueError(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')


def check_unkown_kwargs(kwargs: typing.Any,
                        expected_kwargs: list[str],
                        log: logger.logging.Logger
                        ) -> None:
    """check if there are any unknown kwargs"""
    for key in kwargs:
        if key not in expected_kwargs:
            msg: str = f'\n{key} is unknown!\n'
            log.error(msg)
            raise ValueError(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')
# ----------------


class PhiZeroSigma:
    """compute the phi_0 with respect to sigma"""
    # pylint: disable=too-many-instance-attributes
    info_msg: str = 'Message from PhiZeroSigma:\n'

    configs: AllConfig
    debye_md: float
    debye_exp: list[float]
    charge_density_range: tuple[float, float]

    def __init__(self,
                 log: logger.logging.Logger,
                 configs: AllConfig,
                 **kwargs: typing.Dict[str, typing.Any],
                 ) -> None:
        self.validate_kwargs(kwargs, log)
        self.assign_kwargs(kwargs)
        self.configs = configs
        self.compute_phi_0(log)
        self.write_msg(log)

    def validate_kwargs(self,
                        kwargs: typing.Dict[str, typing.Any],
                        log: logger.logging.Logger
                        ) -> None:
        """Check if all the inputs are valid"""
        expected_kwargs = ['debye_md', 'charge_density_range']

        check_no_kwargs(kwargs, log)
        check_all_kwargs_exist(kwargs, expected_kwargs, log)
        check_unkown_kwargs(kwargs, expected_kwargs, log)

        self.info_msg += f'\tThe following {kwargs.keys() = } are given\n'

    def assign_kwargs(self,
                      kwargs: typing.Dict[str, typing.Any]
                      ) -> None:
        """assign the kwargs to the class"""
        self.debye_md = kwargs['debye_md']
        self.charge_density_range = kwargs['charge_density_range']

    def compute_phi_0(self,
                      log: logger.logging.Logger
                      ) -> None:
        """compute the phi_0 with respect to sigma"""
        configs = self.configs.phi_zero_sigma_config
        self.lambda_exp = self._compute_lambda_exp(configs)
        denisty_range: np.ndarray = \
            np.linspace(*self.charge_density_range, configs.nr_density_points)

    def _compute_lambda_exp(self,
                            configs: "PhiZeroSigmaConfig"
                            ) -> list[float]:
        """compute the Debye length from the experimental data
        from concentrations of salts
        """
        lambda_exp: list[float] = []
        for ionic_strength in configs.exp_salt_concentration:
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
        print(f'{bcolors.OKGREEN}{DLVOPotentialPhiZero.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    PhiZeroSigma(logger.logging.Logger('phi_0_sigma.log'),
                 AllConfig(),
                 debye_md=0.1,
                 charge_density_range=(0.1, 0.2)
                 )
