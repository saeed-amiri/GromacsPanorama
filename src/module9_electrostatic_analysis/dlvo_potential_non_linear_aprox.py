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

from datetime import datetime
import numpy as np

from common import logger
from common.colors_text import TextColor as bcolors
from module9_electrostatic_analysis.dlvo_potential_configs import \
    AllConfig


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
                 phi_0: float,
                 log: logger.logging.Logger,
                 charge: np.ndarray,
                 configs: AllConfig = AllConfig()
                 ) -> None:
        """write and log messages"""
        self.configs = configs
        self.radii, self.phi_r = \
            self.compute_potential(debye_l, phi_0, charge)
        self._write_msg(log)

    def compute_potential(self,
                          debye_l: float,
                          phi_0: float,
                          charge: np.ndarray
                          ) -> None:
        """compute the DLVO potential"""
        box_lim: float = self.configs.phi_parameters['box_xlim']
        r_np: float = self.configs.np_radius / 10.0  # [A] -> [nm]
        radii: np.ndarray = np.linspace(r_np, box_lim, len(charge))
        phi_r: np.ndarray = np.zeros(radii.shape)
        return radii, phi_r

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        now = datetime.now()
        self.info_msg += \
            f'\tTime: {now.strftime("%Y-%m-%d %H:%M:%S")}\n'
        print(f'{bcolors.OKCYAN}{NonLinearPotential.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')


if __name__ == '__main__':
    print(f'{bcolors.CAUTION}This module is not meant to be run '
          f'independently.{bcolors.ENDC}\n')
