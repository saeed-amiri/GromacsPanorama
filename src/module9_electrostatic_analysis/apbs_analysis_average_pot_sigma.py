"""
Compute the charge density based on the Grahame, Loeb, and Ohshima equations
and compare the results with the experimental data
"""

from dataclasses import dataclass, field

import numpy as np

from common import logger
from common.colors_text import TextColor as bcolors


@dataclass
class ComputeSigmaConfig:
    """set the parameters for the computation"""
    parameters: dict[str, float] = field(default_factory=lambda: {
        'T': 298.15,  # Temperature of the system
        'e_charge': 1.602e-19,  # Elementary charge [C]
        'c_salt': 0.00479,   # Bulk concentration of the salt in M(=mol/l)
        'epsilon': 78.5,  # medium  permittivity,
        'epsilon_0': 8.854187817e-12,   # vacuum permittivity, farads per meter
        'n_avogadro': 6.022e23,  # Avogadro's number
        'k_boltzman_JK': 1.380649e-23,  # Joules per Kelvin (J/K)
        'k_boltzman_eVK': 8.617333262145e-5,  # Electronvolts per Kelvin (eV/K)
        'box_xlim': 21.8,  # Length of the box in x direction [nm]
        'box_ylim': 21.8,  # Length of the box in y direction [nm]
        'box_zlim': 22.5,  # Length of the box in z direction [nm] whole box
        'mv_to_v': 1e-3  # mV to V
        })
    equation_type: str = 'Grahame'  # Grahame, Loeb, Ohshima


class ComputeSigma:
    """Compute the charge density"""
    __slots__ = ['info_msg', 'config', 'sigma']
    info_msg: str
    config: ComputeSigmaConfig
    sigma: np.ndarray

    def __init__(self,
                 psi_zero: np.ndarray,
                 lambda_d: np.ndarray,
                 log: logger.logging.Logger,
                 config: ComputeSigmaConfig = ComputeSigmaConfig()
                 ) -> None:
        self.config = config
        self.info_msg = 'Message from ComputeSigma:\n'

        self.sigma = self.compute_sigma(psi_zero, lambda_d)
        self.write_msg(log)

    def compute_sigma(self,
                      psi_zero: np.ndarray,
                      lambda_d: np.ndarray,
                      ) -> np.ndarray:
        """Compute the charge density"""
        sigma: np.ndarray = np.zeros_like(psi_zero)
        if self.config.equation_type == 'Grahame':
            sigma = self.grahame_equation(psi_zero, lambda_d)
        return sigma

    def grahame_equation(self,
                         psi_zero: np.ndarray,
                         lambda_d: np.ndarray
                         ) -> np.ndarray:
        """Compute the charge density based on the Grahame equation"""
        param: dict[str, float] = self.config.parameters
        # Calculate thermal energy k_B * T
        kbt: float = param['T'] * param['k_boltzman_JK']
        # Calculate permittivity epsilon * epsilon_0
        epsilon: float = param['epsilon'] * param['epsilon_0']

        coef_lambda: np.ndarray = \
            2 * epsilon * kbt / param['e_charge'] / lambda_d / 1e-9

        # Calculate the coefficient for the surface charge density term
        arg: np.ndarray = param['e_charge'] * psi_zero * 1e-3 / (2.0 * kbt)
        return coef_lambda * np.sinh(arg)

    def write_msg(self,
                  log: logger.logging.Logger
                  ) -> None:
        """Write the message to the log"""
        print(f'{bcolors.OKCYAN}{ComputeSigmaConfig.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(f'{self.info_msg}\n')
