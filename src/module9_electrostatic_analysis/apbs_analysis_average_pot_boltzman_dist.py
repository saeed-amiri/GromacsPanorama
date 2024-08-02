"""
Compute the Boltzman distribution for ODA at the interface, the selceted
grid where NP cut threogh the oil phase
"""

from dataclasses import dataclass, field

import numpy as np

from common import logger
from common.colors_text import TextColor as bcolors


@dataclass
class BoltzmanConfig:
    """set all the configs and parameters
    """
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
        'mv_to_v': 1e-3,  # mV to V
        'oda_concentration': 0.003  # Molar concentration of the ODA!!
        })
    selected_grid: list[int] = field(default_factory=lambda: [90, 91, 92])


class ComputeBoltzmanDistribution:
    """
    Compute the Boltzman distribution for the input parameters
    """
    # pylint: disable=too-many-arguments
    info_msg: str = 'Message from ComputeBoltzmanDistribution:\n'
    config: BoltzmanConfig
    boltzman_distribution: dict[int, tuple[np.ndarray, np.ndarray]]

    def __init__(self,
                 cut_radial_average: list[np.ndarray],
                 sphere_grid_range: np.ndarray,
                 radii_list: list[np.ndarray],
                 log: logger.logging.Logger,
                 config: BoltzmanConfig = BoltzmanConfig()
                 ) -> None:
        self.config = config
        dict_index_phi: dict[int, tuple[np.ndarray, ...]] = \
            self.make_dict_index_phi(cut_radial_average,
                                     radii_list,
                                     sphere_grid_range)
        self.boltzman_distribution = \
            self.compute_boltzman_distribution(dict_index_phi)
        self.write_msg(log)

    def compute_boltzman_distribution(self,
                                      dict_index_phi: dict[
                                          int, tuple[np.ndarray, ...]],
                                      ) -> dict[int, tuple[np.ndarray, ...]]:
        """Compute the Boltzman distribution"""
        boltzman_dict: dict[int, tuple[np.ndarray, ...]] = {}
        for i, phi_i_radii in dict_index_phi.items():
            if i in self.config.selected_grid:
                dist = self.compute_distribution(phi_i_radii[0])
                boltzman_dict[i] = (dist, phi_i_radii[1])
        return boltzman_dict

    def compute_distribution(self,
                             phi_i: np.float64
                             ) -> np.ndarray:
        """Compute the distribution based on the equation 4.3, pp. 95"""
        param: dict[str, float] = self.config.parameters

        # Calculate thermal energy k_B * T
        kbt: float = param['T'] * param['k_boltzman_JK']

        co_eff: float = param['oda_concentration'] * 1e3 * \
            param['n_avogadro'] * param['e_charge']
        arg: float = param['e_charge'] * phi_i * 1e-3 / kbt
        return co_eff * np.exp(-arg)

    def make_dict_index_phi(self,
                            cut_radial_average: list[np.ndarray],
                            radii_list: list[np.ndarray],
                            sphere_grid_range: np.ndarray
                            ) -> dict[int, tuple[np.ndarray, ...]]:
        """Make a dictionary with the index of the sphere grid range
        and the corresponding phi_i"""
        dict_index_phi: dict[int, tuple[np.ndarray, ...]] = {}
        for i, (phi_i, radii) in enumerate(zip(cut_radial_average,
                                               radii_list)):
            dict_index_phi[sphere_grid_range[i]] = (phi_i, radii)
        return dict_index_phi

    def write_msg(self, log: logger.logging.Logger) -> None:
        """Write the message to the log
        """
        print(f'{bcolors.OKCYAN}{ComputeBoltzmanDistribution.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(f'{self.info_msg}\n')
