"""
Compute the Boltzman distribution for ODA at the interface, the selceted
grid where NP cut threogh the oil phase
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from common import logger
from common import file_writer
from common.colors_text import TextColor as bcolors


@dataclass
class BoltzmanConfig:
    """set all the configs and parameters
    """
    parameters: dict[str, float] = field(default_factory=lambda: {
        'T': 298.15,  # Temperature of the system
        'e_charge': 1.602e-19,  # Elementary charge [C]
        'c_salt': 0.00479,  # Bulk concentration of the salt in M(=mol/l)
        'epsilon': 78.5,  # medium  permittivity,
        'epsilon_0': 8.854187817e-12,  # vacuum permittivity, farads per meter
        'n_avogadro': 6.022e23,  # Avogadro's number
        'k_boltzmann_JK': 1.380649e-23,  # Joules per Kelvin (J/K)
        'k_boltzmann_eVK': 8.617333262145e-5,  # Electronvolts per Kelvin, eV/K
        'box_xlim': 21.8,  # Length of the box in x direction [nm]
        'box_ylim': 21.8,  # Length of the box in y direction [nm]
        'box_zlim': 22.5,  # Length of the box in z direction [nm] whole box
        'mv_to_v': 1e-3,  # mV to V
        'oda_concentration': 0.003  # Molar concentration of the ODA!!
        })
    selected_grid: list[int] = field(default_factory=lambda: [90, 91, 92])
    min_grid_to_write: int = 85
    # if True, the boltzmann coefficient will be calculated, otherwise
    # the concentration will be cnsidered as the input
    if_boltzmann_coeff: bool = True


class ComputeBoltzmanDistribution:
    """
    Compute the Boltzman distribution for the input parameters
    """
    # pylint: disable=too-many-arguments
    __slots__ = [
        'info_msg', 'config', 'boltzmann_distribution', 'all_distribution']
    info_msg: str
    config: BoltzmanConfig
    boltzmann_distribution: dict[int, tuple[np.ndarray, np.ndarray]]
    all_distribution: dict[int, tuple[np.ndarray, np.ndarray]]

    def __init__(self,
                 cut_radial_average: list[np.ndarray],
                 sphere_grid_range: np.ndarray,
                 radii_list: list[np.ndarray],
                 log: logger.logging.Logger,
                 config: BoltzmanConfig = BoltzmanConfig()
                 ) -> None:
        self.info_msg = 'Message from ComputeBoltzmanDistribution:\n'
        self.config = config
        dict_index_phi: dict[int, tuple[np.ndarray, np.ndarray]] = \
            self.make_dict_index_phi(cut_radial_average,
                                     radii_list,
                                     sphere_grid_range)
        self.boltzmann_distribution, self.all_distribution = \
            self.compute_boltzmann_distribution(dict_index_phi)
        self.write_xvg(log)
        self.write_msg(log)

    def compute_boltzmann_distribution(self,
                                       dict_index_phi: dict[
                                          int, tuple[np.ndarray, np.ndarray]],
                                       ) -> tuple[
                                           dict[int, tuple[np.ndarray,
                                                           np.ndarray]],
                                           dict[int, tuple[np.ndarray,
                                                           np.ndarray]]]:
        """Compute the Boltzman distribution"""
        boltzmann_dict: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        boltzmann_dict_to_write: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for i, phi_i_radii in dict_index_phi.items():
            dist = self.compute_distribution(phi_i_radii[0])
            if i > self.config.min_grid_to_write:
                boltzmann_dict_to_write[i] = (dist, phi_i_radii[1])
            if i in self.config.selected_grid:
                boltzmann_dict[i] = (dist, phi_i_radii[1])
        return boltzmann_dict, boltzmann_dict_to_write

    def compute_distribution(self,
                             phi_i: np.ndarray
                             ) -> np.ndarray:
        """Compute the distribution based on the equation 4.3, pp. 95"""
        param: dict[str, float] = self.config.parameters

        # Calculate thermal energy k_B * T
        kbt: float = param['T'] * param['k_boltzmann_JK']

        if not self.config.if_boltzmann_coeff:
            co_eff: float = param['oda_concentration'] * 1e3 * \
                param['n_avogadro'] * param['e_charge']
        else:
            co_eff = 1.0
        arg: np.ndarray = param['e_charge'] * phi_i * 1e-3 / kbt
        return co_eff * np.exp(-arg)

    def make_dict_index_phi(self,
                            cut_radial_average: list[np.ndarray],
                            radii_list: list[np.ndarray],
                            sphere_grid_range: np.ndarray
                            ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Make a dictionary with the index of the sphere grid range
        and the corresponding phi_i"""
        dict_index_phi: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for i, (phi_i, radii) in enumerate(zip(cut_radial_average,
                                               radii_list)):
            dict_index_phi[sphere_grid_range[i]] = (phi_i, radii)
        return dict_index_phi

    def compute_concentration_from_surface_adsorption(self) -> None:
        """Compute the concentration from the surface adsorption
        https://pubs.acs.org/doi/pdf/10.1021/j100873a020

        The surface excess concentration, S_c, can be defined as the
        total surface excess of surfactant, E_s, divided by the total
        surface generated during the determination, S_c = E_a/S, where
        E_B is expressed in moles and S is in square centimeters. The
        total surface excess, in turn, is given by:
        E_a = W_t(C_i — C_b)*10-3,
        where W_t is the weight of the collapsed foam in grams, and C_t
        and C_b are the concentrations of surfactant in moles per liter,
        in the collapsed foam and the bulk solution, respectively.
        This expression assumes that the bulk concentration remains
        constant during the determination.

        """

    def write_xvg(self,
                  log: logger.logging.Logger) -> None:
        """Write the distribution to the xvg file
        """
        dist_dist: dict[int, np.ndarray] = \
            {k: v[1] for k, v in self.all_distribution.items()}
        df_i: pd.DataFrame = pd.DataFrame.from_dict(dist_dist,
                                                    orient='columns')
        file_writer.write_xvg(df_i=df_i,
                              log=log,
                              fname='boltzman_distribution.xvg',
                              extra_comments=['# c/c_0'],
                              xaxis_label='r [nm]',
                              yaxis_label='c/c_0'
                              )
        del df_i
        del dist_dist

    def write_msg(self, log: logger.logging.Logger) -> None:
        """Write the message to the log
        """
        print(f'{bcolors.OKCYAN}{ComputeBoltzmanDistribution.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(f'{self.info_msg}\n')
