"""
DLVO Electrostatic Potential Computation Module

This module calculates the electrostatic potential (phi) around a
nanoparticle (NP) at the water-decane interface using the DLVO model.
The computation requires an initial charge density (sigma) on the
surface of the NP, typically assumed to be uniformly distributed.

Prerequisites:
- The initial charge density is derived from spatially filtered and
    analyzed molecular dynamics simulation outputs, specifically using
    `spatial_filtering_and_analysing_trr.py`.
- It's important to note that the actual charge distribution might not
    be uniform due to the NP's partial immersion in the oil phase,
    affecting the electrostatic interactions.

Key Features:
- This module integrates with other computational modules to refine
    sigma by considering the NP's contact angle at the water-
    decane interface. This angle helps determine the extent of the
    NP's surface exposed to the water phase, where the charges are
    predominantly located.
- In the absence of explicit contact angle data (from `contact.xvg`),
    the module estimates sigma using an average contact angle
    value, ensuring robustness in calculations.
- The main output is the computed electrostatic potential phi,
    essential for understanding the colloidal stability and
    interactions governed by the DLVO theory.

Inputs:
- `charge_df.xvg`: File containing the charge density distribution
    data.
- `contact.xvg` (optional): File containing contact angle measurements.
    If unavailable, an average contact angle is used for calculations.

Physics Overview:
The module employs the linearized Poisson-Boltzmann equation to model
electrostatic interactions, considering the influence of the NP's
geometry and surface charge distribution on the potential phi.
This approach allows for the quantification of repulsive forces
between colloidal particles, crucial for predicting system stability.

Output:
The module calculates and returns the electrostatic potential phi
surrounding the NP, offering insights into the colloidal interactions
within the system.

Usage:
To use this module, ensure the prerequisite files are correctly
formatted and located in the specified directory. The module can be
executed as a standalone script or integrated into larger simulation
frameworks for comprehensive colloidal system analysis.
22 Feb 2024
Saeed
"""

from datetime import datetime
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import matplotlib.pylab as plt

from common import logger, xvg_to_dataframe
from common.colors_text import TextColor as bcolors


@dataclass
class FileConfig:
    """set the name of the input files"""
    charge_fname: str = 'charge_df.xvg'
    contact_fname: str = 'contact.xvg'
    fout: str = 'potential.xvg'


@dataclass
class ParameterConfig:
    """set parameters for the phi calculation
    radius of the nanopartcile is mandatory
    contact angle, is optioanl, it is used in case the contact file is
    not availabel
    """
    np_radius: float = 30.0  # In Ångströms
    avg_contact_angle: float = 36.0  # In Degrees
    # Parameters for the phi computation
    phi_parameters: dict[str, float] = field(default_factory=lambda: {
        'T': 298.15,  # Temperature of the system
        'c_salt': .01,   # Bulk concentration of the salt in M(=mol/l)
        'epsilon': 78.5,  # medium  permittivity,
        'epsilon_0': 8.854187817e-12,   # vacuum permittivity, farads per meter
        'n_avogadro': 6.022e23,  # Avogadro's number
        'k_boltzman_JK': 1.380649e-23,  # Joules per Kelvin (J/K)
        'k_boltzman_eVK': 8.617333262145e-5,  # Electronvolts per Kelvin (eV/K)
        'phi_0': 1.0e-9,  # The potential at zero point (V)
        'box_xlim': 12.3,  # Length of the box in x direction [nm]
        'box_ylim': 12.3,  # Length of the box in y direction [nm]
        'box_zlim': 12.3  # Length of the box in z direction [nm]
    })


@dataclass
class AllConfig(FileConfig, ParameterConfig):
    """set all the parameters and confiurations
    computation types:
    compute_type:
        planar: Linearized Possion-Boltzmann for planar approximation
        sheperical: Linearized Possion-Boltzmann for a sphere
    phi_0_set:
        grahame: Grahame equation
        constant: from a constant value
    """
    compute_type: str = 'sphere'
    phi_0_type: str = 'grahame'


class ElectroStaticComputation:
    """compute the electrostatic potenstial"""

    info_msg: str = 'Message from ElectroStaticComputation:\n'
    configs: AllConfig
    charge_density: np.ndarray
    charge: np.ndarray

    def __init__(self,
                 log: logger.logging.Logger,
                 configs: AllConfig = AllConfig()
                 ) -> None:
        self.configs = configs
        self.initiate(log)
        self.write_msg(log)

    def initiate(self,
                 log: logger.logging.Logger
                 ) -> None:
        """initiate computation by finding debye length"""
        charge_info = ChargeDensity(log, self.configs)
        self.charge, self.charge_density = \
            charge_info.density, charge_info.density
        debye_l: np.ndarray = self.get_debye()
        self.compute_potential(debye_l)

    def get_debye(self) -> np.ndarray:
        """computing the debye length based on Poisson-Boltzmann apprx.
        See:
        pp. 96, Surface and Interfacial Forces, H-J Burr and M.Kappl
        """
        param: dict[str, float] = self.configs.phi_parameters
        debye_l: np.ndarray = np.sqrt(param['T'] * param['k_boltzman_JK'] *
                                      param['epsilon'] * param['epsilon_0'] /
                                      (2 * param['c_salt'])) / self.charge
        return debye_l*1e15

    def compute_potential(self,
                          debye_l: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray]:
        """
        compute phi_r with different approximations
        """
        box_lim: float = self.configs.phi_parameters['box_xlim']
        phi_0: np.ndarray = self._get_phi_zero(debye_l)

        radii: np.ndarray
        phi_r: np.ndarray

        if (compute_type := self.configs.compute_type) == 'planar':
            radii, phi_r = self._linear_planar_possion(debye_l, phi_0, box_lim)
        elif compute_type == 'sphere':
            radii, phi_r = self._linear_shpere(debye_l, phi_0, box_lim)
        plt.plot(radii, phi_r)
        plt.show()
        return radii, phi_r

    def _get_phi_zero(self,
                      debye_l: np.ndarray
                      ) -> np.ndarray:
        """get the value of the phi_0 based on the configuration"""
        phi_0: np.ndarray = np.zeros(debye_l.shape)
        if (phi_0_type := self.configs.phi_0_type) == 'constant':
            phi_0 += self.configs.phi_parameters['phi_0']
        elif phi_0_type == 'grahame':
            phi_0 = self._compute_phi_0_grahame(debye_l)
        self.info_msg += \
            f'\tAvg. {phi_0.mean() = :.3f} from `{phi_0_type}` vlaues\n'
        return phi_0

    def _linear_planar_possion(self,
                               debye_l: np.ndarray,
                               phi_0: np.ndarray,
                               box_lim: float,
                               ) -> tuple[np.ndarray, np.ndarray]:
        """compute the potential based on the linearized Possion-Boltzmann
        relation:
        phi(r) = phi_0 * exp(-r/debye_l)
        pp. 96, Surface and Interfacial Forces, H-J Burr and M.Kappl
        """
        radii: np.ndarray = np.linspace(0, box_lim, len(self.charge))
        phi_r: np.ndarray = phi_0 * np.exp(-radii/debye_l)
        return radii, phi_r

    def _linear_shpere(self,
                       debye_l: np.ndarray,
                       phi_0: np.ndarray,
                       box_lim: float,
                       ) -> tuple[np.ndarray, np.ndarray]:
        """compute the potential based on the linearized Possion-Boltzmann
        relation for a sphere:
        phi(r) = phi_0 * (r_np/r) * exp(-(r-r_np)/debye_l)
        r_np := the nanoparticle radius
        pp. 110, Surface and Interfacial Forces, H-J Burr and M.Kappl
        """
        r_np: float = self.configs.np_radius / 10.0  # [nm]
        radii: np.ndarray = np.linspace(r_np, box_lim, len(self.charge))
        phi_r: np.ndarray = np.zeros(debye_l.shape)
        for phi, debye in zip(phi_0, debye_l):
            phi_r += \
                phi * (r_np/radii) * np.exp(-(radii-r_np)/debye)
        phi_r /= len(debye_l)
        return radii, phi_r

    def _compute_phi_0_grahame(self,
                               debye_l: np.ndarray
                               ) -> np.ndarray:
        """compute the phi_0 based on the linearized Possion-Boltzmann
        relation:
        phi_0 = debye_l * q_density / (epsilon * epsilon_0)
        pp. 102, Surface and Interfacial Forces, H-J Burr and M.Kappl
        """
        phi_0: np.ndarray = debye_l * self.charge_density / (
            self.configs.phi_parameters['epsilon'] *
            self.configs.phi_parameters['epsilon_0'])
        return phi_0*1e-9

    def write_msg(self,
                  log: logger.logging.Logger  # To log
                  ) -> None:
        """write and log messages"""
        now = datetime.now()
        self.info_msg += \
            f'\tTime: {now.strftime("%Y-%m-%d %H:%M:%S")}\n'
        print(f'{bcolors.OKCYAN}{ElectroStaticComputation.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


class ChargeDensity:
    """compute the true density for the part inside water"""
    # pylint: disable=too-few-public-methods

    info_msg: str = 'Message from ChargeDensity:\n'
    density: np.ndarray
    charge: np.ndarray

    def __init__(self,
                 log: logger.logging.Logger,
                 configs: AllConfig
                 ) -> None:
        self.charge, self.density = self._get_density(configs, log)
        self._write_msg(log)

    def _get_density(self,
                     configs: AllConfig,
                     log: logger.logging.Logger
                     ) -> tuple[np.ndarray, np.ndarray]:
        """read the input files and compute the charge desnity"""
        charge: np.ndarray = \
            self._get_column(configs.charge_fname, log, column='total')
        try:
            contact_angle: np.ndarray = \
                self._get_column(configs.contact_fname,
                                 log,
                                 column='contact_angles',
                                 if_exit=False)
            self.info_msg += \
                f'\tContact angle is getting from `{configs.contact_fname}`\n'
        except FileNotFoundError as _:
            self.info_msg += \
                (f'\t`{configs.contact_fname}` not found!\n\tAaverage '
                 f'angle `{configs.avg_contact_angle}` is used.\n')
            contact_angle = np.zeros(charge.shape)
            contact_angle += configs.avg_contact_angle

        cap_surface: np.ndarray = \
            self._compute_under_water_area(configs.np_radius, contact_angle)

        density: np.ndarray = charge / cap_surface
        return charge, density

    def _compute_under_water_area(self,
                                  np_radius: float,
                                  contact_angle: np.ndarray
                                  ) -> np.ndarray:
        """
        Compute the area under water for a NP at a given contact angle.

        The area is calculated based on the NP's radius and the contact
        angle, assuming the contact angle provides the extent of
        exposure to water.

        Parameters:
        - np_radius: Radius of the NP in Ångströms.
        - contact_angle: Contact angle(s) in degrees.

        Returns:
        - The area of the NP's surface exposed to water, in nm^2.
        """
        # Convert angles from degrees to radians for mathematical operations
        radians: np.ndarray = np.deg2rad(contact_angle)

        # Compute the surface area of the cap exposed to water
        # Formula: A = 2 * π * r^2 * (1 + cos(θ))
        # Alco converted from Ångströms^2 to nm^2
        in_water_cap_area: np.ndarray = \
            2 * np.pi * np_radius**2 * (1 + np.cos(radians)) / 100

        return in_water_cap_area

    def _get_column(self,
                    fname: str,
                    log: logger.logging.Logger,
                    column: str,
                    if_exit: bool = True
                    ) -> np.ndarray:
        """read the file and return the column in an array"""
        df_i: pd.DataFrame = \
            xvg_to_dataframe.XvgParser(fname, log, if_exit=if_exit).xvg_df
        return df_i[column].to_numpy(dtype=float)

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        now = datetime.now()
        self.info_msg += \
            f'\tTime: {now.strftime("%Y-%m-%d %H:%M:%S")}\n'
        print(f'{bcolors.OKCYAN}{ChargeDensity.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    ElectroStaticComputation(logger.setup_logger('electro_pot.log'))
