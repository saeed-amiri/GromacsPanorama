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

Adding the a non-linearized version of the Poisson-Boltzmann equation
for the sphere system based on the  Lee.R. White
DOI: https://doi.org/10.1039/F29777300577
It wriiten in a separate file, since this one gettinge too long.
Also the script is splited into separate files for better readability.
11 Jun 2024
"""

from datetime import datetime

import numpy as np

from common import logger
from common.colors_text import TextColor as bcolors
from module9_electrostatic_analysis.dlvo_potential_plot import \
    PlotPotential
from module9_electrostatic_analysis.dlvo_potential_ionic_strength import \
    IonicStrengthCalculation
from module9_electrostatic_analysis.dlvo_potential_charge_density import \
    ChargeDensity
from module9_electrostatic_analysis.dlvo_potential_non_linear_aprox import \
    NonLinearPotential
from module9_electrostatic_analysis.dlvo_potential_phi_zero import \
    DLVOPotentialPhiZero
from module9_electrostatic_analysis.dlvo_potential_phi_0_sigma import \
    PhiZeroSigma
from module9_electrostatic_analysis.dlvo_potential_configs import AllConfig


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
        debye_l: float = self.initiate(log)
        self.compare_experiments(debye_l, log)
        self.write_msg(log)

    def initiate(self,
                 log: logger.logging.Logger
                 ) -> float:
        """initiate computation by finding debye length"""
        charge_info = ChargeDensity(log, self.configs)
        self.charge, self.charge_density = \
            charge_info.charge, charge_info.density
        if self.configs.ionic_type == 'all':
            ionic_strength: float = IonicStrengthCalculation(
                'topol.top', log, self.configs).ionic_strength
            self.info_msg += f'\tUsing all charge groups {ionic_strength = }\n'
        elif self.configs.ionic_type == 'salt':
            ionic_strength = self.configs.phi_parameters['c_salt']
            self.info_msg += \
                f'\tUsing salt concentration: {ionic_strength} Mol\n'
        debye_l: float = self.get_debye(ionic_strength)
        radii: np.ndarray
        phi_r: np.ndarray
        radii, phi_r = self.compute_potential(debye_l, log)

        self.plot_save_phi(radii, phi_r, debye_l, log)
        return debye_l

    def get_debye(self,
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
            f'\t`{debye_l_nm = :.4f}` [nm]\n'
            '\t`computation_radius = '
            f'{self.configs.computation_radius/10.0:.4f}` [nm]\n\t'
            f'kappa * r = {self.configs.computation_radius/10/debye_l_nm:.3f}'
            '\n')
        return float(debye_l_nm)

    def compute_potential(self,
                          debye_l: float,
                          log: logger.logging.Logger
                          ) -> tuple[np.ndarray, np.ndarray]:
        """
        compute phi_r with different approximations
        """
        s_config: "AllConfig" = self.configs.solving_config
        box_lim: float = self.configs.phi_parameters['box_xlim']
        phi_0: np.ndarray
        phi_0_at_denisty_0: np.ndarray
        phi_0, phi_0_at_denisty_0 = self._get_phi_zero(debye_l, log)

        radii: np.ndarray
        phi_r: np.ndarray
        if (compute_type := s_config.compute_type) == 'planar':
            radii, phi_r = self._linear_planar_possion(
                debye_l, phi_0, box_lim)
        elif compute_type == 'sphere':
            radii, phi_r = self._linear_shpere(
                debye_l, phi_0, box_lim/2)
        elif compute_type == 'non_linear':
            radii, phi_r = self._non_linear_sphere_possion(
                debye_l, phi_0, log)
            if self.configs.remove_phi_r_density_0:
                _, phi_r_at_zero = self._non_linear_sphere_possion(
                    debye_l, phi_0_at_denisty_0, log)
                phi_r -= phi_r_at_zero
        else:
            raise ValueError(
                f'\t\n{bcolors.FAIL}Unknown computation type: {compute_type}'
                f'{bcolors.ENDC}\n')
        return radii, phi_r

    def _get_phi_zero(self,
                      debye_l: float,
                      log: logger.logging.Logger
                      ) -> tuple[np.ndarray, np.ndarray]:
        """get the value of the phi_0 based on the configuration"""
        phi_0_compute = DLVOPotentialPhiZero(
            debye_length=debye_l,
            charge=self.charge,
            charge_density=self.charge_density,
            configs=self.configs,
            ion_strength=self.configs.phi_parameters['c_salt'],
            log=log)
        if self.configs.remove_phi_0_density_0:
            phi_0_compute_density_0: np.ndarray = \
                self.compute_phi_0_zero_density(debye_l, log)
            return phi_0_compute.phi_0, phi_0_compute_density_0
        return phi_0_compute.phi_0, np.zeros(self.charge_density.shape)

    def compute_phi_0_zero_density(self,
                                   debye_l: float,
                                   log: logger.logging.Logger
                                   ) -> np.ndarray:
        """remove the potentioal of zero density from the phi_0"""
        zero_density: np.ndarray = np.zeros(self.charge_density.shape)
        zero_charge: np.ndarray = np.zeros(self.charge.shape)
        phi_0_at_denisty_0 = DLVOPotentialPhiZero(
            debye_length=debye_l,
            charge_density=zero_density,
            charge=zero_charge,
            configs=self.configs,
            ion_strength=self.configs.phi_parameters['c_salt'],
            log=log)
        return phi_0_at_denisty_0.phi_0

    def _linear_planar_possion(self,
                               debye_l: float,
                               phi_0: np.ndarray,
                               box_lim: float,
                               ) -> tuple[np.ndarray, np.ndarray]:
        """compute the potential based on the linearized Possion-Boltzmann
        relation:
        phi(r) = phi_0 * exp(-r/debye_l)
        pp. 96, Surface and Interfacial Forces, H-J Burr and M.Kappl
        """
        radii: np.ndarray = np.linspace(0, box_lim, len(self.charge))
        phi_r: np.ndarray = np.zeros(radii.shape)

        for phi in phi_0:
            phi_r += phi * np.exp(-radii/debye_l)

        phi_r /= len(radii)
        return radii, phi_r

    def _linear_shpere(self,
                       debye_l: float,
                       phi_0: np.ndarray,
                       box_lim: float,
                       ) -> tuple[np.ndarray, np.ndarray]:
        """compute the potential based on the linearized Possion-Boltzmann
        relation for a sphere:
        phi(r) = phi_0 * (r_np/r) * exp(-(r-r_np)/debye_l)
        r_np := the nanoparticle radius
        pp. 110, Surface and Interfacial Forces, H-J Burr and M.Kappl
        """
        r_np: float = self.configs.computation_radius / 10.0  # [A] -> [nm]
        radii: np.ndarray = np.linspace(r_np, box_lim, len(self.charge))
        phi_r: np.ndarray = np.zeros(radii.shape)
        for phi in phi_0:
            phi_r += \
                phi * (r_np/radii) * np.exp(-(radii-r_np)/debye_l)
        phi_r /= len(radii)
        return radii, phi_r

    def _non_linear_sphere_possion(self,
                                   debye_l: float,
                                   phi_0: np.ndarray,
                                   log: logger.logging.Logger
                                   ) -> tuple[np.ndarray, np.ndarray]:
        """compute the non-linearized Possion-Boltzmann equation for a
        sphere"""
        non_linear_pot = NonLinearPotential(
            debye_l, phi_0, log, self.charge, self.configs)
        return non_linear_pot.radii, non_linear_pot.phi_r

    def plot_save_phi(self,
                      radii: np.ndarray,
                      phi_r: np.ndarray,
                      debye_l: float,
                      log: logger.logging.Logger
                      ) -> None:
        """plot and save the electostatic potential"""
        PlotPotential(radii, phi_r, debye_l, self.configs, log)

    def compare_experiments(self,
                            debye_l: float,
                            log: logger.logging.Logger
                            ) -> None:
        """compare the experimental and the MD simulation results"""
        if self.configs.compare_phi_0_sigma:
            charge_density_range: tuple[float, float] = \
                (0, self.charge_density.max())
            PhiZeroSigma(log=log,
                         configs=self.configs,
                         kwargs={'debye_md': debye_l,
                                 'charge_density_range': charge_density_range
                                 })

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


if __name__ == '__main__':
    ElectroStaticComputation(logger.setup_logger('electro_pot.log'))
