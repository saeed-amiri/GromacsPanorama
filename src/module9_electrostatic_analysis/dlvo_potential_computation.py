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
from scipy.optimize import fsolve, root


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
from module9_electrostatic_analysis.dlvo_potential_configs import AllConfig


class ElectroStaticComputation:
    """compute the electrostatic potenstial"""

    info_msg: str = 'Message from ElectroStaticComputation:\n'
    configs: AllConfig
    charge_density: np.ndarray
    charge: np.ndarray
    iter_flase_report_flag: bool = False

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
            f'\t`{debye_l_nm = :.4f}` [nm]\n\t'
            rf'$\kappa$ r ={debye_l_nm*self.configs.np_radius/10:.3f}'
            '\n')
        return float(debye_l_nm)

    def compute_potential(self,
                          debye_l: float,
                          log: logger.logging.Logger
                          ) -> tuple[np.ndarray, np.ndarray]:
        """
        compute phi_r with different approximations
        """
        box_lim: float = self.configs.phi_parameters['box_xlim']
        phi_0: np.ndarray = self._get_phi_zero(debye_l, log)

        radii: np.ndarray
        phi_r: np.ndarray

        if (compute_type := self.configs.compute_type) == 'planar':
            radii, phi_r = self._linear_planar_possion(
                debye_l, phi_0, box_lim)
        elif compute_type == 'sphere':
            radii, phi_r = self._linear_shpere(
                debye_l, phi_0, box_lim/2)
        elif compute_type == 'non_linear':
            radii, phi_r = self._non_linear_sphere_possion(
                debye_l, phi_0, log)
        return radii, phi_r

    def _get_phi_zero(self,
                      debye_l: float,
                      log: logger.logging.Logger
                      ) -> np.ndarray:
        """get the value of the phi_0 based on the configuration"""
        phi_0: np.ndarray = np.zeros(self.charge.shape)
        if (phi_0_type := self.configs.phi_0_type) == 'constant':
            phi_0 += self.configs.phi_parameters['phi_0']
        elif phi_0_type == 'grahame_low':
            phi_0 = self._compute_phi_0_grahame_low_potential(debye_l)
        elif phi_0_type == 'grahame':
            phi_0 = self._compute_phi_0_grahame_nonlinear(debye_l, log)
        self.info_msg += (f'\tAvg. {phi_0.mean()*100 = :.3f} [mV] from `'
                          f'{phi_0_type}` values\n')
        return phi_0

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
        r_np: float = self.configs.np_radius / 10.0  # [A] -> [nm]
        radii: np.ndarray = np.linspace(r_np, box_lim, len(self.charge))
        phi_r: np.ndarray = np.zeros(radii.shape)
        for phi in phi_0:
            phi_r += \
                phi * (r_np/radii) * np.exp(-(radii-r_np)/debye_l)
        phi_r /= len(radii)
        return radii, phi_r

    def _compute_phi_0_grahame_low_potential(self,
                                             debye_l: float
                                             ) -> np.ndarray:
        """compute the phi_0 based on the linearized Possion-Boltzmann
        relation:
        phi_0 = debye_l * q_density / (epsilon * epsilon_0)
        pp. 103, Surface and Interfacial Forces, H-J Burr and M.Kappl
        """
        phi_0: np.ndarray = debye_l * 1e-9 * self.charge_density / (
            self.configs.phi_parameters['epsilon'] *
            self.configs.phi_parameters['epsilon_0'])
        return phi_0

    def _compute_phi_0_grahame(self,
                               debye_l: float
                               ) -> np.ndarray:
        """computing the potential at zero from equation 4.32
        phi_0 = (2k_B*T/e) sinh^-1(
            debye_l * q_density * e / (2 * epsilon * epsilon_0 * k_B * T))
        pp. 103, Surface and Interfacial Forces, H-J Burr and M.Kappl
        """
        param: dict[str, float] = self.configs.phi_parameters
        kbt: float = param['T'] * param['k_boltzman_JK']
        epsilon: float = param['epsilon'] * param['epsilon_0']
        args: np.ndarray = \
            debye_l * 1e-9 * param['e_charge'] * self.charge_density / \
            (2 * epsilon * kbt)
        phi_0: np.ndarray = 2 * kbt * np.arcsinh(args) / param['e_charge']
        return phi_0

    def _compute_phi_0_grahame_nonlinear(self,
                                         debye_l: float,
                                         log: logger.logging.Logger
                                         ) -> np.ndarray:
        """
        Compute the phi_0 based on the nonlinearized Poisson-Boltzmann
        relation for a sphere using the Grahame equation.

        This method numerically solves the equation 4.25 from pp. 101,
        Surface and Interfacial Forces, H-J Butt and M.Kappl.

        Parameters:
        debye_l (float): Debye length.

        Returns:
        np.ndarray: Computed phi_0 values.
        """
        param: dict[str, float] = self.configs.phi_parameters

        kbt: float = param['T'] * param['k_boltzman_JK']
        epsilon: float = param['epsilon'] * param['epsilon_0']
        r_np: float = self.configs.np_radius / 10.0  # [A] -> [nm]
        a_kappa: float = r_np / debye_l
        y_0: float = param['e_charge'] / (2.0 * kbt)
        co_factor: float = epsilon * epsilon / (y_0 * debye_l)

        phi_0: np.ndarray = np.zeros(self.charge.shape, dtype=np.float64)

        sigma = self.charge_density.copy()
        y_initial_guess: float = 25.18

        self.iter_flase_report_flag = False

        for i, _ in enumerate(phi_0):
            phi_0[i] = self._root_phi_0(y_0,
                                        a_kappa,
                                        co_factor,
                                        sigma[i],
                                        y_initial_guess,
                                        log)

        self.info_msg += ('\tPhi_0 computed from numerical solution of '
                          'nonlinear equation from Grahame relation\n')

        if all(phi_0) == y_initial_guess:
            print(f"{bcolors.CAUTION}\tWarning: phi_0 did not converge."
                  f"{bcolors.ENDC}\n")
            self.info_msg += 'Warning: phi_0 did not converge.\n'

        return phi_0

    def _fsolve_phi_0(self,
                      y_0: float,
                      a_kappa: float,
                      co_factor: float,
                      sigma: float,
                      y_initial_guess: float,
                      log: logger.logging.Logger
                      ) -> float:
        """
        Solve the nonlinear Grahame equation numerically.

        Parameters:
        y_0 (float): Parameter y_0.
        a_kappa (float): Parameter a_kappa.
        co_factor (float): Coefficient factor.
        sigma (float): Charge density.

        Returns:
        float: Solution for phi_0.
        """
        # pylint: disable=too-many-arguments
        solution = fsolve(self._nonlinear_grahame_equation,
                          y_initial_guess,
                          args=(y_0, a_kappa, co_factor, sigma),
                          full_output=True,
                          maxfev=10000)
        phi_0, _, ier, msg = solution
        if ier != 1 and not self.iter_flase_report_flag:
            log.warning(msg := "\tWarning: fsolve did not converge.\n")
            print(f"{bcolors.WARNING}{msg}{bcolors.ENDC}")
            self.info_msg += msg
            self.iter_flase_report_flag = True
        return phi_0[0]

    def _root_phi_0(self,
                    y_0: float,
                    a_kappa: float,
                    co_factor: float,
                    sigma: float,
                    y_initial_guess: float,
                    log: logger.logging.Logger
                    ) -> float:
        """
        Solve the nonlinear Grahame equation numerically using the root solver.

        Parameters:
        y_0 (float): Parameter y_0.
        a_kappa (float): Parameter a_kappa.
        co_factor (float): Coefficient factor.
        sigma (float): Charge density.

        Returns:
        float: Solution for phi_0.
        """
        # pylint: disable=too-many-arguments
        solution = root(self._nonlinear_grahame_equation,
                        y_initial_guess,
                        args=(y_0, a_kappa, co_factor, sigma),
                        method='hybr',
                        options={'xtol': 1e-10,
                                 'maxfev': 10000})
        if not solution.success and not self.iter_flase_report_flag:
            log.warning(
                msg :=
                f"\tWarning: root did not converge. `{solution.message}`\n")
            print(f"{bcolors.WARNING}{msg}{bcolors.ENDC}")
            self.info_msg += msg
            self.iter_flase_report_flag = True
        return solution.x[0]

    def _nonlinear_grahame_equation(self,
                                    phi_x: float,
                                    y_0: float,
                                    a_kappa: float,
                                    co_factor: float,
                                    sigma: float) -> float:
        """
        The nonlinear Grahame equation to solve.

        Parameters:
        phi_x (float): Potential.
        y_0 (float): Parameter y_0.
        a_kappa (float): Parameter a_kappa.
        co_factor (float): Coefficient factor.
        sigma (float): Charge density.

        Returns:
        float: The result of the Grahame equation.
        """
        # pylint: disable=too-many-arguments
        return (
            co_factor * (
                self.safe_sinh(y_0 * phi_x) -
                (2/a_kappa) * np.tanh(y_0 * phi_x / 2)
                ) - sigma
            )

    @staticmethod
    def safe_sinh(x_in: float
                  ) -> float:
        """
        Safe computation of sinh to avoid overflow.

        Parameters:
        x_in (float): Input value.

        Returns:
        float: Hyperbolic sine of x.
        """
        # Cap x to avoid overflow
        max_x = 700
        if x_in > max_x:
            return np.sinh(max_x)
        if x_in < -max_x:
            return np.sinh(-max_x)
        return np.sinh(x_in)

    @staticmethod
    def _nonlinear_grahame_ohshima_equation(phi_0: float,
                                            y_0: float,
                                            a_kappa: float,
                                            co_factor: float,
                                            sigma: float
                                            ) -> float:
        """equation 4.25 from pp. 101, Surface and Interfacial
        Forces, H-J Burr and M.Kappl
        as solved by Ohshima 1982: doi.org/10.1016/0021-9797(82)90393-9
        see M. Mass, 2022
        """
        arg: float = y_0 * phi_0
        return (
            co_factor * np.sinh(arg) *
            (
                1 +
                1/a_kappa * (2 / (np.cosh(arg)**2)) +
                1/a_kappa**2 * (8 * np.log(np.cosh(arg)) / np.sinh(arg)**2)
            ) ** 0.5 - sigma
            )[0]

    @staticmethod
    def safe_log1p(x_in: float
                   ) -> float:
        """safe computation of log1p to avoid overflow"""
        if x_in > -1.0:
            return np.log1p(x_in)
        return float('-inf')

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
