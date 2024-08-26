"""
Computing phi_0 for the DLVO potential using different approximations
of Grahame equation, which computes the potential at the surface of the
from Possion-Boltzmann equation.
"""

from datetime import datetime

import numpy as np
from enum import Enum
from scipy.optimize import fsolve, root


from common import logger
from common.colors_text import TextColor as bcolors
from module9_electrostatic_analysis.dlvo_potential_configs import (
    AllConfig, SolvingConfig)


class SolverStrategy:
    def solve(self,
              equation: str,
              y_initial_guess: float,
              args: dict[str, float]
              ) -> np.ndarray:
        raise NotImplementedError


class LoebSolver(SolverStrategy):
    def solve(self, equation, y_initial_guess, args):
        return root(equation,
                    y_initial_guess,
                    args=args,
                    method='hybr',
                    options={'xtol': 1e-10, 'maxfev': 10000})


class OhshimaSolver(SolverStrategy):
    def solve(self, equation, y_initial_guess, args):
        return root(equation,
                    y_initial_guess,
                    args=args,
                    method='hybr',
                    options={'xtol': 1e-10, 'maxfev': 10000})


class PhiZeroType(Enum):
    """
    Different methods to compute the phi_0 for the DLVO potential.
    !Must be lower case!
    """
    CONSTANT = 'constant'
    GRAHAME_LOW = 'grahame_low'
    GRAHAME = 'grahame'
    LOEB = 'loeb'


class PhiZeroCalculator:
    """
    Mathematical methods to compute the phi_0 for the DLVO potential.
    """
    config: "AllConfig"

    def __init__(self,
                 configs: AllConfig
                 ) -> None:
        self.configs = configs

    def get_phi_0_method(self):
        """
        Get the method to compute the phi_0 based on the configuration
        """
        _configs: SolvingConfig = self.configs.solving_config
        phi_zero_method: str = _configs.phi_0_type.lower()
        if phi_zero_method == PhiZeroType.CONSTANT:
            return self._constant_phi_zero
        elif phi_zero_method == PhiZeroType.GRAHAME_LOW:
            return self._compute_phi_0_grahame_low_potential
        elif phi_zero_method == PhiZeroType.GRAHAME:
            return self._compute_phi_0_grahame_simple
        elif phi_zero_method == PhiZeroType.LOEB:
            return self._compute_phi_0_loeb_nonlinear

    def _constant_phi_zero(self,
                           charge_density: np.ndarray
                           ) -> np.ndarray:
        """The constant value of phi_0 from configuration"""
        return np.ones(charge_density.shape) * \
            self.config.phi_parameters['phi_0']

    def _compute_phi_0_grahame_low_potential(self,
                                             debye_l: float,  # in nm
                                             charge_density: np.ndarray
                                             ) -> np.ndarray:
        """compute the phi_0 based on the linearized Possion-Boltzmann
        relation:
        phi_0 = debye_l * q_density / (epsilon * epsilon_0)
        pp. 103, Surface and Interfacial Forces, H-J Burr and M.Kappl
        """
        phi_0: np.ndarray = debye_l * 1e-9 * charge_density / \
            (self.configs.phi_parameters['epsilon'] *
             self.configs.phi_parameters['epsilon_0'])
        return phi_0
    
    def _compute_phi_0_grahame_simple(self,
                                      ionic_strength: float,
                                      density: np.ndarray
                                      ) -> np.ndarray:
        """Compute the phi_0 based on the linearized Poisson-Boltzmann
        relation:
        density = sqrt(8 * I * epsilon * epsilon_0 * k_boltzman_JK * T) *
        sinh(e * phi_0 / 2k_BT)
        pp. 102, Surface and Interfacial Forces, H-J Burr and M. Kappl

        Parameters:
        ionic_strength (float): The ionic strength of the solution in
        mol/L:
        I = 1/2 * sum_i(z_i^2 * c_i)

        If c is the concentration in mol/L, then the number of ions
        per cubic meter is:
        Number of ions per m3

        density (np.ndarray): The surface charge density in C/m^2.

        Returns:
        np.ndarray: The surface potential phi_0 in Volts.
        """
        info_msg = '\tPhi_0 computed from Grahame sinh (1) relation\n'

        param: dict[str, float] = self.configs.phi_parameters

        kbt: float = param['T'] * param['k_boltzman_JK']

        epsilon: float = param['epsilon'] * param['epsilon_0']

        y_0: float = 2 * kbt / param['e_charge']

        co_factor: float = np.sqrt(
            8 * ionic_strength * 1e3 * param['n_avogadro'] * epsilon * kbt)

        phi_0: np.ndarray = y_0 * np.arcsinh(density / co_factor)

        return phi_0
    
    def _compute_phi_0_loeb_nonlinear(self,
                                      debye_l: float,  # in nm
                                      log: logger.logging.Logger,
                                      charge_density: np.ndarray,
                                      ) -> np.ndarray:
        """
        Compute the phi_0 based on the nonlinearized Poisson-Boltzmann
        relation for a sphere using the Loeb equation.
        phi_0 = 2 * epsilon * epsilon_0 * k_B *  T / (e * debye_l) * [
            sinh(e * phi_0 / 2k_BT) - 2 * debye_l /a * tanh(e * phi_0 / 2k_BT)
            ]

        Parameters:
        debye_l (float): Debye length.

        Returns:
        np.ndarray: Computed phi_0 values.
        """
        info_msg = '\tPhi_0 computed from Grahame sinh (2) relation\n'

        param: dict[str, float] = self.configs.phi_parameters

        r_np: float = self.configs.computation_radius / 10.0  # [A] -> [nm]
        kbt: float = param['T'] * param['k_boltzman_JK']
        epsilon: float = param['epsilon'] * param['epsilon_0']
        a_kappa: float = r_np / debye_l
        y_0: float = param['e_charge'] / (2.0 * kbt)
        co_factor: float = epsilon / (y_0 * debye_l * 1e-9)
        sigma = charge_density.copy()

        phi_0: np.ndarray = np.zeros(charge_density.shape, dtype=np.float64)


        y_initial_guess: float = 0.1

        self.iter_flase_report_flag = False

        for i, _ in enumerate(phi_0):
            phi_0[i] = self._root_phi_0(y_0=y_0,
                                        a_kappa=a_kappa,
                                        co_factor=co_factor,
                                        sigma=sigma[i],
                                        y_initial_guess=y_initial_guess,
                                        log=log)
        self.info_msg += ('\tPhi_0 computed from numerical solution of '
                          'nonlinear equation from Grahame relation\n')

        if all(phi_0) == y_initial_guess:
            print(f"{bcolors.CAUTION}\tWarning: phi_0 did not converge."
                  f"{bcolors.ENDC}\n")
            info_msg = 'Warning: phi_0 did not converge.\n'

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
        solution = fsolve(self._nonlinear_loeb_equation,
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

    def _nonlinear_loeb_equation(self,
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
                self.safe_sinh(y_0 * phi_x) +
                (2/a_kappa) * np.tanh(y_0 * phi_x / 2)
                ) - sigma
            )

    def _nonlinear_ohshima_equation(self,
                                    phi_0: float,
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
        # pylint: disable=too-many-arguments
        arg: float = y_0 * phi_0
        return (
            co_factor * np.sinh(arg) *
            (
                1 +
                1/a_kappa * (2 / (self.safe_cosh(arg/2.0)**2)) +
                1/a_kappa**2 * (8 * self.safe_log(self.safe_cosh(arg/2.0)) /
                                self.safe_sinh(arg)**2)
            ) ** 0.5 - sigma
            )[0]

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
    def safe_cosh(x_in: float
                  ) -> float:
        """
        Safe computation of sinh to avoid overflow.

        Parameters:
        x_in (float): Input value.

        Returns:
        float: Hyperbolic cosine of x.
        """
        # Cap x to avoid overflow
        max_x = 700
        if x_in > max_x:
            return np.cosh(max_x)
        if x_in < -max_x:
            return np.cosh(-max_x)
        return np.cosh(x_in)

    @staticmethod
    def safe_log1p(x_in: float
                   ) -> float:
        """safe computation of log1p to avoid overflow"""
        if x_in > -1.0:
            return np.log1p(x_in)
        return float('-inf')

    @staticmethod
    def safe_log(x_in: float
                 ) -> float:
        """safe computation of log1p to avoid overflow"""
        if x_in > -1.0:
            return np.log(x_in)
        return float('-inf')


