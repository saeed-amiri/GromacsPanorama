"""
Rad the data for zeta potential and compute the density and also plot
the equation we have for them.
"""

import typing

import numpy as np

from common import logger, elsevier_plot_tools
from common.colors_text import TextColor as bcolors

from module9_electrostatic_analysis.dlvo_potential_charge_density import \
    ChargeDensity
from module9_electrostatic_analysis.dlvo_potential_non_linear_aprox import \
    NonLinearPotential
from module9_electrostatic_analysis.dlvo_potential_phi_zero import \
    DLVOPotentialPhiZero
from module9_electrostatic_analysis.dlvo_potential_phi_0_sigma import \
    PhiZeroSigma
from module9_electrostatic_analysis.dlvo_potential_configs import AllConfig, \
    ComparisonConfigs


class ExperimentComputation:
    """
    Compute the density and the potential for the experimental data
    """
    info_msg: str = 'Message from ExperimentComputation:\n'
    configs: AllConfig
    charge_density: typing.Union[float, np.ndarray]

    def __init__(self,
                 log: logger.logging.Logger,
                 configs: AllConfig = AllConfig()
                 ) -> None:
        self.configs = configs
        self.charge_density = self.compute_density()
        self.info_msg += \
            f'\tThe charge density is computed: {self.charge_density}\n'
        self._write_msg(log)

    def compute_density(self) -> typing.Union[float, np.ndarray]:
        """compute the charge density"""
        style: str = self.configs.experiment_configs.density_computatin_style
        style = style.lower()
        parameters: dict[str, float] = self.configs.phi_parameters
        r_np: float = 200
        zeta_potential: np.ndarray = \
            self.configs.experiment_configs.zeta_average[str(r_np)]
        ion_strength: float = \
            self.configs.experiment_configs.salt_concentration[0]
        debye_lengt: float = self._get_debye(ionic_strength=ion_strength,
                                             param=parameters,
                                             np_radius=r_np)
        if style == 'grahame':
            return (self.density_grahame(zeta_potential,
                                         ion_strength,
                                         parameters))
        if style == 'loeb':
            return (self.density_loeb(zeta_potential,
                                      parameters=parameters,
                                      kappa=1.0/debye_lengt,
                                      np_radius=r_np/10.0))
        if style == 'ohshima':
            return zeta_potential
        return self.no_density(zeta_potential, style)

    def no_density(self,
                   zeta_potential: typing.Union[float, np.ndarray],
                   style: str
                   ) -> typing.Union[float, np.ndarray]:
        """compute the density based on the Grahame equation"""
        self.info_msg += (
            '\tNo density is computed\n'
            f'\n\tAn unknown relation is asked for `{style}`\n\n')
        return np.zeros_like(zeta_potential)

    def density_grahame(self,
                        zeta_potential: typing.Union[float, np.ndarray],
                        ionic_strength: float,
                        parameters: dict[str, float]
                        ) -> typing.Union[float, np.ndarray]:
        """compute the density based on the Grahame equation"""
        self.info_msg += \
            '\tComputing the density based on the Grahame equation\n'
        return (
            np.sqrt(8 * parameters['epsilon'] * parameters['epsilon_0'] *
                    parameters['T'] * parameters['k_boltzman_JK'] *
                    ionic_strength * 1e3 * parameters['n_avogadro']
                    ) *
            np.sinh(parameters['e_charge'] *
                    zeta_potential * parameters['mv_to_v'] /
                    (2.0 * parameters['k_boltzman_JK'] * parameters['T'])
                    )
            )

    def density_loeb(self,
                     zeta_potential: typing.Union[float, np.ndarray],
                     parameters: dict[str, float],
                     kappa: float,  # in nm^-1
                     np_radius: float  # in nm
                     ) -> typing.Union[float, np.ndarray]:
        """compute the density based on the Loeb equation"""
        self.info_msg += \
            '\tComputing the density based on the Loeb equation\n'

        co_factor: float = \
            2.0 * \
            parameters['k_boltzman_JK'] * parameters['T'] * \
            parameters['epsilon_0'] * parameters['epsilon'] / \
            parameters['e_charge']

        # argument of the hyperbolic sine and tangent
        arg: typing.Union[float, np.ndarray] = \
            parameters['e_charge'] * zeta_potential * parameters['mv_to_v'] / \
            (2 * parameters['k_boltzman_JK'] * parameters['T'])

        # general term
        general_term: typing.Union[float, np.ndarray] = \
            co_factor * np.sinh(arg) * kappa

        # effect of the np size
        np_size_term: typing.Union[float, np.ndarray] = \
            2.0 * co_factor * np.tanh(arg/2.0) / np_radius

        return general_term + np_size_term

    def _get_debye(self,
                   ionic_strength: float,
                   param: dict[str, float],
                   np_radius: float
                   ) -> float:
        """computing the debye length based on Poisson-Boltzmann apprx.
        See:
        pp. 96, Surface and Interfacial Forces, H-J Burr and M.Kappl
        """

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
            f'{np_radius/10.0:.4f}` [nm]\n\t'
            f'kappa * r = {np_radius/10/debye_l_nm:.3f}'
            '\n')
        return float(debye_l_nm)

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{ExperimentComputation.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    ExperimentComputation(logger.setup_logger('experiment_computation.log'),
                          AllConfig())
