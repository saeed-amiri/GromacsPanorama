"""
Compute and compare the DLVO potential
The comparison are:
  1. The value of psi_0 for different computation radius, also compare
        the value of psi_0 from different PBE approximations
  2. The value of psi_r for different computation radius
  3. Comparison of psi_r for different cut radius in the APBS calculation

The experimental data also should be plotted!
"""

import numpy as np

import matplotlib.pyplot as plt

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


class CompareChrages:
    """Comapre charge density of the systms in different cut off radius"""

    info_msg: str = 'Message from CompareChrages:\n'
    configs: AllConfig
    charge_density_dict: dict[str, np.ndarray]
    charge_dict: dict[str, np.ndarray]
    r_cuts: list[float]
    densities_ave: list[np.float64]

    def __init__(self,
                 log: logger.logging.Logger,
                 configs: AllConfig = AllConfig()
                 ) -> None:
        self.configs = configs
        self.compare_systems(log)
        self._write_msg(log)

    def compare_systems(self,
                        log: logger.logging.Logger
                        ) -> None:
        """compare the charge density of the systems"""
        s_configs: "ComparisonConfigs" = self.configs.comparison_configs
        self.charge_density_dict, self.charge_dict = \
            self.get_density_charge(s_configs, log)
        self.r_cuts, self.densities_ave = self.get_average_charge_density()
        self.plot_charge_density()

    def get_density_charge(self,
                           s_configs: "ComparisonConfigs",
                           log: logger.logging.Logger
                           ) -> tuple[dict[str, np.ndarray],
                                      dict[str, np.ndarray]]:
        """get the charge density"""
        charge_density_dict: dict[str, np.ndarray] = {}
        charge_dict: dict[str, np.ndarray] = {}

        for key, charges_fname in s_configs.charge_files.items():
            _configs: AllConfig = self.configs
            _configs.charge_fname = charges_fname
            _configs.computation_radius = float(key)*10.0
            density_charge = ChargeDensity(
                log=log, configs=_configs)
            charge_density_dict[key] = density_charge.density
            charge_dict[key] = density_charge.charge

        return charge_density_dict, charge_dict

    def get_average_charge_density(self
                                   ) -> tuple[list[float], list[np.float64]]:
        """get the average charge density"""
        r_cuts: list[float] = \
            [float(i) for i in list(self.charge_density_dict.keys())]
        denisties: list[np.float64] = \
            [np.mean(i) for i in list(self.charge_density_dict.values())]
        return r_cuts, denisties

    def plot_charge_density(self,
                            ) -> None:
        """plot the charge density"""
        fig_i: plt.Figure
        ax_i: plt.Axes
        fig_i, ax_i = elsevier_plot_tools.mk_canvas(size_type='single_column')
        ax_i.plot(self.r_cuts,
                  self.densities_ave,
                  'o:',
                  label='Charge Density',
                  color='black',
                  lw=0.5,
                  markersize=3)
        ax_i.set_xlabel('Cut off radius [nm]')
        ax_i.set_ylabel(r'$\sigma$ [C/m$^2$]')
        elsevier_plot_tools.save_close_fig(
            fig_i, fname := 'charge_density_comparison.jpg')
        self.info_msg += (
            f'\tThe charge density is plotted and saved as `{fname}`\n')

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{CompareChrages.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


class ComparePhiZero:
    """Compare the phi_0 for different cut off radius"""

    info_msg: str = 'Message from ComparePhiZero:\n'
    configs: AllConfig
    phi_0_dict_loeb: dict[str, np.ndarray]
    phi_loeb_avg: list[np.float64]
    phi_0_dict_grahame: dict[str, np.ndarray]
    phi_grahame_avg: list[np.float64]
    debye_length: float

    def __init__(self,
                 r_cuts: list[float],
                 densities_ave: list[np.float64],
                 charges: dict[str, np.ndarray],
                 charge_density: dict[str, np.ndarray],
                 log: logger.logging.Logger,
                 configs: AllConfig = AllConfig()
                 ) -> None:
        # pylint: disable=too-many-arguments
        self.configs = configs
        self.compare_systems(
            r_cuts, densities_ave, charges, charge_density, log)
        self._write_msg(log)

    def compare_systems(self,
                        r_cuts: list[float],
                        densities_ave: list[np.float64],
                        charges: dict[str, np.ndarray],
                        charge_density: dict[str, np.ndarray],
                        log: logger.logging.Logger
                        ) -> None:
        """compare the phi_0 of the systems"""
        # pylint: disable=too-many-arguments
        self.phi_0_dict_loeb = \
            self.get_phi_0_loeb(charges, charge_density, log)
        self.phi_0_dict_grahame = \
            self._get_phi_0_grahame(charges, charge_density, log)
        self.phi_loeb_avg = self._get_avergae_phi_0(self.phi_0_dict_loeb)
        self.phi_grahame_avg = self._get_avergae_phi_0(self.phi_0_dict_grahame)
        phi_loeb_mv: list[np.float64] = [i * 1000 for i in self.phi_loeb_avg]
        phi_grahame_mv: list[np.float64] = \
            [i * 1000 for i in self.phi_grahame_avg]
        self.plot_phi_0_r_cut(r_cuts, phi_loeb_mv, phi_grahame_mv)
        self.plot_phi_0_denisty(densities_ave, phi_loeb_mv, phi_grahame_mv)
        if self.configs.comparison_configs.plot_brocken_graph:
            self.plot_phi_brocken_axis(
                r_cuts, phi_loeb_mv, phi_grahame_mv, 'r_cut')
            self.plot_phi_brocken_axis(
                densities_ave, phi_loeb_mv, phi_grahame_mv, 'density')

    def get_phi_0_loeb(self,
                       charges: dict[str, np.ndarray],
                       charge_density: dict[str, np.ndarray],
                       log: logger.logging.Logger
                       ) -> dict[str, np.ndarray]:
        """get the phi_0 for the systems"""
        phi_0_dict_loeb: dict[str, np.ndarray] = {}
        _configs: AllConfig = self.configs
        _configs.solving_config.phi_0_type = 'grahame'
        ion_strength = _configs.phi_parameters['c_salt']
        debye_length = self._get_debye(ion_strength)
        for key, charge in charges.items():
            _configs.computation_radius = float(key)*10.0
            density = charge_density[key]
            phi_0: np.ndarray = DLVOPotentialPhiZero(
                debye_length=debye_length,
                charge=charge,
                charge_density=density,
                configs=_configs,
                ion_strength=ion_strength,
                log=log).phi_0
            phi_0_dict_loeb[key] = phi_0
        return phi_0_dict_loeb

    def _get_phi_0_grahame(self,
                           charges: dict[str, np.ndarray],
                           charge_density: dict[str, np.ndarray],
                           log: logger.logging.Logger
                           ) -> dict[str, np.ndarray]:
        """get the phi_0 for the systems"""
        phi_0_dict_grahame: dict[str, np.ndarray] = {}
        _configs: AllConfig = self.configs
        _configs.solving_config.phi_0_type = 'grahame_simple'
        ion_strength = _configs.phi_parameters['c_salt']
        debye_length = self._get_debye(ion_strength)
        for key, charge in charges.items():
            density = charge_density[key]
            phi_0: np.ndarray = DLVOPotentialPhiZero(
                debye_length=debye_length,
                charge=charge,
                charge_density=density,
                configs=_configs,
                ion_strength=ion_strength,
                log=log).phi_0
            phi_0_dict_grahame[key] = phi_0
        return phi_0_dict_grahame

    def plot_phi_0_r_cut(self,
                         r_cuts: list[float],
                         phi_loeb_mv: list[np.float64],
                         phi_grahame_mv: list[np.float64]
                         ) -> None:
        """plot the phi_0"""
        fig_i: plt.Figure
        ax_i: plt.Axes
        fig_i, ax_i = elsevier_plot_tools.mk_canvas(size_type='single_column')
        ax_i.plot(r_cuts,
                  phi_loeb_mv,
                  'o:',
                  label='Loeb approx.',
                  color='black',
                  lw=0.5,
                  markersize=3)
        ax_i.plot(r_cuts,
                  phi_grahame_mv,
                  '^:',
                  label='Grahame approx.',
                  color='grey',
                  lw=0.5,
                  markersize=3)
        ax_i.set_xlabel('Cut off radius [nm]')
        ax_i.set_ylabel(r'$\psi_0$ [mV]')
        elsevier_plot_tools.save_close_fig(
            fig_i, fname := 'phi_0_comparison.jpg')
        self.info_msg += (
            f'\tThe phi_0 is plotted and saved as `{fname}`\n')

    def plot_phi_brocken_axis(self,
                              x_data: list[float],
                              phi_loeb_mv: list[np.float64],
                              phi_grahame_mv: list[np.float64],
                              plot_type: str
                              ) -> None:
        """plot the phi_0 with a broken y-axis using matplotlib"""
        _, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 4))

        # Set the y-axis limits for each subplot to create the 'break'
        # effect
        ax1.set_ylim(228, 232)  # Upper plot
        ax2.set_ylim(153.5, 158)  # Lower plot

        # Plot the data on both subplots
        ax1.plot(x_data,
                 phi_loeb_mv,
                 'o:',
                 label='Loeb approx.',
                 color='black',
                 lw=0.5,
                 markersize=3)
        ax1.plot(x_data,
                 phi_grahame_mv,
                 '^:',
                 label='Grahame approx.',
                 color='grey',
                 lw=0.5,
                 markersize=3)
        ax2.plot(x_data,
                 phi_grahame_mv,
                 '^:',
                 color='grey',
                 lw=0.5,
                 markersize=3)

        # Hide the spines between ax and ax2
        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax1.xaxis.tick_top()
        ax1.tick_params(labeltop=False)
        ax2.xaxis.tick_bottom()

        # Add diagonal lines to indicate the 'break' in the axis
        d_size = .015  # how big to make the diagonal lines in axes coordinates
        kwargs = {"transform": ax1.transAxes, "color": 'k', "clip_on": False}
        ax1.plot((-d_size, +d_size), (-d_size, +d_size), **kwargs)
        ax1.plot((1 - d_size, 1 + d_size), (-d_size, +d_size), **kwargs)

        # switch to the bottom axes
        kwargs.update(transform=ax2.transAxes)
        ax2.plot((-d_size, +d_size), (1 - d_size, 1 + d_size), **kwargs)
        ax2.plot((1 - d_size, 1 + d_size), (1 - d_size, 1 + d_size), **kwargs)

        # Labels, titles, and legends
        ax2.set_xlabel('Cut off radius [nm]')
        if plot_type == 'density':
            ax2.set_xlabel(r'$\sigma$ [C/m$^2$]')
        ax1.set_ylabel(r'$\psi_0$ [mV]')
        ax2.set_ylabel(r'$\psi_0$ [mV]')
        ax1.legend(loc='upper right')

        # Adjust layout
        plt.subplots_adjust(hspace=0.1)

        # Save the figure
        plt.savefig(f'phi_0_{plot_type}_comparison_broken_axis.jpg')
        self.info_msg += (
            f'\tThe phi_0 vs {plot_type} is plotted with a broken y-axis '
            'and saved as `phi_0_comparison_broken_axis.jpg`\n')

    def plot_phi_0_denisty(self,
                           densities_ave: list[np.float64],
                           phi_loeb_mv: list[np.float64],
                           phi_grahame_mv: list[np.float64]
                           ) -> None:
        """plot the phi_0"""
        fig_i: plt.Figure
        ax_i: plt.Axes
        fig_i, ax_i = elsevier_plot_tools.mk_canvas(size_type='single_column')
        ax_i.plot(densities_ave,
                  phi_loeb_mv,
                  'o:',
                  label='Loeb approx.',
                  color='black',
                  lw=0.5,
                  markersize=3)
        ax_i.plot(densities_ave,
                  phi_grahame_mv,
                  '^:',
                  label='Grahame approx.',
                  color='grey',
                  lw=0.5,
                  markersize=3)
        ax_i.set_xlabel(r'$\sigma$ [C/m$^2$]')
        ax_i.set_ylabel(r'$\psi_0$ [mV]')
        elsevier_plot_tools.save_close_fig(
            fig_i, fname := 'phi_0_vs_density_comparison.jpg',
            loc='lower right')
        self.info_msg += (
            f'\tThe phi_0 vs density is plotted and saved as `{fname}`\n')

    @staticmethod
    def _get_avergae_phi_0(phi_dict) -> list[np.float64]:
        """get the average phi_0"""
        phi_0_ave: list[np.float64] = \
            [np.mean(i) for i in list(phi_dict.values())]
        return phi_0_ave

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
            f'\t`{debye_l_nm = :.4f}` [nm]\n'
            '\t`computation_radius = '
            f'{self.configs.computation_radius/10.0:.4f}` [nm]\n\t'
            f'kappa * r = {self.configs.computation_radius/10/debye_l_nm:.3f}'
            '\n')
        return float(debye_l_nm)

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{ComparePhiZero.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    LOG: logger.logging.Logger = logger.setup_logger('compare_charges.log')

    CHARGE_INFO = CompareChrages(log=LOG)
    R_CUTS: list[float] = CHARGE_INFO.r_cuts
    DENISTY_AVG: list[np.float64] = CHARGE_INFO.densities_ave
    CHARGE: dict[str, np.ndarray] = CHARGE_INFO.charge_dict
    CHARGE_DENSITY: dict[str, np.ndarray] = CHARGE_INFO.charge_density_dict

    PHI_ZERO = ComparePhiZero(R_CUTS,
                              DENISTY_AVG,
                              CHARGE,
                              CHARGE_DENSITY,
                              log=LOG)

