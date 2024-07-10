"""
Plot phi_0 with respect to sigma
=================================
This script is used to plot the DLVO potential for different values of
phi_0 and sigma.
The aim is the plot values of phi_0 with respect to sigma, for different
vlaues of lambda_D.
Lambda_D is for two different values: from simulation and from the
experimental data.
To determine the rationality of the DLVO modleing
"""

import typing
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from common import logger, elsevier_plot_tools
from common.colors_text import TextColor as bcolors
from module9_electrostatic_analysis.dlvo_potential_phi_zero import \
    DLVOPotentialPhiZero
from module9_electrostatic_analysis.dlvo_potential_configs import \
    AllConfig, PhiZeroSigmaConfig


# Helper functions
# ----------------
def check_no_kwargs(kwargs: typing.Any,
                    log: logger.logging.Logger
                    ) -> None:
    """raise an error if the kwargs are not given"""
    if not kwargs:
        msg: str = '\nNo kwargs are given!\n'
        log.error(msg)
        raise ValueError(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')


def check_all_kwargs_exist(kwargs: typing.Any,
                           expected_kwargs: list[str],
                           log: logger.logging.Logger
                           ) -> None:
    """check if all the expected kwargs are given"""
    for key in expected_kwargs:
        if key not in kwargs:
            msg: str = f'\n{key} is not given!\n'
            log.error(msg)
            raise ValueError(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')


def check_unkown_kwargs(kwargs: typing.Any,
                        expected_kwargs: list[str],
                        log: logger.logging.Logger
                        ) -> None:
    """check if there are any unknown kwargs"""
    for key in kwargs:
        if key not in expected_kwargs:
            msg: str = f'\n{key} is unknown!\n'
            log.error(msg)
            raise ValueError(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')
# ----------------


class PhiZeroSigma:
    """compute the phi_0 with respect to sigma"""
    # pylint: disable=too-many-instance-attributes
    info_msg: str = 'Message from PhiZeroSigma:\n'

    configs: AllConfig
    debye_md: float
    debye_exp: list[float]
    charge_density_range: tuple[float, float]

    def __init__(self,
                 log: logger.logging.Logger,
                 configs: AllConfig,
                 **kwargs: typing.Dict[str, typing.Any],
                 ) -> None:
        self.validate_kwargs(kwargs, log)
        self.assign_kwargs(kwargs)
        self.configs = configs
        self.compute_phi_0(log)
        self.plot_phi_0_sigma_relation(log)
        self.phi_vs_np_radius(log)
        self.write_msg(log)

    def validate_kwargs(self,
                        kwargs: typing.Dict[str, typing.Any],
                        log: logger.logging.Logger
                        ) -> None:
        """Check if all the inputs are valid"""
        expected_kwargs = ['debye_md', 'charge_density_range']

        check_no_kwargs(kwargs, log)
        check_all_kwargs_exist(kwargs, expected_kwargs, log)
        check_unkown_kwargs(kwargs, expected_kwargs, log)

        self.info_msg += f'\tThe following {kwargs.keys() = } are given\n'

    def assign_kwargs(self,
                      kwargs: typing.Dict[str, typing.Any]
                      ) -> None:
        """assign the kwargs to the class"""
        self.debye_md = kwargs['debye_md']
        self.charge_density_range = kwargs['charge_density_range']

    def compute_phi_0(self,
                      log: logger.logging.Logger
                      ) -> None:
        """compute the phi_0 with respect to sigma"""
        phi_list: dict[str, np.ndarray] = {}
        configs = self.configs.phi_zero_sigma_config
        self.debye_exp = self._compute_debye_exp(configs)
        denisty_range: np.ndarray = \
            np.linspace(*self.charge_density_range, configs.nr_density_points)
        for debye, ion_strength in zip(self.debye_exp,
                                       configs.exp_salt_concentration):
            phi_list[f'{ion_strength}'] = self._compute_phi_0_sigma(
                debye, denisty_range, ion_strength, log)

        self._plot_phi_0_sigma(denisty_range, phi_list, configs)

    def _compute_phi_0_sigma(self,
                             debye: float,
                             denisty_range: np.ndarray,
                             ion_strength: float,
                             log: logger.logging.Logger
                             ) -> np.ndarray:
        """compute the phi_0 with respect to sigma"""
        phi_sigma: np.ndarray = DLVOPotentialPhiZero(
            debye_length=debye,
            charge=np.zeros(denisty_range.shape),
            charge_density=denisty_range,
            configs=self.configs,
            ion_strength=ion_strength,
            log=log
        ).phi_0
        return phi_sigma

    def phi_vs_np_radius(self,
                         log: logger.logging.Logger
                         ) -> None:
        """compute the phi_0 with respect to changing the np_radius
        in the simulation salt concentration
        """
        phi_list: dict[str, np.ndarray] = {}
        configs = self.configs.phi_zero_sigma_config
        denisty_range: np.ndarray = \
            np.linspace(*self.charge_density_range, configs.nr_density_points)
        for np_radius in configs.exp_np_radii:
            phi_list[f'{np_radius}'] = self._compute_phi_0_np_radius(
                self.debye_md, denisty_range, np_radius, log)
        self.plot_0_np_radius(denisty_range, phi_list, configs)

    def _compute_phi_0_np_radius(self,
                                 debye: float,
                                 denisty_range: np.ndarray,
                                 np_radius: float,
                                 log: logger.logging.Logger
                                 ) -> np.ndarray:
        """compute the phi_0 with respect to sigma"""
        _configs = self.configs
        _configs.computation_radius = np_radius
        phi_np: np.ndarray = DLVOPotentialPhiZero(
            debye_length=debye,
            charge=np.zeros(denisty_range.shape),
            charge_density=denisty_range,
            configs=_configs,
            ion_strength=self.configs.phi_parameters['c_salt'],
            log=log
        ).phi_0
        return phi_np

    def plot_0_np_radius(self,
                         denisty_range: np.ndarray,
                         phi_list: dict[str, np.ndarray],
                         configs: "PhiZeroSigmaConfig"
                         ) -> None:
        """plot the phi_0 with respect to np radius"""
        fig, ax_i = elsevier_plot_tools.mk_canvas('single_column')
        for i, (key, phi) in enumerate(phi_list.items()):
            ax_i.plot(denisty_range,
                      phi*1000,
                      c=elsevier_plot_tools.BLACK_SHADES[i*2],
                      label=f'{float(key)/10:.1f} [nm]',)
        ax_i.set_xlabel(r'$\sigma$ [C/m$^2$]')
        ax_i.set_ylabel(r'$\psi_0$ [mV]')
        ax_i.set_ylim(configs.y_lims[0], configs.y_lims[1])
        c_salt: str = f'{self.configs.phi_parameters["c_salt"]*1000:.2f}'
        if (
           c_type := self.configs.solving_config.phi_0_type
           ) == 'grahame_simple':
            ax_i.text(0.0,
                      130,
                      s=(r'$\psi_0 = \frac{2k_BT}{e}\sinh^{-1}('
                         r'\frac{\sigma}{8c_0\epsilon\epsilon_0k_BT})$'),
                      fontsize=6)
            ax_i.text(
                -0.0,
                110,
                s=fr'$c_0$ = {c_salt} [mM]',
                fontsize=6
                )
            out_name: str = 'phi_0_np_radius_grahame_simple.jpg'
        elif c_type == 'grahame':
            ax_i.text(
                -0.0,
                120,
                s=(r'$\sigma = \frac{\epsilon\epsilon_0k_BT}{e\lambda}$'
                   r'$(2\sinh(\frac{e\psi_0}{2k_BT})\,+\,$'
                   r'$\frac{4\lambda}{a}\tanh(\frac{e\psi_0}{4k_BT}))$'),
                fontsize=6)
            ax_i.text(
                -0.0,
                100,
                s=fr'$c_0$ = {c_salt} [mM]',
                fontsize=6
                )
            out_name = 'phi_0_np_radius_grahame.jpg'

        elsevier_plot_tools.save_close_fig(fig,
                                           fname=out_name,
                                           loc='lower right')

    def _plot_phi_0_sigma(self,
                          denisty_range: np.ndarray,
                          phi_list: dict[str, np.ndarray],
                          configs: "PhiZeroSigmaConfig"
                          ) -> None:
        """plot the phi_0 with respect to sigma"""
        fig, ax_i = elsevier_plot_tools.mk_canvas('single_column')
        for i, (key, phi) in enumerate(phi_list.items()):
            ax_i.plot(denisty_range,
                      phi*1000,
                      c=elsevier_plot_tools.BLACK_SHADES[i*2],
                      label=f'{key} [M]')
        ax_i.set_xlabel(r'$\sigma$ [C/m$^2$]')
        ax_i.set_ylabel(r'$\psi_0$ [mV]')
        ax_i.set_ylim(configs.y_lims[0], configs.y_lims[1])
        if (
             c_type := self.configs.solving_config.phi_0_type
             ) == 'grahame_simple':
            # ax_i.set_ylim(configs.y_lims[0]*100, configs.y_lims[1]*100)
            ax_i.text(0.0,
                      130,
                      s=(r'$\psi_0 = \frac{2k_BT}{e}\sinh^{-1}('
                         r'\frac{\sigma}{8c_0\epsilon\epsilon_0k_BT})$'),
                      fontsize=6)
            out_name: str = 'phi_0_sigma_grahame_simple.jpg'
        elif c_type == 'grahame':
            ax_i.text(-0.0,
                      120,
                      s=(r'$\sigma = \frac{\epsilon\epsilon_0k_BT}{e\lambda}$'
                         r'$(2\sinh(\frac{e\psi_0}{2k_BT})\,+\,$'
                         r'$\frac{4\lambda}{a}\tanh(\frac{e\psi_0}{4k_BT}))$'),
                      fontsize=6)
            ax_i.text(-0.0,
                      100,
                      s=f'a = {self.configs.computation_radius/10} [nm]',
                      fontsize=6
                      )
            out_name = 'phi_0_sigma_grahame.jpg'

        elsevier_plot_tools.save_close_fig(fig,
                                           fname=out_name,
                                           loc='lower right')

    def _compute_debye_exp(self,
                           configs: "PhiZeroSigmaConfig"
                           ) -> list[float]:
        """compute the Debye length from the experimental data
        from concentrations of salts
        """
        debye_exp: list[float] = []
        for ionic_strength in configs.exp_salt_concentration:
            debye_exp.append(float(self._get_debye(ionic_strength)))
        return debye_exp

    def _get_debye(self,
                   ionic_strength: typing.Union[float, np.ndarray]
                   ) -> typing.Union[np.float64, np.ndarray]:
        """computing the debye length based on Poisson-Boltzmann apprx.
        See:
        pp. 96, Surface and Interfacial Forces, H-J Burr and M.Kappl
        """
        param: dict[str, float] = self.configs.phi_parameters

        # ionnic strength in mol/m^3
        ionic_str_mol_m3: typing.Union[float, np.ndarray] = \
            ionic_strength * 1e3

        # Getting debye length
        debye_l: typing.Union[np.float64, np.ndarray] = np.sqrt(
            param['T'] * param['k_boltzman_JK'] *
            param['epsilon'] * param['epsilon_0'] /
            (2 * ionic_str_mol_m3 * param['n_avogadro'] * param['e_charge']**2
             ))

        # convert to nm
        debye_l_nm: typing.Union[np.float64, np.ndarray] = debye_l * 1e9
        if isinstance(ionic_strength, np.float64):
            self.info_msg += (
                f'\tSalt C is: `{ionic_strength}` -> '
                f'`{debye_l_nm = :.4f}` [nm]\n')
        return debye_l_nm

    def plot_phi_0_sigma_relation(self,
                                  log: logger.logging.Logger
                                  ) -> None:
        """plot the phi_0 with respect to sigma"""
        if self.configs.phi_zero_sigma_config.plot_bare_equation:
            ion_strength: np.ndarray = \
                np.linspace(
                    0.01,
                    100,
                    self.configs.phi_zero_sigma_config.nr_density_points)
            debye: typing.Union[np.float64, np.ndarray] = \
                self._get_debye(ion_strength)
            self._plot_bare_equation()
            self._plot_debye_ion_strength(ion_strength=ion_strength,
                                          debye=debye)
            self._plot_kappa_a_ion_strength(ion_strength=ion_strength,
                                            debye=debye)
            self._plot_kappa_a_c_colormap()

    def _plot_bare_equation(self) -> None:
        """plot the bare equation"""
        fig, ax_i = elsevier_plot_tools.mk_canvas('single_column')
        reversed_black_shades: list[str] = \
            list(reversed(elsevier_plot_tools.BLACK_SHADES))
        x_data: np.ndarray = \
            np.linspace(-10, 10,
                        self.configs.phi_zero_sigma_config.nr_density_points)
        for i, c_0 in enumerate([0.05, 0.1, 1, 10, 100]):
            y_data: np.ndarray = np.arcsinh(x_data*i)
            ax_i.plot(x_data,
                      y_data,
                      c=reversed_black_shades[i+2],
                      label=rf'y = {c_0} $\cdot$ x')
        ax_i.set_xlabel('x')
        ax_i.set_ylabel(r'sinh$^{-1}$(y)')
        elsevier_plot_tools.save_close_fig(fig,
                                           'bare_equation_arcsinh_overx.jpg',
                                           loc='lower right')

    def _plot_debye_ion_strength(self,
                                 ion_strength: np.ndarray,
                                 debye: typing.Union[float, np.ndarray]
                                 ) -> None:
        """plot the debye length with respect to sigma"""
        fig, ax_i = elsevier_plot_tools.mk_canvas('single_column')
        ax_i.plot(ion_strength,
                  debye,
                  c=elsevier_plot_tools.BLACK_SHADES[0],
                  label='Debye length')
        ax_i.set_xlabel(r'$c_0$ [mM]')
        ax_i.set_ylabel(r'$\lambda$ [nm]')
        elsevier_plot_tools.save_close_fig(fig,
                                           'debye_length_sigma.jpg',
                                           loc='upper right')

    def _plot_kappa_a_ion_strength(self,
                                   ion_strength: np.ndarray,
                                   debye: typing.Union[float, np.ndarray]
                                   ) -> None:
        """plot the kappa * a with respect to sigma"""
        fig, ax_i = elsevier_plot_tools.mk_canvas('single_column')
        radii: list[float] = [self.configs.computation_radius, 100.0]

        for i, radius in enumerate(radii):
            kappa_a: np.ndarray = radius / 10.0 / debye
            ax_i.plot(ion_strength,
                      kappa_a,
                      c=elsevier_plot_tools.BLACK_SHADES[i*4],
                      label=rf'$a\,=\,${radius/10}')
        ax_i.set_xlabel(r'$c_0$ [mM]')
        ax_i.set_ylabel(r'$\kappa \cdot a$')
        x_lim: tuple[float, float] = ax_i.get_xlim()
        ax_i.hlines(0.1, x_lim[0], x_lim[1], ls='dashed', lw=1.0, color='grey')
        ax_i.hlines(100, x_lim[0], x_lim[1], ls='dashed', lw=1.0, color='grey')
        ax_i.set_xlim(x_lim)
        elsevier_plot_tools.save_close_fig(fig,
                                           'kappa_a_sigma.jpg',
                                           loc='upper left')
        
    def _plot_kappa_a_c_colormap(self) -> None:
        """plot the kappa * a with respect to concentration"""
       # Range of salt concentrations (mol/L) and colloidal radii (m)
        c_0_values = np.logspace(-5, 0, 200)  # Concentration from 10^-5 to 1 M
        a_values = np.logspace(-9, -6, 200)  # Radius from 1 nm to 1 µm
        # Constants
        epsilon = 80 * 8.854e-12  # Dielectric constant of water at room temperature (F/m)
        k_B = 1.38e-23  # Boltzmann constant (J/K)
        T = 298  # Temperature (K)
        N_A = 6.022e23  # Avogadro's number (1/mol)
        e = 1.602e-19  # Elementary charge (C)
        def debye_length(c_0):
            I = c_0  # Ionic strength for 1:1 electrolyte is equal to the concentration
            return np.sqrt(epsilon * k_B * T / (2 * N_A * e**2 * I))
        
        # Create meshgrid for plotting
        C_0, A = np.meshgrid(c_0_values, a_values)
        Debye = debye_length(C_0)
        print(Debye)
        ratio = A / Debye
        # Plotting
        plt.figure(figsize=(10, 8))
        plt.contourf(C_0, A, ratio, levels=np.logspace(-2, 2, 20), cmap='viridis')
        plt.colorbar(label='a / Debye length')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Salt Concentration (mol/L)')
        plt.ylabel('Colloidal Radius (m)')
        plt.title('Colloidal Radius / Debye Length Ratio')

        # Overlay regions for Hückel and Smoluchowski models
        plt.contour(C_0, A, ratio, levels=[0.1], colors='white', linestyles='--', linewidths=2)
        plt.contour(C_0, A, ratio, levels=[100], colors='red', linestyles='--', linewidths=2)
        plt.text(1e-4, 1e-7, 'Hückel Model', color='white', fontsize=12)
        plt.text(1e-1, 1e-8, 'Smoluchowski Model', color='red', fontsize=12)

        plt.show()
    def write_msg(self,
                  log: logger.logging.Logger  # To log
                  ) -> None:
        """write and log messages"""
        now = datetime.now()
        self.info_msg += \
            f'\tTime: {now.strftime("%Y-%m-%d %H:%M:%S")}\n'
        print(f'{bcolors.OKGREEN}{DLVOPotentialPhiZero.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    PhiZeroSigma(logger.logging.Logger('phi_0_sigma.log'),
                 AllConfig(),
                 debye_md=0.1,
                 charge_density_range=(0.1, 0.2))
