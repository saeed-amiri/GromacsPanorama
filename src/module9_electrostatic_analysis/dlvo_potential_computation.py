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

import sys
import typing
from datetime import datetime
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import matplotlib.pylab as plt

from common import logger, xvg_to_dataframe, my_tools, plot_tools
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
    np_core_charge: int = -8  # Number of charge inside the NP
    all_aptes_charges: int = 322  # Protonated APTES
    # Parameters for the phi computation
    phi_parameters: dict[str, float] = field(default_factory=lambda: {
        'T': 298.15,  # Temperature of the system
        'e_charge': 1.602e-19,  # Elementary charge [C]
        'c_salt': 0.0046,   # Bulk concentration of the salt in M(=mol/l)
        'epsilon': 78.5,  # medium  permittivity,
        'epsilon_0': 8.854187817e-12,   # vacuum permittivity, farads per meter
        'n_avogadro': 6.022e23,  # Avogadro's number
        'k_boltzman_JK': 1.380649e-23,  # Joules per Kelvin (J/K)
        'k_boltzman_eVK': 8.617333262145e-5,  # Electronvolts per Kelvin (eV/K)
        'phi_0': 1.0e-9,  # The potential at zero point (V)
        'box_xlim': 21.7,  # Length of the box in x direction [nm]
        'box_ylim': 21.7,  # Length of the box in y direction [nm]
        'box_zlim': 11.3  # Length of the box in z direction [nm] (water)
    })
    charge_sings: dict[str, int] = field(default_factory=lambda: ({
        'SOL': 0,
        'D10': 0,
        'CLA': -1,
        'ODN': +1,
        'POT': +1,
        'APT_COR': +1
    }))

    def __post_init__(self) -> None:
        self.nr_aptes_charges: int = \
            self.np_core_charge + self.all_aptes_charges


@dataclass
class PlotConfig:
    """
    Basic configurations and setup for the plots.
    """
    # pylint: disable=too-many-instance-attributes
    graph_suffix: str = 'els_potential.png'

    labels: dict[str, str] = field(default_factory=lambda: {
        'title': 'ELS potential',
        'ylabel': 'ELS potential [mV]',
        'xlabel': 'r [nm]'
    })

    graph_styles: dict[str, typing.Any] = field(default_factory=lambda: {
        'label': '15Oda',
        'color': 'black',
        'marker': 'o',
        'linestyle': '-',
        'markersize': 0,
    })

    line_styles: list[str] = \
        field(default_factory=lambda: ['-', ':', '--', '-.'])
    colors: list[str] = \
        field(default_factory=lambda: ['black', 'red', 'blue', 'green'])

    height_ratio: float = (5 ** 0.5 - 1) * 1.5

    y_unit: str = ''

    legend_loc: str = 'lower right'


@dataclass
class AllConfig(FileConfig, ParameterConfig):
    """set all the parameters and confiurations
    computation types:
    compute_type:
        planar: Linearized Possion-Boltzmann for planar approximation
        sphere: Linearized Possion-Boltzmann for a sphere
    phi_0_set:
        grahame: Grahame equation
        grahame_low: Grahame equation for low potential
        constant: from a constant value
    ionic strength:
        salt: use the slac concentration
        all: compute it from all charge groups in the system
    """
    compute_type: str = 'sphere'
    phi_0_type: str = 'grahame'
    ionic_type: str = 'salt'
    plot_config: PlotConfig = field(default_factory=PlotConfig)


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
        radii, phi_r = self.compute_potential(debye_l)
        self.plot_save_phi(radii, phi_r, debye_l)

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
            rf'$kappa$ r ={debye_l_nm*self.configs.np_radius/10:.3f}'
            '\n')
        return float(debye_l_nm)

    def compute_potential(self,
                          debye_l: float
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
            radii, phi_r = self._linear_shpere(debye_l, phi_0, box_lim/2)
        return radii, phi_r

    def _get_phi_zero(self,
                      debye_l: float
                      ) -> np.ndarray:
        """get the value of the phi_0 based on the configuration"""
        phi_0: np.ndarray = np.zeros(self.charge.shape)
        if (phi_0_type := self.configs.phi_0_type) == 'constant':
            phi_0 += self.configs.phi_parameters['phi_0']
        elif phi_0_type == 'grahame_low':
            phi_0 = self._compute_phi_0_grahame_low_potential(debye_l)
        elif phi_0_type == 'grahame':
            phi_0 = self._compute_phi_0_grahame(debye_l)
        self.info_msg += \
            f'\tAvg. {phi_0.mean() = :.3f} [V] from `{phi_0_type}` values\n'
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

    def plot_save_phi(self,
                      radii: np.ndarray,
                      phi_r: np.ndarray,
                      debye_l: float
                      ) -> None:
        """plot and save the electostatic potential"""
        configs: PlotConfig = self.configs.plot_config
        phi_mv: np.ndarray = phi_r * 100
        # Find the index of the closest value in radii to debye_l
        idx_closest = np.abs(radii - debye_l).argmin()
        # Get the corresponding phi_r value
        phi_value = phi_mv[idx_closest]
        ax_i: plt.axes
        fig_i: plt.figure
        fig_i, ax_i = plot_tools.mk_canvas(x_range=(0, radii.max()),
                                           height_ratio=configs.height_ratio,
                                           num_xticks=20)
        ax_i.plot(radii, phi_mv, **configs.graph_styles)
        ax_i.grid(True, 'both', ls='--', color='gray', alpha=0.5, zorder=2)
        ax_i.set_xlabel(configs.labels.get('xlabel'))
        ax_i.set_ylabel(configs.labels.get('ylabel'))
        # Plot vertical line at debye_l
        y_lims: tuple[float, float] = ax_i.get_ylim()
        x_lims: tuple[float, float] = ax_i.get_xlim()
        y_lim_min: float = -0.85
        ax_i.vlines(x=debye_l,
                    ymin=y_lim_min,
                    ymax=phi_value,
                    color=configs.colors[1],
                    linestyle=configs.line_styles[1],
                    label=f'Debye Length: {debye_l:.2f} [nm]')
        # Plot horizontal line from phi_value to the graph
        ax_i.hlines(y=phi_value,
                    xmin=0,
                    xmax=debye_l,
                    color=configs.colors[2],
                    linestyle=configs.line_styles[2],
                    label=f'Potential: {phi_value: .2f} [mV]')
        ax_i.set_xlim(x_lims)
        ax_i.set_ylim((y_lim_min, y_lims[1]))
        plot_tools.save_close_fig(fig_i, ax_i, fname=configs.graph_suffix)

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


class IonicStrengthCalculation:
    """computation of the ionic strength in the system by considering
    all charge groups in the system.
    The number of charge group are a lot, and effect of the salt on
    the results is very small. The results are almost the same without
    any salt in the box. If the concentration of the salt is zero, then
    the computing the debye length is impossible!
    """
    # pylint: disable=too-few-public-methods

    info_msg: str = 'Message from IonicStrengthCalculation:\n'
    ionic_strength: float

    def __init__(self,
                 topol_fname: str,  # Name of the topology file
                 log: logger.logging.Logger,
                 configs: AllConfig
                 ) -> None:
        res_nr: dict[str, int] = my_tools.read_topol_resnr(topol_fname, log)
        self.ionic_strength = self.compute_ionic_strength(res_nr, configs, log)
        self._write_msg(log)

    def compute_ionic_strength(self,
                               res_nr: dict[str, int],
                               configs: AllConfig,
                               log: logger.logging.Logger
                               ) -> float:
        """compute the ionic strength based on the number of charge
        groups"""
        ionic_strength: float = 0.0
        volume: float = self._get_water_volume(configs)
        concentration_dict = \
            self._compute_concentration(res_nr, volume, configs)
        self._check_electroneutrality(
            res_nr, concentration_dict, configs.charge_sings, log)
        for res, _ in res_nr.items():
            ionic_strength += \
                concentration_dict[res] * configs.charge_sings[res]**2
        self.info_msg += f'\t{ionic_strength = :.5f} [mol/l]\n'
        return ionic_strength

    def _compute_concentration(self,
                               res_nr: dict[str, int],
                               volume: float,
                               configs: AllConfig
                               ) -> dict[str, float]:
        """compute the concentration for each charge group"""
        # pylint: disable=invalid-name
        concentration: dict[str, float] = {}
        for res, nr in res_nr.items():
            if res not in ['D10', 'APT_COR']:
                pass
            elif res == 'APT_COR':
                nr = configs.nr_aptes_charges
            elif res == 'D10':
                concentration[res] = 0.0
            concentration[res] = nr / (
                    volume * configs.phi_parameters['n_avogadro'])
        return concentration

    def _check_electroneutrality(self,
                                 res_nr: dict[str, int],  # nr. of charges
                                 concentration: dict[str, float],
                                 charge_sings: dict[str, int],
                                 log: logger.logging.Logger
                                 ) -> None:
        """check the electroneutrality of the system:
        must: \\sum_i c_i Z_i = 0
        """
        # pylint: disable=invalid-name
        electroneutrality: float = 0
        for res, _ in res_nr.items():
            electroneutrality += concentration[res] * charge_sings[res]
        if round(electroneutrality, 3) != 0.0:
            log.error(
                msg := f'\tError! `{electroneutrality = }` must be zero!\n')
            sys.exit(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')

    def _get_water_volume(self,
                          configs: AllConfig
                          ) -> float:
        """return volume of the water section in the box
        dimensions are in nm, the return should be in liters, so:
        nm^3 -> liters: (1e-9)^3 * 1000 = 1e-24 liters
        """
        volume: float = \
            configs.phi_parameters['box_xlim'] * \
            configs.phi_parameters['box_ylim'] * \
            configs.phi_parameters['box_zlim'] * 1e-24
        self.info_msg += f'\tWater section `{volume = :.4e}` liters\n'
        return volume

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        now = datetime.now()
        self.info_msg += \
            f'\tTime: {now.strftime("%Y-%m-%d %H:%M:%S")}\n'
        print(f'{bcolors.OKCYAN}{IonicStrengthCalculation.__name__}:\n'
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
        density: np.ndarray = \
            (charge * configs.phi_parameters['e_charge']) / cap_surface
        self.info_msg += (f'\tAve. `{charge.mean() = :.3f}` [e]\n'
                          f'\t`{cap_surface.mean()*1e18 = :.3f}` [nm^2]\n'
                          f'\tAve. `charge_{density.mean() = :.3f}` '
                          f'[C/m^2] or [As/m^2] \n')
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
        # Formula: A = 2 * pi * r^2 * (1 + cos(θ))
        # Alco converted from Ångströms^2 to m^2
        in_water_cap_area: np.ndarray = \
            2 * np.pi * np_radius**2 * (1 + np.cos(radians))
        return in_water_cap_area * 1e-20

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
