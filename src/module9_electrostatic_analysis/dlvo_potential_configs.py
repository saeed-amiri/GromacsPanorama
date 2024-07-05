"""
This module contains the configurations for the DLVO potential calculation.
"""

import typing
from dataclasses import dataclass, field

from common import elsevier_plot_tools


@dataclass
class FileConfig:
    """set the name of the input files"""
    charge_fname: str = 'charge_df_3_6.xvg'
    total_charge_coloumn: str = 'total_charge'
    contact_fname: str = 'contact.xvg'
    fout: str = 'potential.xvg'
    radial_avg_files: dict[str, str] = field(default_factory=lambda: {
        'numerical, 3.6': 'radial_average_potential_nonlinear_3_6.xvg',
        'numerical, 3.2': 'radial_average_potential_nonlinear_3_2.xvg'})


@dataclass
class ParameterConfig:
    """set parameters for the phi calculation
    radius of the nanopartcile is mandatory
    contact angle, is optioanl, it is used in case the contact file is
    not availabel
    """
    # pylint: disable=too-many-instance-attributes
    np_radius: float = 30.0  # In Ångströms
    stern_layer: float = 36.0  # In Ångströms
    computation_radius: float = 36.0  # In Ångströms
    avg_contact_angle: float = 38.0  # In Degrees
    np_core_charge: int = -8  # Number of charge inside the NP
    all_aptes_charges: int = 322  # Protonated APTES
    # Parameters for the phi computation
    phi_parameters: dict[str, float] = field(default_factory=lambda: {
        'T': 298.15,  # Temperature of the system
        'e_charge': 1.602e-19,  # Elementary charge [C]
        'c_salt': 0.00479,   # Bulk concentration of the salt in M(=mol/l)
        'epsilon': 78.5,  # medium  permittivity,
        'epsilon_0': 8.854187817e-12,   # vacuum permittivity, farads per meter
        'n_avogadro': 6.022e23,  # Avogadro's number
        'k_boltzman_JK': 1.380649e-23,  # Joules per Kelvin (J/K)
        'k_boltzman_eVK': 8.617333262145e-5,  # Electronvolts per Kelvin (eV/K)
        'phi_0': 1.0e-9,  # The potential at zero point (V)
        'box_xlim': 21.8,  # Length of the box in x direction [nm]
        'box_ylim': 21.8,  # Length of the box in y direction [nm]
        'box_zlim': 22.5  # Length of the box in z direction [nm] whole box
    })
    charge_sings: dict[str, int] = field(default_factory=lambda: ({
        'SOL': 0,
        'D10': 0,
        'CLA': -1,
        'ODN': +1,
        'POT': +1,
        'APT_COR': +1
    }))

    grids: list[int] = field(default_factory=lambda: [161, 161, 161])
    z_gird_up_limit: int = 161

    def __post_init__(self) -> None:
        self.nr_aptes_charges: int = \
            self.np_core_charge + self.all_aptes_charges


@dataclass
class PlotConfig(FileConfig):
    """
    Basic configurations and setup for the plots.
    """
    # pylint: disable=invalid-name
    # pylint: disable=too-many-instance-attributes
    graph_suffix: str = \
        f'els_potential_nonlinear.{elsevier_plot_tools.IMG_FORMAT}'

    labels: dict[str, str] = field(default_factory=lambda: {
        'title': 'potential',
        'ylabel': r'potential $\psi$ [mV]',
        'xlabel': 'distance X [nm]'
    })

    graph_styles: dict[str, typing.Any] = field(default_factory=lambda: {
        'label': 'analytical solution',
        'color': 'black',
        'marker': 'o',
        'linestyle': ':',
        'markersize': 0,
        'linewidth': elsevier_plot_tools.LINE_WIDTH,
    })

    line_styles: list[str] = \
        field(default_factory=lambda: ['-', ':', '--', '-.'])
    colors: list[str] = \
        field(default_factory=lambda: ['black',
                                       'darkred',
                                       'royalblue',
                                       'darkgreen',
                                       'dimgrey'])

    angstrom_to_nm: float = 0.1
    voltage_to_mV: float = 1000

    y_unit: str = ''
    y_lims: tuple[float, float] = (0, 200)
    x_lims: tuple[float, float] = (2.8, 7.7)

    x_ticks: list[float] = field(default_factory=lambda: [3, 5])
    y_ticks: list[float] = field(default_factory=lambda: [])

    legend_loc: str = 'upper right'
    if_np_radius_line: bool = True
    if_stern_line: bool = False
    if_debye_line: bool = False
    if_2nd_debye: bool = False

    if_title: bool = False
    if_grid: bool = False
    if_mirror_axes: bool = False
    plot_radial_avg: bool = True

    scheme_fig_path_0: str = \
        '/scratch/saeed/GÖHBP/PRE_DFG_7May24/single_np/15Oda/'
    scheme_fig_path_1: str = 'electrostatic_potential/exclusion_zone_edgy.jpg'
    scheme_fig_path: str = f'{scheme_fig_path_0}{scheme_fig_path_1}'

    isosurface_fig_path: str = \
        '/scratch/saeed/GÖHBP/PRE_DFG_7May24/single_np/15Oda/'
    isosurface_fig_name: str = 'electrostatic_potential/isosurface.png'
    isosurface_fig: str = f'{isosurface_fig_path}{isosurface_fig_name}'


@dataclass
class SolvingConfig(FileConfig, ParameterConfig):
    """set the parameters for the numerical solution
    Option for the computation type:
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
    phi_euation_aprx: (Poission-Boltzmann approximation eqution for phi)
        simple: simple linearized Poission-Boltzmann equation
        Loeb: Loeb equation
        Ohshima: Ohshima equation
    solver:
        _fsolve: use the fsolve from scipy
        _root: use the root from scipy
    """

    compute_type: str = 'non_linear'

    phi_0_type: str = 'grahame'

    ionic_strength: str = 'salt'

    phi_equation_aprx: str = 'Ohshima'

    solver: str = '_fsolve'


@dataclass
class PhiZeroSigmaConfig:
    """
    Set the for the experimental data
    such as concentration of the salt and the temperature
    and also the radii of the nanoparticles
    concentration of the salt is in M(=mol/l)
    radii in Angstrom
    temperature in Kelvin
    """
    exp_salt_concentration: list[float] = field(default_factory=lambda: [
        0.0048, 0.01, 0.1, 0.5, 1.0])
    exp_np_radii: list[float] = \
        field(default_factory=lambda: [30.0, 40.0, 50.0])
    exp_temperature: float = 298.15

    nr_density_points: float = 1000
    plot_bare_equation: bool = True

    y_lims: tuple[float, float] = field(default_factory=lambda: (1, 1.6))
    x_lims: tuple[float, float] = field(default_factory=lambda: (-0.001, 0.02))


@dataclass
class ComparisonConfigs:
    """set the parameters for the comparison of the data"""
    charge_column: str = 'total_charge'

    charge_files: dict[str, str] = field(default_factory=lambda: {
        '3.0': 'charge_df_3_0.xvg',
        '3.2': 'charge_df_3_2.xvg',
        '3.4': 'charge_df_3_4.xvg',
        '3.6': 'charge_df_3_6.xvg'})
    
    
    radial_avg_files: dict[str, str] = field(default_factory=lambda: {
        '3.0': 'radial_average_potential_nonlinear_3_0.xvg',
        '3.2': 'radial_average_potential_nonlinear_3_2.xvg',
        '3.4': 'radial_average_potential_nonlinear_3_4.xvg',
        '3.6': 'radial_average_potential_nonlinear_3_6.xvg',
        })


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
    ionic_type: str = 'salt'

    remove_phi_0_density_0: bool = False
    remove_phi_r_density_0: bool = False

    plot_config: PlotConfig = field(default_factory=PlotConfig)

    solving_config: SolvingConfig = field(default_factory=SolvingConfig)

    phi_r_calculater: str = 'non_linear'

    compare_phi_0_sigma: bool = True
    phi_zero_sigma_config: PhiZeroSigmaConfig = field(
        default_factory=PhiZeroSigmaConfig)

    plot_comparisons: bool = False
    comparison_configs: ComparisonConfigs = field(
        default_factory=ComparisonConfigs)
