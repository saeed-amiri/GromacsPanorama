"""
This module contains the configurations for the DLVO potential calculation.
"""

import typing
from dataclasses import dataclass, field

from common import elsevier_plot_tools


@dataclass
class FileConfig:
    """set the name of the input files"""
    charge_fname: str = 'charge_df.xvg'
    contact_fname: str = 'contact.xvg'
    fout: str = 'potential.xvg'
    radial_avg_files: dict[str, str] = field(default_factory=lambda: {
        'numerical, nonlinear': 'radial_average_potential_nonlinear.xvg',
        'numerical, linear': 'radial_average_potential_linear.xvg'})


@dataclass
class ParameterConfig:
    """set parameters for the phi calculation
    radius of the nanopartcile is mandatory
    contact angle, is optioanl, it is used in case the contact file is
    not availabel
    """
    # pylint: disable=too-many-instance-attributes
    np_radius: float = 30.0  # In Ångströms
    stern_layer: float = 30.0  # In Ångströms
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
class PlotConfig(FileConfig):
    """
    Basic configurations and setup for the plots.
    """
    # pylint: disable=invalid-name
    # pylint: disable=too-many-instance-attributes
    graph_suffix: str = f'els_potential_nonlinear.{elsevier_plot_tools.IMG_FORMAT}'

    labels: dict[str, str] = field(default_factory=lambda: {
        'title': 'potential',
        'ylabel': r'potential $\psi$ [mV]',
        'xlabel': 'distance X [nm]'
    })

    graph_styles: dict[str, typing.Any] = field(default_factory=lambda: {
        'label': 'analytic average',
        'color': 'black',
        'marker': 'o',
        'linestyle': '-',
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
    y_lims: tuple[float, float] = (0, 260)
    x_lims: tuple[float, float] = (2, 12)

    x_ticks: list[float] = field(default_factory=lambda: [3, 7, 9])
    y_ticks: list[float] = field(default_factory=lambda: [])

    legend_loc: str = 'upper right'
    if_stern_line: bool = True
    if_debye_line: bool = True
    if_2nd_debye: bool = False

    if_title: bool = False
    if_grid: bool = False
    if_mirror_axes: bool = False
    plot_radial_avg: bool = True

    scheme_fig_path_0: str = \
        '/scratch/saeed/GÖHBP/PRE_DFG_7May24/single_np/15Oda/'
    scheme_fig_path_1: str = 'electrostatic_potential/dlvo_sphere.jpg'
    scheme_fig_path: str = f'{scheme_fig_path_0}{scheme_fig_path_1}'

    isosurface_fig_path: str = \
        '/scratch/saeed/GÖHBP/PRE_DFG_7May24/single_np/15Oda/'
    isosurface_fig_name: str = 'electrostatic_potential/isosurface.png'
    isosurface_fig: str = f'{isosurface_fig_path}{isosurface_fig_name}'


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
    compute_type: str = 'non_linear'
    phi_0_type: str = 'grahame'
    ionic_type: str = 'salt'
    plot_config: PlotConfig = field(default_factory=PlotConfig)
