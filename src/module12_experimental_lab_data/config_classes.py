"""
Configure classes for ploting lab data
This plot was seperated, incase later need to to some analysis besides
plotting.
InputFiles:
    data_file: str
    The data is taken from ppt send by the lab

"""

import sys

from dataclasses import dataclass, field

from common import elsevier_plot_tools


@dataclass
class InputFiles:
    """Input files dataclass"""
    data_file: str = field(init=False)

    def __post_init__(self) -> None:
        """Post init function"""
        self.data_file = sys.argv[1]


@dataclass
class BasePlotConfig:
    """Base plot configuration dataclass"""
    # pylint: disable=too-many-instance-attributes
    fout: str = f'plot.{elsevier_plot_tools.IMG_FORMAT}'

    ytick_labels: list[float] = \
        field(default_factory=lambda: [30, 40, 50, 60, 70])
    marker_size: int = \
        field(default_factory=lambda: elsevier_plot_tools.MARKER_SIZE)
    colors: list[str] = \
        field(default_factory=lambda: elsevier_plot_tools.MARKER_COLORS)
    marker_shape: list[str] = \
        field(default_factory=lambda: elsevier_plot_tools.MARKER_SHAPES)
    line_style: list[str] = \
        field(default_factory=lambda: elsevier_plot_tools.LINE_STYLES)
    line_width: int = elsevier_plot_tools.LINE_WIDTH

    y_label: str = ''
    x_label: str = ''

    y_lims: tuple[int, int] = field(default_factory=lambda: (30, 71))

    show_mirror_axis: bool = False
    show_guid_lines: bool = True
    show_y_label: bool = False


@dataclass
class CaCovConfig(BasePlotConfig):
    """Contact angle and coverage dataclass"""
    fout: str = f'ca_coverage.{elsevier_plot_tools.IMG_FORMAT}'

    x_label: str = r'c$_{ODA}$ [mM/L]'
    y_label: str = 'Contact angle and coverage'

    y_lims: tuple[int, int] = field(default_factory=lambda: (30, 71))

    show_mirror_axis: bool = False
    show_guid_lines: bool = True
    show_y_label: bool = False


@dataclass
class ToroidalConfig(BasePlotConfig):
    """toroidal radius dataclass"""
    # pylint: disable=too-many-instance-attributes
    fout: str = f'toroidal_radius.{elsevier_plot_tools.IMG_FORMAT}'

    y_label: str = r'S$_{ex}$ [nm]'
    x_label_surfactant: str = r'c$_{ODA}$ [mM/L]'
    x_label_salt: str = r'c$_{NaCl}$ [mM/L]'

    y_lims: tuple[int, int] = field(default_factory=lambda: (-3, 15))

    show_mirror_axis: bool = False
    show_guid_lines: bool = True
    show_y_label: bool = False

    schem_fig_path: str = \
        '/scratch/saeed/GÖHBP/PRE_DFG_7May24/experiment_data/'
    schem_fig_name: str = 'exclusion_zone_schem.png'
    schem_fig: str = f'{schem_fig_path}{schem_fig_name}'


@dataclass
class AllConfig(InputFiles):
    """Analysis configuration dataclass"""
    inf_replacement: float = -3.0
    nan_replacement: float = 0.0
    ca_cov: CaCovConfig = field(default_factory=CaCovConfig)
    toroidal: ToroidalConfig = field(default_factory=ToroidalConfig)
