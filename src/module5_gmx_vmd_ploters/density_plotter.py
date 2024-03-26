"""
Plotting density of the residues in the system
it reads the xvg output files from GRMACS and plot them

"""

import typing
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pylab as plt
import matplotlib.ticker as tck
from matplotlib.ticker import FuncFormatter

from common import logger, plot_tools, xvg_to_dataframe
from common.colors_text import TextColor as bcolors


@dataclass
class BaseGraphConfig:
    """base class for the graph configuration"""
    # pylint: disable=too-many-instance-attributes
    graph_suffix: str = 'denisty.png'
    y_col_name: str = 'density'
    xcol_name: str = 'coordinate (nm)'

    labels: dict[str, str] = field(default_factory=lambda: {
        'ylabel': 'denisty',
        'xlabel': 'coordinates (z) [nm]'
    })

    titles: dict[str, typing.Any] = field(default_factory=lambda: {
    })

    graph_styles: dict[str, typing.Any] = field(default_factory=lambda: {
        'marker': 'o',
        'markersize': 0,
    })

    legends: dict[str, str] = \
        field(default_factory=lambda: {
            'CLA': 'Cl',
            'amino_n': 'N (APTES)',
            'N': 'N (APTES)',
            'amino_charge': r'H$^+$ (APTES)',
            'SOL': 'O (Water)',
            'D10': 'Decane',
            'C5': r'C$_5$ (Decane)',
            'APT': 'N (APTES)',
            'NH2': 'N (ODA)',
            'POT': 'Na',
            'OH2': 'O (water)',
            'ODN': 'ODA'})

    line_styles: dict[str, str] = \
        field(default_factory=lambda: {
            'CLA': '-',
            'amino_n': '--',
            'N': '--',
            'amino_charge': '--',
            'SOL': '-',
            'D10': '-',
            'C5': '-',
            'APT': '-',
            'POT': '-',
            'OH2': '-',
            'NH2': '-.',
            'ODN': '-.'})

    colors: dict[str, str] = \
        field(default_factory=lambda: {
            'CLA': 'green',
            'amino_n': 'blue',
            'N': 'blue',
            'amino_charge': 'blue',
            'SOL': 'red',
            'D10': 'cyan',
            'C5': 'cyan',
            'APT': 'k',
            'POT': 'brown',
            'OH2': 'red',
            'ODN': 'orange',
            'NH2': 'orange'})

    height_ratio: float = (5 ** 0.5 - 1) * 1.5

    y_unit: str = ''

    legend_loc: str = 'lower right'


@dataclass
class FileConfig:
    """setups for the input files names and their columns name"""
    # pylint: disable=too-many-instance-attributes

    normalize_type: str = 'max'  # or any other words neside max!

    files: dict[str, dict[str, typing.Any]] = field(
        default_factory=lambda: {
            'dens_0': {'fname': 'APT.xvg', 'y_col': 'APT'},
            'dens_1': {'fname': 'CLA.xvg', 'y_col': 'CLA'},
            'dens_2': {'fname': 'COR_APT.xvg', 'y_col': 'COR_APT'},
            'dens_3': {'fname': 'COR.xvg', 'y_col': 'COR'},
            'dens_4': {'fname': 'D10.xvg', 'y_col': 'D10'},
            'dens_5': {'fname': 'ODN.xvg', 'y_col': 'ODN'},
            'dens_6': {'fname': 'POT.xvg', 'y_col': 'POT'},
            'dens_7': {'fname': 'SOL.xvg', 'y_col': 'SOL'}
            })
    plot_list: list[int] = \
        field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    legend_loc: str = 'lower right'
    window_legend_loc: str = 'upper left'
    max_indicator: str = 'dens_5'


@dataclass
class AllGraphConfig(BaseGraphConfig, FileConfig):
    """all the configurations for the graph"""


class MultiDensityPlotter:
    """"plotting density of the residues in the system"""

    info_msg: str = 'Message from MultiDensityPlotter:\n'
    configs: AllGraphConfig

    def __init__(self,
                 log: logger.logging.Logger,
                 configs: AllGraphConfig = AllGraphConfig()
                 ) -> None:
        self.configs = configs
        self.write_msg(log)

    def write_msg(self,
                  log: logger.logging.Logger
                  ) -> None:
        """Write and log messages."""
        print(f'{bcolors.OKCYAN}{MultiDensityPlotter.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    AllGraphConfig(logger.setup_logger('mutli_density_plot.log'))
