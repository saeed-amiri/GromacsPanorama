"""
Plot the surface potential and the density of the system along the z-axis.
The goal is to compare the surface potential and the charge density
of the system and see where is the surface we selected for averaging
the surface potential.
It read the density of the systems in different xvg files and plot them.
The potential will be using the function:
    plot_debye_surface_potential
in the [...]_plots.py file.
"""

import typing
from dataclasses import field
from dataclasses import dataclass

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from common import logger
from common import xvg_to_dataframe
from common import elsevier_plot_tools
from common.colors_text import TextColor as bcolors

from module9_electrostatic_analysis.apbs_analysis_average_pot_plots import \
    plot_debye_surface_potential


@dataclass
class DensityFileConfig:
    """setups for the input files names and their columns name"""
    # pylint: disable=too-many-instance-attributes

    normalize_type: str = 'max'  # or any other words neside max!

    files: dict[str, dict[str, typing.Any]] = field(
        default_factory=lambda: {
            'dens_0': {'fname': 'SOL.xvg', 'y_col': 'SOL'},
            'dens_1': {'fname': 'D10.xvg', 'y_col': 'D10'},
            'dens_2': {'fname': 'ODN.xvg', 'y_col': 'ODN'},
            'dens_3': {'fname': 'APT.xvg', 'y_col': 'APT'},
            'dens_4': {'fname': 'CLA.xvg', 'y_col': 'CLA'},
            'dens_5': {'fname': 'POT.xvg', 'y_col': 'POT'},
            'dens_6': {'fname': 'COR.xvg', 'y_col': 'COR'},
            'dens_7': {'fname': 'COR_APT.xvg', 'y_col': 'COR_APT'},
            'dens_8': {'fname': 'ODN_N.xvg', 'y_col': 'NH2'},
            })

    plot_list: list[int] = \
        field(default_factory=lambda: [0, 1, 7, 8])


@dataclass
class DenityPlotConfiguration:
    """The configuration for the density plot"""
    # pylint: disable=missing-function-docstring
    # pylint: disable=invalid-name
    # pylint: disable=too-many-arguments


class SurfacePotentialAndDensityPlot:
    """
    Plot the density of the system along the z-axis
    """

    info_msg: str = 'Message from SurfacePotentialAndDensityPlot:\n'
    configs: DensityFileConfig

    def __init__(self,
                 log: logger.logging.Logger,
                 configs: DensityFileConfig = DensityFileConfig()
                 ) -> None:
        self.file_configs = configs
        self.plot_density(log)
        self.write_msg(log)

    def plot_density(self,
                     log: logger.logging.Logger
                     ) -> None:
        """plot the density of the system"""
        density_dict: dict[str, pd.DataFrame] = self.procces_file(log)

    def procces_file(self,
                     log: logger.logging.Logger
                     ) -> dict[str, pd.DataFrame]:
        """read the files and return the normalized density"""
        density_dict: dict[str, pd.DataFrame] = {}
        for i in self.file_configs.plot_list:
            fname: str = self.file_configs.files[f'dens_{i}']['fname']
            y_col: str = self.file_configs.files[f'dens_{i}']['y_col']
            df: pd.DataFrame = xvg_to_dataframe.XvgParser(fname, log).xvg_df
            denity_i: pd.DataFrame = df[[y_col]]
            density_dict[y_col] = denity_i/np.max(denity_i)
        return density_dict


    def write_msg(self,
                  log: logger.logging.Logger  # To log
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{SurfacePotentialAndDensityPlot.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    pass
