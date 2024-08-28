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
                 potentials: dict[np.int64, float],
                 type_data: str,
                 z_grid_spacing: tuple[float, float, float],
                 np_z_offset: float,
                 log: logger.logging.Logger,
                 configs: DensityFileConfig = DensityFileConfig()
                 ) -> None:
        self.file_configs = configs
        self.plot_density(
            potentials, type_data, z_grid_spacing, np_z_offset, log)
        self.write_msg(log)

    def plot_density(self,
                     potentials: dict[np.int64, float],
                     type_data: str,
                     z_grid_spacing: tuple[float, float, float],
                     np_z_offset: float,
                     log: logger.logging.Logger,
                     ) -> None:
        """plot the density of the system"""
        max_potential: np.float64 = self.get_max_potential(potentials)
        density_dict: dict[str, pd.DataFrame] = self.procces_file(log)
        fig_i, ax_i = plot_debye_surface_potential(potentials,
                                                   type_data,
                                                   z_grid_spacing,
                                                   np_z_offset,
                                                   close_fig=False)
        for i in self.file_configs.plot_list:
            y_col: str = self.file_configs.files[f'dens_{i}']['y_col']
            self.plot_density_i(ax_i,
                                density_dict[y_col],
                                y_col,
                                max_potential)
        # plt.legend()
        ax_i.xaxis.set_major_locator(plt.MaxNLocator())
        ax_i.set_xlim(5, 12)
        plt.savefig(f'density_{type_data}.png')

    def get_max_potential(self,
                          potentials: dict[np.int64, float]
                          ) -> float:
        """get the max potential"""
        return np.max(list(potentials.values()))

    def plot_density_i(self,
                       ax_i: plt.Axes,
                       density_dict: pd.DataFrame,
                       y_col: str,
                       max_potential: np.float64
                       ) -> None:
        """plot the density of the system"""
        ax_i.plot(density_dict.iloc[:, 0],
                  density_dict.iloc[:, 1] * float(max_potential),
                  label=f'{y_col}')

    def procces_file(self,
                     log: logger.logging.Logger
                     ) -> dict[str, pd.DataFrame]:
        """read the files and return the normalized density"""
        density_dict: dict[str, pd.DataFrame] = {}
        for i in self.file_configs.plot_list:
            fname: str = self.file_configs.files[f'dens_{i}']['fname']
            y_col: str = self.file_configs.files[f'dens_{i}']['y_col']
            df: pd.DataFrame = \
                xvg_to_dataframe.XvgParser(fname, log, x_type=float).xvg_df
            df[[y_col]] /= np.max(df[[y_col]])
            density_dict[y_col] = df
        return density_dict

    def write_msg(self,
                  log: logger.logging.Logger  # To log
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{SurfacePotentialAndDensityPlot.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    print('This script is not intended to be run independently!')
