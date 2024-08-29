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

import matplotlib as mpl
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
        field(default_factory=lambda: [0, 8, 1, 7])


class DenityPlotConfiguration:
    """The configuration for the density plot"""
    # pylint: disable=missing-function-docstring
    # pylint: disable=invalid-name
    # pylint: disable=too-many-arguments

    @property
    def DENSITY(self) -> dict[str, str | tuple[float, float] | list[float]]:
        return {'label': r'$\rho$',
                'ylable': r'Density ($\rho$) a.u.',
                'output_file': 'surface_potential.jpg',
                'legend_loc': 'lower left',
                }

    @property
    def PSI_0(self) -> dict[str, str | float | int]:
        return {'label': r'$\psi_0$',
                'ylable': r'Edge potential ($\psi_0$) [mV]',
                'legend_loc': 'lower left',
                'ls': ':',
                'linewidth': 2.0,
                'marker': 'o',
                'marker_size': 3,
                'color': 'darkgreen',
                }

    @property
    def DENSITY_RESIDUE(self) -> dict[str, str]:
        return {
            'SOL': 'Water',
            'D10': 'Oil',
            'ODN': 'ODA',
            'APT': 'APTES',
            'CLA': 'Cl',
            'POT': 'Na',
            'COR': 'COR',
            'COR_APT': 'NP',
            'NH2': 'N',
        }

    @property
    def DENSITY_COLOR(self) -> dict[str, str]:
        return {
            'SOL': 'darkred',
            'D10': 'black',
            'ODN': 'orange',
            'APT': 'royalblue',
            'CLA': 'darkgreen',
            'POT': 'red',
            'COR': 'brown',
            'COR_APT': 'brown',
            'NH2': 'royalblue',
        }

    @property
    def DENSITY_LINESTYLE(self) -> dict[str, str]:
        return {
            'SOL': ':',
            'D10': '--',
            'ODN': '-',
            'APT': '-',
            'CLA': '-.',
            'POT': ':',
            'COR': '-',
            'COR_APT': '--',
            'NH2': '-',
        }

    @property
    def YLIMS(self) -> tuple[float, float]:
        return (-15, 130)

    @property
    def XLIMS(self) -> tuple[float, float]:
        return (5.5, 12.5)

    @property
    def XTICKS(self) -> list[float]:
        return [6, 8, 10, 12]

    @property
    def YTICKS(self) -> list[float]:
        return [0, 60, 120]

    @property
    def X_LABEL(self) -> str:
        return 'z [nm]'

    @property
    def Y_LABEL(self) -> str:
        return r'Edge Potential ($\psi_0$) [mV]'

    @property
    def Y_LABEL_DENSITY(self) -> str:
        return r'Normalized Density ($\rho$) a.u.'


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
        # pylint: disable=too-many-arguments
        self.file_configs = configs
        self.plot_config = DenityPlotConfiguration()
        self.plot_density_phi_zero(
            potentials, type_data, z_grid_spacing, np_z_offset, log)
        self.write_msg(log)

    def plot_density_phi_zero(self,
                              potentials: dict[np.int64, float],
                              type_data: str,
                              z_grid_spacing: tuple[float, float, float],
                              np_z_offset: float,
                              log: logger.logging.Logger,
                              ) -> None:
        """plot the density of the system"""
        # pylint: disable=too-many-arguments
        max_potential: np.float64 = self.get_max_potential(potentials)
        density_dict: dict[str, pd.DataFrame] = self.procces_file(log)
        potential_data: dict[str, typing.Any] = \
            plot_debye_surface_potential(potentials,
                                         type_data,
                                         z_grid_spacing,
                                         np_z_offset,
                                         return_data=True)

        fig_i, ax_i = elsevier_plot_tools.mk_canvas('double_column')
        self.plot_densities(ax_i, density_dict, max_potential)
        z_indicies = self.plot_potential(ax_i, potential_data)
        ax_j: mpl.axes._axes.Axes = self.set_mirror_xaxis(ax_i, z_indicies)
        lgd: mpl.legend.Legend = self.handel_legend(ax_i, ax_j, fig_i)

        self.set_axis_style(ax_i)
        self.set_mirror_axis(ax_i)

        plt.tight_layout()
        plt.savefig(f'density_{type_data}.png',
                    bbox_extra_artists=(lgd,),
                    bbox_inches='tight')

    def set_axis_style(self,
                       ax_i: plt.Axes
                       ) -> None:
        """set the style of the axis"""
        plt.rcParams.update({
            'axes.labelsize': 10,  # Font size for x and y labels
            'xtick.labelsize': 8,  # Font size for x tick labels
            'ytick.labelsize': 8,  # Font size for y tick labels
            'legend.fontsize': 10,  # Font size for legend
            'font.size': 10  # General font size
        })
        ax_i.set_xlim(self.plot_config.XLIMS)
        ax_i.set_ylim(self.plot_config.YLIMS)

        ax_i.set_xlabel(self.plot_config.X_LABEL, fontsize=10)
        ax_i.set_ylabel(self.plot_config.Y_LABEL, fontsize=10)

        ax_i.set_yticks(self.plot_config.YTICKS)
        ax_i.set_xticks(self.plot_config.XTICKS)
        ax_i.tick_params(axis='both', which='major', labelsize=10)

    def set_mirror_axis(self,
                        ax_i: plt.Axes
                        ) -> None:
        """set the mirror axis"""
        ax_f = ax_i.twinx()
        ax_f.tick_params(axis='y',
                         which='both',
                         )
        ax_f.set_yticks([])
        ax_f.set_ylabel(self.plot_config.Y_LABEL_DENSITY)

    def get_max_potential(self,
                          potentials: dict[np.int64, float]
                          ) -> np.float64:
        """get the max potential"""
        return np.max(list(potentials.values()))

    def plot_densities(self,
                       ax_i: plt.Axes,
                       density_dict: dict[str, pd.DataFrame],
                       max_potential: np.float64,
                       ) -> None:
        """plot the density of the system"""
        colors: dict[str, str] = self.plot_config.DENSITY_COLOR
        linestyles: dict[str, str] = self.plot_config.DENSITY_LINESTYLE
        residue: dict[str, str] = self.plot_config.DENSITY_RESIDUE
        for ind in self.file_configs.plot_list:
            y_col: str = self.file_configs.files[f'dens_{ind}']['y_col']
            self.plot_density_i(ax_i,
                                density_dict[y_col],
                                max_potential,
                                colors[y_col],
                                linestyles[y_col],
                                residue[y_col]
                                )

    def plot_density_i(self,
                       ax_i: plt.Axes,
                       density_dict: pd.DataFrame,
                       max_potential: np.float64,
                       color: str,
                       linestyle: str,
                       residue: str,
                       ) -> None:
        """plot the density of the system"""
        # pylint: disable=too-many-arguments
        label: str = rf'$\rho_{{{residue}}}$'
        ax_i.plot(density_dict.iloc[:, 0],
                  density_dict.iloc[:, 1] * float(max_potential),
                  ls=linestyle,
                  lw=2.0,
                  color=color,
                  label=label
                  )

    def plot_potential(self,
                       ax_i: plt.Axes,
                       potential_data: dict[str, typing.Any],
                       ) -> np.ndarray:
        """plot the potential of the system"""
        xdata: np.ndarray = potential_data['xdata']
        ydata: np.ndarray = potential_data['ydata']
        oda_bound: tuple[float, float] = potential_data['oda_bound']
        z_indicies: np.ndarray = potential_data['z_indicies']
        _configs: dict[str, str | int | float] = self.plot_config.PSI_0
        ax_i.plot(xdata,
                  ydata,
                  ls=_configs['ls'],
                  lw=_configs['linewidth'],
                  marker=_configs['marker'],
                  markersize=_configs['marker_size'],
                  color=_configs['color'],
                  label=_configs['label']
                  )
        # Shade the area between ODA_BOUND
        ax_i.fill_betweenx(ylims := ax_i.get_ylim(),
                           oda_bound[0],
                           oda_bound[1],
                           color='gray',
                           edgecolor=None,
                           alpha=0.5,
                           label='Water surface',
                           )
        ax_i.set_ylim(ylims)
        return z_indicies

    def set_mirror_xaxis(self,
                         ax_i: plt.Axes,
                         z_indicies: np.ndarray,
                         ) -> mpl.axes._axes.Axes:
        """set the mirror x-axis"""
        ax_j = ax_i.twiny()
        ax_j.plot(z_indicies, np.zeros_like(z_indicies), alpha=0.0)
        ax_j.set_xticks([60, 70, 80, 90, 100])
        return ax_j

    def handel_legend(self,
                      ax_i: plt.Axes,
                      ax_j: plt.Axes,
                      fig_i: plt.Figure,
                      ) -> mpl.legend.Legend:
        """handle the legend"""
        handles_i, labels_i = ax_i.get_legend_handles_labels()
        handles_j, labels_j = ax_j.get_legend_handles_labels()
        handles = handles_i + handles_j
        labels = labels_i + labels_j

        # Set the combined legend on the primary axis
        return ax_i.legend(handles,
                           labels,
                           bbox_to_anchor=(1.08, 0.5),
                           loc='center right',
                           bbox_transform=fig_i.transFigure,
                           ncol=1,
                           borderaxespad=0.4)

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
