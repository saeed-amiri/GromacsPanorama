"""
Plotting density of the residues in the system
it reads the xvg output files from GRMACS and plot them

"""

import typing
from dataclasses import dataclass, field

import pandas as pd

import matplotlib.pylab as plt

from common import logger, plot_tools, xvg_to_dataframe
from common.colors_text import TextColor as bcolors


@dataclass
class BaseGraphConfig:
    """base class for the graph configuration"""
    # pylint: disable=too-many-instance-attributes
    graph_suffix: str = 'density.png'
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
            'SOL': 'Water',
            'D10': 'Decane',
            'C5': r'C$_5$ (Decane)',
            'NH2': 'N (ODA)',
            'POT': 'Na',
            'OH2': 'O (water)',
            'ODN': 'ODA',
            'COR': 'Silica',
            'APT': 'APTES',
            'COR_APT': 'CORE_APTES',
            })

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
            'ODN': '-.',
            'COR': '-',
            'COR_APT': '-'
            })

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
            'NH2': 'orange',
            'COR': 'grey',
            'COR_APT': 'purple'
            })

    height_ratio: float = (5 ** 0.5 - 1) * 1.5

    y_unit: str = ''

    legend_loc: str = 'center left'


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
        field(default_factory=lambda: [0, 1, 3, 4, 5, 6, 7])
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
    dens_data: dict[str, pd.DataFrame]

    def __init__(self,
                 log: logger.logging.Logger,
                 configs: AllGraphConfig = AllGraphConfig()
                 ) -> None:
        self.configs = configs
        self.dens_data = self.initate_data(log)
        self.initate_plot()
        self.write_msg(log)

    def initate_data(self,
                     log: logger.logging.Logger
                     ) -> dict[str, pd.DataFrame]:
        """initiate the data for the plotting"""
        dens_data: dict[str, pd.DataFrame] = {}
        for key, value in self.configs.files.items():
            dens_data[key] = \
                xvg_to_dataframe.XvgParser(value['fname'],
                                           x_type=float,
                                           log=log).xvg_df
        return dens_data

    def initate_plot(self) -> None:
        """initiate the plot for the density"""
        ax_i: plt.axes
        ax_i = self.plot_density()

        self._save_plot(ax_i)
        self._save_plot_zoom(ax_i)

    def _save_plot(self,
                   ax_i: plt.axes
                   ) -> None:
        """save the plot"""
        plot_tools.save_close_fig(ax_i.figure,
                                  ax_i,
                                  fname=self.configs.graph_suffix,
                                  loc=self.configs.legend_loc)

    def _save_plot_zoom(self,
                        ax_i: plt.axes,
                        ) -> None:
        """save the zoomed plot"""
        ax_i.set_xlim(2, 21)
        ax_i.set_ylim(-5, 90)
        plot_tools.save_close_fig(ax_i.figure,
                                  ax_i,
                                  fname='density_zoom.png',
                                  legend_font_size=10,
                                  if_close=False,
                                  loc='upper left')

    def plot_density(self) -> plt.axes:
        """plot the density of the residues"""
        first_key: str = next(iter(self.dens_data))
        x_range: tuple[float, float] = (
            self.dens_data[first_key]['Coordinate_nm'].iat[0],
            self.dens_data[first_key]['Coordinate_nm'].iat[-1])
        ax_i: plt.axes
        _, ax_i = plot_tools.mk_canvas(
            x_range,
            height_ratio=self.configs.height_ratio,
            num_xticks=7)
        for key in self.configs.plot_list:
            dens_data: pd.DataFrame = self.dens_data[f'dens_{key}']
            ax_i = self.plot_single_density(ax_i, dens_data, key)
        ax_i.set_xlabel(self.configs.labels['xlabel'])
        ax_i.set_ylabel(self.configs.labels['ylabel'])
        ax_i.legend(loc=self.configs.legend_loc)
        ax_i.grid(True, 'both', ls='--', color='gray', alpha=0.5, zorder=2)
        return ax_i

    def plot_single_density(self,
                            ax_i: plt.axes,
                            data: pd.DataFrame,
                            key: int
                            ) -> plt.axes:
        """plot a single density"""
        column_name: str = self.configs.files[f'dens_{key}']['y_col']
        ax_i.plot(data.iloc[:, 0],
                  data.iloc[:, 1],
                  label=self.configs.legends[column_name],
                  color=self.configs.colors[column_name],
                  linestyle=self.configs.line_styles[column_name])
        return ax_i

    def write_msg(self,
                  log: logger.logging.Logger
                  ) -> None:
        """Write and log messages."""
        print(f'{bcolors.OKCYAN}{MultiDensityPlotter.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    MultiDensityPlotter(logger.setup_logger('density_plot.log'))
