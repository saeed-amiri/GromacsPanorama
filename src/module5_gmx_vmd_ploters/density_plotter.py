"""
Plotting density of the residues in the system
it reads the xvg output files from GRMACS and plot them

"""

import typing
from dataclasses import dataclass, field

import pandas as pd

import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
            'CLA': ':',
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
            'APT': 'grey',
            'POT': 'brown',
            'OH2': 'red',
            'ODN': 'orange',
            'NH2': 'orange',
            'COR': '#DAA520',  # goldenrod
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
            'dens_0': {'fname': 'SOL.xvg', 'y_col': 'SOL'},
            'dens_1': {'fname': 'D10.xvg', 'y_col': 'D10'},
            'dens_2': {'fname': 'ODN.xvg', 'y_col': 'ODN'},
            'dens_3': {'fname': 'APT.xvg', 'y_col': 'APT'},
            'dens_4': {'fname': 'CLA.xvg', 'y_col': 'CLA'},
            'dens_5': {'fname': 'POT.xvg', 'y_col': 'POT'},
            'dens_6': {'fname': 'COR.xvg', 'y_col': 'COR'},
            'dens_7': {'fname': 'COR_APT.xvg', 'y_col': 'COR_APT'},
            })
    plot_list: list[int] = \
        field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])
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
            self._get_bulk_density(key, dens_data[key].copy())
        return dens_data

    def _get_bulk_density(self,
                          key: str,
                          data: pd.DataFrame
                          ) -> None:
        """get the bulk density
        sorting the data, then get the bulk density of the  20 max
        values
        """
        data.sort_values(by=data.iloc[:, 1].name,
                         ascending=False,
                         inplace=True)
        bulk_density: float = data.iloc[:20, 1].mean()
        file_data = self.configs.files.get(key)
        residue_name: str = file_data.get('y_col')
        self.info_msg += \
            f'\tThe bulk density of {residue_name} is: {bulk_density:.3f}\n'

    def initate_plot(self) -> None:
        """initiate the plot for the density"""
        ax_i: plt.axes
        ax_i = self.plot_density(self.dens_data)

        self._save_plot(ax_i)
           # Store original limits
        original_xlim = ax_i.get_xlim()
        original_ylim = ax_i.get_ylim()
        self._save_plot_zoom(ax_i)
        self._save_plot_normalized(original_xlim, yticks=False)

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
                                  if_close=True,
                                  loc='upper left')

    def _save_plot_normalized(self,
                              xlim: tuple[float, float],
                              yticks: bool = True
                              ) -> None:
        """save the normalized plot"""
        ax_i: plt.axes
        dens_data = self.normaliz_density(self.dens_data)
        ax_i = self.plot_density(dens_data, grids=False)
        label_x_loc: float = -0.008
        ax_i.set_xlim(-0.5, xlim[1])
        ax_i.set_ylim(-0.0, 1.05)
        ax_i.set_yticks([])
        if yticks:
            yticks: list[int] = [0, 0.5, 1.0]
            ax_i.set_yticks(yticks)
            label_x_loc: float = -0.09
            ax_i.hlines(
                0, xlim[0], xlim[1], color='grey', ls='--', lw=1, alpha=0.5)
            ax_i.hlines(
                1, xlim[0], xlim[1], color='grey', ls='--', lw=1, alpha=0.5)

        ax_i.text(label_x_loc,
                  1,
                  'a)',
                  ha='right',
                  va='top',
                  transform=ax_i.transAxes,
                  fontsize=22)
        ax_i.set_ylabel('normalized density', fontsize=18)
        plot_tools.save_close_fig(ax_i.figure,
                                  ax_i,
                                  fname='density_normalized.png',
                                  loc='best',
                                  bbox_to_anchor=(0.9, 0.5))

    def normaliz_density(self,
                         dens_dict: dict[str, pd.DataFrame]
                         ) -> dict[str, pd.DataFrame]:
        """normalize the density"""
        for key in self.configs.plot_list:
            dens_data: pd.DataFrame = dens_dict[f'dens_{key}']
            max_value: float = dens_data.iloc[:, 1].max()
            dens_data.iloc[:, 1] = dens_data.iloc[:, 1] / max_value
            dens_dict[f'dens_{key}'] = dens_data

        return dens_dict

    def plot_density(self,
                     dens_dict: dict[str, pd.DataFrame],
                     grids: bool = True
                     ) -> plt.axes:
        """plot the density of the residues"""
        first_key: str = next(iter(self.dens_data))
        x_range: tuple[float, float] = (
            dens_dict[first_key]['Coordinate_nm'].iat[0],
            dens_dict[first_key]['Coordinate_nm'].iat[-1])
        ax_i: plt.axes
        _, ax_i = plot_tools.mk_canvas(
            x_range,
            height_ratio=self.configs.height_ratio,
            num_xticks=7)
        for key in self.configs.plot_list:
            dens_data: pd.DataFrame = dens_dict[f'dens_{key}']
            ax_i = self.plot_single_density(ax_i, dens_data, key)
        ax_i.set_xlabel(self.configs.labels['xlabel'], fontsize=18)
        ax_i.set_ylabel(self.configs.labels['ylabel'], fontsize=18)
        ax_i.legend(loc=self.configs.legend_loc)
        if grids:
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
                  linestyle=self.configs.line_styles[column_name],
                  lw=2)
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
