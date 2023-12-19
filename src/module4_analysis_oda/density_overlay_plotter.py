"""
plot the Oda and Cl denisty on the top of each other in different styles
styles:
    - Normal
    - Normalized to one
    - Separate Y-axis
For average and rdf
The data will be read from data files
"""

import typing
from collections import namedtuple
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pylab as plt

from common import logger
from common import plot_tools
from common import xvg_to_dataframe as xvg
from common.colors_text import TextColor as bcolors


FitTurns: "namedtuple" = \
    namedtuple('FitTurns', ['first_turn', 'midpoint', 'second_turn'])


@dataclass
class BaseDataConfig:
    """basic configurations in reading the data"""
    xvg_column: dict[str, str] = field(default_factory=lambda: {
        'regions': 'regions',
        'avg_density': 'avg_density_per_region',
        'smooth': 'smoothed_rdf',
        'rdf': 'rdf_2d'
        })


@dataclass
class DataConfig(BaseDataConfig):
    """set data config"""
    selected_columns: list[str] = field(default_factory=lambda: [
        'rdf'
        ])


@dataclass
class BasePlotConfig:
    """
    Basic configurations on the plots
    """
    height_ratio: float = (5**0.5-1)*1
    graphs_sets: dict[str, typing.Any] = field(default_factory=lambda: {
        'colors': ['k', 'r', 'b', 'g']
    })
    vlines_sets: dict[str, typing.Any] = field(default_factory=lambda: {
        'colors': ['g', 'b', 'r'],
        'linestyles': (':', '--', ':')
    })
    add_vlines: bool = True


@dataclass
class PlotConfig(BasePlotConfig):
    """
    set configuration for plot
    """


class OverlayPlotDensities:
    """plot the overlay of densities on one plot"""

    info_msg: str = 'Messege from OverlayPlotDensities:\n'

    file_names: dict[str, str]
    data_config: "DataConfig"
    plot_config: "PlotConfig"
    fit_turns: "FitTurns"

    x_data: np.ndarray
    xvg_dfs: dict[str, np.ndarray]

    def __init__(self,
                 file_names: dict[str, str],  # Filenames and their lables
                 fit_turns: "FitTurns",  # Fit parameters for vlines
                 log: logger.logging.Logger,
                 data_config: "DataConfig" = DataConfig(),
                 plot_config: "PlotConfig" = PlotConfig()
                 ) -> None:
        self.fit_turns = fit_turns
        self.file_names = file_names
        self.data_config = data_config
        self.plot_config = plot_config
        self.xvg_dfs, self.x_data = self.initiate_data(log)
        self.initiate_plots()
        self._write_msg(log)

    def initiate_data(self,
                      log: logger.logging.Logger
                      ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """initiate reading data files"""
        x_data: np.ndarray
        xvg_dfs: dict[str, np.ndarray] = {}

        for i, f_xvg in enumerate(self.file_names.keys()):
            fanme: str = f_xvg.split('.')[0]
            df_column: str = self.data_config.xvg_column[
                self.data_config.selected_columns[0]]
            df_i = xvg.XvgParser(f_xvg, log).xvg_df
            xvg_dfs[fanme] = df_i[df_column].to_numpy()
            if i == 0:
                x_data = \
                    df_i[self.data_config.xvg_column['regions']].to_numpy()
        self.info_msg += (f'\tThe file names are:\n\t\t`{self.file_names}`\n'
                          '\tThe selected columns are:\n'
                          f'\t\t`{self.data_config.selected_columns}`\n')
        return xvg_dfs, x_data

    def initiate_plots(self) -> None:
        """plots the densities in different styles"""
        self._plot_save_normalized_plot()
        self._plot_save_double_yaxis_plot()

    def _plot_save_normalized_plot(self) -> None:
        """plot and save the normalized densities in one grpah"""
        x_range: tuple[float, float] = (min(self.x_data), max(self.x_data))
        fig_i, ax_i = plot_tools.mk_canvas(
            x_range, height_ratio=self.plot_config.height_ratio)
        for i, (residue, density) in enumerate(self.xvg_dfs.items()):
            ax_i.plot(self.x_data,
                      density/max(density),
                      ls='-',
                      marker='o',
                      markersize=4,
                      color=self.plot_config.graphs_sets['colors'][i],
                      label=self.file_names[f'{residue}.xvg'])
        ax_i.grid(True, linestyle='--', color='gray', alpha=0.5)
        ax_i.set_xlabel('Distance from Nanoparticle [A]')
        ax_i.set_ylabel('g(r) (normalized)')
        ax_i.set_title('Each rdf normalized to its max')
        if self.plot_config.add_vlines:
            ax_i = self.add_vlines(ax_i)
        plot_tools.save_close_fig(
            fig_i, ax_i, fname := 'normalized_rdf.png', loc='lower right')
        self.info_msg += f'\tThe figure is saved as {fname}\n'

    def _plot_save_double_yaxis_plot(self) -> None:
        """plot the rdf for both oda and cl with seperate axis"""
        if len(self.xvg_dfs) < 2:
            raise ValueError(
                "Insufficient datasets to plot with double y-axis.")
        x_range = (min(self.x_data), max(self.x_data))
        fig, ax1 = plot_tools.mk_canvas(
            x_range, height_ratio=self.plot_config.height_ratio)

        files: list[str] = list(self.xvg_dfs.keys())
        dataset1, dataset2 = files[0], files[1]

        color1 = self.plot_config.graphs_sets['colors'][0]
        label1: str = self.file_names[f'{dataset1}.xvg']
        ax1.plot(self.x_data,
                 self.xvg_dfs[dataset1],
                 ls='-',
                 marker='o',
                 markersize=4,
                 color=color1)
        ax1.set_xlabel('Distance from Nanoparticle [A]')
        ax1.set_ylabel(rf'$g_{{{label1}}}(r)$', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, linestyle='--', color='gray', alpha=0.5)

        # Create a second y-axis
        ax2 = ax1.twinx()
        color2: str = self.plot_config.graphs_sets['colors'][1]
        label2: str = self.file_names[f'{dataset2}.xvg']
        ax2.plot(self.x_data,
                 self.xvg_dfs[dataset2],
                 ls='-',
                 marker='o',
                 markersize=4,
                 color=color2)
        ax2.set_ylabel(rf'$g_{{{label2}}}(r)$', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)

        fig.tight_layout()  # Adjust layout

        if self.plot_config.add_vlines:
            ax1 = self.add_vlines(ax1)

        fname = 'separate_y_axes_plot.png'
        plot_tools.save_close_fig(fig, ax1, fname, loc='center right')
        self.info_msg += \
            f'\tThe figure with separate y-axes is saved as {fname}\n'

    def add_vlines(self,
                   ax_i: plt.axes
                   ) -> plt.axes:
        """add the fitted vlines"""
        ylims: tuple[float, float] = ax_i.get_ylim()
        for i, (turn, x_loc) in enumerate(self.fit_turns._asdict().items()):
            if turn == 'first_turn':
                label_i = '1st'
            elif turn == 'midpoint':
                label_i = 'c'
            elif turn == 'second_turn':
                label_i = '2nd'
            ax_i.vlines(x=x_loc,
                        ymin=ylims[0],
                        ymax=ylims[1],
                        ls=self.plot_config.vlines_sets['linestyles'][i],
                        color=self.plot_config.vlines_sets['colors'][i],
                        label=f'{label_i}={x_loc:.1f}')
        ax_i.set_ylim(ylims)
        return ax_i

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    print("This module must be plot from inside of module: "
          "trajectory_oda_analysis.py\n")
