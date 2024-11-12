"""
Plot the 2d rdf for the different ODA concentrations.
the needed files are density xvg files:
X_oda_densities.xvg
in which X is the nominal number of ODA at the interface
"""

from dataclasses import field
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import logger
from common import xvg_to_dataframe
from common import elsevier_plot_tools
from common.colors_text import TextColor as bcolors


@dataclass
class FileConfig:
    """
    set the names of the files and the layers to get average from.
    defines dict of info for files.
    """
    density_files: list[dict[str, str | int]] = \
        field(default_factory=lambda: [
            {'fname': '5_oda_densities.xvg',
             'nr_oda': 5,
             },
            {'fname': '10_oda_densities.xvg',
             'nr_oda': 10,
             },
            {'fname': '15_oda_densities.xvg',
             'nr_oda': 15,
             },
            {'fname': '20_oda_densities.xvg',
             'nr_oda': 20,
             },
            {'fname': '30_oda_densities.xvg',
             'nr_oda': 30,
             },
            {'fname': '40_oda_densities.xvg',
             'nr_oda': 40,
             },
            {'fname': '50_oda_densities.xvg',
             'nr_oda': 50,
             },
            ])
    x_data: str = 'regions'
    y_data: str = 'rdf_2d'
    normalize_data: bool = True
    norm_range: tuple[float, float] = (-10, -1)


@dataclass
class PlotCOnfoig:
    """Set the plot configurations"""
    # pylint: disable=too-many-instance-attributes
    title: str = '2D RDF for different ODA concentrations'
    xlabel: str = 'r [nm]'
    ylabel: str = 'g(r), a.u.'
    xlim: list[float] = field(default_factory=lambda: [0, 12])
    ylim: list[float] = field(default_factory=lambda: [0, 1.05])
    legend_loc: str = 'upper right'
    legend_title: str = 'ODA/nm$^2$'
    nr_columns: int = 3
    nr_rows: int = 3
    ylims: tuple[float, float] = (-0.05, 1.1)


class Plot2dRdf:
    """read data and plot the 2d rdf"""

    __slots__ = ['info_msg', 'config', 'data', 'plot_config']

    info_msg: str
    config: FileConfig
    data: pd.DataFrame
    plot_config: PlotCOnfoig

    def __init__(self,
                 log: logger.logging.Logger,
                 config: FileConfig = FileConfig(),
                 plot_config: PlotCOnfoig = PlotCOnfoig()
                 ) -> None:
        self.info_msg = 'Message from Plot2dRdf:\n'
        self.config = config
        self.plot_config = plot_config
        self.read_data(log)
        self.normalize_data()
        self.plot_data(log)

    def read_data(self,
                  log: logger.logging.Logger
                  ) -> None:
        """read the data"""
        data: dict[str, pd.DataFrame] = {}
        for i, file_info in enumerate(self.config.density_files):
            fname = file_info['fname']
            nr_oda = file_info['nr_oda']
            df_i: pd.DataFrame = \
                xvg_to_dataframe.XvgParser(fname, log, x_type=float).xvg_df
            if i == 0:
                data['regions'] = df_i[self.config.x_data]
            data[str(nr_oda)] = df_i[self.config.y_data]

        self.data = pd.concat(data, axis=1)

    def normalize_data(self) -> None:
        """normalize the data"""
        if not self.config.normalize_data:
            return
        for i, (oda, rdf) in enumerate(self.data.items()):
            if i == 0:
                continue
            mean_value: float = \
                rdf[self.config.norm_range[0]:self.config.norm_range[1]].mean()
            rdf_norm = rdf / mean_value
            self.data[oda] = rdf_norm

    def plot_data(self,
                  log: logger.logging.Logger
                  ) -> None:
        """plot the data"""
        fig_i, ax_i = self._make_axis()
        for i, (oda, rdf) in enumerate(self.data.items()):
            if i == 0:
                x_data: pd.DataFrame = self.data['regions']
            else:
                self._plot_axis(ax_i[i-1], x_data, y_data=rdf)
                self._set_or_remove_ticks(i, ax_i)
                self._add_legend(ax_i[i-1], oda)
        self._save_figure(fig_i)

    def _make_axis(self) -> tuple[plt.Figure, plt.Axes]:
        """make the axis"""
        return elsevier_plot_tools.mk_canvas_multi(
            'double_height',
            n_rows=self.plot_config.nr_columns,
            n_cols=self.plot_config.nr_rows,
            aspect_ratio=2,
            )

    def _plot_axis(self,
                   ax_i: plt.Axes,
                   x_data: pd.DataFrame,
                   y_data: pd.DataFrame,
                   ) -> None:
        """plot the axis"""
        ax_i.plot(x_data,
                  y_data,
                  marker='o',
                  markersize=0.5,
                  ls='',
                  lw=0.5,
                  color='k',
                  )
        ax_i.set_ylim(self.plot_config.ylims)

    def _set_or_remove_ticks(self,
                             ind: int,
                             ax_i: plt.Axes,
                             ) -> None:
        """set or remove the ticks"""
        # Remove y-ticks for axes not in the first column
        if ind % self.plot_config.nr_rows != 0:
            ax_i[ind].set_yticks([])
        # Remove x-ticks for axes not in the third row
        if ind < (self.plot_config.nr_columns - 1) * self.plot_config.nr_rows:
            ax_i[ind].set_xticks([])

    def _add_legend(self,
                    ax_i: plt.Axes,
                    oda: str,
                    ) -> None:
        """add the legend"""
        ax_i.text(1.0,
                  .02,
                  f'{oda} ODA/nm$^2$',
                  ha='right',
                  va='bottom',
                  fontsize=5,
                  transform=ax_i.transAxes,
                  )

    def _save_figure(self,
                     fig_i: plt.Figure,
                     ) -> None:
        """save the figure"""
        elsevier_plot_tools.save_close_fig(
            fig_i, 'multi_2d_rdf.jpg', show_legend=False)


if __name__ == '__main__':
    Plot2dRdf(log=logger.setup_logger('multi_2d_rdf_plotter.log'))
