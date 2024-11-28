"""
Plot the 2d rdf for the different ODA concentrations.
the needed files are density xvg files:
X_oda_densities.xvg
in which X is the nominal number of ODA at the interface
"""

import string
from dataclasses import field
from dataclasses import dataclass

import numpy as np
import pandas as pd

import matplotlib as mp
from matplotlib import patches
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
    fit_data: str = 'fitted_rdf'
    normalize_data: bool = True


@dataclass
class PlotConfoig:
    """Set the plot configurations"""
    # pylint: disable=too-many-instance-attributes
    nr_rows: int = 3
    nr_columns: int = 3
    xlabel: str = 'r$^\star$ [nm]'
    ylabel: str = 'g$^\star$(r$^\star$), a.u.'
    legend_loc: str = 'upper right'
    legend_title: str = 'ODA/nm$^2$'
    ylims: tuple[float, float] = (-0.05, 1.1)
    xlim: list[float] = field(default_factory=lambda: [0, 12])
    ylim: list[float] = field(default_factory=lambda: [0, 1.05])
    colors: list[str] = field(
        default_factory=lambda: elsevier_plot_tools.CLEAR_COLOR_GRADIENT)
    linestyle: list[str] = field(
        default_factory=lambda: [item[1] for item in
                                 elsevier_plot_tools.LINESTYLE_TUPLE][::-1])
    markers: list[str] = field(
        default_factory=lambda: elsevier_plot_tools.MARKER_STYLES)
    add_indices_label: bool = True


class Plot2dRdf:
    """read data and plot the 2d rdf"""

    __slots__ = ['info_msg',
                 'config',
                 'data',
                 'fit_data',
                 'plot_config',
                 ]

    info_msg: str
    config: FileConfig
    data: pd.DataFrame
    fit_data: pd.DataFrame
    plot_config: PlotConfoig

    def __init__(self,
                 log: logger.logging.Logger,
                 config: FileConfig = FileConfig(),
                 plot_config: PlotConfoig = PlotConfoig()
                 ) -> None:
        self.info_msg = 'Message from Plot2dRdf:\n'
        self.config = config
        self.plot_config = plot_config
        self.read_data(log)
        self.data = self.normalize_data(self.data, self.config.normalize_data)
        self.fit_data = self.normalize_data(self.fit_data,
                                            self.config.normalize_data)
        self.plot_data()

    def read_data(self,
                  log: logger.logging.Logger
                  ) -> None:
        """read the data"""
        data: dict[str, pd.Series] = {}
        fit_data: dict[str, pd.Series] = {}
        for i, file_info in enumerate(self.config.density_files):
            fname = file_info['fname']
            nr_oda = file_info['nr_oda']
            df_i: pd.DataFrame = \
                xvg_to_dataframe.XvgParser(fname, log, x_type=float).xvg_df
            if i == 0:
                data['regions'] = df_i[self.config.x_data] * 0.1  # nm
                fit_data['regions'] = df_i[self.config.x_data] * 0.1  # nm
            data[str(nr_oda)] = df_i[self.config.y_data]
            fit_data[str(nr_oda)] = df_i[self.config.fit_data]

        self.data = pd.concat(data, axis=1)
        self.fit_data = pd.concat(fit_data, axis=1)

    @staticmethod
    def normalize_data(data: pd.DataFrame,
                       normalize: bool
                       ) -> pd.DataFrame:
        """normalize the data"""
        if not normalize:
            return data
        for i, (oda, rdf) in enumerate(data.items()):
            if i == 0:
                continue
            # Calculate the mean of the top 10 maximum points
            top_10_mean = rdf.nlargest(2).mean()
            rdf_norm = rdf / top_10_mean
            data[oda] = rdf_norm
        return data

    def plot_data(self) -> None:
        """plot the data"""
        fig_i: plt.Figure
        axes: np.ndarray
        oda: str
        fig_i, axes = self._make_axis()
        last_ind: int = len(self.data.columns)
        for i, (oda, rdf) in enumerate(self.data.items()):
            if i == 0:
                x_data: pd.Series = self.data['regions']
            else:
                self._plot_axis(axes[i-1],
                                x_data,
                                y_data=rdf,
                                fit_data=self.fit_data[oda],
                                color=self.plot_config.colors[i-1],
                                line_style=self.plot_config.linestyle[i-1],
                                marker=self.plot_config.markers[i-1],
                                )
                oda_per_nm2: float = float(oda) / (21.7**2)
                self._add_label(axes[i-1], f'~{oda_per_nm2:.2f} ODA/nm$^2$')

        self._plot_all_rdf(self.data, axes[last_ind - 1], x_data, 'rdf')
        self._add_label(axes[last_ind - 1], r'All g$^*$(r$^*$)')
        axes[last_ind - 1].set_ylim(self.plot_config.ylims)

        self._plot_all_rdf(self.fit_data, axes[last_ind], x_data)
        self._add_label(axes[last_ind], r'Fitted g$^*$(r$^*$)')
        axes[last_ind].set_ylim(self.plot_config.ylims)

        self._set_or_remove_ticks(axes)

        self._add_axis_labels(axes)

        self._add_grid(axes)

        self._set_rectangel_box(fig_i)

        self._add_indices_label(axes)

        self._save_figure(fig_i)

    def _make_axis(self) -> tuple[plt.Figure, np.ndarray]:
        """make the axis"""
        return elsevier_plot_tools.mk_canvas_multi(
            'double_column',
            n_rows=self.plot_config.nr_columns,
            n_cols=self.plot_config.nr_rows,
            aspect_ratio=1,
            )

    def _plot_axis(self,
                   ax_i: mp.axes._axes.Axes,
                   x_data: pd.Series,
                   y_data: pd.Series,
                   fit_data: pd.Series,
                   color: str,
                   line_style: str,
                   marker: str,
                   ) -> None:
        """plot the axis"""
        # pylint: disable=too-many-arguments
        ax_i.plot(x_data,
                  y_data,
                  marker=marker,
                  markersize=1,
                  ls='',
                  lw=1,
                  color=color,
                  label=r'g$^*$(r$^*$)',
                  )
        ax_i.plot(x_data,
                  fit_data,
                  marker='',
                  markersize=0,
                  ls=line_style,
                  lw=1,
                  color=color,
                  label='Fitted',
                  )
        ax_i.legend(fontsize=elsevier_plot_tools.LABEL_FONT_SIZE_PT-3,
                    loc='upper left')
        ax_i.set_ylim(self.plot_config.ylims)

    def _plot_all_rdf(self,
                      data: pd.DataFrame,
                      ax_i: mp.axes._axes.Axes,
                      x_data: pd.Series,
                      style: str = 'fitted_rdf',
                      ) -> None:
        """plot the fitted rdf"""
        for i, (_, rdf) in enumerate(data.items()):
            if i == 0:
                continue
            if style == 'fitted_rdf':
                marker = ''
                linestyle = self.plot_config.linestyle[i-1]
                lw: float = 1.0
            else:
                marker = self.plot_config.markers[i-1]
                linestyle = '--'
                lw = 0.5
            ax_i.plot(x_data,
                      rdf,
                      markersize=0.5,
                      color=self.plot_config.colors[i-1],
                      linestyle=linestyle,
                      lw=lw,
                      marker=marker,
                      )
        ax_i.set_yticklabels([])

    def _set_or_remove_ticks(self,
                             axes: np.ndarray  # of plt.Axes
                             ) -> None:
        """set or remove the ticks"""
        for ind, ax_i in enumerate(axes):
            # Remove y-ticks for axes not in the first column
            if ind % self.plot_config.nr_rows != 0:
                ax_i.set_yticklabels([])
            # Remove x-ticks for axes not in the third row
            if ind < (self.plot_config.nr_columns - 1) * \
               self.plot_config.nr_rows \
               or ind == 0:
                ax_i.set_xticklabels([])

    def _add_label(self,
                   ax_i: mp.axes._axes.Axes,
                   label: str,
                   ) -> None:
        """add the legend"""
        ax_i.text(1.0,
                  .02,
                  label,
                  ha='right',
                  va='bottom',
                  fontsize=elsevier_plot_tools.LABEL_FONT_SIZE_PT - 2,
                  transform=ax_i.transAxes,
                  )

    def _add_grid(self,
                  axes: np.ndarray  # of plt.Axes
                  ) -> None:
        """add the grid"""
        for ax_i in axes:
            ax_i.grid(True, which='both', linestyle='--', lw=0.5)

    def _add_axis_labels(self,
                         axes: np.ndarray  # of plt.Axes
                         ) -> None:
        """add the axis labels"""
        for ind, ax_i in enumerate(axes):
            if ind % self.plot_config.nr_rows == 0:
                ax_i.set_ylabel(
                    self.plot_config.ylabel,
                    fontsize=elsevier_plot_tools.LABEL_FONT_SIZE_PT)
            if ind >= (self.plot_config.nr_columns - 1) * \
                self.plot_config.nr_rows:
                ax_i.set_xlabel(
                    self.plot_config.xlabel,
                    fontsize=elsevier_plot_tools.LABEL_FONT_SIZE_PT)

    def _save_figure(self,
                     fig_i: plt.Figure,
                     ) -> None:
        """save the figure"""
        elsevier_plot_tools.save_close_fig(
            fig_i, 'multi_2d_rdf.jpg', show_legend=True, loc='upper left')

    def _set_rectangel_box(self,
                           fig_i: plt.Figure,
                           ) -> None:
        """set the rectangle box"""
        rect = patches.Rectangle((0.05, -0.005),
                                 0.86, 0.9,
                                 transform=fig_i.transFigure,
                                 linewidth=2,
                                 edgecolor='black',
                                 facecolor='none')
        fig_i.patches.append(rect)

    def _add_indices_label(self,
                           axes: np.ndarray  # of plt.Axes
                           ) -> None:
        """add the indices label"""
        if not self.plot_config.add_indices_label:
            return
        for ind, ax_i in enumerate(axes):
            label = string.ascii_lowercase[ind % 26]
            ax_i.text(0.935, 0.40,
                      f'({label})',
                      fontsize=elsevier_plot_tools.LABEL_FONT_SIZE_PT,
                      transform=ax_i.transAxes,
                      horizontalalignment='center',
                      verticalalignment='center',
                      )

    def write_msg(self,
                  log: logger.logging.Logger
                  ) -> None:
        """Write and log messages."""
        print(f'{bcolors.OKCYAN}{Plot2dRdf.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    Plot2dRdf(log=logger.setup_logger('multi_2d_rdf_plotter.log'))
