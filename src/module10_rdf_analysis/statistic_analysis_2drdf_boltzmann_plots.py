"""
Plot the 2d rdf with the Boltzmann distribution
Since the style is quite different than the other plots, it is
implemented in a separate class.
"""

import string
import numpy as np
import pandas as pd

import matplotlib as mp
import matplotlib.pyplot as plt
from matplotlib import patches

from common import logger
from common import elsevier_plot_tools

from module10_rdf_analysis.config import StatisticsConfig


class PlotRdfBoltzmann:
    """
    Plot the 2d rdf with the Boltzmann distribution
    """
    # pylint: disable=too-many-arguments

    __slots__ = ['config', 'info_msg', 'linestyles']

    def __init__(self,
                 rdf_x: pd.Series,
                 rdf_data: pd.DataFrame,
                 rdf_fit_data: pd.DataFrame,
                 boltzmann_data: pd.DataFrame,
                 vlines_data: pd.DataFrame,
                 log: logger.logging.Logger,
                 config: StatisticsConfig
                 ) -> None:
        # pylint: disable=unused-argument
        self.info_msg: str = "Message from PlotRdfBoltzmann:\n"
        self.config = config
        self.set_styles()
        self.plot_data(
            rdf_x, rdf_data, boltzmann_data, vlines_data)

    def set_styles(self) -> None:
        """set the styles
        cant set the linestyle to the confg since hydra is not able to
        handle the tuple of tuples
        """
        self.linestyles = [
            (offset, tuple(dashpattern)) if dashpattern else 'solid'
            for (_, (offset, dashpattern)) in
            elsevier_plot_tools.LINESTYLE_TUPLE][::-1]
        self.config.colors = elsevier_plot_tools.CLEAR_COLOR_GRADIENT
        self.config.markers = elsevier_plot_tools.MARKER_STYLES

    def plot_data(self,
                  rdf_x: pd.Series,
                  rdf_data: pd.DataFrame,
                  boltzmann_data: pd.DataFrame,
                  vlines_data: pd.DataFrame,
                  ) -> None:
        """
        Plot the 2d rdf with the Boltzmann distribution
        """
        # pylint: disable=too-many-locals

        fig_i: plt.Figure
        axes: np.ndarray
        fig_i, axes = self._make_axis()

        boltzmann_x: pd.Series = boltzmann_data['r_nm']

        last_ind: int = len(rdf_data.columns)

        for i, (oda, rdf) in enumerate(rdf_data.items()):
            bolzmann: pd.Series = boltzmann_data[int(oda)]
            self._plot_axis(axes[i],
                            rdf_x,
                            rdf=rdf,
                            boltzmann_x=boltzmann_x,
                            boltzmann=bolzmann,
                            color=self.config.colors[i],
                            linestyle=self.linestyles[i],
                            marker=self.config.markers[i],
                            )
            self.add_vlines(axes[i], vlines_data[oda])
            oda_per_nm2: float = float(oda) / (21.7**2)
            self._add_label(axes[i], f'~{oda_per_nm2:.2f} ODA/nm$^2$')

        self._plot_all_rdf(rdf_data, axes[last_ind], rdf_x, 'rdf')
        self._add_label(axes[last_ind], r'All g$^*$(r$^*$)')
        axes[last_ind].set_ylim(self.config.ylim)
        rdf_x_lims: tuple[float, float] = axes[last_ind].get_xlim()

        self._plot_all_rdf(
            boltzmann_data, axes[last_ind + 1], boltzmann_x, 'boltzmann')
        self._add_label(axes[last_ind + 1], r'All $\psi$(r$^*$)')
        axes[last_ind + 1].set_ylim(self.config.ylim)
        axes[last_ind + 1].set_xlim(rdf_x_lims)

        self._set_rectangel_box(fig_i)
        self._set_or_remove_ticks(axes)
        self._add_grid(axes)
        self._add_axis_labels(axes)
        self._add_indices_label(axes)
        self._save_figure(fig_i)

    def _make_axis(self) -> tuple[plt.Figure, np.ndarray]:
        """make the axis"""
        return elsevier_plot_tools.mk_canvas_multi(
            'double_column',
            n_rows=self.config.nr_rows,
            n_cols=self.config.nr_cols,
            aspect_ratio=1,
            )

    def _plot_axis(self,
                   ax_i: mp.axes._axes.Axes,
                   x_data: pd.Series,
                   rdf: pd.Series,
                   boltzmann_x: pd.Series,
                   boltzmann: pd.Series,
                   color: str,
                   linestyle: str,
                   marker: str,
                   ) -> None:
        """plot the axis"""
        # pylint: disable=too-many-arguments
        ax_i.plot(x_data,
                  rdf,
                  label=self.config.rdf_label,
                  ls='',
                  lw=1,
                  marker=marker,
                  markersize=1,
                  color=color,
                  )
        ax_i.plot(boltzmann_x,
                  boltzmann,
                  label=self.config.boltzmann_label,
                  ls=linestyle,
                  lw=1,
                  marker='',
                  markersize=0,
                  color=color,
                  )
        ax_i.legend(fontsize=elsevier_plot_tools.LABEL_FONT_SIZE_PT-3,
                    loc='upper left')
        ax_i.set_ylim(self.config.ylim)

    def add_vlines(self,
                   ax_i: mp.axes._axes.Axes,
                   vlines_data: pd.Series,
                   ) -> None:
        """add the vertical lines"""
        ax_i.axvline(vlines_data.iloc[0],
                     color='black',
                     linestyle='-',
                     lw=0.5,
                     label=self.config.contact_radii_label,
                     )
        ax_i.axvline(vlines_data.iloc[1],
                     color='black',
                     linestyle='--',
                     lw=0.5,
                     label=self.config.r_half_max_label,
                     )

    def _plot_all_rdf(self,
                      rdf_data: pd.DataFrame,
                      ax_i: mp.axes._axes.Axes,
                      x_data: pd.Series,
                      style: str = 'boltzmann',
                      ) -> None:
        """plot the fitted rdf"""
        for i, (_, rdf) in enumerate(rdf_data.items()):
            if style == 'boltzmann':
                if i == 0:
                    continue
                marker = ''
                linestyle = self.linestyles[i-1]
                color = self.config.colors[i-1]
                lw: float = 1.0
            else:
                marker = self.config.markers[i]
                linestyle = '--'
                lw = 0.5
                color = self.config.colors[i]

            ax_i.plot(x_data,
                      rdf,
                      markersize=0.5,
                      color=color,
                      linestyle=linestyle,
                      lw=lw,
                      marker=marker,
                      )
        ax_i.set_yticklabels([])

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

    def _set_or_remove_ticks(self,
                             axes: np.ndarray  # of plt.Axes
                             ) -> None:
        """set or remove the ticks"""
        for ind, ax_i in enumerate(axes):
            # Remove y-ticks for axes not in the first column
            if ind % self.config.nr_rows != 0:
                ax_i.set_yticklabels([])
            # Remove x-ticks for axes not in the third row
            if ind < (self.config.nr_cols - 1) * \
               self.config.nr_rows \
               or ind == 0:
                ax_i.set_xticklabels([])

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
            if ind % self.config.nr_rows == 0:
                ax_i.set_ylabel(
                    self.config.ylabel,
                    fontsize=elsevier_plot_tools.LABEL_FONT_SIZE_PT)
            if ind >= (self.config.nr_cols - 1) * self.config.nr_rows:
                ax_i.set_xlabel(
                    self.config.xlabel,
                    fontsize=elsevier_plot_tools.LABEL_FONT_SIZE_PT)

    def _add_indices_label(self,
                           axes: np.ndarray  # of plt.Axes
                           ) -> None:
        """add the indices label"""
        if not self.config.add_indices_label:
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

    def _set_rectangel_box(self,
                           fig_i: plt.Figure,
                           ) -> None:
        """set the rectangle box"""
        if 'add_rectangles' not in self.config or \
           not self.config.add_rectangles:
            return
        rect = patches.Rectangle((0.05, -0.005),
                                 0.86, 0.9,
                                 transform=fig_i.transFigure,
                                 linewidth=2,
                                 edgecolor='black',
                                 facecolor='none')
        fig_i.patches.append(rect)

    def _save_figure(self,
                     fig_i: plt.Figure,
                     ) -> None:
        """save the figure"""
        elsevier_plot_tools.save_close_fig(
            fig_i, self.config.savefig, show_legend=True, loc='upper left')
