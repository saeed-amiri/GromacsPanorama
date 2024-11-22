"""
Plot the 2d rdf with the Boltzmann distribution
Since the style is quite different than the other plots, it is
implemented in a separate class.
"""

import numpy as np
import pandas as pd

import matplotlib as mp
import matplotlib.pyplot as plt

from common import logger
from common import elsevier_plot_tools

from module10_rdf_analysis.config import StatisticsConfig


class PlotRdfBoltzmann:
    """
    Plot the 2d rdf with the Boltzmann distribution
    """
    # pylint: disable=too-many-arguments

    __slots__ = ['config', 'info_msg']

    def __init__(self,
                 rdf_x: pd.Series,
                 rdf_data: pd.DataFrame,
                 rdf_fit_data: pd.DataFrame,
                 boltzmann_data: pd.DataFrame,
                 vlines_data: pd.DataFrame,
                 log: logger.logging.Logger,
                 config: StatisticsConfig
                 ) -> None:
        self.info_msg: str = "Message from PlotRdfBoltzmann:\n"
        self.config = config
        self.plot_data(
            rdf_x, rdf_data, rdf_fit_data, boltzmann_data, vlines_data, log)

    def plot_data(self,
                  rdf_x: pd.Series,
                  rdf_data: pd.DataFrame,
                  rdf_fit_data: pd.DataFrame,
                  boltzmann_data: pd.DataFrame,
                  vlines_data: pd.DataFrame,
                  log: logger.logging.Logger
                  ) -> None:
        """
        Plot the 2d rdf with the Boltzmann distribution
        """
        # pylint: disable=too-many-locals
        fig_i: plt.Figure
        axes: np.ndarray
        oda: str

        fig_i, axes = self._make_axis()
        # last_ind: int = len(self.data.columns)
        boltzmann_x: pd.Series = boltzmann_data['r_nm']
        colors: list[str] = elsevier_plot_tools.CLEAR_COLOR_GRADIENT
        linestyle: list[str] = [item[1] for item in
                                elsevier_plot_tools.LINESTYLE_TUPLE][::-1]
        markers: list[str] = elsevier_plot_tools.MARKER_STYLES
        for i, (oda, rdf) in enumerate(rdf_data.items()):
            bolzmann: pd.Series = boltzmann_data[int(oda)]
            self._plot_axis(axes[i],
                            rdf_x,
                            rdf=rdf,
                            boltzmann_x=boltzmann_x,
                            boltzmann=bolzmann,
                            color=colors[i],
                            linestyle=linestyle[i],
                            marker=markers[i],
                            )
            self.add_vlines(axes[i], vlines_data[oda])
            oda_per_nm2: float = float(oda) / (21.7**2)
            self._add_label(axes[i], f'{oda_per_nm2: .2f} ODA/nm$^2$')
        self._set_or_remove_ticks(axes)
        self._add_grid(axes)
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

    def _save_figure(self,
                     fig_i: plt.Figure,
                     ) -> None:
        """save the figure"""
        elsevier_plot_tools.save_close_fig(
            fig_i, self.config.savefig, show_legend=True, loc='upper left')
