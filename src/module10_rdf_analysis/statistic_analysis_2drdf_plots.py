"""
A script to plot all the statistical analysis results for 2D RDF data.
"""

import pandas as pd
import matplotlib.pyplot as plt

from common import logger
from common import elsevier_plot_tools

from module10_rdf_analysis.config import StatisticsConfig


class PlotStatistics:
    """plots the statistics"""

    info_msg: str = "Message from PlotStatistics:\n"

    def __init__(self,
                 ydata: pd.DataFrame,
                 log: logger.logging.Logger,
                 config: StatisticsConfig
                 ) -> None:
        self.ydata = ydata
        self.plot_statistics(log, config.plot_config)

    def plot_statistics(self,
                        log: logger.logging.Logger,
                        config: StatisticsConfig
                        ) -> None:
        """
        Plot the statistics
        """
        figure: tuple[plt.Figure, plt.Axes] = elsevier_plot_tools.mk_canvas(
            'single_column', aspect_ratio=1)
        fig_i, ax_i = figure
        for idx, col in enumerate(self.ydata.columns):
            ax_i.plot(self.ydata.index,
                      self.ydata[col],
                      label=col.replace('_', ' '),
                      color=config.colors[idx],
                      linestyle=config.linestyles[idx],
                      marker=config.markers[idx],
                      markersize=config.markersizes[idx],
                      )
        ax_i.set_xlabel(config.xlabel)
        ax_i.set_ylabel(config.ylabel)
        ax_i.legend(loc=config.legend_loc)
        ax_i.set_xlim(config.xlim)
        ax_i.set_ylim(config.ylim)
        elsevier_plot_tools.save_close_fig(fig_i,
                                           config.savefig,
                                           loc=config.legend_loc,
                                           show_legend=config.legend)
        log.info(f"{self.info_msg}")
        log.info(f"Statistics plot saved as {config.savefig}\n")
