"""
A script to plot all the statistical analysis results for 2D RDF data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import logger
from common import elsevier_plot_tools

from module10_rdf_analysis.config import StatisticsConfig


class PlotStatistics:
    """plots the statistics"""
    # pylint: disable=too-many-arguments

    info_msg: str = "Message from PlotStatistics:\n"

    def __init__(self,
                 ydata: pd.DataFrame,
                 log: logger.logging.Logger,
                 config: StatisticsConfig,
                 err_plot: bool = False,
                 paper_plot: bool = False
                 ) -> None:
        self.ydata = ydata
        if err_plot:
            self.plot_statistics_with_error(log, config.plot_config)
        elif paper_plot:
            self.plot_statistics_with_error_paper(
                log, config.plot_config)
        else:
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
        ax_i.set_xlabel(config.xlabel,
                        fontsize=elsevier_plot_tools.LABEL_FONT_SIZE_PT)
        ax_i.set_ylabel(config.ylabel,
                        fontsize=elsevier_plot_tools.LABEL_FONT_SIZE_PT)
        ax_i.legend(loc=config.legend_loc)
        ax_i.set_xlim(config.xlim)
        ax_i.set_ylim(config.ylim)
        self.add_text(ax_i, config)
        self.add_grids(ax_i, config)
        elsevier_plot_tools.save_close_fig(fig_i,
                                           config.savefig,
                                           loc=config.legend_loc,
                                           show_legend=config.legend)
        log.info(f"{self.info_msg}")
        log.info(f"Statistics plot saved as {config.savefig}\n")

    def plot_statistics_with_error_paper(self,
                                         log: logger.logging.Logger,
                                         config: StatisticsConfig
                                         ) -> None:
        """
        Plot the statistics for the paper
        """
        figure: tuple[plt.Figure, plt.Axes] = elsevier_plot_tools.mk_canvas(
            'single_column', aspect_ratio=1)
        fig_i, ax_i = figure
        xdata = [item/21.7**2 for item in self.ydata.index]

        for idx, col in enumerate(self.ydata.columns[::2]):
            if col not in config.col_to_plot:
                continue
            ax_i.errorbar(xdata,
                          self.ydata[col],
                          yerr=self.ydata[self.ydata.columns[1::2][idx]],
                          label=config.labels[col],
                          color=config.colors[idx],
                          linestyle=config.linestyles[idx],
                          marker=config.markers[idx],
                          markersize=config.markersizes[idx],
                          )
        ax_i.set_xlabel(config.xlabel,
                        fontsize=elsevier_plot_tools.FONT_SIZE_PT)
        ax_i.set_ylabel(config.ylabel,
                        fontsize=elsevier_plot_tools.FONT_SIZE_PT)
        ax_i.legend(loc=config.legend_loc)
        ax_i.set_xlim(config.xlim)
        ax_i.set_ylim(config.ylim)
        self.add_text(ax_i, config)
        self.add_grids(ax_i, config)
        self.add_paper_label(ax_i, config)
        if 'yticklabels' in config:
            yticks_labels = [int(item) for item in config.yticklabels]
            ax_i.set_yticks(yticks_labels)
            ax_i.set_yticklabels(yticks_labels)

        # set the ticks fontsize
        ax_i.tick_params(axis='both',
                         which='major',
                         labelsize=elsevier_plot_tools.FONT_SIZE_PT)
        elsevier_plot_tools.save_close_fig(fig_i,
                                           config.savefig,
                                           loc=config.legend_loc,
                                           show_legend=config.legend)
        log.info(f"{self.info_msg}")
        log.info(f"Statistics plot saved as {config.savefig}\n")

    def plot_statistics_with_error(self,
                                   log: logger.logging.Logger,
                                   config: StatisticsConfig
                                   ) -> None:
        """
        Plot the statistics with error bars
        """
        if 'y_err' not in config:

            self.info_msg += ("\nNo error data found in the ydata, returning"
                              " plot without error bars")
            self.plot_statistics(log, config)
        figure: tuple[plt.Figure, plt.Axes] = elsevier_plot_tools.mk_canvas(
            'single_column', aspect_ratio=1)
        fig_i, ax_i = figure
        x_data = [item/21.7**2 for item in self.ydata.index]
        y_data = self.ydata[config.y_data]
        y_err = self.ydata[config.y_err]
        ax_i.errorbar(x_data,
                      y_data,
                      yerr=y_err,
                      label=config.label,
                      color=config.colors[0],
                      linestyle=config.linestyles[0],
                      marker=config.markers[0],
                      markersize=config.markersizes[0],
                      )
        ax_i.set_xlabel(config.xlabel)
        ax_i.set_ylabel(config.ylabel)

        if 'yticklabels' in config:
            yticks_labels = [int(item) for item in config.yticklabels]
            ax_i.set_yticks(yticks_labels)
            ax_i.set_yticklabels(yticks_labels)

        # set the ticks fontsize
        ax_i.tick_params(axis='both',
                         which='major',
                         labelsize=elsevier_plot_tools.FONT_SIZE_PT)
        ax_i.legend(loc=config.legend_loc)
        ax_i.set_xlim(config.xlim)
        ax_i.set_ylim(config.ylim)
        self.add_text(ax_i, config)
        self.add_grids(ax_i, config)
        self.add_paper_label(ax_i, config)
        elsevier_plot_tools.save_close_fig(fig_i,
                                           config.savefig,
                                           loc=config.legend_loc,
                                           show_legend=config.legend)
        log.info(f"{self.info_msg}")
        log.info(f"Statistics err plot saved as {config.savefig}\n")

    def add_text(self,
                 ax_i: plt.Axes,
                 config: StatisticsConfig,
                 text_x: float = 0.5,
                 text_y: float = 0.85,
                 ) -> None:
        """
        Add text to the plot
        """
        if 'text' not in config:
            return
        ax_i.text(text_x,
                  text_y,
                  config.text,
                  fontsize=elsevier_plot_tools.LABEL_FONT_SIZE_PT,
                  transform=ax_i.transAxes,
                  )

    def add_paper_label(self,
                        ax_i: plt.Axes,
                        config: StatisticsConfig,
                        text_x: float = -0.11,
                        text_y: float = 0.92,
                        ) -> None:
        """
        Add text to the plot
        """
        if 'add_paper_label' not in config or not config.add_paper_label:
            return
        ax_i.text(text_x,
                  text_y,
                  config.paper_label,
                  fontsize=elsevier_plot_tools.LABEL_FONT_SIZE_PT,
                  transform=ax_i.transAxes,
                  )

    def add_grids(self,
                  ax_i: plt.Axes,
                  config: StatisticsConfig,
                  ) -> None:
        """
        Add grids to the plot
        """
        if 'grid' not in config or not config.grid:
            return
        ax_i.grid(True, which='both', linestyle='--', linewidth=0.5)
        if 'microgrid' not in config or not config.microgrid:
            return
        ax_i.minorticks_on()
        ax_i.grid(
            True, which='minor', axis='both', linestyle=':', linewidth=0.5)
