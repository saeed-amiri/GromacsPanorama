"""
Plot the Contact angle and the coverage of the nanoparticles
Particle coverage and contact angle at the interface as obtained from
synchrotron x-ray reflectometry measurements.
There are 4 different salt concentrations and 4 different surfactant
concentrations.

"""

import pandas as pd

import matplotlib.pyplot as plt

from module12_experimental_lab_data.config_classes import AllConfig
from common.colors_text import TextColor as bcolors
from common import logger, elsevier_plot_tools


class PlotCaCoverage:
    """plot the several plots here"""

    info_msg: str = 'Message from PlotCaCoverage:\n'
    data: pd.DataFrame

    def __init__(self,
                 log: logger.logging.Logger,
                 data: pd.DataFrame,
                 config: AllConfig = AllConfig()
                 ) -> None:
        self.config = config.ca_cov
        self.plot_data(data)
        self.write_msg(log)

    def plot_data(self,
                  data: pd.DataFrame
                  ) -> None:
        """plot the data first by orderring the data"""
        grouped: "pd.core.groupby.generic.DataFrameGroupBy" = \
            self._split_data(data)
        self._plot_data(grouped)

    def _plot_data(self,
                   grouped: "pd.core.groupby.generic.DataFrameGroupBy"
                   ) -> None:
        """Plot the data"""
        fig_i: plt.Figure
        axs: plt.Axes
        fig_i, axs = elsevier_plot_tools.mk_canvas_multi('double_column',
                                                         n_rows=1,
                                                         n_cols=4)
        for i, (name, group) in enumerate(grouped):
            ax_i = axs[i]
            self._select_plot_type(ax_i, group)
            self._set_x_ticks(ax_i, group)
            self._set_y_ticks(ax_i)
            self._set_axis_labels(ax_i, i)
            self._ax_add_fig_labels(ax_i, i, name)
            self._ax_add_fig_legend(ax_i, i)
            self._mirror_axes(ax_i)
        self._save_fig(fig_i)

    def _select_plot_type(self,
                          ax_i: plt.Axes,
                          group: pd.DataFrame
                          ) -> None:
        """Select the plot type"""
        if self.config.show_guid_lines:
            self._ax_guid_lines(ax_i, group)
        else:
            self._ax_scatter(ax_i, group)

    def _ax_scatter(self,
                    ax_i: plt.Axes,
                    group: pd.DataFrame
                    ) -> None:
        """Scatter plot"""
        ax_i.scatter(group['log_surfactant_concentration_mM_L'],
                   group['contact_angle_deg'],
                   marker=self.config.marker_shape[2],
                   s=self.config.marker_size,
                   color=self.config.colors[0],
                   )
        ax_i.scatter(group['log_surfactant_concentration_mM_L'],
                   group['particle_coverage'],
                   marker=self.config.marker_shape[3],
                   s=self.config.marker_size,
                   color=self.config.colors[1],
                   )

    def _ax_guid_lines(self,
                       ax_i: plt.Axes,
                       group: pd.DataFrame
                       ) -> None:
        """Scatter plot"""
        ax_i.plot(group['log_surfactant_concentration_mM_L'],
                group['contact_angle_deg'],
                linestyle=self.config.line_style[3],
                marker=self.config.marker_shape[2],
                ms=self.config.marker_size,
                lw=self.config.line_width / 2,
                color=self.config.colors[0],
                label='CA [deg]'
                )
        ax_i.plot(group['log_surfactant_concentration_mM_L'],
                group['particle_coverage'],
                linestyle=self.config.line_style[3],
                lw=self.config.line_width / 2,
                marker=self.config.marker_shape[3],
                ms=self.config.marker_size,
                color=self.config.colors[1],
                label='Coverage [%]'
                )

    def _set_x_ticks(self,
                     ax_i: plt.Axes,
                     group: pd.DataFrame
                     ) -> None:
        """Set the xticks"""
        x_values = group['log_surfactant_concentration_mM_L']
        x_labels = group['surfactant_concentration_mM_L']
        ax_i.set_xticks(x_values)
        ax_i.set_xticklabels(x_labels)

    def _set_y_ticks(self,
                     ax_i: plt.Axes
                     ) -> None:
        """Set the yticks"""
        ax_i.set_ylim(self.config.y_lims)
        ax_i.set_yticks(self.config.ytick_labels)

    def _set_axis_labels(self,
                         ax_i: plt.Axes,
                         ax_index: int
                         ) -> None:
        """Set the axis labels"""
        ax_i.set_xlabel(self.config.x_label,
                      fontsize=elsevier_plot_tools.FONT_SIZE_PT)
        if ax_index == 0 and self.config.show_y_label:
            ax_i.set_ylabel(self.config.y_label,
                          fontsize=elsevier_plot_tools.FONT_SIZE_PT)

    def _ax_add_fig_labels(self,
                           ax_i: plt.Axes,
                           ax_index: int,
                           name: str
                           ) -> None:
        """Add figure labels"""
        alphabet = chr(97 + ax_index)  # 97 is the Unicode code point for 'a'
        ax_i.text(0.05,
                0.98,
                f'{alphabet}) {name} [mM]',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax_i.transAxes,
                fontsize=elsevier_plot_tools.LABEL_FONT_SIZE_PT-2,
                bbox={"facecolor": 'white', "alpha": 0.}
                )

    def _ax_add_fig_legend(self,
                           ax_i: plt.Axes,
                           ax_index: int
                           ) -> None:
        """Add a legend to the figure"""
        if ax_index == 0:
            ax_i.legend(loc='upper right',
                      fontsize=elsevier_plot_tools.FONT_SIZE_PT,
                      frameon=True,
                      bbox_to_anchor=(0.9, .90),
                      ncol=1
                      )

    def _mirror_axes(self,
                     ax_i: plt.Axes
                     ) -> None:
        """keep or remove the mirror the axes"""
        if not self.config.show_mirror_axis:
            elsevier_plot_tools.remove_mirror_axes(ax_i)

    def _split_data(self,
                    data: pd.DataFrame
                    ) -> "pd.core.groupby.generic.DataFrameGroupBy":
        """Split the data by the salt concentration"""
        group_col: str = 'salt_concentration_mM_L'
        self.info_msg += f'\tData is grouped by `{group_col}`\n'
        return data.groupby(group_col)

    def _save_fig(self,
                  fig_i: plt.Figure
                  ) -> None:
        """Save the figure"""
        elsevier_plot_tools.save_close_fig(fig_i,
                                           self.config.fout,
                                           loc='lower right',
                                           show_legend=False)
        self.info_msg += f'\tA figure saved as `{self.config.fout}`\n'

    def write_msg(self,
                  log: logger.logging.Logger
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{PlotCaCoverage.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)
