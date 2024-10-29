"""
Ploting the toroidal radius of the exlusion zone based on the
concertration of salt and surfactant.
The figure is two panels, one for the salt and the other for the surfactant
But, here we add an extra panel for the toroidal radius scheme.
The data is read from the 'data.xvg' file and plotted.
"""

import pandas as pd

import matplotlib.pyplot as plt

from module12_experimental_lab_data.config_classes import AllConfig
from common.colors_text import TextColor as bcolors
from common import logger, elsevier_plot_tools


class ToroidalRadiusPlot:
    """plot the toroidal radius"""

    info_msg: str = 'Message from ToroidalRadiusPlot:\n'
    data: pd.DataFrame

    def __init__(self,
                 log: logger.logging.Logger,
                 data: pd.DataFrame,
                 config: AllConfig = AllConfig()
                 ) -> None:
        self.config = config.toroidal
        self.plot_data(data)
        self.write_msg(log)

    def plot_data(self,
                  data: pd.DataFrame
                  ) -> None:
        """plot the data"""
        grouped_over_salt: "pd.core.groupby.generic.DataFrameGroupBy" = \
            self._split_data(data, 'salt_concentration_mM_L')
        grouped_over_surfactant: "pd.core.groupby.generic.DataFrameGroupBy" = \
            self._split_data(data, 'surfactant_concentration_mM_L')
        self._plot_data(grouped_over_salt, grouped_over_surfactant)

    def _plot_data(self,
                   grouped_over_salt:
                   "pd.core.groupby.generic.DataFrameGroupBy",
                   grouped_over_surfactant:
                   "pd.core.groupby.generic.DataFrameGroupBy"
                   ) -> None:
        """Plot the data"""
        fig_i: plt.Figure
        axs: plt.Axes
        fig_i, axs = elsevier_plot_tools.mk_canvas_multi('double_column',
                                                         n_rows=1,
                                                         n_cols=3,
                                                         aspect_ratio=2
                                                         )
        self._plot_panel_a(axs[0], grouped_over_salt)
        self._plot_panel_b(axs[1], grouped_over_surfactant)
        self._plot_panel_c(axs[2])

        self._save_fig(fig_i)

    def _plot_panel_a(self,
                      ax_i: plt.Axes,
                      grouped: "pd.core.groupby.generic.DataFrameGroupBy"
                      ) -> None:
        """Plot the toroidal radius for the different salt concentration"""
        self._plot_toroidal_radius(ax_i,
                                   grouped,
                                   'log_surfactant_concentration_mM_L')
        self._set_x_ticks(ax_i,
                          grouped,
                          'surfactant_concentration_mM_L')
        self._set_axis_labels(ax_i, 0, self.config.x_label_surfactant)
        self._set_y_ax(ax_i, self.config.y_lims)
        self._ax_add_fig_labels(ax_i, 0, r'c$_{NaCl}$ [mM/L]')
        self._mirror_axes(ax_i)

    def _plot_panel_b(self,
                      ax_i: plt.Axes,
                      grouped: "pd.core.groupby.generic.DataFrameGroupBy"
                      ) -> None:
        """Plot the toroidal radius for the different surfactant
        concentration"""
        self._plot_toroidal_radius(ax_i,
                                   grouped,
                                   'salt_concentration_mM_L')
        self._set_x_ticks(ax_i,
                          grouped,
                          'salt_concentration_mM_L')
        self._set_axis_labels(ax_i, 1, self.config.x_label_salt)
        self._set_y_ax(ax_i, self.config.y_lims)
        self._ax_add_fig_labels(ax_i, 1, r'c$_{ODA}$ [mM/L]')
        self._mirror_axes(ax_i)

    def _plot_panel_c(self,
                      ax_i: plt.Axes
                      ) -> None:
        """Plot the toroidal radius scheme"""
        self._ax_add_fig_labels(ax_i, 2, 'Exclusion zone')
        ax_i.imshow(plt.imread(self.config.schem_fig))
        self.info_msg += f'\tThe scheme figure is `{self.config.schem_fig}`\n'
        ax_i.axis('off')

    def _plot_toroidal_radius(self,
                              ax_i: plt.Axes,
                              grouped:
                              "pd.core.groupby.generic.DataFrameGroupBy",
                              x_column: str
                              ) -> None:
        """Plot the toroidal radius"""
        for i, (name, group) in enumerate(grouped):
            i_index: int = 0
            if x_column != 'salt_concentration_mM_L':
                i_index = 1
            elif i == 0:
                continue
            ax_i.plot(group[x_column][i_index:],
                      group['toroidal_radius_nm'][i_index:],
                      label=f'{name}',
                      linestyle=self.config.line_style[3],
                      marker=self.config.marker_shape[i],
                      ms=self.config.marker_size,
                      lw=self.config.line_width / 2,
                      color=self.config.colors[i])
            ax_i.legend(loc='upper right',
                        fontsize=elsevier_plot_tools.FONT_SIZE_PT)

    def _set_x_ticks(self,
                     ax_i: plt.Axes,
                     grouped: "pd.core.groupby.generic.DataFrameGroupBy",
                     x_column: str
                     ) -> None:
        """Set the x ticks"""
        _, first_group = next(iter(grouped))
        if x_column == 'salt_concentration_mM_L':
            xticks = first_group[x_column]
            xticks_labels = first_group[x_column]
        else:
            xticks = first_group[f'log_{x_column}'][1:]
            xticks_labels = first_group[x_column][1:]
        ax_i.set_xticks(xticks)
        ax_i.set_xticklabels(xticks_labels)

    def _set_axis_labels(self,
                         ax_i: plt.Axes,
                         ax_index: int,
                         x_label: str
                         ) -> None:
        """Set the axis labels"""
        if ax_index == 0:
            ax_i.set_ylabel(self.config.y_label)
        ax_i.set_xlabel(x_label)

    def _set_y_ax(self,
                  ax_i: plt.Axes,
                  y_lims: tuple[int, int]
                  ) -> None:
        """Set the y axis"""
        ax_i.set_ylim(y_lims)

    def _mirror_axes(self,
                     ax_i: plt.Axes
                     ) -> None:
        """Mirror the axes"""
        if not self.config.show_mirror_axis:
            ax_i.set_axis_on()
            ax_i.spines['top'].set_visible(False)
            ax_i.spines['right'].set_visible(False)
        else:
            ax_i.set_axis_on()

    def _ax_add_fig_labels(self,
                           ax_i: plt.Axes,
                           ax_index: int,
                           name: str
                           ) -> None:
        """Add figure labels"""
        alphabet = chr(97 + ax_index)  # 97 is the Unicode code point for 'a'
        x_pos = 0.05
        y_pos = 0.98
        if ax_index == 2:
            x_pos = -0.05
            y_pos = 1.0
        ax_i.text(x_pos,
                  y_pos,
                  f'{alphabet}) {name}',
                  horizontalalignment='left',
                  verticalalignment='top',
                  transform=ax_i.transAxes,
                  fontsize=elsevier_plot_tools.LABEL_FONT_SIZE_PT-2,
                  bbox={"facecolor": 'white', "alpha": 0.}
                  )

    def _save_fig(self,
                  fig_i: plt.Figure
                  ) -> None:
        """Save the figure"""
        elsevier_plot_tools.save_close_fig(fig_i,
                                           self.config.fout,
                                           loc='upper right',
                                           show_legend=True)
        self.info_msg += f'\tA figure saved as `{self.config.fout}`\n'

    def _split_data(self,
                    data: pd.DataFrame,
                    group_by: str
                    ) -> "pd.core.groupby.generic.DataFrameGroupBy":
        """Split the data based on the group_by"""
        self.info_msg += f'\tData is grouped by `{group_by}`\n'
        return data.groupby(group_by)

    def write_msg(self,
                  log: logger.logging.Logger
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{ToroidalRadiusPlot.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    print(f'\n{bcolors.WARNING}This script is not meant to be run '
          f'directly, it called by `plot_lab_data.py`{bcolors.ENDC}\n')
