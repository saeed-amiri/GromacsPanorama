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
                                                         n_cols=3)
        self._plot_toroidal_radius(grouped_over_surfactant,
                                   axs[1],
                                   'surfactant_concentration_mM_L')
        self._plot_toroidal_radius(grouped_over_salt,
                                   axs[0],
                                   'salt_concentration_mM_L')
        self._save_fig(fig_i)

    def _plot_toroidal_radius(self,
                              grouped:
                              "pd.core.groupby.generic.DataFrameGroupBy",
                              ax_i: plt.Axes,
                              x_column: str
                              ) -> None:
        """Plot the toroidal radius"""
        for name, group in grouped:
            ax_i.plot(group[x_column],
                      group['toroidal_radius_nm'],
                      label=f'{name} mM/L',
                      linestyle=self.config.line_style[3],
                      marker=self.config.marker_shape[2],
                      ms=self.config.marker_size,
                      lw=self.config.line_width / 2,
                      color=self.config.colors[0])

    def _save_fig(self,
                  fig_i: plt.Figure
                  ) -> None:
        """Save the figure"""
        elsevier_plot_tools.save_close_fig(fig_i,
                                           self.config.fout,
                                           loc='lower right',
                                           show_legend=False)
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
