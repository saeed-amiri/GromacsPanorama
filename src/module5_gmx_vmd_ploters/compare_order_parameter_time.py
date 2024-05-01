"""
Comparing the order parameter of the surfactant molecules in the system
over time in the systems with and without the presence of the surfactant
inputs:
    order_parameter.xvg from:
      module7_analysis_brushes/trajectory_brushes_analysis.py
outputs:
    order_parameter_comparison image
May 1 2024
Saeed
"""

import typing
from dataclasses import dataclass, field

import pandas as pd
import matplotlib.pyplot as plt

from common import logger, elsevier_plot_tools, xvg_to_dataframe
from common.colors_text import TextColor as bcolors


@dataclass
class BasePlotConfig:
    """
    Base class for the graph configuration
    """
    # pylint: disable=too-many-instance-attributes
    linewidth: float = elsevier_plot_tools.LINE_WIDTH

    linecolors: list[str] = field(init=False)
    line_styles: list[str] = field(init=False)
    output_file: str = \
        f'order_parameter_comparison.{elsevier_plot_tools.IMG_FORMAT}'

    xlabel: str = 'Time [ns]'
    ylabel: str = r'$S_z$'

    legend_loc: str = 'upper right'

    show_grid: bool = False
    show_multi_label: bool = True
    multi_label: typing.Union[str, None] = 'a)'
    show_nr_oda_label: bool = True

    def __post_init__(self) -> None:
        """Post init function"""
        self.linecolors = elsevier_plot_tools.LINE_COLORS
        self.line_styles = elsevier_plot_tools.LINE_STYLES


@dataclass
class DataConfig:
    """set the name of the files and labels"""
    xvg_files: dict[str, str] = field(default_factory=lambda: {
        'order_parameter_50Oda.xvg': 'with NP',
        'order_parameter_50Oda_no_np.xvg': 'without NP',
    })


@dataclass
class AllConfig(BasePlotConfig, DataConfig):
    """All configurations"""


class OrderParameterComparison:
    """compare the order parameter of the surfactant molecules in the system
    over time in the systems with and without the presence of the surfactant
    """

    info_msg: str = 'Message from OrderParameterComparison:\n'
    configs: AllConfig

    def __init__(self,
                 log: logger.logging.Logger,
                 configs: AllConfig = AllConfig()
                 ) -> None:
        self.configs = configs
        data: dict[str, pd.DataFrame] = self.initiate_data(log)
        self.plot_data(data)
        self.write_msg(log)

    def initiate_data(self,
                      log: logger.logging.Logger
                      ) -> dict[str, pd.DataFrame]:
        """read the data files"""
        data = {}
        for file_name, label in self.configs.xvg_files.items():
            data[label] = xvg_to_dataframe.XvgParser(file_name, log).xvg_df
            self.info_msg += f'\t`{file_name}` is read as `{label}`\n'
        return data

    def plot_data(self,
                  data: dict[str, pd.DataFrame]
                  ) -> None:
        """plot the data"""
        fig_i: plt.Figure
        ax_i: plt.Axes
        fig_i, ax_i = self._make_canvas()
        self._plot_data_on_ax(ax_i, data)
        self._set_labels(ax_i)
        self._set_multi_label(ax_i)
        self._add_nr_oda_label(ax_i)
        self._save_fig(fig_i)

    def _make_canvas(self) -> tuple[plt.Figure, plt.Axes]:
        fig_i: plt.Figure
        ax_i: plt.Axes
        fig_i, ax_i = elsevier_plot_tools.mk_canvas(size_type='single_column')
        return fig_i, ax_i

    def _plot_data_on_ax(self,
                         ax_i: plt.Axes,
                         data: dict[str, pd.DataFrame]
                         ) -> None:
        for i, (label, df_i) in enumerate(data.items()):
            ax_i.plot(df_i['Frame_index'] / 10,  # convert to ns
                      df_i['order_parameter'],
                      label=label,
                      linestyle=self.configs.line_styles[i],
                      color=self.configs.linecolors[i],
                      linewidth=self.configs.linewidth)

    def _set_labels(self,
                    ax_i: plt.Axes
                    ) -> None:
        ax_i.set_xlabel(self.configs.xlabel)
        ax_i.set_ylabel(self.configs.ylabel)

    def _set_multi_label(self,
                         ax_i: plt.Axes
                         ) -> None:
        if self.configs.show_multi_label:
            ax_i.text(-0.085,
                      1,
                      'a)',
                      ha='right',
                      va='top',
                      transform=ax_i.transAxes,
                      fontsize=elsevier_plot_tools.LABEL_FONT_SIZE_PT)

    def _add_nr_oda_label(self,
                          ax_i: plt.Axes
                          ) -> None:
        if self.configs.show_nr_oda_label:
            ax_i.text(0.4,
                      0.98,
                      r'Nr. Oda: 0.11 [1/nm$^2$]',
                      ha='right',
                      va='top',
                      transform=ax_i.transAxes,
                      fontsize=elsevier_plot_tools.FONT_SIZE_PT)

    def _save_fig(self,
                  fig_i: plt.Figure
                  ) -> None:
        fname: str = self.configs.output_file
        elsevier_plot_tools.save_close_fig(fig=fig_i,
                                           fname=fname)
        self.info_msg += f'\tThe figure is saved as `{fname}`\n'

    def write_msg(self,
                  log: logger.logging.Logger
                  ) -> None:
        """Write and log messages."""
        print(f'{bcolors.OKCYAN}{OrderParameterComparison.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    OrderParameterComparison(
        log=logger.setup_logger('order_parameter_comparison.log'))
