"""
Plot Tension from Simulation Data

This script plots the computed tension from simulation data.
It includes several plot types:

    1. Separate plots for 'no NP' and 'singles NP' data.
    2. A combined plot of both 'no NP' and 'singles NP'.
    3. Plots in both normal and logarithmic scales.
    4. Plots with error bars, based on bootstrap data if available.

The data is expected in a simple columnar format, generated by a bash
script. The columns are as follows:

    Name: Identifier of the data point (e.g., '0Oda', '5Oda').
    nr. Oda: Numerical value associated with the 'Name'.
    tension: Computed tension value.
    errorbar: Error value, if available (used for plotting error bars).

Notes:
    - The '0Oda' value is crucial as it represents the baseline for
        plotting differences.
    - The presence of an 'errorbar' value determines whether error bars
        are included in the plot.
Opt. by ChatGpt
Saeed
17 Jan 2023
"""


import typing
from dataclasses import dataclass, field

import pandas as pd
import matplotlib.pyplot as plt

from common import logger
from common import elsevier_plot_tools
from common import my_tools
from common.colors_text import TextColor as bcolors


@dataclass
class BaseConfig:
    """
    Basic configurations and setup for the plots.
    """
    # pylint: disable=too-many-instance-attributes
    graph_suffix: str = 'tension.png'

    labels: dict[str, str] = field(default_factory=lambda: {
        'title': 'Computed Tension',
        'ylabel': r'$\Delta\gamma$',
        'xlabel': 'Nr. Oda'
    })

    graph_styles: dict[str, typing.Any] = field(default_factory=lambda: {
        'label': r'$\gamma$ [tot. Oda]',
        'color': 'black',
        'marker': 'o',
        'linestyle': '--',
        'linewidth': elsevier_plot_tools.LINE_WIDTH,
        'markersize': elsevier_plot_tools.MARKER_SIZE,
    })

    graph_styles2: dict[str, typing.Any] = field(default_factory=lambda: {
        'label': r'$\gamma$ [surf. Oda]',
        'color': 'r',
        'marker': 'o',
        'linestyle': '--',
        'linewidth': elsevier_plot_tools.LINE_WIDTH,
        'markersize': elsevier_plot_tools.MARKER_SIZE,
    })

    line_styles: list[str] = \
        field(default_factory=lambda: ['-', ':', '--', '-.'])
    colors: list[str] = \
        field(default_factory=lambda: ['black', 'red', 'blue', 'green'])

    y_unit: str = r'[mN/nm$^2$]'

    log_x_axis: bool = False
    show_title: bool = False


@dataclass
class SimpleGraph(BaseConfig):
    """
    Parameters for simple data plots.
    """
    graph_suffix: str = 'converted_tension.png'


@dataclass
class RawGraph(BaseConfig):
    """
    Parameters for simple data plots.
    """
    graph_suffix: str = 'raw_tension.png'
    labels: dict[str, str] = field(default_factory=lambda: {
        'title': 'Computed Tension',
        'ylabel': r'$\gamma$',
        'xlabel': 'Nr. Oda'
    })
    y_unit: str = '[bar/nm]'


@dataclass
class LogGraph(BaseConfig):
    """
    Parameters for the semilog plots.
    """
    graph_suffix: str = 'log_xscale.png'
    labels: dict[str, str] = field(default_factory=lambda: {
        'title': 'Computed Tension',
        'ylabel': r'$\Delta\gamma$',
        'xlabel': r'ODA/$nm^2$'
    })
    log_x_axis: bool = True


@dataclass
class DoubleDataLog(LogGraph):
    """plot both data"""
    graph_suffix: str = 'log_xscale_both.png'
    graph_styles: dict[str, typing.Any] = field(default_factory=lambda: {
        'label': r'$\Delta\gamma$ (nominal)',
        'color': 'black',
        'marker': 'o',
        'linestyle': '--',
        'linewidth': elsevier_plot_tools.LINE_WIDTH,
        'markersize': elsevier_plot_tools.MARKER_SIZE,
    })
    label_b: str = r'$\Delta\gamma_{np}$'


@dataclass
class PreprintDataLog(LogGraph):
    """plot both data"""
    graph_suffix: str = 'log_xscale_both.png'
    graph_styles: dict[str, typing.Any] = field(default_factory=lambda: {
        'label': 'without NP',
        'color': 'black',
        'marker': 'o',
        'linestyle': '--',
        'linewidth': elsevier_plot_tools.LINE_WIDTH,
        'markersize': elsevier_plot_tools.MARKER_SIZE,
    })
    label_b: str = r'$\Delta\gamma_{np}$'


@dataclass
class ErrorBarGraph(BaseConfig):
    """
    Parameters for plots with error bars.
    """
    plot_errorbars: bool = True


@dataclass
class FileConfig:
    """
    Set the name of the input files for plot with labels say what are
    those
    """
    fnames: dict[str, str] = field(default_factory=lambda: {
        'no_np': 'tension',
        })


@dataclass
class ParameterConfig:
    """
    Parameters for the plots and conversions
    """
    tension_conversion: float = 20.0  # Convert bar/nm to mN/nm^2
    box_dimension: tuple[float, float] = (21.7, 21.7)  # xy in nm


@dataclass
class AllConfig(FileConfig, ParameterConfig):
    """
    Consolidates all configurations for different graph types.
    """
    # pylint: disable=too-many-instance-attributes
    simple_config: SimpleGraph = field(default_factory=SimpleGraph)
    raw_config: RawGraph = field(default_factory=RawGraph)
    log_config: LogGraph = field(default_factory=LogGraph)
    errbar_config: ErrorBarGraph = field(default_factory=ErrorBarGraph)
    double_config: DoubleDataLog = field(default_factory=DoubleDataLog)
    preprint_config: PreprintDataLog = field(default_factory=PreprintDataLog)
    if_publish: bool = False
    if_label: bool = False


class PlotTension:
    """
    Plot all the graphes
    """

    info_msg: str = 'Message from PlotTension:\n'
    configs: "AllConfig"

    def __init__(self,
                 log: logger.logging.Logger,
                 configs: "AllConfig" = AllConfig()
                 ) -> None:
        self.configs = configs
        tension_dict: dict[str, pd.DataFrame] = self._initiate_data(log)
        self.initiate_plots(tension_dict)
        self.write_msg(log)

    def _initiate_data(self,
                       log: logger.logging.Logger
                       ) -> pd.DataFrame:
        """read files and return the data in dataframes format"""
        tension_dict: dict[str, pd.DataFrame] = {}
        for key, fname in self.configs.fnames.items():
            my_tools.check_file_exist(fname, log)
            tension_dict[key] = self.read_file(fname)
        return tension_dict

    def read_file(self,
                  fname: str
                  ) -> pd.DataFrame:
        """read data file and return as a dataframe"""
        columns: list[str] = \
            ['Name', 'nr.Oda', 'surf.Oda', 'tension', 'tension_with_np']
        df_in: pd.DataFrame = \
            pd.read_csv(fname, delim_whitespace=True, names=columns)
        return df_in

    def initiate_plots(self,
                       tension_dict: dict[str, pd.DataFrame]
                       ) -> None:
        """plots the graphes"""
        nr_files: int = len(tension_dict)
        converted_dict: dict[str, pd.DataFrame] = {}
        for key, tension in tension_dict.items():
            self.plot_graph(key, tension, self.configs.raw_config)
            converted_tension: pd.DataFrame = self.convert_tension(tension)
            converted_dict[key] = converted_tension
            self.plot_graph(key,
                            converted_tension,
                            self.configs.simple_config,
                            ycol_name='converted_tension',
                            xcol_name='nr.Oda')
            self.plot_graph(key,
                            converted_tension,
                            self.configs.log_config,
                            ycol_name='converted_tension',
                            xcol_name='oda_per_area')
        if nr_files > 1:
            self.plot_all_tensions_log(converted_dict)
        self._surf_actula_log(converted_dict)
        self._surf_preprint_log(converted_dict)

    def _surf_actula_log(self,
                         converted_dict: dict[str, pd.DataFrame]
                         ) -> None:
        """plot all the input in a same graph"""
        returned_fig = self.plot_graph('np_np',
                                       converted_dict['no_np'],
                                       self.configs.double_config,
                                       ycol_name='converted_tension',
                                       xcol_name='oda_per_area',
                                       return_ax=True,
                                       add_key_to_title=False)
        if returned_fig is not None:
            fig_i, ax_i = returned_fig
            ax_i.plot(converted_dict['no_np']['surf_oda_per_area'],
                      converted_dict['no_np']['converted_tension'],
                      c='#ff7f0e',
                      marker='o',
                      linestyle='--',
                      markersize=elsevier_plot_tools.MARKER_SIZE,
                      label=r'$\Delta\gamma$ (actual)')
            elsevier_plot_tools.save_close_fig(
                fig_i, fname := 'actual.png', loc='lower left')
            self.info_msg += \
                f'\tThe raw tension plot for both data is saved as `{fname}`\n'

    def _surf_preprint_log(self,
                           converted_dict: dict[str, pd.DataFrame]
                           ) -> None:
        """plot all the input in a same graph"""
        returned_fig = self.plot_graph('no_np',
                                       converted_dict['no_np'],
                                       self.configs.preprint_config,
                                       ycol_name='converted_tension',
                                       xcol_name='surf_oda_per_area',
                                       return_ax=True,
                                       add_key_to_title=False)
        if returned_fig is not None:
            fig_i, ax_i = returned_fig
            ax_i.plot(converted_dict['no_np']['surf_oda_per_area'],
                      converted_dict['no_np']['converted_tension_with_np'],
                      c='#ff7f0e',
                      marker='o',
                      linestyle='--',
                      markersize=elsevier_plot_tools.MARKER_SIZE,
                      linewidth=elsevier_plot_tools.LINE_WIDTH,
                      label='wiht NP')
            if self.configs.if_label:
                ax_i.text(-0.11,
                          1,
                          'a)',
                          ha='right',
                          va='top',
                          transform=ax_i.transAxes,
                          fontsize=elsevier_plot_tools.LABEL_FONT_SIZE_PT)
            fname: str = \
                f'interface_tension_log.{elsevier_plot_tools.IMG_FORMAT}'
            elsevier_plot_tools.save_close_fig(
                fig_i, fname=fname, loc='lower left')
            self.info_msg += \
                f'\tThe raw tension plot for both data is saved as `{fname}`\n'

    def plot_all_tensions_log(self,
                              converted_dict: dict[str, pd.DataFrame]
                              ) -> None:
        """plot all the input in a same graph"""
        double_config: DoubleDataLog = self.configs.double_config
        returned_fig = self.plot_graph('no_np',
                                       converted_dict['no_np'],
                                       self.configs.double_config,
                                       ycol_name='converted_tension',
                                       xcol_name='oda_per_area',
                                       return_ax=True,
                                       add_key_to_title=False)
        if returned_fig:
            fig_i, ax_i = returned_fig

            ax_i.plot(converted_dict['with_np']['oda_per_area'],
                      converted_dict['with_np']['converted_tension'],
                      c=double_config.colors[1],
                      ls=double_config.graph_styles['linestyle'],
                      ms=double_config.graph_styles['markersize'],
                      marker=double_config.graph_styles['marker'],
                      linewidth=double_config.graph_styles['linewidth'],
                      label=double_config.label_b
                      )
            elsevier_plot_tools.save_close_fig(fig_i, fname := 'double.png')
        self.info_msg += \
            f'\tThe raw tension plot for both data is saved as `{fname}`\n'

    def plot_graph(self,
                   key: str,
                   tension: pd.DataFrame,
                   configs: typing.Union[
                     SimpleGraph, RawGraph, LogGraph, DoubleDataLog],
                   ycol_name: str = 'tension',
                   xcol_name: str = 'nr.Oda',
                   return_ax: bool = False,
                   add_key_to_title: bool = True,
                   xcol_name2: typing.Union[str, None] = None
                   ) -> typing.Union[tuple[plt.figure, plt.axis], None]:
        """plot the raw data for later conviniance"""
        # pylint: disable=too-many-arguments

        fig_i: plt.figure
        ax_i: plt.axes
        fig_i, ax_i = elsevier_plot_tools.mk_canvas(size_type='single_column')

        if configs.log_x_axis:
            ax_i.set_xscale('log')

        ax_i.plot(
            tension[xcol_name], tension[ycol_name], **configs.graph_styles)

        if xcol_name2 is not None:
            ax_i.plot(tension[xcol_name2],
                      tension[ycol_name],
                      **configs.graph_styles2)

        ax_i.set_xlabel(configs.labels['xlabel'],
                        fontsize=elsevier_plot_tools.FONT_SIZE_PT)
        ax_i.set_ylabel(f'{configs.labels["ylabel"]} {configs.y_unit}',
                        fontsize=elsevier_plot_tools.FONT_SIZE_PT)
        if self.configs.if_publish:
            if add_key_to_title:
                ax_i.set_title(f'{configs.labels["title"]} ({key})')
            else:
                ax_i.set_title(f'{configs.labels["title"]}')
        ax_i.grid(True, which='both', linestyle='--', color='gray', alpha=0.5)

        if return_ax:
            return fig_i, ax_i

        elsevier_plot_tools.save_close_fig(
            fig_i, fname := f'{key}_{configs.graph_suffix}')

        self.info_msg += \
            f'\tThe raw tension plot for `{key}` is saved as `{fname}`\n'

        return None

    def convert_tension(self,
                        tension: pd.DataFrame
                        ) -> pd.DataFrame:
        """convert the tension and subtract the amount at zero"""
        tension['converted_tension_with_np'] = \
            tension['tension_with_np'] - tension['tension_with_np'][0]
        tension['converted_tension_with_np'] /= self.configs.tension_conversion

        tension['converted_tension'] = \
            tension['tension'] - tension['tension'][0]
        tension['converted_tension'] /= self.configs.tension_conversion

        tension['oda_per_area'] = tension['nr.Oda'] / \
            (self.configs.box_dimension[0] * self.configs.box_dimension[1])
        tension['surf_oda_per_area'] = tension['surf.Oda'] / \
            (self.configs.box_dimension[0] * self.configs.box_dimension[1])
        return tension

    def write_msg(self,
                  log: logger.logging.Logger  # To log
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{PlotTension.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    PlotTension(log=logger.setup_logger('plot_tension.log'))
