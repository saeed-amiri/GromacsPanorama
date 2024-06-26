"""
Plotting Order Parameters

This script plots the computed order parameters in a system with and
without nanoparticles (NP). It supports a variety of plots:

    1. Simple plot of the order parameter.
    2. Plot of the order parameter with error bars.
    3. Comparative plot of order parameters for systems both with and
        without NP, if data for both are available.
    4. Column plot showing the nominal number of Oda molecules with
        the actual number at the interface.

Data Format:
The data should be in a columnar format with the following columns:
    - 'name': Identifier of the data point (e.g., '0Oda', '5Oda').
    - 'nominal_oda': Nominal number of Oda molecules at the interface.
    - 'actual_oda': Average number of Oda molecules at the interface.
    - 'order_parameter': Computed order parameter.
    - 'std': Standard deviation of the order parameter.

Note: The inclusion of an error bar in the plot is optional.
Opt. by ChatGpt
Saeed
18 Jan 2024
"""


import typing
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import logger, my_tools, elsevier_plot_tools
from common.colors_text import TextColor as bcolors


@dataclass
class BaseConfig:
    """
    Basic configurations and setup for the plots.
    """
    # pylint: disable=too-many-instance-attributes
    graph_suffix: str = 'order_parameter.png'
    ycol_name: str = 'order_parameter'
    xcol_name: str = 'nominal_oda'

    labels: dict[str, str] = field(default_factory=lambda: {
        'title': 'Computed order parameter',
        'ylabel': r'$S_z$',
        'xlabel': 'Nr. Oda'
    })

    graph_styles: dict[str, typing.Any] = field(default_factory=lambda: {
        'label': 'without NP',
        'color': 'black',
        'marker': 'o',
        'linestyle': '--',
        'linewidth': 1,
        'markersize': 2,
    })

    line_styles: list[str] = \
        field(default_factory=lambda: ['-', ':', '--', '-.'])
    colors: list[str] = \
        field(default_factory=lambda: ['black', 'darkred', 'blue', 'green'])

    y_unit: str = ''

    log_x_axis: bool = False
    with_error: bool = False

    legend_loc: str = 'lower right'
    show_title: bool = False


@dataclass
class SimpleGraph(BaseConfig):
    """
    Parameters for simple data plots.
    """


@dataclass
class ErrorBarGraph(BaseConfig):
    """
    Parameters for simple data with error bar
    """
    graph_suffix: str = 'order_parameter_err.png'
    xcol_name: str = 'actual_per_area'
    log_x_axis: bool = True
    with_error: bool = True

    def __post_init__(self) -> None:
        self.graph_styles['ecolor'] = 'darkred'
        self.graph_styles['linestyle'] = self.line_styles[1]
        self.labels['xlabel'] = r'ODA/nm$^2$'


@dataclass
class DoubleDataGraph(BaseConfig):
    """
    Parameters for the plotting both the data with and without NP
    """
    graph_suffix: str = 'order_parameter_both.png'
    xcol_name: str = 'nominal_per_area'
    log_x_axis: bool = True


@dataclass
class BarPlotConfig(BaseConfig):
    """
    Parameters for the number of oda at the interface
    """
    graph_suffix: str = 'bar_oda_nr.png'
    log_x_axis: bool = False
    xcol_name: str = 'nominal_per_area'
    bar_label: bool = False
    legend_loc: str = 'upper left'
    bar_width: float = 0.35

    def __post_init__(self) -> None:
        self.labels['title'] = 'Nr. Oda at the interface'
        self.labels['xlabel'] = ''
        self.labels['ylabel'] = 'Nr. Oda'


@dataclass
class FileConfig:
    """
    Set the name of the input files for plot with labels say what are
    those
    """
    fnames: dict[str, str] = field(default_factory=lambda: {
        'no_np': 'order_parameter'
        })


@dataclass
class ParameterConfig:
    """
    Parameters for the plots
    """
    box_dimension: tuple[float, float] = (21.7, 21.7)  # xy in nm@dataclass


@dataclass
class AllConfig(FileConfig, ParameterConfig):
    """
    Consolidates all configurations for different graph types.
    """
    simple_config: SimpleGraph = field(default_factory=SimpleGraph)
    errbar_config: ErrorBarGraph = field(default_factory=ErrorBarGraph)
    double_config: DoubleDataGraph = field(default_factory=DoubleDataGraph)
    bar_config: BarPlotConfig = field(default_factory=BarPlotConfig)


class PlotOrderParameter:
    """
    Plot all the graphs for order parameter analysis.
    """

    info_msg: str = 'Message from PlotOrderParameter:\n'

    def __init__(self,
                 log: logger.logging.Logger,
                 configs: AllConfig = AllConfig()
                 ) -> None:
        self.configs = configs
        order_param_dict = self.initiate_data(log)
        self.initiate_plots(order_param_dict)
        self.write_msg(log)

    def initiate_data(self,
                      log: logger.logging.Logger
                      ) -> dict[str, pd.DataFrame]:
        """Read files and return the data in dataframes format."""
        order_param_dict: dict[str, pd.DataFrame] = {}
        for key, fname in self.configs.fnames.items():
            my_tools.check_file_exist(fname, log)
            data: pd.DataFrame = self._read_file(fname)
            order_param_dict[key] = self._convert_nr_to_ratio(data)
        return order_param_dict

    def _read_file(self,
                   fname: str
                   ) -> pd.DataFrame:
        """Read data file and return as a dataframe."""
        columns: list[str] = \
            ['name', 'nominal_oda', 'actual_oda', 'order_parameter', 'std']
        df_in = pd.read_csv(fname, delim_whitespace=True, names=columns)
        return df_in

    def initiate_plots(self,
                       order_param_dict: dict[str, pd.DataFrame]
                       ) -> None:
        """Create the plots based on the provided data."""
        nr_files: int = len(order_param_dict)

        for key, data in order_param_dict.items():
            # Plot simple graph
            self._plot_graph(key, data, self.configs.simple_config)
            # Plot with error bars if 'std' column is present
            if 'std' in data.columns:
                self._plot_graph(key, data, self.configs.errbar_config)
            self.plot_bar_graph(key, data, self.configs.bar_config)

        if nr_files > 1:
            # To plot the multi data graphes
            pass

    def _plot_graph(self,
                    key: str,
                    data: pd.DataFrame,
                    config: BaseConfig
                    ) -> None:
        """Plot the order parameter data."""
        fig_i: plt.figure
        ax_i: plt.axes
        fig_i, ax_i = elsevier_plot_tools.mk_canvas(size_type='single_column')

        if config.log_x_axis:
            ax_i.set_xscale('log')

        if config.with_error:
            ax_i.errorbar(data[config.xcol_name],
                          data[config.ycol_name],
                          yerr=data['std'],
                          **config.graph_styles)
            ax_i.text(-0.085,
                      1,
                      'b)',
                      ha='right',
                      va='top',
                      transform=ax_i.transAxes,
                      fontsize=elsevier_plot_tools.LABEL_FONT_SIZE_PT)
        else:
            ax_i.plot(data[config.xcol_name],
                      data[config.ycol_name],
                      **config.graph_styles)

        ax_i.set_xlabel(config.labels['xlabel'],
                        fontsize=elsevier_plot_tools.FONT_SIZE_PT)
        ax_i.set_ylabel(f'{config.labels["ylabel"]} {config.y_unit}',
                        fontsize=elsevier_plot_tools.FONT_SIZE_PT)
        if config.show_title:
            ax_i.set_title(f'{config.labels["title"]} ({key})')
        ax_i.grid(True, which='both', linestyle='--', color='gray', alpha=0.5)
        ax_i.set_yticks([0.1, 0.3, 0.5])

        elsevier_plot_tools.save_close_fig(
            fig_i, fname := f'{key}_{config.graph_suffix}', loc='lower right')
        self.info_msg += f'\tThe plot for `{key}` is saved as `{fname}`\n'

    def _convert_nr_to_ratio(self,
                             order_parameter: pd.DataFrame
                             ) -> pd.DataFrame:
        """convert the order_parameter and subtract the amount at zero"""
        area: float = \
            self.configs.box_dimension[0] * self.configs.box_dimension[1]
        order_parameter['nominal_per_area'] = \
            order_parameter['nominal_oda'] / area
        order_parameter['actual_per_area'] = \
            order_parameter['actual_oda'] / area
        return order_parameter

    def plot_bar_graph(self,
                       key: str,
                       data: pd.DataFrame,
                       config: BarPlotConfig
                       ) -> None:
        """Plot a bar graph for nominal and actual Oda numbers."""
        # Set the positions of the bars on the x-axis
        x_data = np.arange(len(data))

        fig_i: plt.figure
        ax_i: plt.axes

        fig_i, ax_i = \
            elsevier_plot_tools.mk_canvas(size_type='single_column')

        if config.log_x_axis:
            ax_i.set_xscale('log')

        bars1 = ax_i.bar(x_data - config.bar_width/2,
                         data['nominal_oda'],
                         config.bar_width,
                         label='Nominal Oda',
                         zorder=3)
        bars2 = ax_i.bar(x_data + config.bar_width/2,
                         data['actual_oda'],
                         config.bar_width,
                         label='Actual Oda',
                         zorder=3)

        ax_i.set_xlabel(config.labels['xlabel'],
                        fontsize=elsevier_plot_tools.FONT_SIZE_PT)
        ax_i.set_ylabel(config.labels['ylabel'],
                        fontsize=elsevier_plot_tools.FONT_SIZE_PT)
        ax_i.set_title(config.labels['title'])

        ax_i.set_xticks(x_data)
        ax_i.set_xticklabels(['']*len(data))

        ax_i.grid(True, 'both', ls='--', color='gray', alpha=0.5, zorder=2)

        # Optional: Add value labels on top of bars
        if config.bar_label:
            self.add_value_labels(ax_i, bars1)
            self.add_value_labels(ax_i, bars2)

        elsevier_plot_tools.save_close_fig(
            fig_i, fname := config.graph_suffix, loc=config.legend_loc)
        self.info_msg += f'\tThe plot for `{key}` is saved as `{fname}`\n'

    @staticmethod
    def add_value_labels(ax_i: plt.axes,
                         bars,
                         rotation: float = 90.0
                         ) -> None:
        """
        Attach a text label above each bar displaying its height, with
        an option to rotate the label."""
        for bar_i in bars:
            height = bar_i.get_height()
            ax_i.annotate(f'{height:.1f}',
                          xy=(bar_i.get_x() + bar_i.get_width() / 2, height),
                          xytext=(0, 3),  # 3 points vertical offset
                          textcoords="offset points",
                          ha='center',
                          va='bottom',
                          rotation=rotation,
                          fontsize=elsevier_plot_tools.FONT_SIZE_PT)

    def write_msg(self,
                  log: logger.logging.Logger
                  ) -> None:
        """Write and log messages."""
        print(f'{bcolors.OKCYAN}{PlotOrderParameter.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    PlotOrderParameter(log=logger.setup_logger('plot_order_parameter.log'))
