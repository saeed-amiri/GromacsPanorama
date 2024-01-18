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
18 Jan 2023
"""


import typing
from dataclasses import dataclass, field

import pandas as pd
import matplotlib.pyplot as plt

from common import logger, plot_tools, my_tools
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
        'ylabel': 'S',
        'xlabel': 'Nr. Oda'
    })

    graph_styles: dict[str, typing.Any] = field(default_factory=lambda: {
        'label': 'Order Parameter',
        'color': 'black',
        'marker': 'o',
        'linestyle': '--',
        'markersize': 5,
    })

    line_styles: list[str] = \
        field(default_factory=lambda: ['-', ':', '--', '-.'])
    colors: list[str] = \
        field(default_factory=lambda: ['black', 'red', 'blue', 'green'])

    height_ratio: float = (5 ** 0.5 - 1) * 2

    y_unit: str = ''

    log_x_axis: bool = False
    with_error: bool = False

    legend_loc: str = 'lower right'


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
    xcol_name: str = 'nominal_per_area'
    log_x_axis: bool = True
    with_error: bool = True

    def __post_init__(self) -> None:
        self.graph_styles['ecolor'] = 'red'
        self.graph_styles['linestyle'] = self.line_styles[1]
        self.labels['xlabel'] = r'log(Nr. Oda) [1/nm$^2$]'


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
    log_x_axis: bool = True


@dataclass
class FileConfig:
    """
    Set the name of the input files for plot with labels say what are
    those
    """
    fnames: dict[str, str] = field(default_factory=lambda: {
        'no_np': 'order_parameter.log'
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
        if nr_files > 1:
            # To plot the multi data graphes
            pass

    def _plot_graph(self,
                    key: str,
                    data: pd.DataFrame,
                    config: BaseConfig
                    ) -> None:
        """Plot the order parameter data."""
        x_range: tuple[float, float] = \
            (min(data[config.xcol_name]), max(data[config.xcol_name]))

        fig_i: plt.figure
        ax_i: plt.axes
        fig_i, ax_i = \
            plot_tools.mk_canvas(x_range, height_ratio=config.height_ratio)
        if config.log_x_axis:
            ax_i.set_xscale('log')

        if config.with_error:
            ax_i.errorbar(data[config.xcol_name],
                          data[config.ycol_name],
                          yerr=data['std'],
                          **config.graph_styles)
        else:
            ax_i.plot(data[config.xcol_name],
                      data[config.ycol_name],
                      **config.graph_styles)

        ax_i.set_xlabel(config.labels['xlabel'])
        ax_i.set_ylabel(f'{config.labels["ylabel"]} {config.y_unit}')
        ax_i.set_title(f'{config.labels["title"]} ({key})')
        ax_i.grid(True, which='both', linestyle='--', color='gray', alpha=0.5)

        plot_tools.save_close_fig(fig_i,
                                  ax_i,
                                  fname := f'{key}_{config.graph_suffix}',
                                  loc=config.legend_loc)
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

    def write_msg(self,
                  log: logger.logging.Logger
                  ) -> None:
        """Write and log messages."""
        print(f'{bcolors.OKCYAN}{PlotOrderParameter.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    PlotOrderParameter(log=logger.setup_logger('plot_order_parameter.log'))
