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
from common import plot_tools
from common import my_tools
from common.colors_text import TextColor as bcolors


@dataclass
class BaseConfig:
    """
    Basic configurations and setup for the plots.
    """
    graph_suffix: str = 'tension.plot'

    labels: dict[str, str] = field(default_factory=lambda: {
        'title': 'Computed Tension',
        'ylabel': r'$\gamma$',
        'xlabel': 'Nr. Oda'
    })

    graph_styles: dict[str, typing.Any] = field(default_factory=lambda: {
        'legend': r'$\gamma$',
        'color': 'black',
        'marker': 'o',
        'linestyle': '-',
        'markersize': 6,
        'second_markersize': 4
    })

    line_styles: list[str] = \
        field(default_factory=lambda: ['-', ':', '--', '-.'])
    colors: list[str] = \
        field(default_factory=lambda: ['black', 'red', 'blue', 'green'])

    height_ratio: float = 5 ** 0.5 - 1


@dataclass
class SimpleGraph(BaseConfig):
    """
    Parameters for simple data plots.
    """


@dataclass
class LogGraph(BaseConfig):
    """
    Parameters for the semilog plots.
    """
    labels: dict[str, str] = field(default_factory=lambda: {
        'title': 'Computed Tension',
        'ylabel': r'log($\gamma$)',
        'xlabel': 'Nr. Oda'
    })


@dataclass
class ErrorBarGraph(BaseConfig):
    """
    Parameters for plots with error bars.
    """
    plot_errorbars: bool = True


@dataclass
class FileConfigs:
    """
    Set the name of the input files for plot with labels say what are
    those
    """
    fnames: dict[str, str] = field(default_factory=lambda: {
        'with_np': 'tension.log'})


@dataclass
class AllConfig(FileConfigs):
    """
    Consolidates all configurations for different graph types.
    """
    simple_config: SimpleGraph = field(default_factory=SimpleGraph)
    log_config: LogGraph = field(default_factory=LogGraph)
    errbar_config: ErrorBarGraph = field(default_factory=ErrorBarGraph)


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
        columns: list[str] = ['Name', 'nr.Oda', 'tension', 'errorbar']
        df_in: pd.DataFrame = \
            pd.read_csv(fname, delim_whitespace=True, names=columns)
        return df_in


if __name__ == '__main__':
    PlotTension(log=logger.setup_logger('plot_tension.log'))
