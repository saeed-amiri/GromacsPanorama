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


@dataclass
class BaseConfig:
    """
    Basic configurations and setup for the plots.
    """
    # pylint: disable=too-many-instance-attributes
    graph_suffix: str = 'order_parameter.png'

    labels: dict[str, str] = field(default_factory=lambda: {
        'title': 'Computed order parameter',
        'ylabel': 'S',
        'xlabel': 'Nr. Oda'
    })

    graph_styles: dict[str, typing.Any] = field(default_factory=lambda: {
        'label': 'S',
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

    y_unit: str = 'S'

    log_x_axis: bool = False


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


@dataclass
class DoubleDataGraph(BaseConfig):
    """
    Parameters for the plotting both the data with and without NP
    """
    graph_suffix: str = 'order_parameter_both.png'


@dataclass
class BarPlotConfig(BaseConfig):
    """
    Parameters for the number of oda at the interface
    """
    graph_suffix: str = 'bar_oda_nr.png'


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
