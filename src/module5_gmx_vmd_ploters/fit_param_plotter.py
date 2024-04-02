"""
Plotting fittted parameters of 2D rdf of the oda around the nanoparticle
Each data set will be plotted separately and saved in the same directory
- Turnning points will be marked on the plot:
    Input:
        fit_parameters.xvg
    Output:
        turn_points.png
- Fitted rdf, overlayed of all the fitted one to compare
    Input:
        fit_rdf_<nr>_oda.xvg
    Output:
        fitted_rdf.png
"""

import typing
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import logger, plot_tools, xvg_to_dataframe
from common.colors_text import TextColor as bcolors


@dataclass
class BaseConfig:
    """
    Base class for the graph configuration
    """
    # pylint: disable=too-many-instance-attributes
    grp_suffix: str = field(init=False)
    ycol_name: typing.Union[str, list[str]] = field(init=False)
    xcol_name: str = field(init=False)

    labels: dict[str, str] = field(default_factory=lambda: {
        'title': 'title',
        'ylabel': 'ylabel',
        'xlabel': 'xlabel'
    })

    graph_styles: dict[str, typing.Any] = field(default_factory=lambda: {
        'label': 'number',
        'color': 'black',
        'marker': 'o',
        'linestyle': '-',
        'markersize': 5,
    })

    line_styles: typing.Union[list[str], dict[str, str]] = \
        field(default_factory=lambda: ['-', ':', '--', '-.'])
    colors: typing.Union[list[str], dict[str, str]] = \
        field(default_factory=lambda: ['black', 'red', 'blue', 'green'])

    height_ratio: float = (5 ** 0.5 - 1) * 1.5

    y_unit: str = ''

    legend_loc: str = 'lower right'


@dataclass
class FitPlotConfig(BaseConfig):
    """
    Configuration for the fit plot
    """
    out_suffix: str = 'turn_points.png'
    xcol_name: str = 'nr_oda'
    ycol_name: list[str] = field(default_factory=lambda: ['contact_radius',
                                                          'first_turn',
                                                          'midpoint',
                                                          'second_turn'
                                                          ])

    labels: dict[str, str] = field(default_factory=lambda: {
        'title': 'Fitted parameters',
        'ylabel': 'Distance from NP COM [nm]',
        'xlabel': 'Nr. ODA'
    })
    fit_param_fname: str = 'fit_parameters.xvg'

    legends: dict[str, str] = \
        field(default_factory=lambda: {
            'contact_radius': r'r$_{c}$',
            'first_turn': 'a',
            'midpoint': 'b',
            'second_turn': 'c'
        })
    line_styles: dict[str, str] = \
        field(default_factory=lambda: {
            'contact_radius': ':',
            'first_turn': ':',
            'midpoint': '--',
            'second_turn': ':'})
    colors: dict[str, str] = \
        field(default_factory=lambda: {
            'contact_radius': 'black',
            'first_turn': 'green',
            'midpoint': 'blue',
            'second_turn': 'red'})
    graph_max_col: str = 'second_turn'


@dataclass
class FitRdfPlotConfig(BaseConfig):
    """
    Configuration for the fit rdf plot
    """
    out_suffix: str = 'fitted_rdf.png'
    xcol_name: str = 'regions'
    ycol_name: list[str] = field(default_factory=lambda: ['fitted_rdf'])

    labels: dict[str, str] = field(default_factory=lambda: {
        'title': 'Fitted RDF',
        'ylabel': 'g(r), a.u.',
        'xlabel': 'r [nm]'
    })
    fit_rdf_fname: list[str] = field(default_factory=lambda: [
        '5_oda_densities.xvg',
        '15_oda_densities.xvg',
        '50_oda_densities.xvg',
        ])

    legends: dict[str, str] = \
        field(default_factory=lambda: {
            '5_oda_densities.xvg': '5 ODA',
            '15_oda_densities.xvg': '15 ODA',
            '50_oda_densities.xvg': '50 ODA'
        })


@dataclass
class AllConfig:
    """
    Configuration for the all plot
    """
    fit_param: FitPlotConfig = field(default_factory=FitPlotConfig)
    fit_rdf: FitRdfPlotConfig = field(default_factory=FitRdfPlotConfig)


class PlotFitted:
    """
    Plotting fittted 2d rdf and their parameters of the oda around
    the nanoparticle
    """

    info_msg: str = 'Message from PlotFitted:\n'
    config: AllConfig

    def __init__(self,
                 log: logger.logging.Logger,
                 config: "AllConfig" = AllConfig()
                 ) -> None:
        self.config = config
        self.log = log

        self.plot_graphs(log)
        self.write_msg(log)

    def plot_graphs(self,
                    log: logger.logging.Logger
                    ) -> None:
        """
        Plot the graphs
        """
        self.plot_turn_points(log, self.config.fit_param)
        self.plot_fitted_rdf(log, self.config.fit_rdf)

    def plot_turn_points(self,
                         log: logger.logging.Logger,
                         config: FitPlotConfig
                         ) -> None:
        """
        Plot the turning points
        """
        fit_data: pd.DataFrame = \
            xvg_to_dataframe.XvgParser(config.fit_param_fname, log).xvg_df
        x_range: tuple[float, float] = \
            fit_data[config.xcol_name].agg(['min', 'max'])

        ax_i: plt.axes
        fig_i: plt.figure
        fig_i, ax_i = \
            plot_tools.mk_canvas(x_range, height_ratio=config.height_ratio)

        for _, ycol in enumerate(config.ycol_name):
            ax_i.plot(fit_data[config.xcol_name],
                      fit_data[ycol]/10.0,  # Convert to nm
                      label=config.legends.get(ycol),
                      color=config.colors.get(ycol),
                      linestyle=config.line_styles.get(ycol),
                      lw=2)

        ax_i.set_xlabel(config.labels['xlabel'])
        ax_i.set_ylabel(config.labels['ylabel'], fontsize=12)

        xticks: list[float] = fit_data[config.xcol_name].unique().tolist()
        ax_i.set_xticks(xticks)

        y_max: float = \
            max(fit_data[config.graph_max_col].unique().tolist()) / 10.0
        y_ticks: list[np.float64] = list(np.linspace(0, y_max, 4))
        y_ticks = [round(item, 1) for item in y_ticks]
        ax_i.set_yticks(y_ticks)

        ax_i.grid(True, 'both', ls='--', color='gray', alpha=0.5, zorder=2)

        ax_i.text(-0.085,
                  1,
                  'd)',
                  ha='right',
                  va='top',
                  transform=ax_i.transAxes,
                  fontsize=18)

        plot_tools.save_close_fig(
            fig_i, ax_i, config.out_suffix, loc='upper right')
        self.log.info(f"Saved plot: {config.out_suffix}")

    def plot_fitted_rdf(self,
                        log: logger.logging.Logger,
                        config: FitRdfPlotConfig
                        ) -> None:
        """
        Plot the fitted rdf
        """
        ax_i: plt.axes
        fig_i: plt.figure
        x_range: tuple[float, float] = (0, 12)

        fig_i, ax_i = \
            plot_tools.mk_canvas(x_range, height_ratio=config.height_ratio)

        for idx, rdf_fname in enumerate(config.fit_rdf_fname):
            rdf_data: pd.DataFrame = \
                xvg_to_dataframe.XvgParser(rdf_fname, log).xvg_df
            rdf = rdf_data[config.ycol_name[0]]
            rdf_max: float = rdf.max()
            rdf_norm = rdf / rdf_max
            ax_i.plot(rdf_data[config.xcol_name] / 10.0,  # Convert to nm
                      rdf_norm,
                      label=config.legends.get(rdf_fname),
                      color=config.colors[idx],
                      linestyle=config.line_styles[idx])

        ax_i.set_xlabel(config.labels['xlabel'])
        ax_i.set_ylabel(config.labels['ylabel'])

        xticks: list[float] = [0, 5, 10]
        ax_i.set_xticks(xticks)

        y_ticks = [0, 0.5, 1]
        ax_i.set_yticks(y_ticks)

        ax_i.grid(True, 'both', ls='--', color='gray', alpha=0.5, zorder=2)

        ax_i.text(-0.09,
                  1,
                  'c)',
                  ha='right',
                  va='top',
                  transform=ax_i.transAxes,
                  fontsize=18)
        plot_tools.save_close_fig(
            fig_i, ax_i, config.out_suffix, loc='lower right')
        self.log.info(f"Saved plot: {config.out_suffix}")

    def write_msg(self,
                  log: logger.logging.Logger
                  ) -> None:
        """Write and log messages."""
        print(f'{bcolors.OKCYAN}{PlotFitted.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    PlotFitted(log=logger.setup_logger('plot_fitted.log'))
