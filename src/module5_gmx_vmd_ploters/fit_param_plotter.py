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

from common import logger, xvg_to_dataframe, elsevier_plot_tools
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
        field(default_factory=lambda: [':', '-', '--', '-.'])
    colors: typing.Union[list[str], dict[str, str]] = \
        field(default_factory=lambda: ['black', 'dimgrey', 'darkgrey', 'red'])

    height_ratio: float = (5 ** 0.5 - 1) * 1.5

    y_unit: str = ''

    legend_loc: str = 'lower right'

    show_grid: bool = True
    plot_contact_radius: bool = False
    show_mirror_axes: bool = True


@dataclass
class FitPlotConfig(BaseConfig):
    """
    Configuration for the fit plot
    The columns are:
    'first_turn',
    'midpoint',
    'second_turn',
    'pure_first_turn',
    'pure_midpoint',
    'pure_second_turn',
    """
    # pylint: disable=too-many-instance-attributes
    out_suffix: str = f'turn_points.{elsevier_plot_tools.IMG_FORMAT}'
    xcol_name: str = 'nr_oda'
    ycol_name: list[str] = field(default_factory=lambda: ['first_turn',
                                                          'midpoint',
                                                          'second_turn'
                                                          ])

    labels: dict[str, str] = field(default_factory=lambda: {
        'title': 'Fitted parameters',
        'ylabel': r'$r^\star$ [nm]',
        'xlabel': r'ODA/nm$^2$'
    })
    fit_param_fname: str = 'fit_parameters.xvg'

    legends: dict[str, str] = \
        field(default_factory=lambda: {
            'first_turn': 'a',
            'midpoint': 'b',
            'second_turn': 'c'
        })
    line_styles: dict[str, str] = \
        field(default_factory=lambda: {
            'first_turn': ':',
            'midpoint': '--',
            'second_turn': '-.'})
    colors: dict[str, str] = \
        field(default_factory=lambda: {
            'first_turn': 'darkred',
            'midpoint': 'darkred',
            'second_turn': 'darkred'})
    graph_max_col: str = 'second_turn'

    yticks: list[float] = field(default_factory=lambda: [0, 3.0, 6.0])

    def __post_init__(self) -> None:
        """Post init function"""
        if self.plot_contact_radius:
            self.ycol_name.insert(0, 'contact_radius')
            self.legends['contact_radius'] = r'$r^\star_{c}$'
            self.line_styles['contact_radius'] = ':'
            self.colors['contact_radius'] = 'black'


@dataclass
class FitRdfPlotConfig(BaseConfig):
    """
    Configuration for the fit rdf plot
    """
    out_suffix: str = f'fitted_rdf.{elsevier_plot_tools.IMG_FORMAT}'
    xcol_name: str = 'regions'
    ycol_name: list[str] = field(default_factory=lambda: ['fitted_rdf'])

    labels: dict[str, str] = field(default_factory=lambda: {
        'title': 'Fitted RDF',
        'ylabel': r'$g^\star(r^\star)$, a.u.',
        'xlabel': r'$r^\star$ [nm]'
    })
    fit_rdf_fname: list[str] = field(default_factory=lambda: [
        '5_oda_densities.xvg',
        '10_oda_densities.xvg',
        '15_oda_densities.xvg',
        '20_oda_densities.xvg',
        '30_oda_densities.xvg',
        '40_oda_densities.xvg',
        '50_oda_densities.xvg',
        ])

    legends: dict[str, str] = \
        field(default_factory=lambda: {
            '5_oda_densities.xvg': r'0.01 ODA/$nm^2$',  # 5ODA
            '10_oda_densities.xvg': r'0.02 ODA/$nm^2$',  # 10ODA
            '15_oda_densities.xvg': r'0.03 ODA/$nm^2$',  # 15ODA
            '20_oda_densities.xvg': r'0.04 ODA/$nm^2$',  # 20ODA
            '30_oda_densities.xvg': r'0.06 ODA/$nm^2$',  # 30ODA
            '40_oda_densities.xvg': r'0.09 ODA/$nm^2$',  # 40ODA
            '50_oda_densities.xvg': r'0.11 ODA/$nm^2$'  # 50ODA
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

        ax_i: plt.axes
        fig_i: plt.figure
        fig_i, ax_i = \
            elsevier_plot_tools.mk_canvas(size_type='single_column')

        for _, ycol in enumerate(config.ycol_name):
            ax_i.plot(fit_data[config.xcol_name],
                      fit_data[ycol]/10.0,  # Convert to nm
                      label=config.legends.get(ycol),
                      color=config.colors.get(ycol),
                      linestyle=config.line_styles.get(ycol),
                      lw=1)

        ax_i.set_xlabel(config.labels['xlabel'],
                        fontsize=elsevier_plot_tools.FONT_SIZE_PT)
        ax_i.set_ylabel(config.labels['ylabel'],
                        fontsize=elsevier_plot_tools.FONT_SIZE_PT)

        xticks: list[float] = fit_data[config.xcol_name].unique().tolist()
        ax_i.set_xticks(xticks)
        xticklabels: list[np.float64] = \
            [np.round(item/(21.7*21.7), 2) for item in ax_i.get_xticks()]
        ax_i.set_xticklabels(xticklabels)

        ax_i.set_yticks(config.yticks)
        ax_i.set_yticklabels([f'{item:.1f}' for item in config.yticks])

        if config.show_grid:
            ax_i.grid(True, 'both', ls='--', color='gray', alpha=0.5, zorder=2)

        ax_i.text(-0.09,
                  1,
                  'd)',
                  ha='right',
                  va='top',
                  transform=ax_i.transAxes,
                  fontsize=elsevier_plot_tools.LABEL_FONT_SIZE_PT)
        if not config.show_mirror_axes:
            self.remove_mirror_axes(ax_i)
        elsevier_plot_tools.save_close_fig(
            fig_i, config.out_suffix, loc='lower right')
        self.info_msg += f"\tSaved plot: {config.out_suffix}\n"

    def plot_fitted_rdf(self,
                        log: logger.logging.Logger,
                        config: FitRdfPlotConfig
                        ) -> None:
        """
        Plot the fitted rdf
        """
        ax_i: plt.axes
        fig_i: plt.figure

        fig_i, ax_i = \
            elsevier_plot_tools.mk_canvas('single_column')

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
                      linestyle=config.line_styles[idx],
                      lw=1)

        ax_i.set_xlabel(config.labels['xlabel'],
                        fontsize=elsevier_plot_tools.FONT_SIZE_PT)
        ax_i.set_ylabel(config.labels['ylabel'],
                        fontsize=elsevier_plot_tools.FONT_SIZE_PT)

        xticks: list[float] = [0, 5, 10]
        ax_i.set_xticks(xticks)

        y_ticks = [0, 0.5, 1]
        ax_i.set_yticks(y_ticks)

        if config.show_grid:
            ax_i.grid(True, 'both', ls='--', color='gray', alpha=0.5, zorder=2)

        ax_i.text(-0.09,
                  1,
                  'c)',
                  ha='right',
                  va='top',
                  transform=ax_i.transAxes,
                  fontsize=elsevier_plot_tools.LABEL_FONT_SIZE_PT)
        if not config.show_mirror_axes:
            self.remove_mirror_axes(ax_i)
        elsevier_plot_tools.save_close_fig(
            fig_i, config.out_suffix, loc='lower right')

        self.info_msg += f"\tSaved plot: {config.out_suffix}\n"

    def remove_mirror_axes(self,
                           ax: plt.axes
                           ) -> None:
        """Remove the top and right spines and ticks from a matplotlib Axes"""
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

    def write_msg(self,
                  log: logger.logging.Logger
                  ) -> None:
        """Write and log messages."""
        print(f'{bcolors.OKCYAN}{PlotFitted.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    PlotFitted(log=logger.setup_logger('plot_fitted.log'))
