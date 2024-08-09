"""
Plot combination data
plot the selcted Boltzman factor for ODA ans the 2d RDF of the ODA at
the interface
"""
from typing import Dict, Union, Tuple, List
from dataclasses import field
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import logger
from common import xvg_to_dataframe
from common import elsevier_plot_tools
from common.colors_text import TextColor as bcolors


@dataclass
class FileConfig:
    """set all the configs and parameters
    """
    rdf_file: Dict[str, str] = field(default_factory=lambda: {
        'fname': '15_oda_densities.xvg',
        'data': 'fitted_rdf',
        'radii': 'regions',
        })
    boltzman_file: Dict[str, str | List[float]] = field(
         default_factory=lambda: {
              'fname': 'boltzman_distribution.xvg',
              'data': [90, 91, 92],
              'radii': 'r_nm',
              })


@dataclass
class FitParameres:
    """
    paramters from the fit of the rdf and the size of the contact
    radius at the interface
    """
    add_fit_vlines: bool = True
    fit_pram: dict[str, float] = field(default_factory=lambda: {
        'contact_radius': 1.75,
        'a': 3.04,
        'b': 4.62,
        'c': 6.14,
        })

    line_style: dict[str, str] = field(default_factory=lambda: {
        'contact_radius': '-',
        'a': ':',
        'b': '--',
        'c': '-.',
        })

    line_width: dict[str, float] = field(default_factory=lambda: {
        'contact_radius': 1.0,
        'a': 1.0,
        'b': 1.0,
        'c': 1.0,
        })

    colors: dict[str, str] = field(default_factory=lambda: {
        'contact_radius': 'black',
        'a': 'darkred',
        'b': 'darkred',
        'c': 'darkred',
        })

    labels: dict[str, str] = field(default_factory=lambda: {
        'contact_radius': r'$r^*_c$',
        'a': 'a',
        'b': 'b',
        'c': 'c',
        })


class PlotBolzmannRdfConfiguratio:
    """
    set all the configs and parameters and properties for the plot
    """
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=invalid-name
    @property
    def PLOT_PROPERTIES(self) -> Dict[str, Union[str, int, float]]:
        """set the plot properties"""
        return {'y_label': None,
                'xlabel': 'r (nm)',
                'title': 'Boltzman factor and RDF',
                'x_lims': [-0.5, 10.5],
                'y_lims': [-0.1, 1.1],
                'x_ticks': [0, 2, 4, 6, 8, 10],
                'y_ticks': [0, 0.5, 1],
                'legend_loc': 'lower right',
                'output_fname': 'boltzman_rdf.jpg',
                }

    @property
    def RDF_PROP(self) -> Dict[str, str | float]:
        """set the line style for RDF"""
        return {
            'linestyle': '-',
            'color': 'darkred',
            'label': r'$g^*(r^*)$, a.u.',
            'linewidth': 1.5,
            }

    @property
    def BOLTZMAN_PROP(self) -> Dict[str, str | float]:
        """set the line style for Boltzman factor"""
        return {
            'linestyle': ':',
            'color': 'darkblue',
            'label': r'c/c$_0$, a.u.',
            'linewidth': 1.5,
            }


class PlotBolzmannRdf:
    """
    Read the files and plot the data
    """
    __solts__ = ['rdf_radii',
                 'rdf_data',
                 'boltzman_radii',
                 'boltzman_data',
                 'config',
                 'info_msg']
    rdf_data: np.ndarray
    rdf_radii: np.ndarray  # in Angstrom
    boltzman_data: np.ndarray
    boltzman_radii: np.ndarray  # in nm
    config: FileConfig
    info_msg: str

    def __init__(self,
                 log: logger.logging.Logger,
                 config: FileConfig = FileConfig()
                 ) -> None:
        self.info_msg = 'Message from PlotBolzmannRdf:\n'
        self.config = config
        self.process_files(log)
        self.plot_data()
        self.write_msg(log)

    def process_files(self,
                      log: logger.logging.Logger
                      ) -> None:
        """process the files"""
        self.set_rdf_data(log)
        self.set_boltzman_data(log)

    def set_rdf_data(self,
                     log: logger.logging.Logger
                     ) -> None:
        """set the RDF data"""
        rdf_data, self.rdf_radii = self.parse_xvg(
            fname=self.config.rdf_file['fname'],
            data_column=self.config.rdf_file['data'],
            radii_column=self.config.rdf_file['radii'],
            log=log)
        self.rdf_data = rdf_data / np.max(rdf_data)

    def set_boltzman_data(self,
                          log: logger.logging.Logger
                          ) -> None:
        """set the Boltzman factor data"""
        boltzman_data_list: List[np.ndarray] = []
        for i in self.config.boltzman_file['data']:
            boltzman_data, boltzman_radii = self.parse_xvg(
                fname=self.config.boltzman_file['fname'],
                data_column=f'{i}',
                radii_column=self.config.boltzman_file['radii'],
                log=log)
            boltzman_data_list.append(boltzman_data)
        boltzman_data = np.mean(boltzman_data_list, axis=0)
        self.boltzman_data = boltzman_data / np.max(boltzman_data)
        self.boltzman_radii = boltzman_radii

    @staticmethod
    def parse_xvg(fname: str,
                  data_column: str,
                  radii_column: str,
                  log: logger.logging.Logger
                  ) -> Tuple[np.ndarray, np.ndarray]:
        """parse the xvg file"""
        df_i: pd.DataFrame = xvg_to_dataframe.XvgParser(
            fname=fname, log=log, x_type=float).xvg_df
        data = np.asanyarray(df_i[data_column].values)
        radii = np.asanyarray(df_i[radii_column].values)
        return data, radii / 10.0  # convert to nm

    def plot_data(self) -> None:
        """plot the data"""
        figure: Tuple[plt.Figure, plt.Axes] = \
            elsevier_plot_tools.mk_canvas('single_column')
        fig_i, ax_i = figure

        plt_config = PlotBolzmannRdfConfiguratio()
        self.plot_rdf(ax_i, plt_config.RDF_PROP)
        self.plot_boltzman(ax_i, plt_config.BOLTZMAN_PROP)
        self.plot_vlines(ax_i)

        self.set_axis_properties(ax_i, plt_config)

        elsevier_plot_tools.save_close_fig(
            fig=fig_i,
            loc=plt_config.PLOT_PROPERTIES['legend_loc'],
            fname=(fname := plt_config.PLOT_PROPERTIES['output_fname']),
            )
        self.info_msg += f'\tThe plot is saved as {fname}\n'

    def plot_rdf(self,
                 ax_i: plt.Axes,
                 kwargs: Dict[str, str | float]
                 ) -> None:
        """plot the RDF"""
        ax_i.plot(self.rdf_radii,
                  self.rdf_data,
                  scalex=True,
                  scaley=True,
                  **kwargs)

    def plot_boltzman(self,
                      ax_i: plt.Axes,
                      kwargs: Dict[str, str | float]
                      ) -> None:
        """plot the Boltzman factor"""
        ax_i.plot(self.boltzman_radii,
                  self.boltzman_data,
                  scalex=True,
                  scaley=True,
                  **kwargs)

    def set_axis_properties(self,
                            ax_i: plt.Axes,
                            plt_config: PlotBolzmannRdfConfiguratio
                            ) -> None:
        """set the axis properties"""
        elsevier_plot_tools.remove_mirror_axes(ax_i)
        ax_i.set_xlabel(plt_config.PLOT_PROPERTIES['xlabel'])
        ax_i.set_ylabel(plt_config.PLOT_PROPERTIES['y_label'])
        ax_i.set_yticks(plt_config.PLOT_PROPERTIES['y_ticks'])
        ax_i.set_xticks(plt_config.PLOT_PROPERTIES['x_ticks'])

    def plot_vlines(self,
                    ax_i: plt.Axes
                    ) -> None:
        """plot the vertical lines"""
        fit_param = FitParameres()
        for key_i, value_i in fit_param.fit_pram.items():
            ax_i.axvline(x=value_i,
                         linestyle=fit_param.line_style[key_i],
                         color=fit_param.colors[key_i],
                         linewidth=fit_param.line_width[key_i],
                         label=fit_param.labels[key_i])

    def write_msg(self,
                  log: logger.logging.Logger  # To log
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{PlotBolzmannRdf.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    PlotBolzmannRdf(logger.setup_logger('combine_plots.log'))
