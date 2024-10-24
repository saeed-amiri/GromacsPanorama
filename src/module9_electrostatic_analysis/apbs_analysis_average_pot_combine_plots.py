"""
Plot combination data
plot the selcted Boltzman factor for ODA ans the 2d RDF of the ODA at
the interface
"""
from typing import Dict, Union, Tuple, List, Any
from dataclasses import field
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import logger
from common import plot_tools
from common import xvg_to_dataframe
from common import elsevier_plot_tools
from common.colors_text import TextColor as bcolors


@dataclass
class FileConfig:
    """set all the configs and parameters
    """
    oda_concentration: float = 0.003  # ODA/nm^2
    rdf_file: Dict[str, str] = field(default_factory=lambda: {
        'fname': '15_oda_densities.xvg',
        'data': 'fitted_rdf',
        'raw_data': 'rdf_2d',
        'radii': 'regions',
        })
    boltzman_file: Dict[str, str | List[float]] = field(
         default_factory=lambda: {
              'fname': 'boltzman_distribution.xvg',
              'data': [90, 91, 92, 93, 94, 95],
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
                'xlabel': r'r$^*$ [nm]',
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
            'label': r'$g^*_{fitted}(r^*)$, a.u.',
            'linewidth': 1.5,
            }

    @property
    def BOLTZMAN_PROP(self) -> Dict[str, str | float]:
        """set the line style for Boltzman factor"""
        return {
            'linestyle': '-',
            'color': 'darkred',
            'label': r'c$^\star(r^\star)$, norm.',
            'linewidth': 1.5,
            }

    @property
    def ADD_TEXT(self) -> bool:
        """add text for ODA concentration to the plot"""
        return True


class PlotBolzmannRdf:
    """
    Read the files and plot the data
    """
    # pylint: disable=too-many-instance-attributes
    __solts__ = ['rdf_radii',
                 'rdf_data',
                 'raw_rdf_data',
                 'boltzman_radii',
                 'boltzman_dict',
                 'boltzman_data',
                 'config',
                 'info_msg']
    rdf_data: np.ndarray
    raw_rdf_data: np.ndarray
    rdf_radii: np.ndarray  # in Angstrom
    boltzman_data: np.ndarray
    boltzman_dict: dict[int, np.ndarray]
    boltzman_radii: np.ndarray  # in nm
    config: FileConfig
    info_msg: str

    def __init__(self,
                 log: logger.logging.Logger,
                 cut_radius: float | None = None,
                 config: FileConfig = FileConfig()
                 ) -> None:
        self.info_msg = 'Message from PlotBolzmannRdf:\n'
        self.config = config
        self.process_files(cut_radius, log)
        self.plot_data()
        self.plot_data_bpm()
        self.plot_data_paper()
        self.plot_fit_rdf_with_distribution(cut_radius)
        self.plot_raw_rdf_with_distribution(cut_radius)
        self.write_msg(log)

    def process_files(self,
                      cut_radius: float | None,
                      log: logger.logging.Logger
                      ) -> None:
        """process the files"""
        self.set_rdf_data(cut_radius, log)
        self.set_raw_rdf_data(cut_radius, log)
        self.set_boltzman_data(cut_radius, log)

    def set_rdf_data(self,
                     cut_radius: float | None,
                     log: logger.logging.Logger
                     ) -> None:
        """set the RDF data"""
        rdf_data, rdf_radii = self.parse_xvg(
            fname=self.config.rdf_file['fname'],
            data_column=self.config.rdf_file['data'],
            radii_column=self.config.rdf_file['radii'],
            log=log)
        rdf_data /= np.max(rdf_data)
        self.rdf_radii, self.rdf_data = \
            self.cut_radii(rdf_radii, rdf_data, cut_radius)

    def set_raw_rdf_data(self,
                         cut_radius: float | None,
                         log: logger.logging.Logger
                         ) -> None:
        """set the RDF data"""
        rdf_data, rdf_radii = self.parse_xvg(
            fname=self.config.rdf_file['fname'],
            data_column=self.config.rdf_file['raw_data'],
            radii_column=self.config.rdf_file['radii'],
            log=log)
        rdf_data /= np.max(rdf_data)
        self.rdf_radii, self.raw_rdf_data = \
            self.cut_radii(rdf_radii, rdf_data, cut_radius)

    def set_boltzman_data(self,
                          cut_radius: float | None,
                          log: logger.logging.Logger
                          ) -> None:
        """set the Boltzman factor data"""
        boltzman_data_dict: dict[int, np.ndarray] = {}
        boltzman_data_list: list[np.ndarray] = []
        for i in self.config.boltzman_file['data']:
            boltzman_data, boltzman_radii = self.parse_xvg(
                fname=self.config.boltzman_file['fname'],
                data_column=f'{i}',
                radii_column=self.config.boltzman_file['radii'],
                log=log)
            boltzman_data_dict[i] = boltzman_data
            boltzman_data_list.append(boltzman_data)
        boltzman_data_dict['radii'] = boltzman_radii
        boltzman_data = np.mean(boltzman_data_list, axis=0)
        self.boltzman_radii, boltzman_data = self.cut_radii(
            boltzman_radii, boltzman_data, cut_radius)
        self.boltzman_data = boltzman_data / np.max(boltzman_data)
        self.boltzman_dict = boltzman_data_dict
        del boltzman_data_dict
        del boltzman_data_list

    @staticmethod
    def cut_radii(radii: np.ndarray,
                  data: np.ndarray,
                  cut_radius: float | None
                  ) -> Tuple[np.ndarray, np.ndarray]:
        """cut the radii and data"""
        if cut_radius is None:
            return radii, data
        cut_index = np.argmin(np.abs(radii - cut_radius))
        return radii[:cut_index], data[:cut_index]

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
        self.plot_rdf(ax_i, self.rdf_data, plt_config.RDF_PROP)
        self.plot_boltzman(ax_i, self.boltzman_data, plt_config.BOLTZMAN_PROP)
        self.plot_vlines(ax_i)

        self.set_axis_properties(ax_i, plt_config)
        self.add_text(ax_i, plt_config)

        elsevier_plot_tools.save_close_fig(
            fig=fig_i,
            loc=plt_config.PLOT_PROPERTIES['legend_loc'],
            fname=(fname := plt_config.PLOT_PROPERTIES['output_fname']),
            )
        self.info_msg += f'\tThe plot is saved as {fname}\n'

    def plot_data_bpm(self) -> None:
        """plot the data"""
        figure: Tuple[plt.Figure, plt.Axes] = \
            elsevier_plot_tools.mk_canvas('single_column')
        fig_i, ax_i = figure

        _plt_config = PlotBolzmannRdfConfiguratio()
        _plt_config.BOLTZMAN_PROP['label'] = r'c^\star/c$_0 (norm)$'
        _plt_config.PLOT_PROPERTIES['linestyle'] = '-'
        golden_ratio: float = (1 + 5 ** 0.5) / 2
        hight: float = 2.35
        width: float = hight * golden_ratio
        self.plot_boltzman(ax_i, self.boltzman_data, _plt_config.BOLTZMAN_PROP)
        fig_i.set_size_inches(width, hight)
        ax_i.set_xlabel(r'$r^\star$ [nm]', fontsize=14)
        ax_i.set_ylabel(ax_i.get_ylabel(), fontsize=14)
        ax_i.set_yticks([0.0, 0.5, 1.0])
        ax_i.set_xticks([0, 2, 4, 6, 8, 10])
        ax_i.tick_params(axis='x', labelsize=14)  # X-ticks
        ax_i.tick_params(axis='y', labelsize=14)  # Y-ticks
        plot_tools.save_close_fig(fig_i,
                                  ax_i,
                                  fout := 'boltzman_bpm.jpg',
                                  loc='lower right',
                                  legend_font_size=11,
                                  if_close=False,
                                  )
        _plt_config.RDF_PROP['linesytle'] = ':'
        _plt_config.RDF_PROP['marker'] = 's'
        _plt_config.RDF_PROP['markersize'] = '3'
        ax_i.plot(self.rdf_radii,
                  self.raw_rdf_data,
                  scalex=True,
                  scaley=True,
                  marker='o',
                  markersize=3,
                  linestyle=':',
                  color='k',
                  label=r'$g^\star(r^\star)$, a.u.',
                  )

        ax_i.axvline(x=1.75,
                     ymin=0,
                     ymax=1,
                     linestyle='--',
                     color='darkred',
                     linewidth=1.0,
                     label=r'r$^\star_c$',
                     )

        plot_tools.save_close_fig(fig_i,
                                  ax_i,
                                  fout := 'boltzman_rdf_bpm.jpg',
                                  loc='lower right',
                                  legend_font_size=11,
                                  if_close=False,
                                  )

        self.info_msg += f'\tThe plot is saved as {fout}\n'

    def plot_data_paper(self) -> None:
        """plot the data"""
        figure: Tuple[plt.Figure, plt.Axes] = \
            elsevier_plot_tools.mk_canvas('single_column')
        fig_i, ax_i = figure

        _plt_config = PlotBolzmannRdfConfiguratio()
        _plt_config.BOLTZMAN_PROP['label'] = r'c$^\star(norm)$'
        _plt_config.PLOT_PROPERTIES['linestyle'] = '-'
        self.plot_boltzman(ax_i, self.boltzman_data, _plt_config.BOLTZMAN_PROP)
        ax_i.set_xlabel(r'$r^\star$ [nm]',
                        fontsize=elsevier_plot_tools.FONT_SIZE_PT)
        ax_i.set_ylabel(ax_i.get_ylabel(),
                        fontsize=elsevier_plot_tools.FONT_SIZE_PT)
        ax_i.set_yticks([0.0, 0.5, 1.0])
        ax_i.set_xticks([0, 2, 4, 6, 8, 10])
        ax_i.tick_params(axis='x',
                         labelsize=elsevier_plot_tools.FONT_SIZE_PT)  # X-ticks
        ax_i.tick_params(axis='y',
                         labelsize=elsevier_plot_tools.FONT_SIZE_PT)  # Y-ticks
        plot_tools.save_close_fig(fig_i,
                                  ax_i,
                                  fout := 'boltzman_paper.jpg',
                                  loc='lower right',
                                  if_close=False,
                                  )
        _plt_config.RDF_PROP['linesytle'] = ':'
        _plt_config.RDF_PROP['marker'] = 's'
        _plt_config.RDF_PROP['markersize'] = '2'
        ax_i.plot(self.rdf_radii,
                  self.raw_rdf_data,
                  scalex=True,
                  scaley=True,
                  marker='o',
                  markersize=2,
                  linestyle=' ',
                  color='k',
                  label=r'$g^\star(r^\star)$, a.u.',
                  )

        ax_i.axvline(x=1.75,
                     ymin=0,
                     ymax=0.9,
                     linestyle='-',
                     color='gray',
                     linewidth=1.0,
                     label=r'r$^\star_c$=1.75',
                     )
        ax_i.grid(True, linestyle='--', color='gray', alpha=0.5)
        ax_i.set_xlim(-0.5, 10.5)

        ax_i.text(0.3,
                  0.98,
                  '0.03 ODA/nm$^2$',
                  ha='right',
                  va='top',
                  transform=ax_i.transAxes,
                  bbox=dict(facecolor='white',
                            edgecolor='white',
                            boxstyle='round,pad=0.1'),
                  fontsize=elsevier_plot_tools.LABEL_FONT_SIZE_PT - 2)

        plot_tools.save_close_fig(fig_i,
                                  ax_i,
                                  fout := 'boltzman_rdf_paper.jpg',
                                  loc='lower right',
                                  if_close=False,
                                  )

        self.info_msg += f'\tThe plot is saved as {fout}\n'


    def plot_fit_rdf_with_distribution(self,
                                       cut_radius: float | None = None
                                       ) -> None:
        """
        plot the raw rdf as dots along with potential of each point
        """
        for layer in self.config.boltzman_file['data']:
            figure: Tuple[plt.Figure, plt.Axes] = \
                elsevier_plot_tools.mk_canvas('single_column')
            fig_i, ax_i = figure

            plt_config = PlotBolzmannRdfConfiguratio()
            self.plot_rdf(ax_i, self.rdf_data, plt_config.RDF_PROP)
            boltzman_data = self.cut_radii(self.boltzman_dict['radii'],
                                           self.boltzman_dict[layer],
                                           cut_radius)[1]
            self.plot_boltzman(ax_i,
                               boltzman_data / np.max(boltzman_data),
                               plt_config.BOLTZMAN_PROP)

            self.set_axis_properties(ax_i, plt_config)
            self.add_text(ax_i, plt_config)
            self.add_text(ax_i,
                          plt_config,
                          text=f'z index = {layer}',
                          loc=(0.14, 0.9))

            elsevier_plot_tools.save_close_fig(
                fig=fig_i,
                loc=plt_config.PLOT_PROPERTIES['legend_loc'],
                fname=(fname := f'{layer}_boltzman_rdf.jpg'),
                )
            self.info_msg += f'\tThe plot is saved as {fname}\
                for the layer {layer}\n'

    def plot_raw_rdf_with_distribution(self,
                                       cut_radius: float | None = None
                                       ) -> None:
        """
        plot the raw rdf as dots along with potential of each point
        """
        plt_config = PlotBolzmannRdfConfiguratio()

        rdf_config: Dict[str, Any] = {
            'linestyle': ':',
            'linewidth': 1.5,
            'marker': 'o',
            'markersize': 3,
            'color': 'darkred',
            'label': r'$g^*(r^*)$, a.u.',
            }
        boltzman_config: Dict[str, Any] = {
            'linestyle': '-',
            'color': 'darkblue',
            'label': r'c/c$_0$, norm.',
            'linewidth': 1.5,
            }
        for layer in self.config.boltzman_file['data']:
            figure: Tuple[plt.Figure, plt.Axes] = \
                elsevier_plot_tools.mk_canvas('single_column')
            fig_i, ax_i = figure

            self.plot_rdf(ax_i, self.raw_rdf_data, rdf_config)
            boltzman_data = self.cut_radii(self.boltzman_dict['radii'],
                                           self.boltzman_dict[layer],
                                           cut_radius)[1]
            self.plot_boltzman(ax_i,
                               boltzman_data / np.max(boltzman_data),
                               boltzman_config)

            self.set_axis_properties(ax_i, plt_config)
            self.add_text(ax_i, plt_config)
            self.add_text(ax_i,
                          plt_config,
                          text=f'z index = {layer}',
                          loc=(0.14, 0.9))

            elsevier_plot_tools.save_close_fig(
                fig=fig_i,
                loc=plt_config.PLOT_PROPERTIES['legend_loc'],
                fname=(fname := f'raw_{layer}_boltzman_rdf.jpg'),
                )
            self.info_msg += f'\tThe plot is saved as {fname}\
                for the layer {layer}\n'

    def plot_rdf(self,
                 ax_i: plt.Axes,
                 rdf_data: np.ndarray,
                 kwargs: Dict[str, str | float]
                 ) -> None:
        """plot the RDF"""
        ax_i.plot(self.rdf_radii,
                  rdf_data,
                  scalex=True,
                  scaley=True,
                  **kwargs)

    def plot_boltzman(self,
                      ax_i: plt.Axes,
                      boltzman_data: np.ndarray,
                      kwargs: Dict[str, str | float]
                      ) -> None:
        """plot the Boltzman factor"""
        ax_i.plot(self.boltzman_radii,
                  boltzman_data,
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
        ylims: Tuple[float, float] = ax_i.get_ylim()
        for key_i, value_i in fit_param.fit_pram.items():
            if key_i == 'contact_radius':
                ylo = ylims[0]
                yhi = ylims[1] * 0.86
            else:
                ylo = ylims[0]
                yhi = ylims[1]
            ax_i.axvline(x=value_i,
                         ymin=ylo,
                         ymax=yhi,
                         linestyle=fit_param.line_style[key_i],
                         color=fit_param.colors[key_i],
                         linewidth=fit_param.line_width[key_i],
                         label=f'{fit_param.labels[key_i]} = {value_i}',
                         )

    def add_text(self,
                 ax_i: plt.Axes,
                 plt_config: PlotBolzmannRdfConfiguratio,
                 text: str | None = None,
                 loc: Tuple[float, float] = (0.16, 1.0),
                 font_size: int = elsevier_plot_tools.LABEL_FONT_SIZE_PT - 3
                 ) -> None:
        """add text to the plot"""
        # pylint: disable=too-many-arguments
        if text is not None:
            pattern = text
        else:
            pattern = f'{self.config.oda_concentration} ODA/nm$^2$'
        if plt_config.ADD_TEXT:
            ax_i.text(loc[0],
                      loc[1],
                      pattern,
                      transform=ax_i.transAxes,
                      fontsize=font_size,
                      verticalalignment='top',
                      horizontalalignment='center',
                      )

    def write_msg(self,
                  log: logger.logging.Logger  # To log
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{PlotBolzmannRdf.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    PlotBolzmannRdf(logger.setup_logger('combine_plots.log'), cut_radius=9.8)
