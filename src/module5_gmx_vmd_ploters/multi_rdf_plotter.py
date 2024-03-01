"""
Plot Radial Distribution Function (RDF) Calculated from GROMACS

This script plots the radial distribution function (RDF) and cumulative
distribution function (CDF) for Chloroacetate (CLA) at the surface of
a nanoparticle (NP). It utilizes data generated by GROMACS.

GROMACS offers two methods for calculating RDF:
    1. Based on the center of mass (COM) of the NP.
    2. Based on the outermost residues of the NP, specifically APTES
        (APTES being the functional groups on the NP).

The script generates the following plots:
    - RDF plots for both COM-based and outermost residue-based
        calculations.
    - CDF plots corresponding to both calculation methods.

Inputs:
    The script requires RDF and CDF data files for each calculation
        method. It will generate plots if these files are present.

Notes:
    - The script is specifically designed for RDF and CDF analysis in
        the context of nanoparticles and their surface functional
        groups.
    - Ensure that the input data files are in the correct format as
        expected by the script.

Here plots all the data from one viewpoint on one canvas
Opt. by ChatGPt
Saeed
30 Jan 2024
"""

import typing
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pylab as plt

from common import logger, plot_tools, xvg_to_dataframe
from common.colors_text import TextColor as bcolors


@dataclass
class BaseGraphConfig:
    """Basic setups for graphs"""

    # pylint: disable=too-many-instance-attributes
    graph_suffix: str = 'mda.png'
    y_col_name: str = 'density'
    xcol_name: str = 'r_nm'

    labels: dict[str, str] = field(default_factory=lambda: {
        'ylabel': 'rdf',
        'xlabel': 'r [nm]'
    })

    titles: dict[str, typing.Any] = field(default_factory=lambda: {
        'com': 'From COM',
        'shell': 'From Shell'
    })

    graph_styles: dict[str, typing.Any] = field(default_factory=lambda: {
        'marker': 'o',
        'markersize': 0,
    })

    legends: dict[str, str] = \
        field(default_factory=lambda: {
            'CLA': 'Cl',
            'amino_n': 'N (APTES)',
            'N': 'N (APTES)',
            'amino_charge': r'H$^+$ (APTES)',
            'SOL': 'Water',
            'D10': 'Decane',
            'APT': 'APTES',
            'POT': 'Na',
            'OH2': 'O (water)',
            'ODN': 'ODA'})

    line_styles: dict[str, str] = \
        field(default_factory=lambda: {
            'CLA': '-',
            'amino_n': '--',
            'N': '--',
            'amino_charge': '--',
            'SOL': '-',
            'D10': '-',
            'APT': '-',
            'POT': '-',
            'OH2': '-',
            'ODN': '-.'})

    colors: dict[str, str] = \
        field(default_factory=lambda: {
            'CLA': 'green',
            'amino_n': 'blue',
            'N': 'blue',
            'amino_charge': 'blue',
            'SOL': 'red',
            'D10': 'grey',
            'APT': 'k',
            'POT': 'brown',
            'OH2': 'red',
            'ODN': 'green'})

    height_ratio: float = (5 ** 0.5 - 1) * 1.5

    y_unit: str = ''

    legend_loc: str = 'lower right'


@dataclass
class FileConfig:
    """setups for the input files names and their columns name"""
    # pylint: disable=too-many-instance-attributes
    # comfile: files which are the viewpoint are the com of the np
    # shellfile: files which are the viewpoint are the shell of the np
    viewpoint: list[str] = field(default_factory=lambda: ['com', 'shell'])

    normalize_type: str = 'max'  # or any other words neside max!

    com_files: dict[str, dict[str, typing.Any]] = field(
        default_factory=lambda: {
            'com_0': {'fname': 'mda_com_OH2.xvg', 'y_col': 'OH2'},
            'com_1': {'fname': 'com_d10.xvg', 'y_col': 'D10'},
            'com_2': {'fname': 'com_sol.xvg', 'y_col': 'SOL'},
            'com_3': {'fname': 'com_odn.xvg', 'y_col': 'ODN'},
            'com_4': {'fname': 'mda_com_POT.xvg', 'y_col': 'POT'},
            'com_5': {'fname': 'mda_com_CLA.xvg', 'y_col': 'CLA'},
            'com_6': {'fname': 'mda_com_N.xvg', 'y_col': 'N'},
            'com_7': {'fname': 'com_apt.xvg', 'y_col': 'APT'},
            'com_8': {'fname': 'com_amino_charge.xvg',
                      'y_col': 'amino_charge'}
            })
    com_plot_list: list[int] = field(default_factory=lambda: [4])
    com_legend_loc: str = 'lower right'
    com_window_legend_loc: str = 'upper left'
    com_max_indicator: str = 'com_5'

    shell_files: dict[str, dict[str, typing.Any]] = field(
        default_factory=lambda: {
            'shell_0': {'fname': 'shell_cla.xvg', 'y_col': 'CLA'},
            'shell_1': {'fname': 'shell_N.xvg', 'y_col': 'amino_n'},
            'shell_2': {'fname': 'shell_sol.xvg', 'y_col': 'SOL'},
            'shell_3': {'fname': 'shell_d10.xvg', 'y_col': 'D10'},
            'shell_4': {'fname': 'shell_odn.xvg', 'y_col': 'ODN'}
            })
    shell_plot_list: list[int] = field(default_factory=lambda: [0, 1, 2])
    shell_legend_loc: str = 'upper right'
    shell_window_legend_loc: str = 'upper right'


@dataclass
class OverlayConfig(BaseGraphConfig):
    """set the parameters for the overlay plots"""
    # Max on the x-axis to set the window
    second_window: dict[str, float] = field(
        default_factory=lambda: {'com': 4.05, 'shell': 1.5})
    nr_xtick_in_window = int = 5


@dataclass
class VerticalLineConfig:
    """set the location and style of the vertical lines"""
    # pylint: disable=too-many-instance-attributes
    nominal_cor: float = 2.5
    nominal_np: float = 3.0
    shell_cor_n_first_pick: float = 0.28
    shell_cor_n_2nd_pick: float = 0.48
    shell_cor_cl_pick: float = 0.33

    v_legends: dict[str, str] = \
        field(default_factory=lambda: {
            'nominal_cor': 'nominal silica',
            'nominal_np': 'nominal NP'})

    v_line_styles: dict[str, str] = \
        field(default_factory=lambda: {
            'nominal_cor': ':',
            'nominal_np': ':'})

    v_colors: dict[str, str] = \
        field(default_factory=lambda: {
            'nominal_cor': 'k',
            'nominal_np': 'brown'})


@dataclass
class AllConfig(FileConfig, VerticalLineConfig):
    """Set the all the configs"""
    if_public: bool = True
    data_sets: str = 'rdf'  # rdf or cdf

    plot_configs: OverlayConfig = field(default_factory=OverlayConfig)
    plot_verticals_single: bool = True
    plot_verticals_overlay: bool = True
    plot_verticals_window: bool = True

    def __post_init__(self) -> None:
        for dic in [self.com_files, self.shell_files]:
            for _, data in dic.items():
                fname: str = data['fname']
                data['fname'] = f'{self.data_sets}_{fname}'
        self.plot_configs.labels['ylabel'] = self.data_sets
        if self.normalize_type == 'max' and self.data_sets == 'rdf':
            self.plot_configs.labels['ylabel'] = 'g(r), a. u. '


class MultiRdfPlotter:
    """Plot multi rdf graphs on canvas"""

    info_msg: str = 'Message from MultiRdfPlotter:\n'
    configs: AllConfig
    rdf_dict: dict[str, pd.DataFrame]

    def __init__(self,
                 log: logger.logging.Logger,
                 configs: AllConfig = AllConfig()
                 ) -> None:
        self.configs = configs
        rdf_dict = self.initiate_data(log)
        self.initiate_plots(rdf_dict)
        self.write_msg(log)

    def initiate_data(self,
                      log: logger.logging.Logger
                      ) -> dict[str, pd.DataFrame]:
        """reading data and return them"""
        rdf_data: dict[str, pd.DataFrame] = {}
        plot_list: list[str] = self._get_list_for_plots(self.configs)
        for file_config in [self.configs.com_files, self.configs.shell_files]:
            for key, config in file_config.items():
                if key in plot_list:
                    rdf_data[key] = xvg_to_dataframe.XvgParser(
                        config['fname'], log, x_type=float).xvg_df
        return rdf_data

    @staticmethod
    def _get_list_for_plots(configs: AllConfig) -> list[str]:
        """make a list from the <config>_plot_list"""
        plot_list: list[str] = []
        for com_i in configs.com_plot_list:
            plot_list.append(f'com_{com_i}')
        for shell_i in configs.shell_plot_list:
            plot_list.append(f'shell_{shell_i}')
        return plot_list

    def initiate_plots(self,
                       rdf_dict: dict[str, pd.DataFrame]
                       ) -> None:
        """initiate plots"""
        for viewpoint in self.configs.viewpoint:
            sources = getattr(self.configs, f'{viewpoint}_files')
            sources = {
                key: value for key, value in sources.items() if
                key in list(rdf_dict.keys())}
            self.plot_overlay_rdf(rdf_dict, sources, viewpoint)
            self.plot_single_rdf(rdf_dict, sources, viewpoint)
            # self.plot_multirows_rdf(rdf_dict, sources, viewpoint)

    def plot_overlay_rdf(self,
                         rdf_dict: dict[str, pd.DataFrame],
                         sources: dict[str, dict[str, str]],
                         viewpoint: str
                         ) -> None:
        """
        Multiple RDFs will be plotted on the same axes, one on top of
        the other
        """
        # pylint: disable=too-many-locals
        x_max: int = -1
        first_key: str = next(iter(rdf_dict))
        x_range: tuple[float, float] = (rdf_dict[first_key]['r_nm'].iat[0],
                                        rdf_dict[first_key]['r_nm'].iat[-1])
        ax_i: plt.axes
        fig_i: plt.figure
        fig_i, ax_i = plot_tools.mk_canvas(
            x_range,
            height_ratio=self.configs.plot_configs.height_ratio,
            num_xticks=7)
        norm_factor: float = 1.0
        for s_i in sources:
            rdf_df: pd.DataFrame = rdf_dict.get(s_i)
            if rdf_df is not None:
                if (self.configs.normalize_type == 'max' and
                   self.configs.data_sets == 'rdf'):
                    col = sources[s_i].get('y_col')
                    norm_factor = rdf_df[col].max()
                ax_i = self._plot_layer(
                    ax_i, rdf_df, viewpoint, s_i, norm_factor)
            if s_i == self.configs.com_max_indicator and \
               self.configs.data_sets == 'rdf':
                max_loc: int = rdf_df[col].argmax()
                x_max = rdf_df['r_nm'][max_loc]

        self._setup_plot_labels(ax_i, viewpoint)

        legend_loc: tuple[str, str] = self._legend_locs(viewpoint)

        tag = self._get_fout_tag(viewpoint)
        fout: str = f'{self.configs.data_sets}_{viewpoint}{tag}'
        fout += f'overlay_{self.configs.plot_configs.graph_suffix}'
        if self.configs.plot_verticals_overlay and viewpoint != 'shell':
            ax_j, vlines = self._plot_vlines(ax_i)
        else:
            ax_j = ax_i

        self._save_plot(
            fig_i, ax_j, fout, viewpoint, close_fig=False, loc=legend_loc[0])
        ax_i = self._plot_window_overlay(
            ax_i, x_range, viewpoint, x_max, norm_factor)

        fout = f'window_{fout}'
        self._save_plot(
            fig_i, ax_i, fout, viewpoint, close_fig=False, loc=legend_loc[1])
        if self.configs.data_sets == 'rdf':
            if viewpoint == 'com':
                ax_i = self._plot_shadow_com(ax_i, vlines)
                fout = f'shadow_{fout}'
                self._save_plot(fig_i,
                                ax_i,
                                fout,
                                viewpoint,
                                close_fig=True,
                                loc=legend_loc[1])
            elif viewpoint == 'shell':
                ax_i = self._plot_shadow_shell(ax_i)
                fout = f'shadow_{fout}'
                self._save_plot(fig_i,
                                ax_i,
                                fout,
                                viewpoint,
                                close_fig=True,
                                loc=legend_loc[1])
        else:
            plt.close(fig_i)

    def plot_single_rdf(self,
                        rdf_dict: dict[str, pd.DataFrame],
                        sources: dict[str, dict[str, str]],
                        viewpoint: str
                        ) -> None:
        """
        Single RDF will be plotted"""
        for s_i in sources:
            ax_i: plt.axes
            fig_i: plt.figure

            rdf_df: pd.DataFrame = rdf_dict.get(s_i)
            if rdf_df is not None:
                x_range: tuple[float, float] = \
                    (rdf_df['r_nm'].iat[0], rdf_df['r_nm'].iat[0])
                fig_i, ax_i = plot_tools.mk_canvas(
                    x_range,
                    height_ratio=self.configs.plot_configs.height_ratio,
                    num_xticks=7)
                ax_i = self._plot_layer(ax_i, rdf_df, viewpoint, s_i)

                self._setup_plot_labels(ax_i, viewpoint)

                legend_loc: tuple[str, str] = self._legend_locs(viewpoint)
                y_column: str = \
                    getattr(self.configs, f'{viewpoint}_files')[s_i]['y_col']
                tag: str = f'{y_column}_single_'
                fout: str = f'{self.configs.data_sets}_{viewpoint}{tag}'
                fout += f'{self.configs.plot_configs.graph_suffix}'
                if self.configs.plot_verticals_single and viewpoint != 'shell':
                    ax_i, _ = self._plot_vlines(ax_i)
                if self.configs.data_sets == 'rdf':
                    ax_i.set_ylabel('g(r)')
                else:
                    ax_i.set_ylabel('cdf')
                self._save_plot(fig_i,
                                ax_i,
                                fout,
                                viewpoint,
                                close_fig=False,
                                loc=legend_loc[0])

    def _get_fout_tag(self,
                      viewpoint: str
                      ) -> str:
        """set the tag for the ouput png based on the plot_list"""
        config_files: dict[str, dict[str, str]] = \
            getattr(self.configs, f'{viewpoint}_files')
        plot_list: list[int] = getattr(self.configs, f'{viewpoint}_plot_list')
        tag: str = '_'
        for i in plot_list:
            tag += config_files[f'{viewpoint}_{i}']['y_col']
            tag += '_'
        return tag

    def _legend_locs(self,
                     viewpoint: str
                     ) -> tuple[str, str]:
        """get the loegend loc for each plot"""
        main_plot_loc: str = getattr(self.configs, f'{viewpoint}_legend_loc')
        window_plot_loc: str = \
            getattr(self.configs, f'{viewpoint}_window_legend_loc')
        return main_plot_loc, window_plot_loc

    def plot_multirows_rdf(self,
                           rdf_dict: dict[str, pd.DataFrame],
                           sources: dict[str, dict[str, str]],
                           viewpoint: str
                           ) -> None:
        """
        Multiple RDFs will be plotted on the same x axis and different
        y axis
        """
        # pylint: disable=unused-argument
        # pylint: disable=unused-variable
        nr_row: int = len(sources)
        first_key: str = next(iter(rdf_dict))
        x_range: tuple[float, float] = (rdf_dict[first_key]['r_nm'].iat[0],
                                        rdf_dict[first_key]['r_nm'].iat[-1])
        # print(viewpoint, len(rdf_dict), len(sources))
        rdf_dict = {k: v for k, v in rdf_dict.items() if v is not None}
        # print(viewpoint, len(rdf_dict), len(sources))
        for k in self.configs.com_plot_list:
            print(k, rdf_dict[f'{viewpoint}_{k}'])

    def _setup_plot_labels(self,
                           ax_i: plt.axes,
                           viewpoint: str
                           ) -> plt.axes:
        """set labels"""
        if not self.configs.if_public:
            ax_i.set_title(self.configs.plot_configs.titles.get(viewpoint))
        ax_i.set_xlabel(self.configs.plot_configs.labels['xlabel'])
        ax_i.set_ylabel(self.configs.plot_configs.labels['ylabel'])
        ax_i.grid(True, 'both', ls='--', color='gray', alpha=0.5, zorder=2)
        return ax_i

    def _save_plot(self,
                   fig_i: plt.figure,
                   ax_i: plt.axes,
                   fout: str,
                   viewpoint: str,
                   loc: str,  # The legend loc
                   close_fig: bool = True) -> None:
        """ Save the plot with a given fout and optionally close it """
        # pylint: disable=too-many-arguments
        self.info_msg += f'\tThe figure for `{viewpoint}` saved as `{fout}`\n'
        plot_tools.save_close_fig(
            fig_i, ax_i, fname=fout, if_close=close_fig, loc=loc)

    def _plot_window_overlay(self,
                             ax_i: plt.axes,
                             x_range: tuple[float, float],
                             viewpoint: str,
                             x_max: float = -1.0,
                             norm_factor: float = 1.0,
                             ) -> plt.axes:
        """plot the graph in the window"""
        # pylint: disable=too-many-arguments
        x_end: typing.Union[float, None] = \
            self.configs.plot_configs.second_window.get(viewpoint)
        if x_end is None:
            x_end = x_range[1]
        x_range = (x_range[0], x_end)
        xticks: list[float] = \
            np.linspace(0,
                        x_range[1],
                        self.configs.plot_configs.nr_xtick_in_window).tolist()
        if x_max != -1.0:
            xticks_width: float = xticks[1] - xticks[0]
            xticks_new: list[float] = []
            for i in range(-3, 3):
                if i != 0:
                    xticks_new.append(round((i*xticks_width + x_max), 1))
                else:
                    xticks_new.append(i*xticks_width + x_max)
            xticks_new.append(0)
        else:
            xticks_new = [np.round(item, 2) for item in xticks]
        ax_i.set_xticks(xticks_new)
        xlims: tuple[float, float] = ax_i.get_xlim()
        ax_i.set_xlim(xlims[0]/5, x_range[1])
        if norm_factor == 1.0:
            ax_i.set_ylim(-0.0, 400)
        else:
            ax_i.set_ylim(-0.0, 1.05)

        return ax_i

    def _plot_layer(self,
                    ax_i: plt.axis,
                    rdf_df: pd.DataFrame,
                    viewpoint: str,
                    s_i: str,
                    norm_factor: float = 1
                    ) -> plt.axes:
        """plot on dataset"""
        # pylint: disable=too-many-arguments

        y_column: str = \
            getattr(self.configs, f'{viewpoint}_files')[s_i]['y_col']
        ax_i.plot(rdf_df['r_nm'],
                  rdf_df[y_column] / norm_factor,
                  c=self.configs.plot_configs.colors[y_column],
                  ls=self.configs.plot_configs.line_styles[y_column],
                  label=self.configs.plot_configs.legends[y_column],
                  **self.configs.plot_configs.graph_styles)
        return ax_i

    def _plot_vlines(self,
                     ax_in: plt.axes
                     ) -> tuple[
                        plt.axes,
                        tuple[matplotlib.collections.LineCollection, ...]]:
        """plot vlines for the np"""
        ylims: tuple[float, float] = ax_in.get_ylim()
        vline1 = ax_in.vlines(x=self.configs.nominal_cor,
                              ymin=ylims[0],
                              ymax=ylims[1],
                              ls=self.configs.v_line_styles['nominal_cor'],
                              color=self.configs.v_colors['nominal_cor'])
        vline2 = ax_in.vlines(x=self.configs.nominal_np,
                              ymin=ylims[0],
                              ymax=ylims[1],
                              ls=self.configs.v_line_styles['nominal_np'],
                              color=self.configs.v_colors['nominal_np'])
        ax_in.set_ylim(ylims)
        return ax_in, (vline1, vline2)

    def _plot_shadow_com(
            self,
            ax_i: plt.axes,
            vlines: tuple[matplotlib.collections.LineCollection, ...]
            ) -> plt.axis:
        """
        Plot the shadow regions for the center of mass (COM) in the given axes.
        Returns:
            plt.axis: The modified axes object.
        """
        for vline in vlines:
            vline.remove()
        x_lims: tuple[float, float] = ax_i.get_xlim()
        y_lims: tuple[float, float] = ax_i.get_ylim()
        ax_i.fill_between(x=[x_lims[0], self.configs.nominal_cor],
                          y1=y_lims[0],
                          y2=y_lims[1],
                          color='grey',
                          alpha=0.2,
                          edgecolor=None)
        ax_i.fill_between(
            x=[self.configs.nominal_cor, self.configs.nominal_np],
            y1=y_lims[0],
            y2=y_lims[1],
            color='red',
            alpha=0.1,
            edgecolor=None)
        ax_i.grid(False, axis='both')
        ax_i.set_yticks([])
        return ax_i

    def _plot_shadow_shell(self, ax_i: plt.axes) -> plt.axis:
        """
        Plot the shadow shell region on the given axes.
        Returns:
            plt.axis: The modified axes object.
        """
        x_lims: tuple[float, float] = ax_i.get_xlim()
        y_lims: tuple[float, float] = ax_i.get_ylim()
        ax_i.fill_between(x=[x_lims[0], self.configs.shell_cor_n_first_pick],
                          y1=y_lims[0],
                          y2=y_lims[1],
                          color='grey',
                          alpha=0.2,
                          edgecolor=None)
        ax_i.fill_between(
            x=[self.configs.shell_cor_n_first_pick,
               self.configs.shell_cor_n_2nd_pick],
            y1=y_lims[0],
            y2=y_lims[1],
            color='red',
            alpha=0.1,
            edgecolor=None)
        ax_i.grid(False, axis='both')
        ax_i.set_yticks([])
        return ax_i

    def write_msg(self,
                  log: logger.logging.Logger
                  ) -> None:
        """Write and log messages."""
        print(f'{bcolors.OKCYAN}{MultiRdfPlotter.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    MultiRdfPlotter(log=logger.setup_logger('multi_rdf_plot.log'))
