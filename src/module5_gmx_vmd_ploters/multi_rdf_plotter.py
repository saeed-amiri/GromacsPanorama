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
import matplotlib.pylab as plt

from common import logger, plot_tools, xvg_to_dataframe
from common.colors_text import TextColor as bcolors


@dataclass
class BaseGraphConfig:
    """Basic setups for graphs"""

    # pylint: disable=too-many-instance-attributes
    graph_suffix: str = 'gmx.png'
    y_col_name: str = 'density'
    xcol_name: str = 'r_nm'

    labels: dict[str, str] = field(default_factory=lambda: {
        'ylabel': 'g(r)',
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
            'amino_charge': r'H$^+$ (APTES)',
            'SOL': 'Water',
            'D10': 'Decane',
            'APT': 'APTES',
            'POT': 'Na',
            'sol_oxygen': 'O (water)',
            'ODN': 'ODA'})

    line_styles: dict[str, str] = \
        field(default_factory=lambda: {
            'CLA': '-',
            'amino_n': '--',
            'amino_charge': '--',
            'SOL': ':',
            'D10': ':',
            'APT': '-',
            'POT': '-',
            'sol_oxygen': '--',
            'ODN': '-.'})

    colors: dict[str, str] = \
        field(default_factory=lambda: {
            'CLA': 'green',
            'amino_n': 'blue',
            'amino_charge': 'blue',
            'SOL': 'red',
            'D10': 'grey',
            'APT': 'k',
            'POT': 'brown',
            'sol_oxygen': 'blue',
            'ODN': 'green'})

    height_ratio: float = (5 ** 0.5 - 1) * 1.5

    y_unit: str = ''

    legend_loc: str = 'lower right'


@dataclass
class FileConfig:
    """setups for the input files names and their columns name"""
    # comfile: files which are the viewpoint are the com of the np
    # shellfile: files which are the viewpoint are the shell of the np
    viewpoint: list[str] = field(default_factory=lambda: ['com', 'shell'])
    com_files: dict[str, dict[str, typing.Any]] = field(
        default_factory=lambda: {
            'com_0': {'fname': 'rdf_com_o_sol.xvg', 'y_col': 'sol_oxygen'},
            'com_1': {'fname': 'rdf_com_d10.xvg', 'y_col': 'D10'},
            'com_2': {'fname': 'rdf_com_sol.xvg', 'y_col': 'SOL'},
            'com_3': {'fname': 'rdf_com_odn.xvg', 'y_col': 'ODN'},
            'com_4': {'fname': 'rdf_com_pot.xvg', 'y_col': 'POT'},
            'com_5': {'fname': 'rdf_com_cla.xvg', 'y_col': 'CLA'},
            'com_6': {'fname': 'rdf_com_n.xvg', 'y_col': 'amino_n'},
            'com_7': {'fname': 'rdf_com_apt.xvg', 'y_col': 'APT'},
            'com_8': {'fname': 'rdf_com_amino_charge.xvg',
                      'y_col': 'amino_charge'}
            })
    com_plot_list: list[int] = field(default_factory=lambda: [0, 1, 2, 3])

    shell_files: dict[str, dict[str, typing.Any]] = field(
        default_factory=lambda: {
            'shell_0': {'fname': 'rdf_shell_cla.xvg', 'y_col': 'CLA'},
            'shell_1': {'fname': 'rdf_shell_N.xvg', 'y_col': 'amino_n'},
            'shell_2': {'fname': 'rdf_shell_sol.xvg', 'y_col': 'SOL'},
            'shell_3': {'fname': 'rdf_shell_odn.xvg', 'y_col': 'ODN'}
            })
    shell_plot_list: list[int] = field(default_factory=lambda: [0, 2, 3])


@dataclass
class OverlayConfig(BaseGraphConfig):
    """set the parameters for the overlay plots"""
    # Max on the x-axis to set the window
    second_window: dict[str, float] = field(
        default_factory=lambda: {'com': 4.2, 'shell': 1.5})
    nr_xtick_in_window = int = 4


@dataclass
class AllConfig(FileConfig):
    """Set the all the configs"""
    plot_configs: OverlayConfig = field(default_factory=OverlayConfig)


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
        plot_list: list[str] = self._get_data_for_plots(self.configs)
        for file_config in [self.configs.com_files, self.configs.shell_files]:
            for key, config in file_config.items():
                if key in plot_list:
                    rdf_data[key] = xvg_to_dataframe.XvgParser(
                        config['fname'], log, x_type=float).xvg_df
        return rdf_data

    @staticmethod
    def _get_data_for_plots(configs: AllConfig) -> list[str]:
        """make a list from the <config>_plot_list"""
        plot_list: list[str] = []
        for com_i, shell_i in zip(configs.com_plot_list,
                                  configs.shell_plot_list):
            plot_list.append(f'com_{com_i}')
            plot_list.append(f'shell_{shell_i}')
        return plot_list

    def initiate_plots(self,
                       rdf_dict: dict[str, pd.DataFrame]
                       ) -> None:
        """initiate plots"""
        for viewpoint in self.configs.viewpoint:
            sources = getattr(self.configs, f'{viewpoint}_files')
            self.plot_overlay_rdf(rdf_dict, sources, viewpoint)
            self.plot_multirows_rdf(rdf_dict, sources, viewpoint)

    def plot_overlay_rdf(self,
                         rdf_dict: dict[str, pd.DataFrame],
                         sources: dict[str, dict[str, str]],
                         viewpoint: str
                         ) -> None:
        """
        Multiple RDFs will be plotted on the same axes, one on top of
        the other
        """
        first_key: str = next(iter(rdf_dict))
        x_range: tuple[float, float] = (rdf_dict[first_key]['r_nm'].iat[0],
                                        rdf_dict[first_key]['r_nm'].iat[-1])
        ax_i: plt.axes
        fig_i: plt.figure
        fig_i, ax_i = plot_tools.mk_canvas(
            x_range,
            height_ratio=self.configs.plot_configs.height_ratio,
            num_xticks=7)
        for s_i in sources:
            rdf_df: pd.DataFrame = rdf_dict.get(s_i)
            if rdf_df is not None:
                ax_i = self._plot_layer(ax_i, rdf_df, viewpoint, s_i)
        self._setup_plot_labels(ax_i, viewpoint)
        tag: typing.Union[str, None] = \
            getattr(self.configs, f'{viewpoint}_fout_prefix')
        tag = self._get_fout_tag(viewpoint)
        fout: str = \
            f'{tag}overlay_{self.configs.plot_configs.graph_suffix}'
        self._save_plot(fig_i, ax_i, fout, viewpoint, close_fig=False)

        self._plot_save_window_overlay(ax_i, x_range, viewpoint)
        fout = f'window_{fout}'
        self._save_plot(fig_i, ax_i, fout, viewpoint, close_fig=True)

    def _get_fout_tag(self,
                      viewpoint: str
                      ) -> str:
        """set the tag for the ouput png based on the plot_list"""
        config_files: dict[str, dict[str, str]] = \
            getattr(self.configs, f'{viewpoint}_files')
        plot_list: list[int] = getattr(self.configs, f'{viewpoint}_plot_list')
        tag: str = ''
        for i in plot_list:
            tag += config_files[f'{viewpoint}_{i}']['y_col']
            tag += '_'
        return tag

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

    def _setup_plot_labels(self,
                           ax_i: plt.axes,
                           viewpoint: str
                           ) -> plt.axes:
        """set labels"""
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
                   close_fig: bool = True) -> None:
        """ Save the plot with a given fout and optionally close it """
        # pylint: disable=too-many-arguments
        self.info_msg += f'\tThe figure for `{viewpoint}` saved as `{fout}`\n'
        plot_tools.save_close_fig(
            fig_i, ax_i, fname=fout, if_close=close_fig)

    def _plot_save_window_overlay(self,
                                  ax_i: plt.axes,
                                  x_range: tuple[float, float],
                                  viewpoint: str
                                  ) -> plt.axes:
        """plot the graph in the window"""
        x_end: typing.Union[float, None] = \
            self.configs.plot_configs.second_window.get(viewpoint)
        if x_end is None:
            x_end = x_range[1]
        x_range = (x_range[0], x_end)
        xticks: list[float] = \
            np.linspace(x_range[0],
                        x_range[1],
                        self.configs.plot_configs.nr_xtick_in_window).tolist()
        ax_i.set_xticks(xticks)
        ax_i.set_xlim(x_range)
        return ax_i

    def _plot_layer(self,
                    ax_i: plt.axis,
                    rdf_df: pd.DataFrame,
                    viewpoint: str,
                    s_i: str
                    ) -> plt.axes:
        """plot on dataset"""
        y_column: str = \
            getattr(self.configs, f'{viewpoint}_files')[s_i]['y_col']
        ax_i.plot(rdf_df['r_nm'],
                  rdf_df[y_column],
                  c=self.configs.plot_configs.colors[y_column],
                  ls=self.configs.plot_configs.line_styles[y_column],
                  label=self.configs.plot_configs.legends[y_column],
                  **self.configs.plot_configs.graph_styles)
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
