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
        'title': 'From shell',
        'ylabel': 'g(r)',
        'xlabel': 'r [nm]'
    })

    graph_styles: dict[str, typing.Any] = field(default_factory=lambda: {
        'marker': 'o',
        'markersize': 0,
    })

    legends: dict[str, str] = \
        field(default_factory=lambda: {
            'CLA': 'Cl',
            'amino_n': 'N (APTES)',
            'oxygen': 'O (Water)',
            'oda_n': 'N (ODA)'})

    line_styles: dict[str, str] = \
        field(default_factory=lambda: {
            'CLA': '-',
            'amino_n': '--',
            'oxygen': ':',
            'oda_n': '-.'})

    colors: dict[str, str] = \
        field(default_factory=lambda: {
            'CLA': 'green',
            'amino_n': 'blue',
            'oxygen': 'red',
            'oda_n': 'green'})

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
            'com_0': {'fname': 'rdf_shell_cla.xvg', 'y_col': 'CLA'},
            'com_1': {'fname': 'rdf_shell_N.xvg', 'y_col': 'amino_n'}
            })

    shell_files: dict[str, dict[str, typing.Any]] = field(
        default_factory=lambda: {
            'shell_0': {'fname': 'rdf_shell_cla.xvg', 'y_col': 'CLA'},
            'shell_1': {'fname': 'rdf_shell_N.xvg', 'y_col': 'amino_n'}
            })


@dataclass
class OverlayConfig(BaseGraphConfig):
    """set the parameters for the overlay plots"""
    second_window: float = 1.5  # Max on the x-axis to set the window
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
        for file_config in [self.configs.com_files, self.configs.shell_files]:
            for key, config in file_config.items():
                rdf_data[key] = xvg_to_dataframe.XvgParser(
                    config['fname'], log, x_type=float).xvg_df
        return rdf_data

    def initiate_plots(self,
                       rdf_dict: dict[str, pd.DataFrame]
                       ) -> None:
        """initiate plots"""
        for viewpoint in self.configs.viewpoint:
            sources = getattr(self.configs, f'{viewpoint}_files')
            self.plot_overlay_rdf(rdf_dict, sources, viewpoint)

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
        x_range: tuple[float, float] = (list(rdf_dict[first_key]['r_nm'])[0],
                                        list(rdf_dict[first_key]['r_nm'])[-1])
        ax_i: plt.axes
        fig_i: plt.figure
        fig_i, ax_i = plot_tools.mk_canvas(
            x_range,
            height_ratio=self.configs.plot_configs.height_ratio,
            num_xticks=7)
        for s_i in sources:
            ax_i = self._plot_layer(ax_i, rdf_dict[s_i], viewpoint, s_i)

        ax_i.set_title(self.configs.plot_configs.labels['title'])
        ax_i.set_xlabel(self.configs.plot_configs.labels['xlabel'])
        ax_i.set_ylabel(self.configs.plot_configs.labels['ylabel'])
        ax_i.grid(True, 'both', ls='--', color='gray', alpha=0.5, zorder=2)
        fout: str = \
            f'{viewpoint}_overlay_{self.configs.plot_configs.graph_suffix}'
        self.info_msg += f'\tThe figure for `{viewpoint}` saved as `{fout}`\n'
        plot_tools.save_close_fig(fig_i, ax_i, fname=fout, if_close=False)

        self._plot_save_window_overlay(ax_i, x_range)
        fout = f'window_{fout}'
        self.info_msg += \
            f'\tThe figure for `{viewpoint}` in window saved as `{fout}`\n'
        plot_tools.save_close_fig(fig_i, ax_i, fname=fout, if_close=True)

    def _plot_save_window_overlay(self,
                                  ax_i: plt.axes,
                                  x_range: tuple[float, float],
                                  ) -> plt.axes:
        """plot the graph in the window"""
        x_range = (x_range[0], self.configs.plot_configs.second_window)
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
