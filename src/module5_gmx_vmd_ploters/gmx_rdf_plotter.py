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
Opt. by ChatGPt
Saeed
19 Jan 2024
"""

import typing
from dataclasses import dataclass, field

import pandas as pd
import matplotlib.pyplot as plt

from common import logger, plot_tools, xvg_to_dataframe
from common.colors_text import TextColor as bcolors


@dataclass
class BaseConfig:
    """
    Basic configurations and setup for the plots.
    """
    # pylint: disable=too-many-instance-attributes
    graph_suffix: str = 'gmx.png'
    ycol_name: str = 'density'
    xcol_name: str = 'r_nm'

    labels: dict[str, str] = field(default_factory=lambda: {
        'title': 'Computed density',
        'ylabel': 'g(r)',
        'xlabel': 'r [nm]'
    })

    graph_styles: dict[str, typing.Any] = field(default_factory=lambda: {
        'label': 'density',
        'color': 'black',
        'marker': 'o',
        'linestyle': '-',
        'markersize': 0,
    })

    line_styles: list[str] = \
        field(default_factory=lambda: ['-', ':', '--', '-.'])
    colors: list[str] = \
        field(default_factory=lambda: ['black', 'red', 'blue', 'green'])

    height_ratio: float = (5 ** 0.5 - 1) * 1.5

    y_unit: str = ''

    legend_loc: str = 'lower right'


@dataclass
class RdfComConfig(BaseConfig):
    """
    Set parameters for the plotting rdf from com of nanoparticle
    """
    def __post_init__(self) -> None:
        self.graph_suffix: str = 'rdf_com_gmx.png'
        self.ycol_name: str = 'CLA'
        self.xcol_name: str = 'r_nm'
        self.labels['title'] = 'Rdf from COM of NP'


@dataclass
class RdfOutConfig(BaseConfig):
    """
    Set parameters for the plotting rdf from outermost of nanoparticle
    """
    def __post_init__(self) -> None:
        self.graph_suffix: str = 'rdf_out_gmx.png'
        self.ycol_name: str = 'CLA'
        self.xcol_name: str = 'r_nm'
        self.labels['title'] = 'Rdf from COM of NP'


@dataclass
class CdfComConfig(BaseConfig):
    """
    Set parameters for the plotting cdf from com of nanoparticle
    """
    def __post_init__(self) -> None:
        self.graph_suffix: str = 'cdf_com_gmx.png'
        self.ycol_name: str = 'CLA'
        self.xcol_name: str = 'r_nm'
        self.labels['title'] = 'Cdf from COM of NP'


@dataclass
class CdfOutConfig(BaseConfig):
    """
    Set parameters for the plotting cdf from outermost of nanoparticle
    """
    def __post_init__(self) -> None:
        self.graph_suffix: str = 'cdf_out_gmx.png'
        self.ycol_name: str = 'CLA'
        self.xcol_name: str = 'r_nm'
        self.labels['title'] = 'Cdf from OUT of NP'


@dataclass
class FileInConfig:
    """
    Set the names of the input files files
    """
    fnames: dict[str, dict[str, str]] = field(default_factory=lambda: {
            'com': {'rdf': 'gmx_rdf_cla_com.xvg', 'cdf': 'gmx_cdf_cla_com.xvg'}
        })


@dataclass
class ParameterCofig:
    """
    Set the constant and other parameters for the plots
    """


@dataclass
class AllConfig(FileInConfig, ParameterCofig):
    """
    Consolidates all configurations for different graph types.
    """
    rdf_com_config: RdfComConfig = field(default_factory=RdfComConfig)
    cdf_com_config: CdfComConfig = field(default_factory=CdfComConfig)
    rdf_out_config: RdfOutConfig = field(default_factory=RdfOutConfig)
    cdf_out_config: CdfOutConfig = field(default_factory=CdfOutConfig)


class PlotRdfCdf:
    """
    Plot rdf and cdf from gromacs
    """

    info_msg: str = 'Message from PlotRdfCdf:\n'
    configs: "AllConfig"

    def __init__(self,
                 log: logger.logging.Logger,
                 configs: "AllConfig" = AllConfig()
                 ) -> None:
        self.configs = configs
        rdf_cdf_data: dict[str, dict[str, pd.DataFrame]] = \
            self.initiate_data(log)
        self.initiate_plots(rdf_cdf_data)
        self.write_msg(log)

    def initiate_data(self,
                      log: logger.logging.Logger
                      ) -> dict[str, dict[str, pd.DataFrame]]:
        """Read files and return the data in dataframes format."""
        rdf_cdf_data: dict[str, dict[str, pd.DataFrame]] = {}
        for calc_type, files in self.configs.fnames.items():
            rdf_cdf_data[calc_type] = {}
            for data_type, fname in files.items():
                rdf_cdf_data[calc_type][data_type] = \
                    xvg_to_dataframe.XvgParser(fname, log, x_type=float).xvg_df
        return rdf_cdf_data

    def initiate_plots(self,
                       rdf_cdf_data: dict[str, dict[str, pd.DataFrame]]
                       ) -> None:
        """Create the plots based on the provided data."""
        for calc_type, data in rdf_cdf_data.items():
            for data_type, df_i in data.items():
                if data_type == 'rdf':
                    config = getattr(self.configs, f'rdf_{calc_type}_config')
                else:
                    config = getattr(self.configs, f'cdf_{calc_type}_config')
                self._plot_graph(calc_type, data_type, df_i, config)

    def _plot_graph(self,
                    calc_type: str,
                    data_type: str,
                    data: pd.DataFrame,
                    config: BaseConfig) -> None:
        """Plot the RDF or CDF data."""
        x_range: tuple[float, float] = \
            (min(data[config.xcol_name]), max(data[config.xcol_name]))

        fig_i: plt.figure
        ax_i: plt.axes
        fig_i, ax_i = plot_tools.mk_canvas(x_range,
                                           height_ratio=config.height_ratio,
                                           num_xticks=7)
        ax_i.plot(data[config.xcol_name],
                  data[config.ycol_name],
                  **config.graph_styles)
        ax_i.set_xlabel(config.labels['xlabel'])
        ax_i.set_ylabel(config.labels['ylabel'])
        ax_i.set_title(config.labels['title'])

        ax_i.grid(True, 'both', ls='--', color='gray', alpha=0.5, zorder=2)

        plot_tools.save_close_fig(fig_i,
                                  ax_i,
                                  fname := (f'{calc_type}_{data_type}_'
                                            f'{config.graph_suffix}'))
        self.info_msg += \
            f'\tThe plot for `{calc_type}_{data_type}` is saved as `{fname}`\n'

    def write_msg(self,
                  log: logger.logging.Logger
                  ) -> None:
        """Write and log messages."""
        print(f'{bcolors.OKCYAN}{PlotRdfCdf.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    PlotRdfCdf(log=logger.setup_logger('plot_gmx_rdf.log'))
