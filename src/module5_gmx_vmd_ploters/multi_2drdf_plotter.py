"""
Plot the 2d rdf for the different ODA concentrations.
the needed files are density xvg files:
X_oda_densities.xvg
in which X is the nominal number of ODA at the interface
"""

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
    """
    set the names of the files and the layers to get average from.
    defines dict of info for files.
    """
    density_files: list[dict[str, str | int]] = \
        field(default_factory=lambda: [
            {'fname': '5_oda_densities.xvg',
             'nr_oda': 5,
             },
            {'fname': '10_oda_densities.xvg',
             'nr_oda': 10,
             },
            {'fname': '15_oda_densities.xvg',
             'nr_oda': 15,
             },
            {'fname': '20_oda_densities.xvg',
             'nr_oda': 20,
             },
            {'fname': '30_oda_densities.xvg',
             'nr_oda': 30,
             },
            {'fname': '40_oda_densities.xvg',
             'nr_oda': 40,
             },
            {'fname': '50_oda_densities.xvg',
             'nr_oda': 50,
             },
            ])
    x_data: str = 'regions'
    y_data: str = 'rdf_2d'


@dataclass
class PlotCOnfoig:
    """Set the plot configurations"""
    # pylint: disable=too-many-instance-attributes
    title: str = '2D RDF for different ODA concentrations'
    xlabel: str = 'r [nm]'
    ylabel: str = 'g(r), a.u.'
    xlim: list[float] = field(default_factory=lambda: [0, 12])
    ylim: list[float] = field(default_factory=lambda: [0, 1.05])
    legend_loc: str = 'upper right'
    legend_title: str = 'ODA/nm$^2$'
    nr_columns: int = 3
    nr_rows: int = 3


class Plot2dRdf:
    """read data and plot the 2d rdf"""

    __slots__ = ['info_msg', 'config', 'data', 'plot_config']

    info_msg: str
    config: FileConfig
    data: pd.DataFrame
    plot_config: PlotCOnfoig

    def __init__(self,
                 log: logger.logging.Logger,
                 config: FileConfig = FileConfig(),
                 plot_config: PlotCOnfoig = PlotCOnfoig()
                 ) -> None:
        self.info_msg = 'Message from Plot2dRdf:\n'
        self.config = config
        self.plot_config = plot_config
        self.read_data(log)

    def read_data(self,
                  log: logger.logging.Logger
                  ) -> None:
        """read the data"""
        data: dict[str, pd.DataFrame] = {}
        for i, file_info in enumerate(self.config.density_files):
            fname = file_info['fname']
            nr_oda = file_info['nr_oda']
            df_i: pd.DataFrame = \
                xvg_to_dataframe.XvgParser(fname, log, x_type=float).xvg_df
            if i == 0:
                data['regions'] = df_i[self.config.x_data]
            data[str(nr_oda)] = df_i[self.config.y_data]

        self.data = pd.concat(data, axis=1)


if __name__ == '__main__':
    Plot2dRdf(log=logger.setup_logger('multi_2d_rdf_plotter.log'))
