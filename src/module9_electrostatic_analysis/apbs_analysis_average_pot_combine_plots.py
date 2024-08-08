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
import matplotlib as mpl

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
                'legend_loc': 'upper right',
                }

    @property
    def RDF_LS(self) -> str:
        """set the line style for RDF"""
        return '-'

    @property
    def RDF_COLOR(self) -> str:
        """set the color for RDF"""
        return 'darkblue'

    @property
    def RDF_LABEL(self) -> str:
        """set the label for RDF"""
        return 'RDF'

    @property
    def BOLTZMAN_LS(self) -> str:
        """set the line style for Boltzman factor"""
        return ':'

    @property
    def BOLTZMAN_COLOR(self) -> str:
        """set the color for Boltzman factor"""
        return 'darkred'

    @property
    def BOLTZMAN_LABEL(self) -> str:
        """set the label for Boltzman factor"""
        return 'Boltzman factor'


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
        self.plot_data(log)

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
        return data, radii

    def plot_data(self,
                  log: logger.logging.Logger
                  ) -> None:
        """plot the data"""
        fig, ax = plt.subplots()
        self.plot_rdf(ax)
        self.plot_boltzman(ax)
        # self.set_plot_properties(ax)
        plt.show()

    def plot_rdf(self,
                 ax: mpl.axes.Axes
                 ) -> None:
        """plot the RDF"""
        ax.plot(self.rdf_radii, self.rdf_data)

    def plot_boltzman(self,
                      ax: mpl.axes.Axes
                      ) -> None:
        """plot the Boltzman factor"""
        ax.plot(self.boltzman_radii, self.boltzman_data,)


if __name__ == '__main__':
    PlotBolzmannRdf(logger.setup_logger('combine_plots.log'))
