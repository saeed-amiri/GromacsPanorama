"""
plot the Oda and Cl denisty on the top of each other in different styles
styles:
    - Normal
    - Normalized to one
    - Separate Y-axis
For average and rdf
The data will be read from data files
"""

from dataclasses import dataclass, field

import numpy as np
import matplotlib.pylab as plt

from common import logger
from common import xvg_to_dataframe as xvg
from common.colors_text import TextColor as bcolors


@dataclass
class BaseDataConfig:
    """basic configurations in reading the data"""
    xvg_column: dict[str, str] = field(default_factory=lambda: {
        'regions': 'regions',
        'avg_density': 'avg_density_per_region',
        'smooth': 'smoothed_rdf',
        'rdf': 'rdf_2d'
        })


@dataclass
class DataConfig(BaseDataConfig):
    """set data config"""
    selected_columns: list[str] = field(default_factory=lambda: [
        'avg_density'
        ])


@dataclass
class BasePlotConfig:
    """
    Basic configurations on the plots
    """


@dataclass
class PlotConfig(BasePlotConfig):
    """
    set configuration for plot
    """


class OverlayPlotDensities:
    """plot the overlay of densities on one plot"""

    info_msg: str = 'Messege from OverlayPlotDensities:\n'

    file_names: list[str]
    data_config: "DataConfig"
    plot_config: "PlotConfig"
    x_data: np.ndarray
    xvg_df: dict[str, np.ndarray]

    def __init__(self,
                 file_names: list[str],
                 log: logger.logging.Logger,
                 data_config: "DataConfig" = DataConfig(),
                 plot_config: "PlotConfig" = PlotConfig()
                 ) -> None:
        self.file_names = file_names
        self.data_config = data_config
        self.plot_config = plot_config
        self.xvg_df, self.x_data = self.initiate_data(log)
        self.initiate_plots(log)
        self._write_msg(log)

    def initiate_data(self,
                      log: logger.logging.Logger
                      ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """initiate reading data files"""
        x_data: np.ndarray
        xvg_df: dict[str, np.ndarray] = {}
        xvg_df, x_data = self.get_xvg_dict(log)
        return xvg_df, x_data

    def initiate_plots(self,
                       log: logger.logging.Logger
                       ) -> None:
        """plots the densities in different styles"""

    def get_xvg_dict(self,
                     log: logger.logging.Logger
                     ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """return select data from the files"""
        x_data: np.ndarray
        xvg_df: dict[str, np.ndarray] = {}

        for i, f_xvg in enumerate(self.file_names):
            fanme: str = f_xvg.split('.')[0]
            df_column: str = self.data_config.xvg_column[
                self.data_config.selected_columns[0]]
            df_i = xvg.XvgParser(f_xvg, log).xvg_df
            xvg_df[fanme] = df_i[df_column].to_numpy()
            if i == 0:
                x_data = \
                    df_i[self.data_config.xvg_column['regions']].to_numpy()
        self.info_msg += (f'\tThe file names are:\n\t\t`{self.file_names}`\n'
                          '\tThe selected columns are:\n'
                          f'\t\t`{self.data_config.selected_columns}`\n')
        return xvg_df, x_data

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    print("This module must be plot from inside of module: "
          "trajectory_oda_analysis.py\n")
