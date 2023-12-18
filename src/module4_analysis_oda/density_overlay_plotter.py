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

import pandas as pd

from common import logger
from common import xvg_to_dataframe as xvg


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

    def __init__(self,
                 file_names: list[str],
                 log: logger.logging.Logger,
                 data_config: "DataConfig" = DataConfig(),
                 plot_config: "PlotConfig" = PlotConfig()
                 ) -> None:
        self.file_names = file_names
        self.data_config = data_config
        self.plot_config = plot_config
        self.initiate_data(log)

    def initiate_data(self,
                      log: logger.logging.Logger
                      ) -> None:
        """initiate reading data files"""
        xvg_df: dict[str, pd.DataFrame] = {}
        xvg_df = self.get_xvg_dict(log)

    def get_xvg_dict(self,
                     log: logger.logging.Logger
                     ) -> dict[str, pd.DataFrame]:
        """return select data from the files"""
        xvg_df: dict[str, pd.DataFrame] = {}
        for f_xvg in self.file_names:
            fanme: str = f_xvg.split('.')[0]
            df_column: str = self.data_config.xvg_column[
                self.data_config.selected_columns[0]]
            xvg_df[fanme] = xvg.XvgParser(f_xvg, log).xvg_df[df_column]
        return xvg_df


if __name__ == '__main__':
    print("This module must be plot from inside of module: "
          "trajectory_oda_analysis.py\n")
