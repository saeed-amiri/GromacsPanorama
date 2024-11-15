"""
Read rdf data from files

"""

import pandas as pd

from common import logger
from common import xvg_to_dataframe

from module10_rdf_analysis.config import StatisticsConfig

class ReadData:
    """Read the data from the files"""
    # pylint: disable=too-few-public-methods
    _slots__ = ['xdata', 'data', 'fit_data', 'config']

    xdata: pd.Series
    data: pd.DataFrame
    fit_data: pd.DataFrame
    config: StatisticsConfig

    def __init__(self,
                 config: StatisticsConfig,
                 log: logger.logging.Logger
                 ) -> None:
        self.config = config
        self.read_data(log)

    def read_data(self,
                  log: logger.logging.Logger
                  ) -> None:
        """read the data"""
        data: StatisticsConfig[str, pd.Series] = {}
        fit_data: dict[str, pd.Series] = {}
        xdata: str = self.config.files.xdata
        ydata: str = self.config.files.ydata
        fitted_data: str = self.config.files.fitted_data

        for i, (oda, fname) in enumerate(
           self.config.files['file_names'].items()):
            nr_oda = oda
            df_i: pd.DataFrame = \
                xvg_to_dataframe.XvgParser(fname, log, x_type=float).xvg_df
            if i == 0:
                xdata = df_i[xdata] * 0.1  # nm
            data[str(nr_oda)] = df_i[ydata]
            fit_data[str(nr_oda)] = df_i[fitted_data]

        self.data = pd.concat(data, axis=1)
        self.fit_data = pd.concat(fit_data, axis=1)
        self.xdata = xdata


class ProcessData(ReadData):
    """Process the data"""
    # pylint: disable=too-few-public-methods
    __slots__ = ['config', 'data', 'fit_data']

    def __init__(self,
                 config: StatisticsConfig,
                 log: logger.logging.Logger
                 ) -> None:
        super().__init__(config, log)
        self.config = config
        self.process_data()

    def process_data(self) -> None:
        """Process the data"""
        for i in self.data.columns:
            self.data[i] /= self.data[i].max()
            self.fit_data[i] /= self.fit_data[i].max()
