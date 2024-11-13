"""
Read rdf data from files

"""

import pandas as pd
from common import xvg_to_dataframe
from common import logger


class ReadData:
    """Read the data from the files"""
    # pylint: disable=too-few-public-methods
    _slots__ = ['config', 'data', 'fit_data']

    def __init__(self,
                 config: dict,
                 log: logger.logging.Logger
                 ) -> None:
        self.config = config
        self.read_data(log)

    def read_data(self,
                  log: logger.logging.Logger
                  ) -> None:
        """read the data"""
        data: dict[str, pd.Series] = {}
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
                data[xdata] = df_i[xdata] * 0.1  # nm
                fit_data[xdata] = df_i[xdata] * 0.1  # nm
            data[str(nr_oda)] = df_i[ydata]
            fit_data[str(nr_oda)] = df_i[fitted_data]

        self.data = pd.concat(data, axis=1)
        self.fit_data = pd.concat(fit_data, axis=1)
