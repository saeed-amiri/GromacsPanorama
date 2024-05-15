"""
Reading data from a file
"""

import numpy as np
import pandas as pd

from module12_experimental_lab_data.config_classes import AllConfig

from common.colors_text import TextColor as bcolors
from common import logger, xvg_to_dataframe


class ReadData:
    """Read data from a file"""

    info_msg: str = 'Message from ReadData:\n'
    data: pd.DataFrame

    def __init__(self,
                 log: logger.logging.Logger,
                 config: AllConfig = AllConfig()
                 ) -> None:
        self.config = config
        data = self.read_data(log)
        self.data = self.add_log_x_scale_column(data)
        self._replace_nan()
        self.write_msg(log)

    def read_data(self,
                  log: logger.logging.Logger
                  ) -> pd.DataFrame:
        """Read lines from the data file"""
        data: pd.DataFrame = \
            xvg_to_dataframe.XvgParser(self.config.data_file, log=log).xvg_df
        return data

    def add_log_x_scale_column(self,
                               data: pd.DataFrame
                               ) -> pd.DataFrame:
        """Add a column to the data"""
        add_columns_based: list[str] = ['surfactant_concentration_mM_L',
                                        'salt_concentration_mM_L']
        for column in add_columns_based:
            log_values = np.log10(data[column])
            log_values[np.isinf(log_values)] = self.config.inf_replacement
            data[f'log_{column}'] = log_values
            self.info_msg += \
                ('\tlog_surfactant_concentration_mM_L column added\n'
                 f'\tinf replaced with {self.config.inf_replacement}\n')
        return data

    def _replace_nan(self) -> None:
        """Replace NaN values in the data"""
        self.data.fillna(self.config.nan_replacement, inplace=True)
        self.info_msg += \
            (f'\tNaN values replaced with {self.config.nan_replacement}\n')

    def write_msg(self,
                  log: logger.logging.Logger
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{ReadData.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    # Read the input file
    ReadData(logger.setup_logger('read_lab_date.log'))
