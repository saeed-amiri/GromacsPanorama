"""
Reading data from a file
"""

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
        self.data = self.read_data(log)
        self.write_msg(log)

    def read_data(self,
                  log: logger.logging.Logger
                  ) -> pd.DataFrame:
        """Read lines from the data file"""
        data: pd.DataFrame = \
            xvg_to_dataframe.XvgParser(self.config.data_file, log=log).xvg_df
        return data

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
