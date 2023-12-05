"""must be updated
for PRE
"""

from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt

from common import logger
from common import my_tools
from common import plot_tools
from common import static_info as stinfo
from common import xvg_to_dataframe as xvg
from common.colors_text import TextColor as bcolors


@dataclass
class XvgBaseConfig:
    """Patameters for the plotting the xvg files"""
    f_names: list[str]  # Names of the xvg files
    x_column: str  # Name of the x data
    y_columns: list[str]  # Names of the second columns to plots
    out_suffix: str = 'xvg.png'  # Suffix for the output file


@dataclass
class XvgPlotterConfig(XvgBaseConfig):
    """set the parameters"""
    f_names: list[str] = field(default_factory=lambda: ['coord.xvg'])
    x_column: str = 'frame'
    y_columns: list[str] = field(default_factory=lambda: ['COR_APT_Z'])


class PlotXvg:
    """plot xvg here"""

    info_msg: str = 'Messeges from PlotXvg:\n'

    def __init__(self,
                 log: logger.logging.Logger,
                 configs: "XvgPlotterConfig" = XvgPlotterConfig()
                 ) -> None:
        self.configs = configs
        self.initiate_data(log)
        self._write_msg(log)

    def initiate_data(self,
                      log: logger.logging.Logger
                      ) -> None:
        """check the data and return them in a pd.Dataframe"""
        self.file_existence(self.configs.f_names, log)

    @staticmethod
    def file_existence(f_names: list[str],
                       log: logger.logging.Logger
                       ) -> None:
        """check if the files exist"""
        for fname in f_names:
            my_tools.check_file_exist(fname, log)

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == "__main__":
    PlotXvg(log=logger.setup_logger('plot_xvg.log'))
