"""
Plotting output files from GROMACS (xvg) or self prepared xvg files,
output from vmd and also other simple data files.
It should supprt multi files, and multi columns plotting.
still thinking about the structures...
"""

import sys
from dataclasses import dataclass, field

import pandas as pd
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
    f_names: list[str] = field(
        default_factory=lambda: ['coord.xvg', 'coord_cp.xvg'])
    x_column: str = 'frame'
    y_columns: list[str] = field(
        default_factory=lambda: ['COR_APT_Z', 'COR_APT_X'])


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
        self.make_plot_df()

    def make_plot_df(self) -> pd.DataFrame:
        """make a data frame of the columns of a same x_data and
        the same y_column name from different files"""
        if (l_f := len(self.configs.f_names)) == 1:
            columns: list[str] = [self.configs.x_column]
            columns.extend(self.configs.y_columns)
        elif l_f > 1:
            if (l_y := len(self.configs.y_columns)) == 1:
                columns = [self.configs.x_column]
                columns.extend(
                    f'{fname}-{self.configs.y_columns[0]}' for
                    fname in self.configs.f_names)
            elif l_y > 1:
                for col in range(l_y):
                    print(f'columns_{col}')

    @staticmethod
    def file_existence(f_names: list[str],
                       log: logger.logging.Logger
                       ) -> None:
        """check if the files exist"""
        if f_names:
            for fname in f_names:
                my_tools.check_file_exist(fname, log)
        else:
            log.error(
                msg := '\n\tError!The list of the file names is empty!\n')
            sys.exit(f'{bcolors.FAIL}{msg}{bcolors.ENDC}\n')

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == "__main__":
    PlotXvg(log=logger.setup_logger('plot_xvg.log'))
