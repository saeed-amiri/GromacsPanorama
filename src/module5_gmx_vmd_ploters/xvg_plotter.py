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
    x_column: str = 'Time_ps'
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
        self.initiate_plots(log)
        self._write_msg(log)

    def initiate_plots(self,
                       log: logger.logging.Logger
                       ) -> None:
        """check the data and return them in a pd.Dataframe"""
        dfs_plot: list[pd.DataFrame] = ProccessData(self.configs, log).dfs_plot

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


class ProccessData:
    """process data based on the number of files and columns"""

    info_msg: str = 'Messeges from ProccessData:\n'
    dfs_plot: list[pd.DataFrame]

    def __init__(self,
                 configs: "XvgPlotterConfig",
                 log: logger.logging.Logger
                 ) -> None:
        self.configs: "XvgPlotterConfig" = configs
        self.dfs_plot = self.process_data(log)
        self._write_msg(log)

    def process_data(self,
                     log: logger.logging.Logger
                     ) -> list[pd.DataFrame]:
        """proccess all the xvg files"""
        xvg_dict: dict[str, pd.DataFrame] = self.get_all_xvg(log)
        return self.make_plot_df(xvg_dict, log)

    def get_all_xvg(self,
                    log: logger.logging.Logger
                    ) -> dict[str, pd.DataFrame]:
        """retrun a dict of all the xvg dataframes"""
        return {fname: xvg.XvgParser(fname, log).xvg_df for
                fname in self.configs.f_names}

    def make_plot_df(self,
                     xvg_dict: dict[str, pd.DataFrame],
                     log: logger.logging.Logger
                     ) -> list[pd.DataFrame]:
        """make a data frame of the columns of a same x_data and
        the same y_column name from different files"""
        num_files = len(self.configs.f_names)
        num_y_cols = len(self.configs.y_columns)
        self.info_msg += (
            f'\tThe xvg files are:\n\t\t`{self.configs.f_names}`\n'
            f'\tThe y columns are:\n\t\t`{self.configs.y_columns}`')

        if num_files == 1:
            return [self._process_single_file(xvg_dict)]

        if num_files > 1:
            if num_y_cols == 1:
                return [self._process_single_ycol_multi_files(xvg_dict)]
            if num_y_cols > 1:
                return self._process_multi_ycol_multi_files(xvg_dict)

        log.error(msg := '\n\tError! The y columns are not set\n')
        sys.exit(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')

    def _process_single_file(self,
                             xvg_dict: dict[str, pd.DataFrame]
                             ) -> list[pd.DataFrame]:
        """Process a single file."""
        df_f = pd.concat(xvg_dict.values(), ignore_index=True)
        return df_f[[self.configs.x_column] + self.configs.y_columns]

    def _process_single_ycol_multi_files(self,
                                         xvg_dict: dict[str, pd.DataFrame]
                                         ) -> list[pd.DataFrame]:
        """Process multiple files with a single y column."""
        y_col = self.configs.y_columns[0]
        return pd.concat({f'{y_col}-{fname}': df[y_col] for \
                         fname, df in xvg_dict.items()}, axis=1)

    def _process_multi_ycol_multi_files(self,
                                        xvg_dict: dict[str, pd.DataFrame]
                                        ) -> list[pd.DataFrame]:
        """Process multiple files with multiple y columns."""
        dfs: list[pd.DataFrame] = []
        for fname, df_i in xvg_dict.items():
            df_f: pd.DataFrame = \
                df_i[[self.configs.x_column] + self.configs.y_columns].copy()
            df_f.columns = [
                f'{col}-{fname}' if col != self.configs.x_column else
                col for col in df_f.columns]
            dfs.append(df_f)
        return dfs

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == "__main__":
    PlotXvg(log=logger.setup_logger('plot_xvg.log'))
