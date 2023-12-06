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
from common import plot_tools
from common import xvg_to_dataframe as xvg
from common.colors_text import TextColor as bcolors


@dataclass
class XvgBaseConfig:
    """Patameters for the plotting the xvg files"""
    f_names: list[str]  # Names of the xvg files
    x_column_labels: dict[str, str]
    y_columns_labels: dict[str, str]
    x_column: str  # Name of the x data
    y_columns: list[str]  # Names of the second columns to plots
    x_axis_label: str = 'frame index'
    y_axis_label: str = r'$\Delta$X'
    out_suffix: str = 'xvg.png'  # Suffix for the output file
    line_colors: list[str] = \
        field(default_factory=lambda: ['k', 'r', 'b', 'g', 'y'])
    line_style: list[str] = \
        field(default_factory=lambda: ['-', '--', ':', '-.'])
    subtract_initial: bool = False
    frames_to_time: bool = False
    frames_to_time_ratio: float = 0.001


@dataclass
class XvgPlotterConfig(XvgBaseConfig):
    """set the parameters"""
    f_names: list[str] = field(
        default_factory=lambda: ['coord.xvg'])
    x_column_labels: dict[str, str] = \
        field(default_factory=lambda: {'Time_ps': 'time [ns]'})
    y_columns_labels: dict[str, str] = \
        field(default_factory=lambda: {'COR_APT_Z': r'np$_{com,z}$',
                                       'COR_APT_X': r'np$_{com,x}$'})
    x_column: str = field(init=False)
    y_columns: list[str] = field(init=False)
    subtract_initial: bool = True
    frames_to_time: bool = True

    def __post_init__(self):
        self.x_column = list(self.x_column_labels.keys())[0]
        self.y_columns = list(self.y_columns_labels.keys())


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
        self.plot_xvg(dfs_plot)

    def plot_xvg(self,
                 dfs_plot: list[pd.DataFrame]
                 ) -> None:
        """plot the processed xvg files"""
        num_files: int = len(self.configs.f_names)
        num_y_cols: int = len(self.configs.y_columns)
        if num_files == 1:
            self._plot_single_file(dfs_plot)

    def _plot_single_file(self,
                          dfs_plot: list[pd.DataFrame]
                          ) -> None:
        """plot and save figure when theere is only one file"""
        df_i: pd.DataFrame = dfs_plot[0]
        xrange: tuple[float, float] = (min(df_i[self.configs.x_column]),
                                       max(df_i[self.configs.x_column]))
        fig_i: plt.figure
        ax_i: plt.axes
        fig_i, ax_i = plot_tools.mk_canvas(xrange, height_ratio=(5**0.5-1)*1.8)
        fout: str = (f'{self.configs.f_names[0].split(".")[0]}-'
                     f'{self.configs.out_suffix}')
        for i, col in enumerate(df_i.iloc[:, 1:]):
            ax_i.plot(df_i[self.configs.x_column],
                      df_i[col],
                      label=self.configs.y_columns_labels[col],
                      color=self.configs.line_colors[i])
        ax_i.set_xlabel(self.configs.x_axis_label)
        ax_i.set_ylabel(self.configs.y_axis_label)
        ax_i.grid(True, linestyle='--', color='gray', alpha=0.5)
        plot_tools.save_close_fig(fig_i, ax_i, fname=fout)
        self.info_msg += (f'\tThe fig for {self.configs.f_names} is saved '
                         f'with name {fout}\n')

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
        dfs_plot: list[pd.DataFrame] = self.make_plot_df(xvg_dict, log)
        dfs_plot = self.set_time(dfs_plot)
        return self.subtract_initial(dfs_plot)

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
        num_files: int = len(self.configs.f_names)
        num_y_cols: int = len(self.configs.y_columns)
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

    def set_time(self,
                 dfs_plot: list[pd.DataFrame]
                 ) -> list[pd.DataFrame]:
        """convert index frames to time"""
        if self.configs.frames_to_time:
            updated_dfs = []
            for df_i in dfs_plot:
                df_copy = df_i.copy()
                df_copy[self.configs.x_column] *= \
                    self.configs.frames_to_time_ratio
                updated_dfs.append(df_copy)
            return updated_dfs
        return dfs_plot

    def subtract_initial(self,
                         dfs_plot: list[pd.DataFrame]
                         ) -> list[pd.DataFrame]:
        """
        Subtract the initial value of specific columns from those
        columns.
        """
        if self.configs.subtract_initial:
            updated_dfs = []
            for df_i in dfs_plot:
                df_copy = df_i.copy()
                for col in self.configs.y_columns:
                    initial_value = df_copy[col].iloc[0]
                    df_copy[col] = df_copy[col] - initial_value

                updated_dfs.append(df_copy)
            return updated_dfs
        return dfs_plot

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
        return pd.concat({f'{y_col}-{fname}': df[y_col] for
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
