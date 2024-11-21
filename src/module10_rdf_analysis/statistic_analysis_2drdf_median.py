"""
Read and analyze the median from all the rdf and fitted data
"""

import numpy as np
import pandas as pd
from scipy import stats

from common import logger
from common import xvg_to_dataframe
from common.colors_text import TextColor as bcolors

from module10_rdf_analysis.config import StatisticsConfig


class MedianAnalysis:
    """
    Read and analyze the median data
    """

    __slots__ = ['config', 'info_msg']

    def __init__(self,
                 config: StatisticsConfig,
                 log: logger.logging.Logger
                 ) -> None:
        self.info_msg: str = "Message from MedianAnalysis:\n"
        self.config = config
        self.process_median(log)
        self._write_msg(log)

    def process_median(self,
                       log: logger.logging.Logger
                       ) -> None:
        """
        Read the median data
        """
        rdf_median: pd.DataFrame
        fitted_median: pd.DataFrame
        rdf_median, fitted_median = \
            self.read_medians(log, self.config.files.medians)
        rdf_median = self.clean_df(rdf_median)
        fitted_median = self.clean_df(fitted_median)

    def read_medians(self,
                     log: logger.logging.Logger,
                     f_config: StatisticsConfig
                     ) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
        """
        Read the median data
        """
        rdf_median: dict[str, pd.Series] = {}
        fitted_median: dict[str, pd.Series] = {}
        for oda, fname in f_config.medians_files.items():
            df_i: pd.DataFrame = \
                xvg_to_dataframe.XvgParser(fname, log, x_type=float).xvg_df
            ydata: pd.Series = df_i[f_config.rdf_median_ydata]
            ydata = self.clean_df(ydata)
            rdf_median[str(oda)] = ydata
            # print(oda, df_i, rdf_median)
            fitted_median[str(oda)] = df_i[f_config.fitted_median_ydata]
        return rdf_median, fitted_median

    @staticmethod
    def clean_df(data: pd.Series | pd.DataFrame) -> pd.Series:
        """
        Clean the data
        """
        # drop NaN values
        data = data.dropna()
        # remove outliers
        data = data[(np.abs(stats.zscore(data)) < 3)]
        return data

    def _write_msg(self, log: logger.logging.Logger) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)
