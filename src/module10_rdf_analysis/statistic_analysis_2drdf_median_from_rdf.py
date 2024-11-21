"""
Applying median statistics to the 2D RDF data
"""

import pandas as pd
import numpy as np

from common import logger
from common import file_writer

from module10_rdf_analysis.config import StatisticsConfig
from module10_rdf_analysis.statistic_analysis_2drdf_plots import PlotStatistics


class CalculateMedian:
    """
    Apply Median to the 2D RDF data
    """
    _slots__ = ['data', 'fit_data', 'info_msg']

    data: pd.DataFrame
    fit_data: pd.DataFrame
    info_msg: str = "Message from CalculateMedian::\n"

    def __init__(self,
                 x_data: pd.DataFrame,
                 data: pd.DataFrame,
                 fit_data: pd.DataFrame,
                 log: logger.logging.Logger,
                 config: StatisticsConfig
                 ) -> None:
        # pylint: disable=too-many-arguments
        self.x_data = x_data
        self.data = data
        self.fit_data = fit_data
        self.get_median(log, config)

    def get_median(self,
                   log: logger.logging.Logger,
                   config: StatisticsConfig
                   ) -> None:
        """
        Get the median of the data
        The median is a basic statistical measure that represents the
        middle value of a dataset when it is ordered from smallest to
        largest.
        """
        rdf_median: np.ndarray = self._calculate_median(self.x_data, self.data)
        fitted_median: np.ndarray = self._calculate_median(
            self.x_data, self.fit_data)
        median_df: pd.DataFrame = \
            self.arr_to_df(rdf_median, fitted_median)
        PlotStatistics(median_df, log, config.plots.median)
        self.write_median(median_df, log, config.files.rdf.out_fname)

    @staticmethod
    def _calculate_median(xdata: pd.DataFrame,
                          data: pd.DataFrame,
                          ) -> np.ndarray:
        """
        Get the median of the raw data
        """
        median_arr: np.ndarray = np.zeros((len(data.columns), 2))
        for idx, col in enumerate(data.columns):
            rdf: np.ndarray = np.asanyarray(data[col])
            rdf = np.sort(rdf)
            rdf /= np.max(rdf)
            median = np.median(rdf)
            # find cloest x value to the median
            median_r: float = np.abs(rdf - median).argmin()
            median_arr[idx, 0] = col
            median_arr[idx, 1] = xdata[median_r]
        return median_arr

    @staticmethod
    def arr_to_df(rdf_median: np.ndarray,
                  fitted_median: np.ndarray,
                  ) -> pd.DataFrame:
        """
        Convert the array to a DataFrame
        """
        median_df: pd.DataFrame = pd.DataFrame(
            columns=['oda', 'rdf_median', 'fitted_median'])
        median_df['oda'] = rdf_median[:, 0]
        median_df['rdf_median'] = rdf_median[:, 1]
        median_df['fitted_median'] = fitted_median[:, 1]
        # drop index and set the ODA as the index
        median_df.set_index('oda', inplace=True)
        return median_df

    def write_median(self,
                     median_df: pd.DataFrame,
                     log: logger.logging.Logger,
                     fname: str
                     ) -> None:
        """
        Write the median to a file
        """
        extra_comments: str = ('median of the 2D RDF data, for normalized data'
                               ' and fitted data, by:'
                               f' {self.__module__}')
        file_writer.write_xvg(
            median_df,
            log,
            fname=fname,
            extra_comments=extra_comments,
            xaxis_label='ODA',
            yaxis_label='Median',
            title='Median of the 2D RDF data'
        )
