"""
Applying noraml statistics to the 2D RDF data
"""

import pandas as pd
import numpy as np

from common import logger


class NormalStatistics:
    """
    Apply normal statistics to the 2D RDF data
    """
    _slots__ = ['xdata', 'data', 'fit_data', 'info_msg']

    xdata: pd.Series
    data: pd.DataFrame
    fit_data: pd.DataFrame
    info_msg: str = "Message from NormalStatistics:\n"

    def __init__(self,
                 xdata: pd.Series,
                 data: pd.DataFrame,
                 fit_data: pd.DataFrame,
                 log: logger.logging.Logger
                 ) -> None:
        self.xdata = xdata
        self.data = data
        self.fit_data = fit_data
        self.normal_tests(log)

    def normal_tests(self,
                     log: logger.logging.Logger
                     ) -> None:
        """
        Test the normality of the data
        """
        self.get_median(log)

    def get_median(self,
                   log: logger.logging.Logger
                   ) -> None:
        """
        Get the median of the data
        The median is a basic statistical measure that represents the
        middle value of a dataset when it is ordered from smallest to
        largest.
        """
        rdf_median: np.ndarray = self._calculate_median(self.data)
        fitted_median: np.ndarray = self._calculate_median(self.fit_data)

    @staticmethod
    def _calculate_median(data: pd.DataFrame,
                          ) -> np.ndarray:
        """
        Get the median of the raw data
        """
        print('data shape:', len(data.columns))
        median_arr: np.ndarray = np.zeros((len(data.columns), 2))
        for idx, col in enumerate(data.columns):
            rdf: np.ndarray = np.asanyarray(data[col])
            rdf = np.sort(rdf)
            rdf /= np.max(rdf)
            median = np.median(rdf)
            median_arr[idx, 0] = col
            median_arr[idx, 1] = median
        return median_arr
