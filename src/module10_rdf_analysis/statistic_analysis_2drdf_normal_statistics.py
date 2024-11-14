"""
Applying noraml statistics to the 2D RDF data
"""

import pandas as pd
import numpy as np
import scipy.stats as stats

from common import logger


class NormalStatistics:
    """
    Apply normal statistics to the 2D RDF data
    """

    info_msg: str = "Message from NormalStatistics:\n"

    def __init__(self,
                 data: pd.DataFrame,
                 log: logger.logging.Logger
                 ) -> None:
        self.data = data
        self.normal_tests(log)

    def normal_tests(self,
                     log: logger.logging.Logger
                     ) -> None:
        """
        Test the normality of the data
        """
        self.get_median(log)

    def get_median(self,
                   logger: logger.logging.Logger
                   ) -> None:
        """
        Get the median of the data
        The median is a basic statistical measure that represents the
        middle value of a dataset when it is ordered from smallest to
        largest.
        """
