"""The bootstrap method is a powerful statistical tool used to estimate
the distribution of a sample statistic (like the mean or standard
deviation) by repeatedly resampling with replacement from the data.
It allows for the estimation of the sampling distribution of almost
any statistic, making it a flexible method for uncertainty estimation
"""

import sys
import common.logger as logger
import common.xvg_to_dataframe as xvg_to_df
from common.colors_text import TextColor as bcolors


class BootStrping:
    """do the boostraping sampling and error propagation"""

    info_msg: str = 'Message from BootStrping:\n'  # Meesage in methods to log
    xvg: "xvg_to_df"  # Parsed datafile

    def __init__(self,
                 fname: str,  # Data file xvg format
                 log: logger.logging.Logger
                 ) -> None:
        self.xvg = xvg_to_df.XvgParser(fname=fname, log=log)


if __name__ == "__main__":
    BootStrping(fname=sys.argv[1], log=logger.setup_logger('bootstrap.log'))
