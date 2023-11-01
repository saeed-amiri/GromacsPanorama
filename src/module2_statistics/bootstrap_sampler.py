"""The bootstrap method is a powerful statistical tool used to estimate
the distribution of a sample statistic (like the mean or standard
deviation) by repeatedly resampling with replacement from the data.
It allows for the estimation of the sampling distribution of almost
any statistic, making it a flexible method for uncertainty estimation
"""

import sys
import random
import numpy as np
from common import logger
import common.xvg_to_dataframe as xvg_to_df
from common.colors_text import TextColor as bcolors


class BootStrping:
    """do the boostraping sampling and error propagation"""

    info_msg: str = 'Message from BootStrping:\n'  # Meesage in methods to log
    xvg: "xvg_to_df.XvgParser"  # Parsed datafile

    booter: str = 'Surf_SurfTen'  # Name of the columns to bootstrap them

    def __init__(self,
                 fname: str,  # Data file xvg format
                 log: logger.logging.Logger
                 ) -> None:
        self.xvg = xvg_to_df.XvgParser(fname=fname, log=log)
        self.initiate_normal(log)

    def initiate_normal(self,
                        log: logger.logging.Logger
                        ) -> None:
        """do the sampling here"""
        samples: list[np.float64] = self.random_replacement()

    def random_replacement(self) -> list[np.float64]:
        """Randomly Select With Replacement"""
        samples: list[np.float64] = []
        for _ in range(self.xvg.nr_frames):
            sample_i = random.choices(self.xvg.xvg_df[self.booter],
                                      k=self.xvg.nr_frames)
            samples.append(sum(sample_i)/self.xvg.nr_frames)
        return samples


if __name__ == "__main__":
    BootStrping(fname=sys.argv[1], log=logger.setup_logger('bootstrap.log'))
