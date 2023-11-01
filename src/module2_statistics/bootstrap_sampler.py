"""The bootstrap method is a powerful statistical tool used to estimate
the distribution of a sample statistic (like the mean or standard
deviation) by repeatedly resampling with replacement from the data.
It allows for the estimation of the sampling distribution of almost
any statistic, making it a flexible method for uncertainty estimation
"""

import sys
import json
import random
import typing
from collections import Counter
import numpy as np
import pandas as pd
from common import logger
import common.xvg_to_dataframe as xvg_to_df
from common.colors_text import TextColor as bcolors


class BootStrping:
    """do the boostraping sampling and error propagation"""

    info_msg: str = 'Message from BootStrping:\n'  # Meesage in methods to log

    xvg: "xvg_to_df.XvgParser"  # Parsed datafile
    booter: str = 'Surf_SurfTen'  # Name of the columns to bootstrap them
    raw_stats_dict: dict[str, typing.Any] = {}
    stats_dict: dict[str, typing.Any] = {}
    convert_rate: float = 2*10  # Since we have two interface
    block_size: int = 10  # Size of each block

    def __init__(self,
                 fname: str,  # Data file xvg format
                 log: logger.logging.Logger
                 ) -> None:
        self.xvg = xvg_to_df.XvgParser(fname=fname, log=log)
        self.info_msg += \
            (f'\tBootstraping for `{fname}` on column `{self.booter}`\n'
             f'\tConvert rate is: `{self.convert_rate}`\n')

        self.initiate_normal()
        self.initiate_block()
        self.write_log_msg(log)

    def initiate_normal(self) -> None:
        """do the sampling here"""
        samples: list[np.float64] = self.random_replacement()
        self.analysis_sample(samples, 'normal')
        self.convert_stats('normal')

    def random_replacement(self) -> list[np.float64]:
        """Randomly Select With Replacement"""
        samples: list[np.float64] = []
        for _ in range(self.xvg.nr_frames):
            sample_i = random.choices(self.xvg.xvg_df[self.booter],
                                      k=self.xvg.nr_frames)
            samples.append(sum(sample_i)/self.xvg.nr_frames)
        return samples

    def initiate_block(self) -> None:
        """calculate stats from block bootstraping resamplings"""
        blocks: list[pd.DataFrame] = self.get_blocks_list()
        samples: list[float] = self.sample_blocks(blocks)
        self.analysis_sample(samples, 'block')

    def get_blocks_list(self) -> list[pd.DataFrame]:
        blocks: list[pd.DataFrame] = \
            [pd.DataFrame(arr) for arr in
             np.array_split(self.xvg.xvg_df,
                            len(self.xvg.xvg_df) // self.block_size)]
        return blocks

    def sample_blocks(self,
                      blocks: list[pd.DataFrame]) -> list[float]:
        """get the samples"""
        samples: list[float] = []
        for _ in range(l_1 := len(blocks)):
            sample_i = random.choices(blocks, k=l_1)
            df_tmp: pd.DataFrame = pd.concat(sample_i)
            samples.append(df_tmp[self.booter].mean())
            del df_tmp
        return samples

    def analysis_sample(self,
                        samples: typing.Union[list[np.float64], list[float]],
                        style: str
                        ) -> None:
        """calculate std and averages"""
        sample_arr: np.ndarray = np.asarray(samples)
        self.raw_stats_dict['std'] = np.std(sample_arr)
        self.raw_stats_dict['mean'] = np.mean(sample_arr)
        self.raw_stats_dict['mode'] = \
            self.calc_mode(sample_arr, self.raw_stats_dict['std']/5)
        self.info_msg += \
            (f'\tStats (raw) for `{style}` bootstraping:'
             f'{json.dumps(self.raw_stats_dict, indent=8)}\n')

    @staticmethod
    def calc_mode(samples: np.ndarray,
                  tolerance: np.float64
                  ) -> np.float64:
        """return mode for the sample"""
        # Round the data to the nearest multiple of the tolerance
        rounded_data = [round(x / tolerance) * tolerance for x in samples]
        # Use Counter to count occurrences of rounded values
        counts = Counter(rounded_data)
        max_count: float = max(counts.values())
        modes \
            = [value for value, count in counts.items() if count == max_count]
        return modes[0]

    def convert_stats(self,
                      style: str
                      ) -> None:
        """convert data to the asked unit"""
        for key, value in self.raw_stats_dict.items():
            self.stats_dict[key] = value/self.convert_rate
        self.info_msg += \
            (f'\tStats (Converted) for `{style}` bootstraping:'
             f'{json.dumps(self.stats_dict, indent=8)}\n')

    def write_log_msg(self,
                      log: logger.logging.Logger  # Name of the output file
                      ) -> None:
        """writing and logging messages from methods"""
        log.info(self.info_msg)
        print(f'{bcolors.OKBLUE}{self.__module__}:\n'
              f'\t{self.info_msg}\n{bcolors.ENDC}')


if __name__ == "__main__":
    BootStrping(fname=sys.argv[1], log=logger.setup_logger('bootstrap.log'))
