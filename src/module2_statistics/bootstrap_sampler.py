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
    block_size: int = 10  # Size of each block
    convert_rate: float = 2*10  # Since we have two interface
    booter: str = 'Surf_SurfTen'  # Name of the columns to bootstrap them

    raw_normal: dict[str, typing.Any] = {}
    raw_block: dict[str, typing.Any] = {}
    raw_mbb: dict[str, typing.Any] = {}
    stats_normal: dict[str, typing.Any] = {}
    stats_block: dict[str, typing.Any] = {}
    stats_mbb: dict[str, typing.Any] = {}

    def __init__(self,
                 fname: str,  # Data file xvg format
                 log: logger.logging.Logger
                 ) -> None:
        self.xvg = xvg_to_df.XvgParser(fname=fname, log=log)
        self.info_msg += \
            (f'\tBootstraping for `{fname}` on column `{self.booter}`\n'
             f'\tConvert rate is: `{self.convert_rate}`\n'
             f'\tblock size is `{self.block_size}`\n\n')

        self.check_initial_data()
        self.perform_normal_bootstrap()
        self.perform_block_bootstrap()
        self.perform_moving_block_bootstrap()
        self.write_log_msg(log)

    def check_initial_data(self) -> None:
        """check the mean, std, mode of the initial data"""
        tmp_dict: dict[str, typing.Any] = \
            self.calc_raw_stats(self.xvg.xvg_df[self.booter], 'initial')
        self.convert_stats(tmp_dict, 'initial')

    def perform_normal_bootstrap(self) -> None:
        """do the sampling here"""
        samples: list[np.float64] = self.sample_randomly_with_replacement()
        self.raw_normal = self.calc_raw_stats(samples, 'normal')
        self.stats_normal = self.convert_stats(self.raw_normal, 'normal')

    def sample_randomly_with_replacement(self) -> list[np.float64]:
        """Randomly Select With Replacement"""
        samples: list[np.float64] = []
        for _ in range(self.xvg.nr_frames):
            sample_i = random.choices(self.xvg.xvg_df[self.booter],
                                      k=self.xvg.nr_frames)
            samples.append(sum(sample_i)/self.xvg.nr_frames)
        return samples

    def perform_block_bootstrap(self) -> None:
        """calculate stats from block bootstraping resamplings"""
        blocks: list[pd.DataFrame] = self.split_data_into_blocks()
        samples: list[float] = self.bootstrap_blocks(blocks)
        self.raw_block = self.calc_raw_stats(samples, 'block')
        self.stats_block = self.convert_stats(self.raw_block, style='block')

    def split_data_into_blocks(self) -> list[pd.DataFrame]:
        """make random blocks"""
        blocks: list[pd.DataFrame] = \
            [pd.DataFrame(arr) for arr in
             np.array_split(self.xvg.xvg_df,
                            len(self.xvg.xvg_df) // self.block_size)]
        return blocks

    def bootstrap_blocks(self,
                         blocks: list[pd.DataFrame]
                         ) -> list[float]:
        """get the samples"""
        samples: list[float] = []
        for _ in range(l_1 := len(blocks)):
            sample_i = random.choices(blocks, k=l_1)
            df_tmp: pd.DataFrame = pd.concat(sample_i)
            samples.append(df_tmp[self.booter].mean())
            del df_tmp
        return samples

    def perform_moving_block_bootstrap(self) -> None:
        """The moving block bootstrap (MBB) is a variant of the block
        bootstrap method designed to retain the temporal correlation
        structure of time-series data when generating bootstrap samples.
        Unlike the traditional block bootstrap method, which samples
        blocks of consecutive observations with replacement, the moving
        block bootstrap allows for overlapping blocks."""
        blocks: list[pd.DataFrame] = \
            self.create_overlapping_blocks(self.xvg.xvg_df, self.block_size)
        samples: list[float] = self.bootstrap_blocks(blocks)
        self.raw_mbb = self.calc_raw_stats(samples, 'MMB')
        self.stats_mbb = self.convert_stats(self.raw_mbb, 'MBB')

    @staticmethod
    def create_overlapping_blocks(dataframe: pd.DataFrame,
                                  block_size: int
                                  ) -> list[pd.DataFrame]:
        """for MBB"""
        overlapping_blocks: list[pd.DataFrame] = []

        for i in range(len(dataframe) - block_size + 1):
            block = dataframe.iloc[i:i + block_size]
            overlapping_blocks.append(block)

        return overlapping_blocks

    def calc_raw_stats(self,
                       samples: typing.Union[list[np.float64], list[float]],
                       style: str
                       ) -> dict[str, typing.Any]:
        """calculate std and averages"""
        raw_stats_dict: dict[str, typing.Any] = {}
        sample_arr: np.ndarray = np.array(samples)
        raw_stats_dict['std'] = np.std(sample_arr)
        raw_stats_dict['mean'] = np.mean(sample_arr)
        raw_stats_dict['mode'] = \
            self.calc_mode(sample_arr, raw_stats_dict['std']/5)
        if style == 'initial':
            boots = ''
        else:
            boots = ' bootstraping'
        self.info_msg += \
            (f'\tStats (raw) for `{style}`{boots}:'
             f'{json.dumps(raw_stats_dict, indent=8)}\n')
        return raw_stats_dict

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
                      raw_stats: dict[str, typing.Any],
                      style: str
                      ) -> dict[str, typing.Any]:
        """convert data to the asked unit"""
        stats_dict: dict[str, typing.Any] = {}
        stats_dict = \
            {key: value/self.convert_rate for key, value in raw_stats.items()}
        if style == 'initial':
            boots = ''
        else:
            boots = ' bootstraping'
        self.info_msg += \
            (f'\tStats (Converted) for `{style}`{boots}:'
             f'{json.dumps(stats_dict, indent=8)}\n')
        return stats_dict

    def write_log_msg(self,
                      log: logger.logging.Logger  # Name of the output file
                      ) -> None:
        """writing and logging messages from methods"""
        log.info(self.info_msg)
        print(f'{bcolors.OKBLUE}{self.__module__}:\n'
              f'\t{self.info_msg}\n{bcolors.ENDC}')


if __name__ == "__main__":
    BootStrping(fname=sys.argv[1], log=logger.setup_logger('bootstrap.log'))
