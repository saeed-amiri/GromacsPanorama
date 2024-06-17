"""
Reading output files from ABPS simulations and calculating the average
potential.
The output files are in the format of .dx
"""

import sys
import typing
from dataclasses import dataclass
import multiprocessing as mp

import numpy as np
import pandas as pd

from common import logger
from common.cpuconfig import ConfigCpuNr
from common.colors_text import TextColor as bcolors


@dataclass
class FileConfig:
    """set the name of the input files"""
    header_file: str = 'apt_cor_0.dx'
    number_of_header_lines: int = 11
    number_of_tail_lines: int = 5
    output_file: str = 'average_potential.dx'


class AveragePotential:
    """
    Compute the average potential from the output files of ABPS simulations
    """
    info_msg: str = 'Message from AveragePotential:\n'
    files: list[str]
    num_files: int
    n_cores: int

    def __init__(self,
                 log: logger.logging.Logger,
                 configs: FileConfig = FileConfig()
                 ) -> None:
        """write and log messages"""
        self.configs = configs
        self.get_files(log)
        total_sum: pd.DataFrame
        header: list[str]
        tail: list[str]
        total_sum, header, tail = self.read_files(log)
        averages_potential: pd.DataFrame = \
            self.get_average_potential(total_sum, log)
        self.write_file(averages_potential, header, tail)
        self._write_msg(log)

    def get_files(self,
                  log: logger.logging.Logger
                  ) -> None:
        """get the list of files"""
        files = sys.argv[1:]
        files = [f for f in files if f.endswith('.dx')]
        files = [f for f in files if f != 'average_potential.dx']
        self.files = files[:50]

        self.info_msg += f'\tThe numbers of files is {len(files) = }\n'

        if len(files) == 0:
            msg: str = '\tNo files are provided!\n'
            log.error(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')
            sys.exit(msg)

    def read_files(self) -> tuple[list[np.ndarray], list[str], list[str]]:
        """read the files"""
        data_frames: list[np.ndarray] = []
        for file_name in self.files:
            if file_name == self.configs.header_file:
                header, tail = self.read_header_tail(file_name)
            data_frames.append(self.read_file(file_name))
        return data_frames, header, tail

    def get_average_potential(self,
                              data_frames: list[pd.DataFrame],
                              log: logger.logging.Logger
                              ) -> pd.DataFrame:
        """get the average potential"""
        average_data = self.average_data(data_frames)
        average_data = self.check_average_data(average_data, log)
        return average_data

    def read_file(self,
                  file_name: str
                  ) -> pd.DataFrame:
        """read the file"""
        with open(file_name, 'r', encoding='utf-8') as f_in:
            return self.read_data(f_in)

    def read_header_tail(self,
                         file_name: str
                         ) -> tuple[list[str], list[str]]:
        """read the header"""
        header: list[str] = []
        tail: list[str] = []
        with open(file_name, 'r', encoding='utf-8') as f_in:
            for _ in range(self.configs.number_of_header_lines):
                header.append(f_in.readline().strip())
            lines = f_in.readlines()
            tail = [line.strip() for line in
                    lines[-self.configs.number_of_tail_lines:]]
        return header, tail

    def read_data(self,
                  f_in: typing.TextIO
                  ) -> pd.DataFrame:
        """read the data"""
        return pd.read_csv(f_in,
                           sep=' ',
                           header=None,
                           skiprows=self.configs.number_of_header_lines,
                           skipfooter=self.configs.number_of_tail_lines,
                           engine='python')

    def average_data(self,
                     total_sum: pd.DataFrame
                     ) -> pd.DataFrame:
        """average the data"""
        average_data = total_sum / self.num_files
        return average_data

    def check_average_data(self,
                           average_data: pd.DataFrame,
                           log: logger.logging.Logger
                           ) -> pd.DataFrame:
        """check the average data"""
        if not np.all(average_data):
            msg: str = '\tThe average data is zero!\n'
            log.error(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')
            sys.exit(msg)
        # Assuming df is your DataFrame
        nan_cols: list = \
            average_data.columns[average_data.isna().all()].tolist()
        # Drop these columns from the dataframe
        average_data = average_data.drop(nan_cols, axis=1)
        return average_data

    def write_file(self,
                   average_data: pd.DataFrame,
                   header: list[str],
                   tail: list[str]
                   ) -> None:
        """write the file"""
        with open(self.configs.output_file, 'w', encoding='utf-8') as f_out:
            self.write_header(header, f_out)
            self.write_data(average_data, f_out)
            self.write_tail(tail, f_out)
    
    def write_header(self,
                     header: list[str],
                     f_out: typing.TextIO
                     ) -> None:
        """write the header"""
        for line in header:
            f_out.write(f'{line}\n')

    def write_data(self,
                   data: pd.DataFrame,
                   f_out: typing.TextIO
                   ) -> None:
        """write the data"""
        data.to_csv(f_out, sep=' ', header=False, index=False)

    def write_tail(self,
                   tail: list[str],
                   f_out: typing.TextIO
                   ) -> None:
        """write the tail"""
        for line in tail:
            f_out.write(f'{line}\n')

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{AveragePotential.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    AveragePotential(logger.logging.Logger('average_potential.log'))
