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
        self.files = files
        self.num_files = len(files)
        self.info_msg += f'\tThe numbers of files is {self.num_files = }\n'

        if len(files) == 0:
            msg: str = '\tNo files are provided!\n'
            log.error(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')
            sys.exit(msg)

    def read_and_sum_files(self,
                           file_names: list[str]
                           ) -> pd.DataFrame:
        """read and sum the files"""
        # Initialize the sum
        total_sum = None

        # Read and sum the data in each file
        for file_name in file_names:
            data = self.read_file(file_name)
            if total_sum is None:
                total_sum = data
            else:
                total_sum += data

        return total_sum

    def read_files(self,
                   log: logger.logging
                   ) -> tuple[list[pd.DataFrame], list[str], list[str]]:
        """read the files"""
        self._initiate_cpu(log)
        self._set_number_of_cores()
        file_chunks: list[list[str]] = self._chunk_files()
        header, tail = self.read_header_tail(self.files[0])

        with mp.Pool(self.n_cores) as pools:
            sums = pools.map(self.read_and_sum_files, file_chunks)

        # Sum the results from each chunk
        total_sum = sum(sums)
        return total_sum, header, tail

    def _initiate_cpu(self,
                      log: logger.logging.Logger
                      ) -> None:
        """
        Return the number of core for run based on the data and the machine
        """
        cpu_info = ConfigCpuNr(log)
        self.n_cores: int = min(cpu_info.cores_nr, len(self.files))
        self.info_msg += f'\tThe numbers of using cores: {self.n_cores}\n'

    def _set_number_of_cores(self) -> None:
        """
        Set the number of cores for the multiprocessing baesd on the number of
        files.
        """
        if self.num_files < self.n_cores:
            self.n_cores = self.num_files
            self.info_msg += f'\tThe numbers of cores set to: {self.n_cores}\n'

    def _chunk_files(self) -> list[list[str]]:
        """
        Chunk the list of files based on the number of cores
        """
        self.info_msg += f'\tThe files are chunked to: {self.n_cores} parts\n'
        chunk_size = len(self.files) // self.n_cores
        return [self.files[i:i + chunk_size] for i in
                range(0, len(self.files), chunk_size)]

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
