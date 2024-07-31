"""
Read the dx file and return the info from it
"""

import sys
from dataclasses import dataclass

import numpy as np

from common import logger
from common.colors_text import TextColor as bcolors


@dataclass
class DxFileConfig:
    """set the name of the input files"""
    number_of_header_lines: int = 11
    number_of_tail_lines: int = 5
    pot_unit_conversion: float = 25.2  # Conversion factor to mV


class ProcessDxFile:
    """
    read the dx file and return the info from it
    """
    info_msg: str = 'Message from ProcessDxFile:\n'
    configs: DxFileConfig
    grid_points: list[int]
    grid_spacing: list[float]
    origin: list[float]
    box_size: list[float]
    data_arr: np.ndarray

    def __init__(self,
                 fname_dx: str,
                 log: logger.logging.Logger,
                 configs: DxFileConfig = DxFileConfig()
                 ) -> None:
        self.configs = configs
        self.process_dx_file(fname_dx, log)
        self.write_msg(log)

    def process_dx_file(self,
                        fname_dx: str,
                        log: logger.logging.Logger
                        ) -> None:
        """process the dx file"""
        lines: list[str] = self.read_dx_file(fname_dx)
        self.grid_points, self.grid_spacing, self.origin = self._get_header(
            lines[:self.configs.number_of_header_lines], log)
        _: list[str] = self._get_tail(
            lines[-self.configs.number_of_tail_lines:])
        data: list[float] = self._get_data(lines[
            self.configs.number_of_header_lines:
            -self.configs.number_of_tail_lines])
        self.check_number_of_points(data, self.grid_points, log)
        self.check_number_of_points(data, self.grid_points, log)
        self._get_box_size(self.grid_points, self.grid_spacing, self.origin)
        self.data_arr: np.ndarray = self._reshape_reevaluate_data(
            data, self.grid_points, self.configs.pot_unit_conversion)

    def read_dx_file(self,
                     file_name: str
                     ) -> list[str]:
        """read the file"""
        with open(file_name, 'r', encoding='utf-8') as f_dx:
            lines = f_dx.readlines()
        return [line.strip() for line in lines]

    def _get_header(self,
                    head_lines: list[str],
                    log: logger.logging.Logger
                    ) -> tuple[list[int], list[float], list[float]]:
        """get the header"""
        grid_points: list[int]
        grid_spacing: list[float] = []
        origin: list[float]
        self.check_header(head_lines, log)
        for line in head_lines:
            if 'object 1' in line:
                grid_points = [int(i) for i in line.split()[-3:]]
            if 'origin' in line:
                origin = [float(i) for i in line.split()[-3:]]
            if 'delta' in line:
                grid_spacing.append([
                    float(i) for i in line.split()[-3:] if float(i) != 0.0][0])
        return grid_points, grid_spacing, origin

    @staticmethod
    def check_header(head_lines: list[str],
                     log: logger.logging.Logger
                     ) -> None:
        """check the header"""
        try:
            assert 'counts' in head_lines[4]
            assert 'origin' in head_lines[5]
            assert 'delta' in head_lines[6]
        except AssertionError:
            msg: str = 'The header is not in the correct format!'
            print(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')
            log.error(msg)
            sys.exit(1)

    def _get_tail(self,
                  tail_lines: list[str]
                  ) -> list[str]:
        """get the tail"""
        return tail_lines

    def _get_data(self,
                  data_lines: list[str]
                  ) -> list[float]:
        """get the data"""
        data_tmp: list[list[str]] = [item.split() for item in data_lines]
        data = [float(i) for sublist in data_tmp for i in sublist]
        return data

    @staticmethod
    def check_number_of_points(data: list[float],
                               grid_points: list[int],
                               log: logger.logging.Logger
                               ) -> None:
        """check the number of points"""
        if len(data) != np.prod(grid_points):
            msg: str = ('The number of data points is not correct!\n'
                        f'\t{len(data) = } != {np.prod(grid_points) = }\n')
            print(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')
            log.error(msg)
            sys.exit(1)

    def _get_box_size(self,
                      grid_points: list[int],
                      grid_spacing: list[float],
                      origin: list[float]
                      ) -> None:
        """get the box size"""
        x_size: float = grid_points[0] * grid_spacing[0] - origin[0]
        y_size: float = grid_points[1] * grid_spacing[1] - origin[1]
        z_size: float = grid_points[2] * grid_spacing[2] - origin[2]
        self.info_msg += (
            f'\tThe box size is:\n'
            f'\t{x_size/10.0 = :.5f} [nm]\n'
            f'\t{y_size/10.0 = :.5f} [nm]\n'
            f'\t{z_size/10.0 = :.5f} [nm]\n')
        self.box_size: list[float] = [x_size, y_size, z_size]

    @staticmethod
    def _reshape_reevaluate_data(data: list[float],
                                 grid_points: list[int],
                                 pot_unit_conversion: float
                                 ) -> np.ndarray:
        """reshape and reevaluate the data.
        In NumPy, the default order for reshaping (C order) is row-major,
        which means the last index changes fastest. This aligns with
        the way the data is ordered (z, y, x)."""
        return np.array(data).reshape(grid_points) * pot_unit_conversion

    def write_msg(self,
                  log: logger.logging.Logger  # To log
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{ProcessDxFile.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)
