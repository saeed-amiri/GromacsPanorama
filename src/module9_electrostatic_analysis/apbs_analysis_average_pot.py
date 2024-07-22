"""
The system's average potential, computed by APBS, is analyzed in this
module. It is similar to apbs_radial_averagepotential_dx.py, but
focuses on the average potential along the z-axis. The box is gridded
in the z-axis, and the average potential is computed for every plane
on the z-axis. At each index, the radial average has its own definition
of the EDL layers: surface, Stern, and Diffuse. For every plane that
intersects with the NP, the potential from the NP surface can be
fitted to the planar surface approximation of the PB equation. The
potential from the surface until the end of the diffuse layer exhibits
exponential decay. By fitting the potential, the surface potential and
the decay constant can be computed; the decay constant is the inverse
of the Debye length.

Input:
    Average of the potential along the z-axis, computed by APBS:
    average_potential.dx

Opt. by ChatGpt
22 July 2024
Saeed
"""

import sys
from dataclasses import dataclass, field

import numpy as np

from common import logger
from common.colors_text import TextColor as bcolors


@dataclass
class DxFileConfig:
    """set the name of the input files"""
    average_potential: str = 'average_potential.dx'
    number_of_header_lines: int = 11
    number_of_tail_lines: int = 5
    pot_unit_conversion: float = 25.2  # Conversion factor to mV


@dataclass
class ParameterConfig:
    """set parameters for the average potential analysis
    computaion_radius: the radius of the sphere choosen for the
    computation of the average potential in Ångströms
    """
    computation_radius: float = 36.0


@dataclass
class AllConfig(ParameterConfig):
    """set all the configs and parameters"""

    dx_configs: DxFileConfig = field(default_factory=DxFileConfig)


class DxAttributeWrapper:
    """
    Wrapper for the attributes of the dx file
    """
    # pylint: disable=missing-function-docstring
    # pylint: disable=invalid-name
    def __init__(self,
                 grid_points: list[int],
                 grid_spacing: list[float],
                 origin: list[float],
                 data_arr: np.ndarray
                 ) -> None:
        self._grid_points = grid_points
        self._grid_spacing = grid_spacing
        self._origin = origin
        self._data_arr = data_arr

    @property
    def GRID_POINTS(self) -> list[int]:
        return self._grid_points

    @property
    def GRID_SPACING(self) -> list[float]:
        return self._grid_spacing

    @property
    def ORIGIN(self) -> list[float]:
        return self._origin

    @property
    def DATA_ARR(self) -> np.ndarray:
        return self._data_arr


class AverageAnalysis:
    """
    Reading and analysing the potential along the z-axis
    """

    info_msg: str = 'Message from AverageAnalysis:\n'
    all_config: AllConfig

    def __init__(self,
                 fname_dx: str,
                 log: logger.logging.Logger,
                 configs: AllConfig = AllConfig()
                 ) -> None:
        self.configs = configs
        self.read_dx(fname_dx, log)

    def read_dx(self,
                fname_dx: str,
                log: logger.logging.Logger
                ) -> None:
        """read the dx file"""
        read_dx = ProcessDxFile(fname_dx, log, self.configs.dx_configs)
        print(read_dx.grid_points)


class ProcessDxFile:
    """
    read the dx file and return the info from it
    """
    info_msg: str = 'Message from ProcessDxFile:\n'
    configs: DxFileConfig
    grid_points: list[int]
    grid_spacing: list[float]
    origin: list[float]
    data_arr: np.ndarray

    def __init__(self,
                 fname_dx: str,
                 log: logger.logging.Logger,
                 configs: DxFileConfig
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
            f'\t{x_size = :.5f} [nm]\n'
            f'\t{y_size = :.5f} [nm]\n'
            f'\t{z_size = :.5f} [nm]\n')

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


if __name__ == '__main__':
    AverageAnalysis('average_potential.dx',
                    logger.setup_logger('avarge_potential_analysis.log'))
