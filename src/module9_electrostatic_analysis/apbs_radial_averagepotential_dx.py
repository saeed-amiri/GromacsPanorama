"""
Radial averaging of the DLVO potential calculation
The input calculation is done with ABPS simulations and the output is in the
format of .dx files:

    object 1 class gridpositions counts nx ny nz
    origin xmin ymin zmin
    delta hx 0.0 0.0
    delta 0.0 hy 0.0
    delta 0.0 0.0 hz
    object 2 class gridconnections counts nx ny nz
    object 3 class array type double rank 0 items n data follows
    u(0,0,0) u(0,0,1) u(0,0,2)
    ...
    u(0,0,nz-3) u(0,0,nz-2) u(0,0,nz-1)
    u(0,1,0) u(0,1,1) u(0,1,2)
    ...
    u(0,1,nz-3) u(0,1,nz-2) u(0,1,nz-1)
    ...
    u(0,ny-1,nz-3) u(0,ny-1,nz-2) u(0,ny-1,nz-1)
    u(1,0,0) u(1,0,1) u(1,0,2)
    ...
    attribute "dep" string "positions"
    object "regular positions regular connections" class field
    component "positions" value 1
    component "connections" value 2
    component "data" value 3`

The variables in this format include:
    nx ny nz
        The number of grid points in the x-, y-, and z-directions
    xmin ymin zmin
        The coordinates of the grid lower corner
    hx hy hz
        The grid spacings in the x-, y-, and z-directions.
    n
        The total number of grid points; n=nx*ny*nz
    u(*,*,*)
        The data values, ordered with the z-index increasing most
        quickly, followed by the y-index, and then the x-index.


First the header should be read, than based on number of the grids,
and size of the grids, the data should be read and the radial averaging
should be done.
input files are in the format of .dx
13 Jun 2024
Saeed
"""

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import logger, my_tools
from common.colors_text import TextColor as bcolors


@dataclass
class InputConfig:
    """set the name of the input files"""
    number_of_header_lines: int = 11
    number_of_tail_lines: int = 5
    output_file: str = 'radial_average_potential_nonlinear.xvg'


class RadialAveragePotential:
    """
    Compute the radial average potential from the output files of ABPS
    simulations
    """

    info_msg: str = 'Message from RadialAveragePotential:\n'
    average_index_from: int = 80
    pot_unit_conversion: float = 25.2
    dist_unit_conversion: float = 10.0

    def __init__(self,
                 file_name: str,
                 log: logger.logging.Logger,
                 configs: InputConfig = InputConfig()
                 ) -> None:
        """write and log messages"""
        self.configs = configs
        self.process_file(file_name, log)
        self.write_msg(log)

    def process_file(self,
                     file_name: str,
                     log: logger.logging.Logger
                     ) -> None:
        """process the file"""
        lines: list[str] = self.read_file(file_name)

        grid_points: list[int]
        grid_spacing: list[float]
        origin: list[float]
        grid_points, grid_spacing, origin = self._get_header(
            lines[:self.configs.number_of_header_lines], log)
        _: list[str] = self._get_tail(
            lines[-self.configs.number_of_tail_lines:])
        data: list[float] = self._get_data(lines[
            self.configs.number_of_header_lines:
            -self.configs.number_of_tail_lines])
        self.process_data(data, grid_points, grid_spacing, origin, log)

    def _get_data(self,
                  data_lines: list[str]
                  ) -> list[float]:
        """get the data"""
        data_tmp: list[list[str]] = [item.split() for item in data_lines]
        data = [float(i) for sublist in data_tmp for i in sublist]
        return data

    def process_data(self,
                     data: list[float],
                     grid_points: list[int],
                     grid_spacing: list[float],
                     origin: list[float],
                     log: logger.logging.Logger
                     ) -> None:
        """process the data"""
        # pylint: disable=too-many-arguments
        self.check_number_of_points(data, grid_points, log)
        self._get_box_size(grid_points, grid_spacing, origin)
        data_arr: np.ndarray = np.array(data).reshape(grid_points)
        radii, radial_average = \
            self.radial_average(data_arr, grid_points, grid_spacing, origin)
        self._plot_radial_average(radii, radial_average)
        self.write_radial_average(radii, radial_average, log)

    def radial_average(self,
                       data_arr: np.ndarray,
                       grid_points: list[int],
                       grid_spacing: list[float],
                       origin: list[float]) -> tuple[np.ndarray, list[float]]:
        """Compute and plot the radial average of the potential from
        the center of the box."""
        # pylint: disable=too-many-locals
        # pylint: disable=unused-argument
        self.info_msg += (
            f'\tThe average index is set to {self.average_index_from}\n')
        # Calculate the center of the box in grid units
        center_x = grid_points[0] // 2
        center_y = grid_points[1] // 2
        center_z = grid_points[2] // 2

        # Calculate the maximum radius for the radial average
        max_radius = min(center_x, center_y, center_z) * min(grid_spacing)
        # Create a grid of distances from the center
        x_space = np.linspace(0, grid_points[0] - 1, grid_points[0])
        y_space = np.linspace(0, grid_points[1] - 1, grid_points[1])
        z_space = np.linspace(0, grid_points[2] - 1, grid_points[2])

        grid_x, grid_y, grid_z = \
            np.meshgrid(x_space, y_space, z_space, indexing='ij')

        distances = np.sqrt((grid_x - center_x)**2 +
                            (grid_y - center_y)**2 +
                            (grid_z - center_z)**2) * grid_spacing[0]

        # Calculate the radial average
        radii = np.arange(0, max_radius, grid_spacing[0])

        radial_average: list[float] = []

        for radius in radii:
            mask = (distances >= radius) \
                & (distances < radius + grid_spacing[0]) \
                & (grid_z <= self.average_index_from)
            if np.sum(mask) > 0:
                avg_potential = \
                    np.mean(data_arr[mask]) * self.pot_unit_conversion
                radial_average.append(avg_potential)
            else:
                radial_average.append(0)
        return radii, radial_average

    def _plot_radial_average(self,
                             radii: np.ndarray,
                             radial_average: list[float]) -> None:
        """Plot the radial average of the potential"""
        # Plot the radial average
        plt.figure(figsize=(10, 6))

        plt.plot(radii/self.dist_unit_conversion,
                 radial_average,
                 label='Radial Average of Potential')
        plt.xlabel('Radius [nm]')
        plt.ylabel('Average Potential')
        plt.title('Radial Average of Potential from the Center of the Box')
        plt.legend()
        plt.grid(True)
        plt.show()

    def write_radial_average(self,
                             radii: np.ndarray,
                             radial_average: list[float],
                             log: logger.logging.Logger
                             ) -> None:
        """Write the radial average to a file"""
        # Write the radial average to a file
        convert_to_kt_e = [i/self.pot_unit_conversion for i in radial_average]
        data = {'Radius [nm]': radii/self.dist_unit_conversion,
                'Average Potential [mV]': radial_average,
                'Average Potential [kT/e]': convert_to_kt_e
                }
        extra_msg_0 = ('The radial average is set below the index: '
                       f'{self.average_index_from}')
        extra_msg = \
            [extra_msg_0,
             f'The conversion factor to [meV] is {self.pot_unit_conversion}']
        df_i = pd.DataFrame(data)
        df_i.set_index(df_i.columns[0], inplace=True)
        my_tools.write_xvg(
            df_i, log, extra_msg, fname=self.configs.output_file)

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

    def _get_tail(self,
                  tail_lines: list[str]
                  ) -> list[str]:
        """get the tail"""
        return tail_lines

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

    def read_file(self,
                  file_name: str
                  ) -> list[str]:
        """read the file"""
        with open(file_name, 'r', encoding='utf-8') as f_dx:
            lines = f_dx.readlines()
        return [line.strip() for line in lines]

    def write_msg(self,
                  log: logger.logging.Logger  # To log
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{RadialAveragePotential.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    try:
        RadialAveragePotential(
            sys.argv[1],
            log=logger.setup_logger('radial_average_potential.log'))
    except IndexError:
        print(f'{bcolors.FAIL}No file is provided!{bcolors.ENDC}')
        sys.exit(1)
