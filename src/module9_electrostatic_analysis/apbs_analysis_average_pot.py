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
import typing
from dataclasses import dataclass, field

import numpy as np

import matplotlib.pyplot as plt

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
    bulk_averaging: bool = False  # if Bulk averaging else interface averaging


class DxAttributeWrapper:
    """
    Wrapper for the attributes of the dx file
    """
    # pylint: disable=missing-function-docstring
    # pylint: disable=invalid-name
    # pylint: disable=too-many-arguments
    def __init__(self,
                 grid_points: list[int],
                 grid_spacing: list[float],
                 origin: list[float],
                 box_size: list[float],
                 data_arr: np.ndarray
                 ) -> None:
        self._grid_points = grid_points
        self._grid_spacing = grid_spacing
        self._origin = origin
        self._data_arr = data_arr
        self._box_size = box_size

    @property
    def GRID_POINTS(self) -> list[int]:
        return self._grid_points

    @GRID_POINTS.setter
    def GRID_POINTS(self,
                    grid_points: typing.Union[list[int], typing.Any]
                    ) -> None:
        if not isinstance(grid_points, list):
            raise TypeError('The grid_points should be a list!')
        if not all(isinstance(i, int) for i in grid_points):
            raise TypeError('All elements of the grid_points should be int!')
        self._grid_points = grid_points

    @property
    def GRID_SPACING(self) -> list[float]:
        return self._grid_spacing

    @property
    def ORIGIN(self) -> list[float]:
        """
        It is the origin of the box in the dx file NOT the origin of
        the computational sphere
        """
        return self._origin

    @property
    def DATA_ARR(self) -> np.ndarray:
        return self._data_arr

    @property
    def BOX_SIZE(self) -> list[float]:
        """Box sizes in Angstrom"""
        return self._box_size


class AverageAnalysis:
    """
    Reading and analysing the potential along the z-axis
    """
    # pylint: disable=invalid-name
    info_msg: str = 'Message from AverageAnalysis:\n'
    all_config: AllConfig
    dx: DxAttributeWrapper  # The dx file

    def __init__(self,
                 fname_dx: str,
                 log: logger.logging.Logger,
                 configs: AllConfig = AllConfig()
                 ) -> None:
        self.configs = configs
        self.read_dx(fname_dx, log)
        self.analyse_potential()
        self.write_msg(log)

    def read_dx(self,
                fname_dx: str,
                log: logger.logging.Logger
                ) -> None:
        """read the dx file"""
        self.info_msg += f'\tAnalysing the dx file: {fname_dx}\n'
        read_dx = ProcessDxFile(fname_dx, log, self.configs.dx_configs)
        self.dx = DxAttributeWrapper(
            grid_points=read_dx.grid_points,
            grid_spacing=read_dx.grid_spacing,
            origin=read_dx.origin,
            box_size=read_dx.box_size,
            data_arr=read_dx.data_arr
        )

    def analyse_potential(self) -> None:
        """analyse the potential"""
        center_xyz: tuple[int, int, int] = \
            self.calculate_center(self.dx.GRID_POINTS)

        shpere_gride_range: np.ndarray = \
            self.find_grid_inidices_covers_shpere(center_xyz)

        self.info_msg += (
            f'\tThe centeral grid is: {center_xyz}\n'
            f'\tThe computation radius is: {self.configs.computation_radius}\n'
            f'\tNr. grids cover the sphere: {len(shpere_gride_range)}\n'
            f'\tThe lowest grid index: {shpere_gride_range[0]}\n'
            f'\tThe highest grid index: {shpere_gride_range[-1]}\n'
            )

        for layer in shpere_gride_range:
            center_xyz = (center_xyz[0], center_xyz[1], layer)
            radii, radial_average = self.process_layer(center_xyz)
            plt.plot(radii, radial_average)
        plt.show()

    def process_layer(self,
                      center_xyz: tuple[int, int, int]
                      ) -> tuple[np.ndarray, np.ndarray]:
        """process the layer
        The potential from the surface until the end of the diffuse layer
        """
        max_radius: float = self.calculate_max_radius(
            center_xyz, self.dx.GRID_SPACING)
        # Create the distance grid
        grid_xyz: tuple[np.ndarray, np.ndarray, np.ndarray] = \
            self.create_distance_grid(self.dx.GRID_POINTS)
        # Calculate the distances from the center of the box
        distances: np.ndarray = self.compute_distance(
                self.dx.GRID_SPACING, grid_xyz, center_xyz)
        radii, radial_average = self.calculate_radial_average(
                self.dx.DATA_ARR,
                distances,
                self.dx.GRID_SPACING,
                max_radius,
                grid_xyz[2],
                interface_low_index=grid_xyz[2],
                interface_high_index=grid_xyz[2]+1,
                lower_index_bulk=0,
                )
        return radii, np.asanyarray(radial_average)

    def calculate_radial_average(self,
                                 data_arr: np.ndarray,
                                 distances: np.ndarray,
                                 grid_spacing: list[float],
                                 max_radius: float,
                                 grid_z: np.ndarray,
                                 interface_low_index,
                                 interface_high_index,
                                 lower_index_bulk,
                                 ) -> tuple[np.ndarray, list[float]]:
        """Calculate the radial average of the potential"""
        # pylint: disable=too-many-arguments
        radii = np.arange(0, max_radius, grid_spacing[0])
        radial_average = []

        for radius in radii:
            mask = self.create_mask(distances,
                                    radius,
                                    grid_spacing,
                                    grid_z,
                                    interface_low_index,
                                    interface_high_index,
                                    lower_index_bulk,
                                    )
            if np.sum(mask) > 0:
                avg_potential = np.mean(data_arr[mask])
                radial_average.append(avg_potential)
            else:
                radial_average.append(0)

        return radii, radial_average

    def create_mask(self,
                    distances: np.ndarray,
                    radius: float,
                    grid_spacing: list[float],
                    grid_z: np.ndarray,
                    interface_low_index: int,
                    interface_high_index: int,
                    low_index_bulk: int
                    ) -> np.ndarray:
        """Create a mask for the radial average"""
        # pylint: disable=too-many-arguments
        shell_thickness: float = grid_spacing[0]
        shell_condition: np.ndarray = (distances >= radius) & \
                                      (distances < radius + shell_thickness)

        if self.configs.bulk_averaging:
            z_condition: np.ndarray = self.create_mask_bulk(
                grid_z, interface_low_index, low_index_bulk)
        else:
            z_condition = self.create_mask_interface(
                grid_z, interface_low_index, interface_high_index)

        return shell_condition & z_condition

    @staticmethod
    def create_mask_bulk(grid_z: np.ndarray,
                         interface_low_index: int,
                         low_index_bulk: int,
                         ) -> np.ndarray:
        """Create a mask for the radial average from the bulk"""
        z_condition: np.ndarray = (grid_z <= interface_low_index) & \
                                  (grid_z >= low_index_bulk)
        return z_condition

    @staticmethod
    def create_mask_interface(grid_z: np.ndarray,
                              interface_low_index: int,
                              interface_high_index: int
                              ) -> np.ndarray:
        """Create a mask for the radial average from the interface"""
        z_condition: np.ndarray = (grid_z >= interface_low_index) & \
                                  (grid_z <= interface_high_index)
        return z_condition

    @staticmethod
    def calculate_center(grid_points: list[int]
                         ) -> tuple[int, int, int]:
        """Calculate the center of the box in grid units"""
        center_x: int = grid_points[0] // 2
        center_y: int = grid_points[1] // 2
        center_z: int = grid_points[2] // 2
        return center_x, center_y, center_z

    @staticmethod
    def calculate_max_radius(center_xyz: tuple[float, float, float],
                             grid_spacing: list[float]
                             ) -> float:
        """Calculate the maximum radius for the radial average"""
        return min(center_xyz[:2]) * min(grid_spacing)

    @staticmethod
    def create_distance_grid(grid_points: list[int],
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create the distance grid"""
        x_space = np.linspace(0, grid_points[0] - 1, grid_points[0])
        y_space = np.linspace(0, grid_points[1] - 1, grid_points[1])
        z_space = np.linspace(0, grid_points[2] - 1, grid_points[2])

        grid_x, grid_y, grid_z = \
            np.meshgrid(x_space, y_space, z_space, indexing='ij')
        return grid_x, grid_y, grid_z

    @staticmethod
    def compute_distance(grid_spacing: list[float],
                         grid_xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
                         center_xyz: tuple[float, float, float],
                         ) -> np.ndarray:
        """Calculate the distances from the center of the box"""
        return np.sqrt((grid_xyz[0] - center_xyz[0])**2 +
                       (grid_xyz[1] - center_xyz[1])**2 +
                       (grid_xyz[2] - center_xyz[2])**2) * grid_spacing[0]

    def find_grid_inidices_covers_shpere(self,
                                         center_xyz: tuple[int, int, int],
                                         ) -> np.ndarray:
        """Find the grid points within the NP"""
        grid_size: float = self.dx.BOX_SIZE[2] / self.dx.GRID_POINTS[2]
        radius: float = self.configs.computation_radius
        nr_grids_coveres_sphere_radius: int = int(radius / grid_size) + 1
        lowest_z_index: int = center_xyz[2] - nr_grids_coveres_sphere_radius
        highest_z_index: int = center_xyz[2] + nr_grids_coveres_sphere_radius
        return np.arange(lowest_z_index, highest_z_index)

    def write_msg(self,
                  log: logger.logging.Logger  # To log
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{AverageAnalysis.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


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


if __name__ == '__main__':
    AverageAnalysis('average_potential.dx',
                    logger.setup_logger('avarge_potential_analysis.log'))
