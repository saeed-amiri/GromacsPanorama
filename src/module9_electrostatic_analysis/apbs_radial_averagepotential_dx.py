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
import typing
from dataclasses import field
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from common import logger
from common import my_tools
from common import plot_tools
from common import elsevier_plot_tools
from common.colors_text import TextColor as bcolors


@dataclass
class InputConfig:
    """set the name of the input files"""
    number_of_header_lines: int = 11
    number_of_tail_lines: int = 5
    output_file: str = 'radial_average_potential_nonlinear.xvg'
    interface_file: str = 'interface_radial_average_potential_nonlinear.xvg'
    bulk_averaging: bool = False  # if Bulk averaging else interface averaging
    interface_low_index: int = 89
    interface_high_index: int = 90
    lower_index_bulk: int = 10


@dataclass
class MultilayerPlotConfig:
    """set the parameters for the multilayer plot"""
    delta_z: int = 0  # increment in z grid, it is index
    highest_z: int = 100
    lowest_z: int = 90
    decrement_z: int = 1
    main_z: int = 90
    main_z_linestyle: list[tuple[str, tuple[int, typing.Any]]] = field(
        default_factory=lambda: [('solid', (0, ())),])
    drop_z: list[int] = field(default_factory=lambda: [])
    x_lims: tuple[float, float] = field(default_factory=lambda: (2.8, 7))
    y_lims: tuple[float, float] = field(default_factory=lambda: (10, 135))


@dataclass
class AllConfigs(InputConfig, MultilayerPlotConfig):
    """set the parameters for the radial average potential"""
    pot_unit_conversion: float = 25.7  # kT/e <-> mV
    dist_unit_conversion: float = 10.0  # Angstrom <-> nm


def compute_distance(grid_spacing: list[float],
                     grid_xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
                     center_xyz: tuple[float, float, float],
                     ) -> np.ndarray:
    """Calculate the distances from the center of the box"""
    return np.sqrt((grid_xyz[0] - center_xyz[0])**2 +
                   (grid_xyz[1] - center_xyz[1])**2 +
                   (grid_xyz[2] - center_xyz[2])**2) * grid_spacing[0]


def calculate_max_radius(center_xyz: tuple[float, float, float],
                         grid_spacing: list[float]
                         ) -> float:
    """Calculate the maximum radius for the radial average"""
    return min(center_xyz[:2]) * min(grid_spacing)


def create_distance_grid(grid_points: list[int],
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create the distance grid"""
    x_space = np.linspace(0, grid_points[0] - 1, grid_points[0])
    y_space = np.linspace(0, grid_points[1] - 1, grid_points[1])
    z_space = np.linspace(0, grid_points[2] - 1, grid_points[2])

    grid_x, grid_y, grid_z = \
        np.meshgrid(x_space, y_space, z_space, indexing='ij')
    return grid_x, grid_y, grid_z


def calculate_center(grid_points: list[int]
                     ) -> tuple[int, int, int]:
    """Calculate the center of the box in grid units"""
    center_x = grid_points[0] // 2
    center_y = grid_points[1] // 2
    center_z = grid_points[2] // 2
    return center_x, center_y, center_z


class RadialAveragePotential:
    """
    Compute the radial average potential from the output files of ABPS
    simulations
    """

    info_msg: str = 'Message from RadialAveragePotential:\n'
    configs: AllConfigs

    def __init__(self,
                 configs: AllConfigs = AllConfigs()
                 ) -> None:
        """write and log messages"""
        self.configs = configs

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
        self.write_msg(log)

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
        data_arr: np.ndarray = self._reshape_reevaluate_data(data, grid_points)
        radii, radial_average = \
            self.radial_average(data_arr, grid_points, grid_spacing)
        # self._plot_radial_average(radii, radial_average)
        self.write_radial_average(radii, radial_average, log)
        self.plot_potential_layers_z(
                data_arr, grid_spacing, grid_points, log)

    def _reshape_reevaluate_data(self,
                                 data: list[float],
                                 grid_points: list[int]
                                 ) -> np.ndarray:
        """reshape and reevaluate the data.
        In NumPy, the default order for reshaping (C order) is row-major,
        which means the last index changes fastest. This aligns with
        the way the data is ordered (z, y, x)."""
        return np.array(data).reshape(grid_points) * \
            self.configs.pot_unit_conversion

    def radial_average(self,
                       data_arr: np.ndarray,
                       grid_points: list[int],
                       grid_spacing: list[float],
                       ) -> tuple[np.ndarray, np.ndarray]:
        """Compute and plot the radial average of the potential from
        the center of the box."""
        # pylint: disable=too-many-locals

        # Calculate the center of the box in grid units
        center_xyz: tuple[int, int, int] = calculate_center(grid_points)

        # Calculate the maximum radius for the radial average
        max_radius: float = calculate_max_radius(center_xyz, grid_spacing)

        # Create the distance grid
        grid_xyz: tuple[np.ndarray, np.ndarray, np.ndarray] = \
            create_distance_grid(grid_points)

        # Calculate the distances from the center of the box
        distances: np.ndarray = compute_distance(
            grid_spacing, grid_xyz, center_xyz)

        # Calculate the radial average
        radii_average: "ClaculateRadialAveragePotential" = \
            ClaculateRadialAveragePotential(
                data_arr,
                distances,
                grid_spacing,
                max_radius,
                grid_xyz[2],
                self.configs.interface_low_index,
                self.configs.interface_high_index,
                self.configs.lower_index_bulk,
                self.configs)

        radii, radial_average = \
            radii_average.radii, radii_average.radial_average

        self.info_msg += ('\tThe average index is set to '
                          f'{self.configs.interface_low_index}\n'
                          f'\tThe maximum radius is {max_radius:.5f} [nm]\n')
        return radii, radial_average

    def _plot_radial_average(self,
                             radii: np.ndarray,
                             radial_average: np.ndarray
                             ) -> None:
        """Plot the radial average of the potential"""
        # Plot the radial average
        plt.figure(figsize=(20, 12))
        matplotlib.rcParams.update({'font.size': 22})

        plt.plot(radii/self.configs.dist_unit_conversion,
                 radial_average,
                 label='Radial Average of Potential',
                 color='grey',
                 lw=3,
                 )
        y_lims: tuple[float, float] = plt.ylim()
        plt.vlines(
            3.6, y_lims[0], y_lims[1], color='red', lw=3, linestyle='--')
        plt.ylim(y_lims)
        plt.xlabel('Radius [nm]')
        plt.ylabel('Average Potential')
        plt.title('Radial Average of Potential from the Center of the Box')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_potential_layers_z(self,
                                data_arr: np.ndarray,
                                grid_spacing: list[float],
                                grid_points: list[int],
                                log: logger.logging.Logger
                                ) -> None:
        """
        plot the average potential layers by changing z
        """
        # make sure the computation is in the plane not bulk

        PlotOverlayLayers(
            data_arr, grid_spacing, grid_points, self.configs, log)

    def write_radial_average(self,
                             radii: np.ndarray,
                             radial_average: np.ndarray,
                             log: logger.logging.Logger
                             ) -> None:
        """Write the radial average to a file"""
        # Write the radial average to a file
        convert_to_kj = [i*self.configs.pot_unit_conversion for i in
                         radial_average]
        data = {'Radius [nm]': radii/self.configs.dist_unit_conversion,
                'Average Potential [mV]': convert_to_kj,
                'Average Potential [kT/e]': radial_average
                }
        extra_msg_0 = ('# The radial average is set below the index: '
                       f'{self.configs.interface_low_index}')
        extra_msg = \
            [extra_msg_0,
             '# The conversion factor to [meV] is ',
             f'{self.configs.pot_unit_conversion}']
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


class ClaculateRadialAveragePotential:
    """Calculate the radial average potential"""
    configs: AllConfigs
    radii: np.ndarray
    radial_average: np.ndarray

    def __init__(self,
                 data_arr: np.ndarray,
                 distances: np.ndarray,
                 grid_spacing: list[float],
                 max_radius: float,
                 grid_z: np.ndarray,
                 interface_low_index,
                 interface_high_index,
                 lower_index_bulk,
                 configs: AllConfigs
                 ) -> None:
        """write and log messages"""
        # pylint: disable=too-many-arguments
        self.configs = configs
        self.radii, self.radial_average = \
            self.calculate_radial_average(data_arr,
                                          distances,
                                          grid_spacing,
                                          max_radius,
                                          grid_z,
                                          interface_low_index,
                                          interface_high_index,
                                          lower_index_bulk,
                                          )

    def calculate_radial_average(self,
                                 data_arr: np.ndarray,
                                 distances: np.ndarray,
                                 grid_spacing: list[float],
                                 max_radius: float,
                                 grid_z: np.ndarray,
                                 interface_low_index: int,
                                 interface_high_index: int,
                                 lower_index_bulk: int,
                                 ) -> tuple[np.ndarray, np.ndarray]:
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

        return radii, np.asarray(radial_average)

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


class PlotOverlayLayers:
    """plot potential selected layers"""
    # pylint: disable=too-many-locals
    info_msg: str = 'Message from PlotOverlayLayers:\n'
    configs: AllConfigs

    def __init__(self,
                 data_arr: np.ndarray,
                 grid_spacing: list[float],
                 grid_points: list[int],
                 configs: AllConfigs,
                 log: logger.logging.Logger
                 ) -> None:
        """write and log messages"""
        self.configs = configs
        self.configs.bulk_averaging = False
        radii_average_dict: dict[int | str, np.ndarray] = \
            self.plot_potential_layers_z(data_arr, grid_spacing, grid_points)
        self.plot_bpm_talk(radii_average_dict, log)

    def plot_potential_layers_z(self,
                                data_arr: np.ndarray,
                                grid_spacing: list[float],
                                grid_points: list[int]
                                ) -> dict[int | str, np.ndarray]:
        """
        plot the average potential layers by changing z
        """
        # make sure the computation is in the plane not bulk
        self.configs.bulk_averaging = False

        center_xyz: tuple[int, int, int] = calculate_center(grid_points)

        fig_i: plt.Figure
        ax_i: plt.Axes
        fig_i, ax_i = elsevier_plot_tools.mk_canvas('single_column')

        min_z_value: list[float] = []

        range_z: list[int] = self.get_zrange()
        colors: list[str]
        line_styles: list[tuple[str, tuple[int, typing.Any]]]
        long_list: bool
        colors, line_styles, long_list = self._get_colors_linestyle(range_z)
        radii_average_dict: dict[int | str, np.ndarray] = {}
        for i, z_index in enumerate(range_z):
            # Calculate the center of the box in grid units
            center_xyz = (center_xyz[0], center_xyz[1], z_index)
            # Calculate the maximum radius for the radial average
            max_radius: float = calculate_max_radius(center_xyz, grid_spacing)

            # Create the distance grid
            grid_xyz: tuple[np.ndarray, np.ndarray, np.ndarray] = \
                create_distance_grid(grid_points)

            # Calculate the distances from the center of the box
            distances: np.ndarray = compute_distance(
                grid_spacing, grid_xyz, center_xyz)

            radii_average: "ClaculateRadialAveragePotential" = \
                ClaculateRadialAveragePotential(
                    data_arr,
                    distances,
                    grid_spacing,
                    max_radius,
                    grid_xyz[2],
                    interface_low_index=z_index,
                    interface_high_index=z_index+self.configs.delta_z,
                    lower_index_bulk=0,
                    configs=self.configs)

            label: typing.Union[str, None] = self._get_lable(
                i, z_index, range_z, long_list)
            if z_index == self.configs.main_z:
                line_style = self.configs.main_z_linestyle[0][1]
            else:
                line_style = line_styles[i][1]
            ax_i.plot(radii_average.radii/self.configs.dist_unit_conversion,
                      radii_average.radial_average,
                      lw=1,
                      c=colors[i],
                      ls=line_style,
                      label=label,
                      )
            radii_average_dict[z_index] = radii_average.radial_average
            min_z_value.append(min(radii_average.radial_average))
        radii_average_dict['radii'] = radii_average.radii

        ax_i.set_xlabel('Radius [nm]')
        ax_i.set_ylabel('Average Potential')
        plt.legend()
        elsevier_plot_tools.save_close_fig(
            fig_i, 'potential_layers_z.jpg', close_fig=False)

        ax_i.set_xlim(self.configs.x_lims)
        ax_i.set_ylim(min(min_z_value)-self.configs.y_lims[0],
                      self.configs.y_lims[1])
        elsevier_plot_tools.save_close_fig(
            fig_i, 'potential_layers_z_zoom.jpg', close_fig=True)
        return radii_average_dict

    def _get_lable(self,
                   iteration: int,
                   z_index: int,
                   range_z: list[int],
                   long_list: bool
                   ) -> typing.Union[str, None]:
        """get the label"""
        if long_list:
            return (
                f'index = {z_index}' if (iteration in [0, len(range_z)-1])
                else None)
        return f'index = {z_index}'

    def get_zrange(self) -> list[int]:
        """get the z range for plotting"""
        z_range: list[int] = list(range(self.configs.lowest_z,
                                        self.configs.highest_z,
                                        self.configs.decrement_z))
        if self.configs.drop_z:
            for drop_z in self.configs.drop_z:
                if drop_z in z_range:
                    z_range.remove(drop_z)
        self.info_msg += f'\t{z_range = }\n'
        return z_range

    def _get_colors_linestyle(self,
                              range_z: list[int]) -> tuple[
                              list[str],
                              list[tuple[str, tuple[int, typing.Any]]],
                              bool]:
        """get the colors for plotting"""
        colors: list[str] = elsevier_plot_tools.CLEAR_COLOR_GRADIENT
        line_styles: list[tuple[str, tuple[int, typing.Any]]] = \
            elsevier_plot_tools.LINESTYLE_TUPLE
        long_list: bool = False
        if len(range_z) > len(colors):
            colors = elsevier_plot_tools.generate_shades(len(range_z))
            line_styles = [('solid', (0, ())) for _ in range(len(range_z))]
            long_list = True
            print(f'{bcolors.WARNING}The number of z layers is more than '
                  f'the number of colors!{bcolors.ENDC}')
        return colors, line_styles, long_list

    def plot_bpm_talk(self,
                      radii_average_dict: dict[int | str, np.ndarray],
                      log: logger.logging.Logger
                      ) -> None:
        """plot the potential layers for the talk"""
        average_range: tuple[int, int] = (90, 96)
        average_potetial: np.ndarray = np.zeros(
            radii_average_dict['radii'].shape)
        for z_index, radial_average in radii_average_dict.items():
            if z_index in average_range:
                average_potetial += radial_average
        average_potetial /= len(average_range)
        xdata, ydata = filter_data(
            radii_average_dict['radii']/self.configs.dist_unit_conversion,
            average_potetial)
        fig_i, ax_i = elsevier_plot_tools.mk_canvas('single_column')
        ax_i.plot(xdata,
                  ydata,
                  #   label=r'$\psi(r^\star)$ at interface',
                  lw=2,
                  c='black',
                  )
        golden_ratio: float = (1 + 5 ** 0.5) / 2
        hight: float = 2.35
        width: float = hight * golden_ratio
        fig_i.set_size_inches(width, hight)
        ax_i.set_yticks([0.0, 10.0, 20.0, 30.0])
        ax_i.set_xlabel(r'$r^\star$ [nm]', fontsize=14)
        ax_i.set_ylabel(r'$\psi(r^\star)$ [mV]', fontsize=14)
        ax_i.tick_params(axis='x', labelsize=14)  # X-ticks
        ax_i.tick_params(axis='y', labelsize=14)  # Y-ticks
        plot_tools.save_close_fig(fig_i,
                                  ax_i,
                                  'average_potential_interface_bpm_talk.jpg',
                                  loc='upper right',
                                  legend_font_size=13,
                                  )
        self.write_radial_average(xdata, ydata, log)

    def write_radial_average(self,
                             radii: np.ndarray,
                             radial_average: np.ndarray,
                             log: logger.logging.Logger
                             ) -> None:
        """Write the radial average to a file"""
        # Write the radial average to a file
        data = {'Radius [nm]': radii,
                'Average Potential [mV]': radial_average
                }
        extra_msg_0 = ('# The radial average is set below the index: '
                       f'{self.configs.interface_low_index}')
        extra_msg = [extra_msg_0]
        df_i = pd.DataFrame(data)
        df_i.set_index(df_i.columns[0], inplace=True)
        my_tools.write_xvg(df_i,
                           log,
                           extra_msg,
                           fname=self.configs.interface_file,
                           x_axis_label='radius [nm]')

def filter_data(xdata: np.ndarray,
                ydata: np.ndarray,
                min_radius: float = 1.65,
                max_radius: float = 10.0
                ) -> tuple[np.ndarray, np.ndarray]:
    """
    Filter the xdata and ydata to include only the values where xdata
    is between min_radius and max_radius.

    Parameters:
    xdata (np.ndarray): The radii data.
    ydata (np.ndarray): The potential data.
    min_radius (float): The minimum radius to include.
    max_radius (float): The maximum radius to include.

    Returns:
    tuple[np.ndarray, np.ndarray]: The filtered xdata and ydata.
    """
    # Create a boolean mask for the desired range
    mask = (xdata >= min_radius) & (xdata <= max_radius)

    # Apply the mask to xdata and ydata
    filtered_xdata = xdata[mask]
    filtered_ydata = ydata[mask]

    return filtered_xdata, filtered_ydata


if __name__ == '__main__':
    try:
        RadialAveragePotential().process_file(
            sys.argv[1],
            log=logger.setup_logger('radial_average_potential.log'))
    except IndexError:
        print(f'{bcolors.FAIL}No file is provided!{bcolors.ENDC}')
        sys.exit(1)
