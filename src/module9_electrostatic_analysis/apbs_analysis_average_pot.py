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
from dataclasses import dataclass

import numpy as np

import module9_electrostatic_analysis.apbs_analysis_average_pot_plots as \
    pot_plots

import module9_electrostatic_analysis.apbs_analysis_average_tools as \
    pot_tools

from module9_electrostatic_analysis.apbs_analysis_average_pot_fits import \
    FitPotential
from module9_electrostatic_analysis.apbs_analysis_average_pot_read_dx import \
    ProcessDxFile

from common import logger
from common.colors_text import TextColor as bcolors


@dataclass
class ParameterConfig:
    """set parameters for the average potential analysis
    computaion_radius: the radius of the sphere choosen for the
    computation of the average potential in Ångströms
    """
    computation_radius: float = 36.0
    diffuse_layer_threshold: float = 75.0  # Threshold for the diffuse layer A


@dataclass
class AllConfig(ParameterConfig):
    """set all the configs and parameters
    Also possible of compare them:
    """
    # pylint: disable=too-many-instance-attributes
    bulk_averaging: bool = False  # if Bulk averaging else interface averaging
    debug_plot: bool = False
    fit_potential: bool = True
    plot_interactive: bool = False


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
        read_dx = ProcessDxFile(fname_dx, log)
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
            pot_tools.calculate_center(self.dx.GRID_POINTS)

        sphere_grid_range: np.ndarray = \
            self.find_grid_inidices_covers_shpere(center_xyz)

        self.info_msg += (
            f'\tThe centeral grid is: {center_xyz}\n'
            f'\tThe computation radius is: {self.configs.computation_radius}\n'
            f'\tNr. grids cover the sphere: {len(sphere_grid_range)}\n'
            f'\tThe lowest grid index: {sphere_grid_range[0]}\n'
            f'\tThe highest grid index: {sphere_grid_range[-1]}\n'
            )
        radii_list: list[np.ndarray]
        radial_average_list: list[np.ndarray]
        radii_list, radial_average_list = \
            self.compute_all_layers(center_xyz, sphere_grid_range)
        cut_radii, cut_radial_average, cut_indices, interset_radius = \
            self.cut_average_from_surface(sphere_grid_range,
                                          center_xyz,
                                          radii_list,
                                          radial_average_list)
        self._plot_debug(cut_radii, cut_radial_average, radii_list,
                         radial_average_list, sphere_grid_range)

        self.compute_debye_surface_potential(cut_radii,
                                             cut_radial_average,
                                             cut_indices,
                                             interset_radius,
                                             sphere_grid_range,
                                             radial_average_list
                                             )

    def compute_debye_surface_potential(self,
                                        cut_radii: list[np.ndarray],
                                        cut_radial_average: list[np.ndarray],
                                        cut_indices: np.ndarray,
                                        interset_radius: np.ndarray,
                                        sphere_grid_range: np.ndarray,
                                        radial_average_list: list[np.ndarray]
                                        ) -> None:
        """Compute the surface potential and the decay constant
        The potetial decay part is fitted to the exponential decay
        \\psi = \\psi_0 * exp(-r/\\lambda_d)
        """
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-arguments
        if not self.configs.fit_potential:
            return
        # Drop the cut_radial_average which have zero cut_indices
        cut_radial_average = [cut_radial_average[i] for i in range(
            len(cut_radial_average)) if cut_indices[i] != 0]
        cut_radii = [cut_radii[i] for i in range(
            len(cut_radii)) if cut_indices[i] != 0]
        # Fit the potential to the planar surface approximation
        plots_data = []
        lambda_d_dict: dict[str, float] = {}
        psi_zero_dict: dict[str, float] = {}
        for r_np, radii, radial_average, grid, uncut_psi in zip(
           interset_radius,
           cut_radii,
           cut_radial_average,
           sphere_grid_range,
           radial_average_list):

            psi_inf: float = np.min(uncut_psi)
            fit: "FitPotential" = \
                self._fit_potential(r_np, radii, radial_average, psi_inf)
            plots_data.append((radii,
                               radial_average,
                               fit.fitted_pot,
                               fit.popt,
                               fit.evaluate_fit,
                               grid))
            if fit.evaluate_fit[0] > 0.99:
                lambda_d_dict[grid] = fit.popt[0]
                psi_zero_dict[grid] = fit.popt[1]

        if self.configs.plot_interactive:
            self._interactive_plot(plots_data)

        self.plot_debye_surface_potential(lambda_d_dict, 'lambda_d')
        self.plot_debye_surface_potential(psi_zero_dict, 'psi_0')

    def _interactive_plot(self,
                          plots_data: list[tuple[np.ndarray,
                                                 np.ndarray,
                                                 np.ndarray,
                                                 np.ndarray,
                                                 tuple[float, float, float],
                                                 int]],
                          ) -> None:
        """Interactive plot for the fitted potential"""
        pot_plots.interactive_plot(plots_data, self.dx.GRID_SPACING[2])

    def _fit_potential(self,
                       r_np: float,
                       radii: np.ndarray,
                       radial_average: np.ndarray,
                       psi_inf: float
                       ) -> "FitPotential":
        """Fit the potential to the planar surface approximation"""
        return FitPotential(radii, radial_average, r_np, psi_inf)

    def _plot_debug(self,
                    cut_radii: list[np.ndarray],
                    cut_radial_average: list[np.ndarray],
                    radii_list: list[np.ndarray],
                    radial_average_list: list[np.ndarray],
                    sphere_grid_range: np.ndarray
                    ) -> None:
        """Plot for debugging"""
        # pylint: disable=too-many-arguments
        if not self.configs.debug_plot:
            return
        pot_plots.plot_debug(cut_radii,
                             cut_radial_average,
                             radii_list,
                             radial_average_list,
                             sphere_grid_range)

    def plot_debye_surface_potential(self,
                                     data: dict[str, float],
                                     type_data: str
                                     ) -> None:
        """Plot the Debye length and surface potential"""
        pot_plots.plot_debye_surface_potential(data,
                                               type_data,
                                               self.dx.GRID_SPACING)

    def compute_all_layers(self,
                           center_xyz: tuple[int, int, int],
                           sphere_grid_range: np.ndarray
                           ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Compute the radial average for all layers"""
        radii_list: list[np.ndarray] = []
        radial_average_list: list[np.ndarray] = []
        for layer in sphere_grid_range:
            center_xyz = (center_xyz[0], center_xyz[1], layer)
            radii, radial_average = self.process_layer(center_xyz)
            radii_list.append(radii)
            radial_average_list.append(radial_average)
        return radii_list, radial_average_list

    def cut_average_from_surface(self,
                                 sphere_grid_range: np.ndarray,
                                 center_xyz: tuple[int, int, int],
                                 radii_list: list[np.ndarray],
                                 radial_average_list: list[np.ndarray],
                                 ) -> tuple[list[np.ndarray],
                                            list[np.ndarray],
                                            np.ndarray,
                                            np.ndarray,
                                            ]:
        """Cut the average from the surface based on the circle's radius
        of the intesection of the sphere with the grid in z-axis"""
        # pylint: disable=too-many-locals
        radius: float = self.configs.computation_radius
        center_z: int = center_xyz[2]
        interset_radius: np.ndarray = \
            pot_tools.calculate_grid_sphere_intersect_radius(
                radius, center_z, sphere_grid_range, self.dx.GRID_SPACING[2])
        cut_indices_i: np.ndarray = \
            pot_tools.find_inidices_of_surface(interset_radius, radii_list)
        cut_indices_f: np.ndarray = pot_tools.find_indices_of_diffuse_layer(
            radii_list, self.configs.diffuse_layer_threshold)
        cut_radial_average: list[np.ndarray] = []
        cut_radii: list[np.ndarray] = []
        for i, (radial_average, radii) in enumerate(zip(radial_average_list,
                                                        radii_list)):
            cut_i = int(cut_indices_i[i])
            cut_f = int(cut_indices_f[i])
            cut_radial_average.append(radial_average[cut_i:cut_f])
            cut_radii.append(radii[cut_i:cut_f])
        return cut_radii, cut_radial_average, cut_indices_i, interset_radius

    def process_layer(self,
                      center_xyz: tuple[int, int, int]
                      ) -> tuple[np.ndarray, np.ndarray]:
        """process the layer
        The potential from the surface until the end of the diffuse layer
        """
        max_radius: float = pot_tools.calculate_max_radius(
            center_xyz, self.dx.GRID_SPACING)
        # Create the distance grid
        grid_xyz: tuple[np.ndarray, np.ndarray, np.ndarray] = \
            pot_tools.create_distance_grid(self.dx.GRID_POINTS)
        # Calculate the distances from the center of the box
        distances: np.ndarray = pot_tools.compute_distance(
                self.dx.GRID_SPACING, grid_xyz, center_xyz)
        radii, radial_average = pot_tools.calculate_radial_average(
                self.dx.DATA_ARR,
                distances,
                self.dx.GRID_SPACING,
                max_radius,
                grid_xyz[2],
                interface_low_index=center_xyz[2],
                interface_high_index=center_xyz[2],
                lower_index_bulk=0,
                bulk_averaging=self.configs.bulk_averaging
                )
        return radii, np.asanyarray(radial_average)

    def find_grid_inidices_covers_shpere(self,
                                         center_xyz: tuple[int, int, int],
                                         ) -> np.ndarray:
        """Find the grid points within the NP"""
        grid_size: float = self.dx.BOX_SIZE[2] / self.dx.GRID_POINTS[2]
        self.info_msg += f'\tGird size: {grid_size:.4f}\n'
        radius: float = self.configs.computation_radius
        nr_grids_coveres_sphere_radius: int = int(radius / grid_size)
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


if __name__ == '__main__':
    AverageAnalysis(sys.argv[1],
                    logger.setup_logger('avarge_potential_analysis.log'))
