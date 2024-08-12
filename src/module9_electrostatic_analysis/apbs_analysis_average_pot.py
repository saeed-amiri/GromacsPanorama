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

# pylint: disable=import-error

import sys
import typing
from dataclasses import dataclass

import numpy as np
import pandas as pd

import module9_electrostatic_analysis.apbs_analysis_average_pot_plots as \
    pot_plots

import module9_electrostatic_analysis.apbs_analysis_average_pot_tools as \
    pot_tools

from module9_electrostatic_analysis.apbs_analysis_average_pot_fits \
    import FitPotential
from module9_electrostatic_analysis.apbs_analysis_average_pot_read_dx \
    import ProcessDxFile
from module9_electrostatic_analysis.apbs_analysis_average_pot_sigma \
    import ComputeSigma
from module9_electrostatic_analysis.apbs_analysis_average_pot_boltzman_dist \
    import ComputeBoltzmanDistribution


from common import logger
from common import file_writer
from common.colors_text import TextColor as bcolors


@dataclass
class ParameterConfig:
    """set parameters for the average potential analysis
    computaion_radius: the radius of the sphere choosen for the
    computation of the average potential in Ångströms
    """
    computation_radius: float = 36.0
    diffuse_layer_threshold: float = 75.0  # Threshold for the diffuse layer A
    highest_np_grid_index: int = 99  # The highest grid index of the NP


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
    __slots__ = ['info_msg', 'configs', 'dx']
    info_msg: str
    configs: AllConfig
    dx: DxAttributeWrapper  # The dx file

    def __init__(self,
                 fname_dx: str,
                 log: logger.logging.Logger,
                 configs: AllConfig = AllConfig()
                 ) -> None:
        self.configs = configs
        self.info_msg = 'Message from AverageAnalysis:\n'
        self.read_dx(fname_dx, log)
        self.analyse_potential(log)
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

    def analyse_potential(self,
                          log: logger.logging.Logger
                          ) -> None:
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

        computed_dicts: tuple[dict[np.int64, float],
                              dict[np.int64, float]] | None = \
            self.compute_debye_surface_potential(cut_radii,
                                                 cut_radial_average,
                                                 cut_indices,
                                                 interset_radius,
                                                 sphere_grid_range,
                                                 radial_average_list
                                                 )
        if computed_dicts is None:
            return

        lambda_d, psi_zero = computed_dicts
        del computed_dicts

        self.compute_oda_boltzman_distribution(radial_average_list,
                                               sphere_grid_range,
                                               radii_list,
                                               log)

        sigma = self.compute_charge_density(lambda_d, psi_zero, log)

        self.plot_debye_surface_potential(lambda_d, 'lambda_d')
        self.plot_debye_surface_potential(psi_zero, 'psi_0')
        self.plot_debye_surface_potential(sigma, 'sigma')
        self.write_xvg({'lambda_d [A]': lambda_d,
                        'psi_0 [mV]': psi_zero,
                        'sigma [C/m^2]': sigma}, log)
        self._plot_debug(cut_radii.copy(),
                         cut_radial_average.copy(),
                         radii_list.copy(),
                         radial_average_list.copy(),
                         sphere_grid_range.copy())
        del sigma
        del cut_radii
        del radii_list
        del cut_radial_average

    def compute_debye_surface_potential(self,
                                        cut_radii: list[np.ndarray],
                                        cut_radial_average: list[np.ndarray],
                                        cut_indices: np.ndarray,
                                        interset_radius: np.ndarray,
                                        sphere_grid_range: np.ndarray,
                                        radial_average_list: list[np.ndarray]
                                        ) -> tuple[
                                            dict[np.int64, float],
                                            dict[np.int64, float]] | None:
        """Compute the surface potential and the decay constant
        The potetial decay part is fitted to the exponential decay
        \\psi = \\psi_0 * exp(-r/\\lambda_d)
        """
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-arguments
        if not self.configs.fit_potential:
            return None
        # Drop the cut_radial_average which have zero cut_indices
        cut_radial_average = [cut_radial_average[i] for i in range(
            len(cut_radial_average)) if cut_indices[i] != 0]
        cut_radii = [cut_radii[i] for i in range(
            len(cut_radii)) if cut_indices[i] != 0]
        # Fit the potential to the planar surface approximation
        plots_data = []
        lambda_d_dict: dict[np.int64, float] = {}
        psi_zero_dict: dict[np.int64, float] = {}
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

        del cut_radii
        del cut_indices
        del interset_radius
        del sphere_grid_range
        del cut_radial_average
        del radial_average_list

        return lambda_d_dict, psi_zero_dict

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

    @staticmethod
    def _fit_potential(r_np: float,
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
                                     data: dict[np.int64, float],
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
            radii, radial_average = pot_tools.process_layer(
                center_xyz=center_xyz,
                grid_spacing=self.dx.GRID_SPACING,
                grid_points=self.dx.GRID_POINTS,
                data_arr=self.dx.DATA_ARR,
                bulk_averaging=self.configs.bulk_averaging,
                )
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

    def find_grid_inidices_covers_shpere(self,
                                         center_xyz: tuple[int, int, int],
                                         ) -> np.ndarray:
        """Find the grid points within the NP"""
        grid_size: float = self.dx.BOX_SIZE[2] / self.dx.GRID_POINTS[2]
        self.info_msg += f'\tGird size: {grid_size:.4f}\n'
        radius: float = self.configs.computation_radius
        nr_grids_coveres_sphere_radius: int = int(radius / grid_size)
        lowest_z_index: int = center_xyz[2] - nr_grids_coveres_sphere_radius
        highest_z_index: int = self.configs.highest_np_grid_index
        return np.arange(lowest_z_index, highest_z_index)

    @staticmethod
    def compute_charge_density(lambda_d: dict[np.int64, float],
                               psi_zero: dict[np.int64, float],
                               log: logger.logging.Logger
                               ) -> dict[np.int64, float]:
        """Compute the charge density"""
        lambda_d_arr: np.ndarray = np.array(list(lambda_d.values()))
        psi_arr: np.ndarray = np.array(list(psi_zero.values()))
        sigma: ComputeSigma = ComputeSigma(psi_arr, lambda_d_arr, log)
        z_index: list[np.int64] = list(lambda_d.keys())
        sigma_dict: dict[np.int64, float] = {}
        sigma_dict = {z: sigma.sigma[i] for i, z in enumerate(z_index)}
        return sigma_dict

    @staticmethod
    def compute_oda_boltzman_distribution(cut_radial_average: list[np.ndarray],
                                          sphere_grid_range: np.ndarray,
                                          radii_list: list[np.ndarray],
                                          log: logger.logging.Logger
                                          ) -> None:
        """Compute the Boltzman distribution"""
        dist = ComputeBoltzmanDistribution(cut_radial_average,
                                           sphere_grid_range,
                                           radii_list,
                                           log)
        dist_radii: dict[int, tuple[np.ndarray, np.ndarray]] = \
            dist.boltzmann_distribution
        # get arg of clooses radii to 100
        cut_ind = np.argmin(np.abs(radii_list[0] - 100))

        pot_plots.plot_boltzman_distribution(dist_radii, cut_ind)
        pot_plots.plot_all_boltman_distribution(dist.all_distribution, cut_ind)

        del radii_list
        del cut_radial_average

    def write_xvg(self,
                  data: dict[str, dict[np.int64, float]],
                  log: logger.logging.Logger
                  ) -> None:
        """Write the data to xvg file"""
        column_names: list[str] = list(data.keys())
        z_index: list[np.int64] = list(data[column_names[0]].keys())
        z_loc: list[float] = [
            float(i)*self.dx.GRID_SPACING[2] for i in z_index]
        single_data: dict[str, list[float] | list[np.int64]] = {
            'index': z_index,
            'z': z_loc,
            column_names[0]: list(data[column_names[0]].values()),
            column_names[1]: list(data[column_names[1]].values()),
            column_names[2]: list(data[column_names[2]].values())
            }
        df_i: pd.DataFrame = pd.DataFrame.from_dict(single_data)
        file_writer.write_xvg(df_i=df_i,
                              log=log,
                              fname='debye_surface_pot.xvg',
                              extra_comments='# Debye surface potential data',
                              xaxis_label='z index',
                              )

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
