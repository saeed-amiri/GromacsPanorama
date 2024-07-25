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
import inspect
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import matplotlib as mpl

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
    diffuse_layer_threshold: float = 75.0  # Threshold for the diffuse layer A


@dataclass
class AllConfig(ParameterConfig):
    """set all the configs and parameters
    fit_function: the function used to fit the potential:
        exponential_decay or linear_sphere or non_linear_sphere
    Also possible of compare them:
    fit_comparisons: bool = False or True
    """
    dx_configs: DxFileConfig = field(default_factory=DxFileConfig)
    bulk_averaging: bool = False  # if Bulk averaging else interface averaging
    debug_plot: bool = False
    fit_potential: bool = False
    fit_function: str = 'non_linear_sphere'
    fit_comparisons: bool = False
    debye_intial_guess: float = 75.0


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
                                             )

    def compute_debye_surface_potential(self,
                                        cut_radii: list[np.ndarray],
                                        cut_radial_average: list[np.ndarray],
                                        cut_indices: np.ndarray,
                                        interset_radius: np.ndarray,
                                        ) -> None:
        """Compute the surface potential and the decay constant
        The potetial decay part is fitted to the exponential decay
        \\psi = \\psi_0 * exp(-r/\\lambda_d)
        """
        if not self.configs.fit_potential:
            return
        # Drop the cut_radial_average which have zero cut_indices
        cut_radial_average = [cut_radial_average[i] for i in range(
            len(cut_radial_average)) if cut_indices[i] != 0]
        cut_radii = [cut_radii[i] for i in range(
            len(cut_radii)) if cut_indices[i] != 0]
        # Fit the potential to the planar surface approximation
        for r_np, radii, radial_average in zip(interset_radius,
                                               cut_radii,
                                               cut_radial_average):
            fit: "FitPotential" = \
                self._fit_potential(r_np, radii, radial_average)
            fitted_pot: np.ndarray = fit.fitted_pot
            popt: np.ndarray = fit.popt

    def _fit_potential(self,
                       r_np: float,
                       radii: np.ndarray,
                       radial_average: np.ndarray,
                       ) -> "FitPotential":
        """Fit the potential to the planar surface approximation"""
        return FitPotential(radii, radial_average, r_np, self.configs)

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
        for average, ind in zip(cut_radial_average, sphere_grid_range):
            average -= average[0]
            average += ind
        mpl.rcParams['font.size'] = 20

        for i, radial_average in enumerate(cut_radial_average):
            _, ax = plt.subplots(figsize=(30, 16))
            ax.plot(radii_list[i], radial_average_list[i], 'r:')
            ax.plot(cut_radii[i],
                    radial_average,
                    'k-',
                    label=f'z={sphere_grid_range[i]}')
            plt.legend()
            plt.show()

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
            self._calculate_grid_sphere_intersect_radius(radius,
                                                         center_z,
                                                         sphere_grid_range)
        cut_indices_i: np.ndarray = \
            self._find_inidices_of_surface(interset_radius, radii_list)
        cut_indices_f: np.ndarray = self._find_indices_of_diffuse_layer(
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

    def _calculate_grid_sphere_intersect_radius(self,
                                                radius: float,
                                                center_z: int,
                                                sphere_grid_range: np.ndarray
                                                ) -> np.ndarray:
        """Get radius of the intersection between the sphere and the
        grid plane"""
        radius_index: np.ndarray = np.zeros(len(sphere_grid_range))
        for i, z_index in enumerate(sphere_grid_range):
            height: float = \
                np.abs(center_z - z_index) * self.dx.GRID_SPACING[2]
            if height <= radius:
                radius_index[i] = np.sqrt(radius**2 - height**2)
        return radius_index

    def _find_inidices_of_surface(self,
                                  interset_radius: np.ndarray,
                                  radii_list: list[np.ndarray]
                                  ) -> np.ndarray:
        """Find the indices of the surface by finding the index of the
        closest radius to the intersection radius"""
        cut_indices: np.ndarray = np.zeros(len(interset_radius))
        for i, radius in enumerate(interset_radius):
            cut_indices[i] = np.argmin(np.abs(radii_list[i] - radius))
        return cut_indices

    def _find_indices_of_diffuse_layer(self,
                                       radii_list: list[np.ndarray],
                                       threshold: float) -> np.ndarray:
        """Find the indices of the diffuse layer by finding the index
        of the radial average which is less than the threshold"""
        cut_indices: np.ndarray = np.ones(len(radii_list)) * -1
        for i, radii in enumerate(radii_list):
            cut_indices[i] = np.argmin(radii - threshold <= 0)
        return cut_indices

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
                interface_low_index=center_xyz[2],
                interface_high_index=center_xyz[2],
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
        self.info_msg += f'\tGird size: {grid_size:.4f}\n'
        radius: float = self.configs.computation_radius
        nr_grids_coveres_sphere_radius: int = int(radius / grid_size) + 2
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


class FitPotential:
    """Fitting the decay of the potential"""
    info_msg: str = 'Message from FitPotential:\n'
    config: AllConfig
    fitted_pot: np.ndarray
    popt: np.ndarray

    def __init__(self,
                 radii: np.ndarray,
                 radial_average: np.ndarray,
                 r_np: float,
                 config: AllConfig,
                 ) -> None:
        self.config = config
        fitted_func: typing.Callable[..., np.ndarray]
        fitted_func, self.popt = \
            self.fit_potential(radii, radial_average, r_np)
        self.fitted_pot = fitted_func(radii, *self.popt)

    def fit_potential(self,
                      radii: np.ndarray,
                      radial_average: np.ndarray,
                      r_np: float
                      ) -> tuple[typing.Callable[..., np.ndarray], np.ndarray]:
        """Fit the potential to the planar surface approximation"""
        # Define the exponential decay function

        # Initial guess for the parameters [psi_0, lambda_d]

        # Use curve_fit to find the best fitting parameters
        # popt contains the optimal values for psi_0 and lambda_d
        fit_fun: typing.Callable[..., np.ndarray] = self.get_fit_function()

        initial_guess: list[float] = self.get_initial_guess(
            radial_average[0], self.config.debye_intial_guess, r_np)

        popt, *_ = curve_fit(f=fit_fun,
                             xdata=radii,
                             ydata=radial_average,
                             p0=initial_guess,
                             maxfev=10000)
        return fit_fun, popt

    def get_fit_function(self) -> typing.Callable[..., np.ndarray]:
        """Get the fit function"""
        fit_fun_type: str = self.validate_fit_function()
        return {
            'exponential_decay': self.exp_decay,
            'linear_sphere': self.linear_sphere,
            'non_linear_sphere': self.non_linear_sphere,
        }[fit_fun_type]

    def get_initial_guess(self,
                          phi_0: float,
                          lambda_d: float,
                          r_np: float
                          ) -> list[float]:
        """Get the initial guess for the Debye length"""
        fit_fun_type = self.validate_fit_function()
        return {
            'exponential_decay': [phi_0, lambda_d],
            'linear_sphere': [phi_0, lambda_d, r_np],
            'non_linear_sphere': [phi_0, lambda_d, r_np],
        }[fit_fun_type]

    @staticmethod
    def get_function_args(func: typing.Callable[..., np.ndarray]
                          ) -> list[str]:
        """
        Get the list of argument names for a given function, excluding 'self'.
        """
        signature = inspect.signature(func)
        return [
            name for name, _ in signature.parameters.items() if name != 'self']

    @staticmethod
    def exp_decay(radius: np.ndarray,
                  psi_0: float,
                  lambda_d: float,
                  ) -> np.ndarray:
        """Exponential decay function"""
        return psi_0 * np.exp(-radius / lambda_d)

    @staticmethod
    def linear_sphere(radius: np.ndarray,
                      psi_0: float,
                      lambda_d: float,
                      r_np: float
                      ) -> np.ndarray:
        """Linear approximation of the potential"""
        return psi_0 * np.exp(-(radius - r_np) / lambda_d) * r_np / radius

    @staticmethod
    def non_linear_sphere(radius: np.ndarray,
                          psi_0: float,
                          lambda_d: float,
                          r_np: float
                          ) -> np.ndarray:
        """Non-linear approximation of the potential"""
        parameters: dict[str, float] = {
            'e_charge': 1.602176634e-19,
            'epsilon_0': 8.854187817e-12,
            'k_b': 1.380649e-23,
            'T': 298.15,
            'n_avogadro': 6.022e23,
            }
        alpha: float = np.arctanh(parameters['e_charge'] * psi_0 /
                                  (4 * parameters['k_b'] * parameters['T']))
        radial_term: np.ndarray = r_np / radius
        power_term: np.ndarray = (radius - r_np) / lambda_d
        alpha_term: np.ndarray = alpha * np.exp(-power_term) * radial_term
        co_factor: float = parameters['k_b'] * parameters['T'] / \
            parameters['e_charge']
        return co_factor * np.log((1 + alpha_term) / (1 - alpha_term))

    def validate_fit_function(self) -> str:
        """Validate and return the fit function type from config"""
        fit_fun_type = self.config.fit_function
        valid_functions = [
            'exponential_decay', 'linear_sphere', 'non_linear_sphere']
        if fit_fun_type not in valid_functions:
            raise ValueError(
                f'\n\tThe fit function: `{fit_fun_type}` is not valid!\n'
                f'\tThe valid options are: \n'
                f'\t{" ,".join(valid_functions)}\n')
        return fit_fun_type


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
