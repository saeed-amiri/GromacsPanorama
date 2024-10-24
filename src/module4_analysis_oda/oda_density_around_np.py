"""
For each defined region, calculate the number density of ODA molecules.
This is typically done by counting the number of ODA molecules in each
region and dividing by the volume of that region.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from common import logger
from common import my_tools
from common import static_info as stinfo
from common import xvg_to_dataframe as xvg
from common.colors_text import TextColor as bcolors

from module4_analysis_oda import fit_rdf


@dataclass
class OdaInputFilesConfig:
    """set the input for analysing"""
    contact_xvg: str = 'contact.xvg'
    np_coord_xvg: str = 'coord.xvg'
    box_xvg: str = 'box.xvg'


@dataclass
class ParameterConfig:
    """set the default parameters for the calculataion"""
    number_of_regions: int = 150
    time_dependent_step: int = 100
    xvg_output: str = 'densities.xvg'
    if_clean_np: bool = True  # If true, the ODA before r* will be cleaned


class SurfactantDensityAroundNanoparticle:
    """self explained"""
    info_msg: str = 'Message from SurfactantDensityAroundNanoparticle:\n'
    input_config: "OdaInputFilesConfig"
    param_config: "ParameterConfig"
    contact_data: pd.DataFrame  # The contact data (from module3)
    box: np.ndarray  # Size of the box at each frame (from gromacs)
    np_com: np.ndarray  # COM of the NP at each frame (from gromacs)
    interface_z: np.ndarray  # Z location of the interface (from module3)
    density_per_region: dict[float, list[float]]  # Density per area

    avg_density_per_region: dict[float, float]
    rdf_2d: dict[float, float]  # rdf (g((r)) in 2d
    fitted_rdf: dict[float, float]  # fitted rdf
    smoothed_rdf: dict[float, float]  # smoothed rdf

    time_dependent_rdf: dict[int, dict[float, float]]
    time_dependent_ave_density: dict[int, dict[float, float]]
    midpoint: float  # midpoint of the fit
    first_turn: float
    second_turn: float

    def __init__(self,
                 amino_arr: np.ndarray,  # amino head com of the oda
                 log: logger.logging.Logger,
                 input_config: "OdaInputFilesConfig" = OdaInputFilesConfig(),
                 param_config: "ParameterConfig" = ParameterConfig(),
                 residue: str = 'AMINO_ODN'
                 ) -> None:
        # The two last rows of amino_arr are indicies from main trr file
        amino_arr = amino_arr[:-2]
        self.input_config = input_config
        self.param_config = param_config
        regions: list[float] = self._initiate(amino_arr, log, residue)
        self.write_to_xvg(regions, log)
        self.write_msg(log)

    def _initiate(self,
                  amino_arr: np.ndarray,
                  log: logger.logging.Logger,
                  residue: str
                  ) -> list[float]:
        """Initiate the calculation by checking necessary files."""
        self.check_input_files(log, self.input_config)

        self.contact_data: pd.DataFrame = self.load_contact_data(log)
        np_com_df: pd.DataFrame = self.load_np_com_data(log)
        box_df: pd.DataFrame = self.load_box_data(log)
        self.initialize_data_arrays(np_com_df, box_df, log)
        regions: list[float] = self.generate_regions(
            self.param_config.number_of_regions, update_msg=True)
        self.density_per_region, num_oda_in_radius = \
            self.initialize_calculation(amino_arr, regions, log)

        self.avg_density_per_region = \
            self._comput_avg_density(self.density_per_region)

        self.rdf_2d = \
            self._comput_2d_rdf(self.density_per_region, num_oda_in_radius)

        if residue == 'AMINO_ODN':
            self._clean_np()
            self.time_dependent_rdf, self.time_dependent_ave_density = \
                self.calculate_time_dependent_densities(
                    amino_arr, num_oda_in_radius, regions, log)

            self._fit_and_set_fitted2d_rdf(log)
        self._comput_and_set_moving_average(3)
        return regions

    def initialize_calculation(self,
                               amino_arr: np.ndarray,
                               regions: list[float],
                               log: logger.logging.Logger
                               ) -> tuple[dict[float, list[float]],
                                          list[int]]:
        """getting the density number from the parsed data"""
        z_threshold: np.ndarray = \
            self.compute_surfactant_vertical_bounds(log)
        # Initialize a dictionary to store densities for each region
        density_per_region: dict[float, list[float]] = \
            {region: [] for region in regions}
        num_oda: list[int] = []
        num_oda_in_radius: list[int] = []
        for i, frame_i in enumerate(amino_arr):
            arr_i: np.ndarray = \
                self._get_surfactant_at_interface(frame_i, z_threshold[i])
            distance: np.ndarray = self.compute_pbc_distance(i, arr_i)
            density_per_region, count_in_radius = \
                self._compute_density_per_region(regions,
                                                 distance,
                                                 density_per_region)
            num_oda.append(len(arr_i))
            num_oda_in_radius.append(count_in_radius)
        return density_per_region, num_oda_in_radius

    def _comput_avg_density(self,
                            density_per_region:
                            dict[float, list[float]]
                            ) -> dict[float, float]:
        """self explanatory"""
        avg_density_per_region: dict[float, float] = {}
        for region, densities in density_per_region.items():
            if densities:
                avg_density_per_region[region] = np.mean(densities)
            else:
                avg_density_per_region[region] = 0
        return avg_density_per_region

    def _comput_2d_rdf(self,
                       density_per_region: dict[float, list[float]],
                       num_oda: list[int]
                       ) -> dict[float, float]:
        """set the 2d rdf (g(r))"""
        max_radius_area: float = \
            max(item for item in density_per_region.keys())
        rdf_2d: dict[float, float] = {}
        for region, densities in density_per_region.items():
            if not densities:
                rdf_2d[region] = 0
                continue

            tmp = []
            for j, item in enumerate(densities):
                density: float = num_oda[j]/(np.pi * max_radius_area**2)
                tmp.append(item/density)
            rdf_2d[region] = np.mean(tmp)
        return rdf_2d

    def _clean_np(self) -> dict[np.ndarray, np.ndarray]:
        """clean the rdf before the contact radius by setting them the
        min value of the rdf values"""

        if self.param_config.if_clean_np:
            min_value = min(self.rdf_2d.values())
            contact_radius: float = \
                self.contact_data.loc[:, 'contact_radius'].mean()
            for key in self.rdf_2d.keys():
                if key < contact_radius:
                    self.rdf_2d[key] = min_value

    def _fit_and_set_fitted2d_rdf(self,
                                  log: logger.logging.Logger
                                  ) -> None:
        """
        fit and smooth the rdf using the 5PL2S function with initial
        guesses based on derivatives"""
        fitted_rdf = fit_rdf.FitRdf2dTo5PL2S(self.rdf_2d, log)
        self.fitted_rdf = fitted_rdf.fitted_rdf
        self.midpoint = fitted_rdf.midpoind
        self.first_turn = fitted_rdf.first_turn
        self.second_turn = fitted_rdf.second_turn

    def calculate_time_dependent_densities(self,
                                           amino_arr: np.ndarray,
                                           num_oda_in_radius: list[int],
                                           regions: list[float],
                                           log: logger.logging.Logger
                                           ) -> tuple[dict[int,
                                                      dict[float, float]],
                                                      dict[int,
                                                      dict[float, float]]]:
        """calculate avedensity and rdf as function of time"""
        step: int = self.param_config.time_dependent_step
        time_dependent_rdf: dict[int, dict[float, float]] = {}
        time_dependent_ave_density: dict[int, dict[float, float]] = {}
        for i in range(0, amino_arr.shape[0], step):
            amino_arr_i = amino_arr[:i].copy()
            density_per_region, num_oda_in_radius = \
                self.initialize_calculation(amino_arr_i, regions, log)
            rdf_i = self._comput_2d_rdf(density_per_region, num_oda_in_radius)
            try:
                time_dependent_rdf[i] = \
                    fit_rdf.FitRdf2dTo5PL2S(rdf_i, log).fitted_rdf
                time_dependent_ave_density[i] = \
                    self._comput_avg_density(density_per_region)
            except RuntimeError:
                pass
        return time_dependent_rdf, time_dependent_ave_density

    @staticmethod
    def _compute_density_per_region(regions: list[float],
                                    distance: np.ndarray,
                                    density_per_region:
                                    dict[float, list[float]]
                                    ) -> tuple[dict[float, list[float]], int]:
        """self explanatory"""
        # Here, the code increments the count in the appropriate bin
        # by 2. The increment by 2 is necessary because each pair
        # contributes to two interactions: one for each atom in the
        # pair being considered as the reference atom. In simpler
        # terms, for each pair of atoms, both atoms contribute to the
        # density at this distance, so the count for this bin is
        # increased by 2 to account for both contributions.
        count_in_radius = 2
        for ith in range(len(regions) - 1):
            r_inner = regions[ith]
            r_outer = regions[ith + 1]
            count = np.sum((distance >= r_inner) & (distance < r_outer))
            area = np.pi * (r_outer**2 - r_inner**2)
            density = count / area
            density_per_region[r_outer].append(density)
            count_in_radius += count
        return density_per_region, count_in_radius

    def generate_regions(self,
                         nr_of_regions: int,
                         update_msg: bool = True
                         ) -> list[float]:
        """divide the the area around the np for generating regions"""
        max_box_len: np.float64 = np.max(self.box[0][:2]) / 2
        if update_msg:
            self.info_msg += (
                f'\tThe number of regions is: `{nr_of_regions}`\n'
                f'\tThe half of the max length of box is `{max_box_len:.3f}`\n'
                f'\tThe bin size is: `{max_box_len/nr_of_regions:.3f}`\n'
            )
        return np.linspace(0, max_box_len, nr_of_regions).tolist()

    def _comput_and_set_moving_average(self,
                                       window_size: int
                                       ) -> None:
        """
        Apply moving average on rdf_2d data.

        Parameters:
        window_size (int): The number of data points to include in each
        average.

        Returns:
        ndarray: The rdf_2d data after applying the moving average.
        """
        radii = np.array(list(self.rdf_2d.keys()))
        rdf_values = np.array(list(self.rdf_2d.values()))

        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer.")

        smoothed_rdf = np.convolve(
            rdf_values, np.ones(window_size)/window_size, mode='valid')
        self.smoothed_rdf = dict(zip(radii, smoothed_rdf))

    @staticmethod
    def _get_surfactant_at_interface(frame_i: np.ndarray,
                                     z_threshold_i: np.ndarray
                                     ) -> np.ndarray:
        """return the oda at the interface"""
        frame_reshaped: np.ndarray = frame_i.reshape(-1, 3)
        below_threshold_indices: np.ndarray = \
            np.where((frame_reshaped[:, 2] < z_threshold_i[1]) &
                     (frame_reshaped[:, 2] > z_threshold_i[0]))
        return frame_reshaped[below_threshold_indices]

    def compute_pbc_distance(self,
                             frame_index: int,
                             arr: np.ndarray
                             ) -> np.ndarray:
        """claculating the distance between the np and the surfactants
        at each frame and return an array
        Only considering 2d distance, in the XY plane
        """
        np_com: np.ndarray = self.np_com[frame_index]
        box: np.ndarray = self.box[frame_index]
        dx_i = arr[:, 0] - np_com[0]
        dx_pbc = dx_i - (box[0] * np.round(dx_i/box[0]))
        dy_i = arr[:, 1] - np_com[1]
        dy_pbc = dy_i - (box[1] * np.round(dy_i/box[1]))
        return np.sqrt(dx_pbc*dx_pbc + dy_pbc*dy_pbc)

    def compute_surfactant_vertical_bounds(self,
                                           log: logger.logging.Logger
                                           ) -> np.ndarray:
        """
        computes the vertical bounds for surfactants to be excluded
        from calculations.
        """
        average_frames: int = 10
        self.info_msg += (f'\t`{average_frames}` first frames used for setting'
                          f' thickness of the interface\n')
        contact_angle: np.ndarray = \
            self.parse_contact_data(self.contact_data, 'contact_angles', log)
        cos_contact_angles: np.ndarray = np.cos(np.radians(contact_angle)) + 1

        # Calculate the height of the nanoparticle from the interface
        np_radius = stinfo.np_info['radius']
        h_np: np.float64 = \
            np.mean(np.abs(np_radius * cos_contact_angles[:average_frames]))
        tip_of_np_berg: np.ndarray = 2 * np_radius - h_np

        interface_std: np.float64 = np.std(self.interface_z)
        upper_bounds: np.ndarray = \
            self.interface_z + interface_std + tip_of_np_berg
        lower_bounds: np.ndarray = \
            self.interface_z - interface_std - tip_of_np_berg

        bounds = np.column_stack((lower_bounds, upper_bounds))

        return bounds

    def initialize_data_arrays(self,
                               np_com_df: pd.DataFrame,
                               box_df: pd.DataFrame,
                               log: logger.logging.Logger
                               ) -> None:
        """set the main arrays as attributes for the further calculations"""
        self.interface_z = \
            self.parse_contact_data(self.contact_data, 'interface_z', log)
        self.np_com = self.parse_gmx_xvg(np_com_df)
        self.box = self.parse_gmx_xvg(box_df)

    def load_contact_data(self, log: logger.logging.Logger) -> pd.DataFrame:
        """Load and return the contact data from XVG file."""
        return xvg.XvgParser(self.input_config.contact_xvg, log).xvg_df

    def load_np_com_data(self, log: logger.logging.Logger) -> pd.DataFrame:
        """
        Load and return the nanoparticle center of mass data from XVG
        file."""
        return xvg.XvgParser(self.input_config.np_coord_xvg, log).xvg_df

    def load_box_data(self, log: logger.logging.Logger) -> pd.DataFrame:
        """Load and return the box dimension data from XVG file."""
        return xvg.XvgParser(self.input_config.box_xvg, log).xvg_df

    def write_to_xvg(self,
                     regions: list[float],
                     log: logger.logging.Logger
                     ) -> None:
        """write the densities in the xvg file"""
        xvg_df: pd.DataFrame = self._prepare_arraies_for_xvg(regions, log)
        extra_msg: list[str] = []
        extra_msg.append('# Densities of the system.')
        extra_msg.append('# Data may have interpolated to have same length.')
        my_tools.write_xvg(
            xvg_df, log, extra_msg, fname=self.param_config.xvg_output)

    def _prepare_arraies_for_xvg(self,
                                 regions: list[float],
                                 log: logger.logging.Logger
                                 ) -> pd.DataFrame:
        """convert the arraies to one dataframe to write into xvg file"""
        columns: list[str] = self._get_xvg_columns()
        if not self._check_denisty(log):
            return False

        xvg_df: pd.DataFrame = pd.DataFrame(columns=columns)
        xvg_df['regions'] = regions
        for col in columns[1:]:
            try:
                arr: np.ndarray = self._interpolate_data_to_fix_length(
                    regions, getattr(self, col))
                xvg_df[col] = arr
            except AttributeError:
                log.error(
                    msg := f'\tAttribute {col} not found in the object.\n')
                print(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')
                return None

        return xvg_df

    def _get_xvg_columns(self) -> list[str]:
        """get the list of the columns names for xvg file"""
        columns: list[str] = ['regions', 'avg_density_per_region', 'rdf_2d']
        if hasattr(self, 'fitted_rdf'):
            columns.extend(['fitted_rdf'])
        if hasattr(self, 'smoothed_rdf'):
            columns.extend(['smoothed_rdf'])
        return columns

    def _check_denisty(self,
                       log: logger.logging.Logger
                       ) -> bool:
        """check if there is any data in ave_density"""
        if not self.avg_density_per_region:
            log.warning(msg := '\tThere is no average density computed!\n')
            warnings.warn(msg, UserWarning)
            return False
        return True

    @staticmethod
    def _interpolate_data_to_fix_length(regions: list[float],
                                        density: dict[float, float]
                                        ) -> np.ndarray:
        """interploate data in order to all be in same range"""
        density_regions: list[float] = list(density.keys())
        dens_arr: np.ndarray = np.array(list(density.values()))
        if density_regions == regions:
            return dens_arr
        radii_arr: np.ndarray = np.array(density_regions)
        interpolated_density = interp1d(radii_arr,
                                        dens_arr,
                                        kind='linear',
                                        fill_value="extrapolate")
        return interpolated_density(regions)

    @staticmethod
    def parse_contact_data(contact_data: pd.DataFrame,
                           column_name: str,
                           log: logger.logging.Logger
                           ) -> np.ndarray:
        """return the selected column of the contact data as an array"""
        if column_name not in contact_data.columns.to_list():
            log.error(msg := f'The column {column_name} does not '
                      'exist in the contact.xvg\n')
            raise ValueError(f'{bcolors.FAIL}{msg}{bcolors.ENDC}\n')
        return contact_data[column_name].to_numpy().reshape(-1, 1)

    @staticmethod
    def parse_gmx_xvg(np_com_df: pd.DataFrame
                      ) -> np.ndarray:
        """return the nanoparticle center of mass as an array"""
        unit_nm_to_angestrom: float = 10
        return np_com_df.iloc[:, 1:4].to_numpy() * unit_nm_to_angestrom

    @staticmethod
    def check_input_files(log: logger.logging.Logger,
                          input_config: "OdaInputFilesConfig"
                          ) -> None:
        """check the existence of the input files"""
        my_tools.check_file_exist(input_config.contact_xvg, log)
        my_tools.check_file_exist(input_config.np_coord_xvg, log)
        my_tools.check_file_exist(input_config.box_xvg, log)

    def write_msg(self,
                  log: logger.logging.Logger  # To log
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKGREEN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == "__main__":
    print(f'{bcolors.CAUTION}\tThis script runs within '
          f'trajectory_oda_analysis.py{bcolors.ENDC}')
