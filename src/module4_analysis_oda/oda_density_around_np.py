"""
For each defined region, calculate the number density of ODA molecules.
This is typically done by counting the number of ODA molecules in each
region and dividing by the volume of that region.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

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
    midpoint: float  # midpoint of the fit
    first_turn: float
    second_turn: float

    def __init__(self,
                 amino_arr: np.ndarray,  # amino head com of the oda
                 log: logger.logging.Logger,
                 input_config: "OdaInputFilesConfig" = OdaInputFilesConfig(),
                 param_config: "ParameterConfig" = ParameterConfig()
                 ) -> None:
        # The two last rows of amino_arr are indicies from main trr file
        amino_arr = amino_arr[:-2]
        self.input_config = input_config
        self.param_config = param_config
        self._initiate(amino_arr, log)
        self.write_msg(log)

    def _initiate(self,
                  amino_arr: np.ndarray,
                  log: logger.logging.Logger,
                  ) -> None:
        """Initiate the calculation by checking necessary files."""
        self.check_input_files(log, self.input_config)

        self.contact_data: pd.DataFrame = self.load_contact_data(log)
        np_com_df: pd.DataFrame = self.load_np_com_data(log)
        box_df: pd.DataFrame = self.load_box_data(log)
        self.initialize_data_arrays(np_com_df, box_df, log)
        self.density_per_region = self.initialize_calculation(amino_arr, log)

    def initialize_calculation(self,
                               amino_arr: np.ndarray,
                               log: logger.logging.Logger
                               ) -> dict[float, list[float]]:
        """getting the density number from the parsed data"""
        z_threshold: np.ndarray = self.compute_surfactant_vertical_threshold()
        regions: list[float] = \
            self.generate_regions(self.param_config.number_of_regions)
        # Initialize a dictionary to store densities for each region
        density_per_region: dict[float, list[float]] = \
            {region: [] for region in regions}
        num_oda: list[int] = []
        num_oda_in_raius: list[int] = []
        for i, frame_i in enumerate(amino_arr):
            arr_i: np.ndarray = \
                self._get_surfactant_at_interface(frame_i, z_threshold[i])
            distance: np.ndarray = self.compute_pbc_distance(i, arr_i)
            density_per_region, count_in_radius = \
                self._compute_density_per_region(regions,
                                                 distance,
                                                 density_per_region)
            num_oda.append(len(arr_i))
            num_oda_in_raius.append(count_in_radius)
        self._comput_and_set_avg_density_as_attribute(density_per_region)
        self._comput_and_set_2d_rdf(density_per_region, num_oda_in_raius)
        self._fit_and_set_fitted2d_rdf(log)
        self._comput_and_set_moving_average(3)
        return density_per_region

    def _comput_and_set_avg_density_as_attribute(self,
                                                 density_per_region:
                                                 dict[float, list[float]]
                                                 ) -> None:
        """self explanatory"""
        self.avg_density_per_region = {}
        for region, densities in density_per_region.items():
            if densities:
                self.avg_density_per_region[region] = np.mean(densities)
            else:
                self.avg_density_per_region[region] = 0

    def _comput_and_set_2d_rdf(self,
                               density_per_region: dict[float, list[float]],
                               num_oda: list[int]
                               ) -> None:
        """set the 2d rdf (g(r))"""
        max_radius_area: float = \
            max(item for item in density_per_region.keys())
        self.rdf_2d = {}
        for region, densities in density_per_region.items():
            if not densities:
                self.rdf_2d[region] = 0
                continue

            tmp = []
            for j, item in enumerate(densities):
                density: float = num_oda[j]/(np.pi * max_radius_area**2)
                tmp.append(item/density)
            self.rdf_2d[region] = np.mean(tmp)

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
                         nr_of_regions: int
                         ) -> list[float]:
        """divide the the area around the np for generating regions"""
        max_box_len: np.float64 = np.max(self.box[0][:2]) / 2
        self.info_msg += (
            f'\tThe number of regions is: `{nr_of_regions}`\n'
            f'\tThe half of the max length of the box is `{max_box_len:.3f}`\n'
            f'\tThe bin size, thus, is: `{max_box_len/nr_of_regions:.3f}`\n'
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
                                     z_threshold_i: float
                                     ) -> np.ndarray:
        """return the oda at the interface"""
        frame_reshaped: np.ndarray = frame_i.reshape(-1, 3)
        below_threshold_indices: np.ndarray = \
            np.where(frame_reshaped[:, 2] < z_threshold_i)
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

    def compute_surfactant_vertical_threshold(self) -> np.ndarray:
        """find the vertical threshold for the surfactants, to drop from
        calculation"""
        return (stinfo.np_info['radius'] +
                self.np_com[:, 2] + np.std(self.np_com[:, 2])).reshape(-1, 1)

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
