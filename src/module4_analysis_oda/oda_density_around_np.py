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


@dataclass
class OdaInputFilesConfig:
    """set the input for analysing"""
    contact_xvg: str = 'contact.xvg'
    np_coord_xvg: str = 'coord.xvg'
    box_xvg: str = 'box.xvg'


@dataclass
class ParameterConfig:
    """set the default paramters for the calculataion"""
    number_of_regins: int = 50


class SurfactantDensityAroundNanoparticle:
    """self explained"""
    info_msg: str = '\tMessage from SurfactantDensityAroundNanoparticle:\n'
    input_config: "OdaInputFilesConfig"
    param_config: "ParameterConfig"
    contact_data: pd.DataFrame  # The contact data (from module3)
    box: np.ndarray  # Size of the box at each frame (from gromacs)
    np_com: np.ndarray  # COM of the NP at each frame (from gromacs)
    amino_arr: np.ndarray  # Com of the oda_amino head (from module1)
    interface_z: np.ndarray  # Z location of the interface (from module3)
    density_per_region: dict[float, list[float]]  # Density per area
    avg_density_per_region: dict[float, float]

    def __init__(self,
                 amino_arr: np.ndarray,  # amino head com of the oda
                 log: logger.logging.Logger,
                 input_config: "OdaInputFilesConfig" = OdaInputFilesConfig(),
                 param_config: "ParameterConfig" = ParameterConfig()
                 ) -> None:
        # The two last rows of amino_arr are indicies from main trr file
        self.amino_arr = amino_arr[:-2]
        self.input_config = input_config
        self.param_config = param_config
        self._initiate(log)

    def _initiate(self,
                  log: logger.logging.Logger,
                  ) -> None:
        """Initiate the calculation by checking necessary files."""
        self.check_input_files(log, self.input_config)

        self.contact_data: pd.DataFrame = self.load_contact_data(log)
        np_com_df: pd.DataFrame = self.load_np_com_data(log)
        box_df: pd.DataFrame = self.load_box_data(log)
        self.initialize_data_arrays(np_com_df, box_df, log)
        self.density_per_region = self.initialize_calculation()

    def initialize_calculation(self) -> dict[float, list[float]]:
        """getting the density number from the parsed data"""
        z_threshold: np.ndarray = self.compute_surfactant_vertical_threshold()
        regions: list[float] = \
            self.generate_regions(self.param_config.number_of_regins)
        # Initialize a dictionary to store densities for each region
        density_per_region: dict[float, list[float]] = \
            {region: [] for region in regions}
        for i, frame_i in enumerate(self.amino_arr):
            arr_i: np.ndarray = \
                self._get_surfactant_at_interface(frame_i, z_threshold[i])
            distance: np.ndarray = self.compute_pbc_distance(i, arr_i)
            density_per_region = \
                self._compute_density_per_region(regions,
                                                 distance,
                                                 density_per_region)
        self._comput_and_set_avg_density_as_attibute(density_per_region)
        return density_per_region

    def _comput_and_set_avg_density_as_attibute(self,
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

    @staticmethod
    def _compute_density_per_region(regions: list[float],
                                    distance: np.ndarray,
                                    density_per_region:
                                    dict[float, list[float]]
                                    ) -> dict[float, list[float]]:
        """self explanatory"""
        for ith in range(len(regions) - 1):
            r_inner = regions[ith]
            r_outer = regions[ith + 1]
            count = np.sum((distance >= r_inner) & (distance < r_outer))
            area = np.pi * (r_outer**2 - r_inner**2)
            density = count / area
            density_per_region[r_outer].append(density)
        return density_per_region

    def generate_regions(self,
                         number_of_regions: int
                         ) -> list[float]:
        """divide the the area around the np for generating regions"""
        max_box_len: np.float64 = np.max(self.box[0][:2]) / 2
        return np.linspace(0, max_box_len, number_of_regions).tolist()

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
        """set the main arrays as attibutes for the further calculationsa"""
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


if __name__ == "__main__":
    print(f'{bcolors.CAUTION}\tThis script runs within '
          f'trajectory_oda_analysis.py{bcolors.ENDC}')
