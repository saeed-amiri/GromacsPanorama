"""
This script calculates the density of surfactants in the contact area
and compares it to the density in the bulk.
It focuses on evaluating the concentration of surfactants within a
defined circular area (with a radius equal to the contact radius) and
contrasts this with the overall density in the bulk region, providing
insights into the distribution and aggregation of surfactants in
different zones.
"""

from dataclasses import dataclass
from collections import namedtuple

import numpy as np
import pandas as pd

from common import logger
from common import my_tools
from common import xvg_to_dataframe as xvg
from common.colors_text import TextColor as bcolors


@dataclass
class InputFilesConfig:
    """set the input for analysing"""
    contact_xvg: str = 'contact.xvg'
    np_coord_xvg: str = 'coord.xvg'
    box_xvg: str = 'box.xvg'


DataArrays = namedtuple('DataArrays', ['contact_radius', 'np_com', 'box'])


class SurfactantsLocalizedDensityContrast:
    """
    Analyzing and contrasting the localized density of surfactants
    within a defined area versus the bulk region.
    """

    info_msg: str = 'Messeges from SurfactantsLocalizedDensityContrast:\n'
    amino_arr: pd.DataFrame  # Amino heads location
    input_config: "InputFilesConfig"
    data_arrays: "DataArrays"  # Loaded data

    def __init__(self,
                 amino_arr: np.ndarray,  # amino head com of the oda
                 log: logger.logging.Logger,
                 input_configs: "InputFilesConfig" = InputFilesConfig()
                 ) -> None:
        self.input_config = input_configs
        self.amino_arr = amino_arr[:-2]
        self._prepare_data_and_analysis(log)

    def _prepare_data_and_analysis(self,
                                   log: logger.logging.Logger
                                   ) -> None:
        """Initiate the calculation by checking necessary files."""
        self.data_arrays = \
            self.initialize_data_arrays(*self.load_data(log), log)
        self.initialize_calculation()

    def initialize_calculation(self) -> None:
        """initiate the the calculations"""
        oda_in_zone: np.ndarray = self.get_oda_in_zone()

    def get_oda_in_zone(self) -> np.ndarray:
        """return the numbers of oda on the contact area at each frame"""
        oda_in_zone_dict: dict[int, int] = {}
        for i, frame_i in enumerate(self.amino_arr):
            oda_frame = self.apply_pbc_distance(i, frame_i.reshape(-1, 3))
            oda_in_zone_dict[i] = self.get_oda_nr_in_np_zone(i, oda_frame)
        return np.array(list(oda_in_zone_dict.items()))

    def apply_pbc_distance(self,
                           frame_index: int,
                           arr: np.ndarray
                           ) -> np.ndarray:
        """apply pbc to all the oda with np as refrence"""
        np_com: np.ndarray = self.data_arrays.np_com[frame_index]
        box: np.ndarray = self.data_arrays.box[frame_index]
        arr_pbc: np.ndarray = np.zeros(arr.shape)
        for i in range(arr.shape[1]):
            dx_i = arr[:, i] - np_com[i]
            arr_pbc[:, i] = dx_i - (box[i] * np.round(dx_i/box[i]))
        return arr_pbc

    def get_oda_nr_in_np_zone(self,
                               frame_index: int,
                               arr: np.ndarray
                               ) -> int:
        """find the numbers of the oda in the area of contact radius"""
        contact_radius: float = self.data_arrays.contact_radius[frame_index]
        np_com: np.ndarray = self.data_arrays.np_com[frame_index]
        dx_i: np.ndarray = arr[:, 0] - np_com[0]
        dy_i: np.ndarray = arr[:, 1] - np_com[1]
        distances: np.ndarray = np.sqrt(dx_i*dx_i + dy_i*dy_i)
        mask = distances < contact_radius
        return len(arr[mask])

    def load_data(self,
                  log: logger.logging.Logger
                  ) -> tuple[pd.DataFrame, ...]:
        """load the data in the config"""
        self.check_input_files(log, self.input_config)
        contact_data: pd.DataFrame = self.load_contact_data(log)
        np_com_df: pd.DataFrame = self.load_np_com_data(log)
        box_df: pd.DataFrame = self.load_box_data(log)
        return contact_data, np_com_df, box_df

    def load_contact_data(self, log: logger.logging.Logger) -> pd.DataFrame:
        """Load and return the contact data from XVG file."""
        return xvg.XvgParser(self.input_config.contact_xvg, log).xvg_df

    def load_np_com_data(self, log: logger.logging.Logger) -> pd.DataFrame:
        """Load and return the NP center of mass data from XVG file."""
        return xvg.XvgParser(self.input_config.np_coord_xvg, log).xvg_df

    def load_box_data(self, log: logger.logging.Logger) -> pd.DataFrame:
        """Load and return the box dimension data from XVG file."""
        return xvg.XvgParser(self.input_config.box_xvg, log).xvg_df

    def initialize_data_arrays(self,
                               contact_data: pd.DataFrame,
                               np_com_df: pd.DataFrame,
                               box_df: pd.DataFrame,
                               log: logger.logging.Logger
                               ) -> "DataArrays":
        """set the main arrays as attributes for the further calculations"""
        contact_radius = \
            self.parse_contact_data(contact_data, 'contact_radius', log)
        np_com = self.parse_gmx_xvg(np_com_df)
        box = self.parse_gmx_xvg(box_df)
        return DataArrays(contact_radius, np_com, box)

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
                          input_config: "InputFilesConfig"
                          ) -> None:
        """check the existence of the input files"""
        my_tools.check_file_exist(input_config.contact_xvg, log)
        my_tools.check_file_exist(input_config.np_coord_xvg, log)
        my_tools.check_file_exist(input_config.box_xvg, log)

    def write_msg(self,
                  log: logger.logging.Logger  # To log
                  ) -> None:
        """write and log messages"""
        message: str = f'{self.__class__.__name__}:\n\t{self.info_msg}'
        print(f'{bcolors.OKGREEN}{message}{bcolors.ENDC}')
        log.info(message)


if __name__ == "__main__":
    print(f'{bcolors.CAUTION}\tThis script runs within '
          f'trajectory_oda_analysis.py{bcolors.ENDC}')
