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
class OdaInputConfig:
    """set the input for analysing"""
    contact_xvg: str = 'contact.xvg'
    np_coord_xvg: str = 'coord.xvg'
    box_xvg: str = 'box.xvg'


class SurfactantDensityAroundNanoparticle:
    """self explained"""
    info_msg: str = '\tMessage from SurfactantDensityAroundNanoparticle:\n'
    input_config: "OdaInputConfig"
    box: np.ndarray  # Size of the box at each frame (from gromacs)
    np_com: np.ndarray  # COM of the NP at each frame (from gromacs)
    amino_arr: np.ndarray  # Com of the oda_amino head (from module1)
    interface_z: np.ndarray  # Z location of the interface (from module3)

    def __init__(self,
                 amino_arr: np.ndarray,  # amino head com of the oda
                 log: logger.logging.Logger,
                 input_config: "OdaInputConfig" = OdaInputConfig()
                 ) -> None:
        # The two last columns of amino_arr are indicies from main trr file
        self.amino_arr = amino_arr[:-2]
        self.input_config = input_config
        self._initiate(log)

    def _initiate(self,
                  log: logger.logging.Logger,
                  ) -> None:
        """Initiate the calculation by checking necessary files."""
        self.check_input_files(log, self.input_config)

        contact_data: pd.DataFrame = self.load_contact_data(log)
        np_com_df: pd.DataFrame = self.load_np_com_data(log)
        box_df: pd.DataFrame = self.load_box_data(log)
        self.initialize_data_arrays(contact_data, np_com_df, box_df, log)
        self.initialize_calculation(log)

    def initialize_calculation(self,
                               log: logger.logging.Logger
                               ) -> None:
        """getting the density number from the parsed data"""
        z_threshold: np.ndarray = self.compute_surfactant_vertical_threshold()

    def compute_surfactant_vertical_threshold(self) -> np.ndarray:
        """find the vertical threshold for the surfactants, to drop from
        calculation"""
        return (stinfo.np_info['radius'] +
                self.np_com[:, 2] + np.std(self.np_com[:, 2])).reshape(-1, 1)

    def initialize_data_arrays(self,
                               contact_data: pd.DataFrame,
                               np_com_df: pd.DataFrame,
                               box_df: pd.DataFrame,
                               log: logger.logging.Logger
                               ) -> None:
        """set the main arrays as attibutes for the further calculationsa"""
        self.interface_z = \
            self.parse_contact_data(contact_data, 'interface_z', log)
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
                          input_config: "OdaInputConfig"
                          ) -> None:
        """check the existence of the input files"""
        my_tools.check_file_exist(input_config.contact_xvg, log)
        my_tools.check_file_exist(input_config.np_coord_xvg, log)
        my_tools.check_file_exist(input_config.box_xvg, log)


if __name__ == "__main__":
    print(f'{bcolors.CAUTION}\tThis script runs within '
          f'trajectory_oda_analysis.py{bcolors.ENDC}')
