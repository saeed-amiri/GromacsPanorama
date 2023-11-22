"""
For each defined region, calculate the number density of ODA molecules.
This is typically done by counting the number of ODA molecules in each
region and dividing by the volume of that region.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from common import logger
from common import xvg_to_dataframe as xvg
from common import my_tools
from common.colors_text import TextColor as bcolors


@dataclass
class OdaInputConfig:
    """set the input for analysing"""
    contact_xvg: str = 'contact.xvg'
    np_coord_xvg: str = 'coord.xvg'


class SurfactantDensityAroundNanoparticle:
    """self explained"""
    info_msg: str = '\tMessage from SurfactantDensityAroundNanoparticle:\n'

    def __init__(self,
                 amino_arr: np.ndarray,  # amino head com of the oda
                 box_dims: dict[str, float],  # Dimension of the Box
                 log: logger.logging.Logger,
                 input_config: "OdaInputConfig" = OdaInputConfig()
                 ) -> None:
        self._initiate(amino_arr, box_dims, log, input_config)

    def _initiate(self,
                  amino_arr: np.ndarray,  # amino head com of the oda
                  box_dims: dict[str, float],  # Dimension of the Box
                  log: logger.logging.Logger,
                  input_config: "OdaInputConfig"
                  ) -> None:
        """Initiate the calculation by checking necessary files."""
        self.check_input_files(log, input_config)

        contact_data: pd.DataFrame = \
            xvg.XvgParser(input_config.contact_xvg, log).xvg_df
        np_com_df: pd.DataFrame = \
            xvg.XvgParser(input_config.np_coord_xvg, log).xvg_df

        interface_z: np.ndarray = \
            self.parse_contact_data(contact_data, 'interface_z')
        np_com: np.ndarray = self.parse_np_com(np_com_df)
        amino_coms: np.ndarray = amino_arr[:-2]

    @staticmethod
    def parse_contact_data(contact_data: pd.DataFrame,
                           column_name: str
                           ) -> np.ndarray:
        """return the selected column of the contact data as an array"""
        if column_name not in contact_data.columns.to_list():
            raise ValueError(
                f'{bcolors.FAIL}The column {column_name} does not '
                f'exist in the contact.xvg{bcolors.ENDC}\n')
        return contact_data[column_name].to_numpy().reshape(-1, 1)

    @staticmethod
    def parse_np_com(np_com_df: pd.DataFrame
                     ) -> np.ndarray:
        """return the nanoparticle center of mass as an array"""
        return np_com_df.to_numpy()

    @staticmethod
    def check_input_files(log: logger.logging.Logger,
                          input_config: "OdaInputConfig"
                          ) -> None:
        """check the existence of the input files"""
        my_tools.check_file_exist(input_config.contact_xvg, log)
        my_tools.check_file_exist(input_config.np_coord_xvg, log)


if __name__ == "__main__":
    print(f'{bcolors.CAUTION}\tThis script runs within '
          f'trajectory_oda_analysis.py{bcolors.ENDC}')
