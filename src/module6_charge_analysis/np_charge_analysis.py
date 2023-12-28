"""
Analysis the charge of the nanoparticle
1- Charge in total
2- Partial charge at the interface
"""

import typing
from collections import namedtuple

import numpy as np
import pandas as pd

from common import logger
from common import xvg_to_dataframe as xvg
from common.colors_text import TextColor as bcolors


if typing.TYPE_CHECKING:
    from module6_charge_analysis.charge_analysis_interface_np import \
        ComputeConfigurations

DataArrays = namedtuple('DataArrays', ['contact_radius',
                                       'np_com',
                                       'rdf',
                                       'cdf',
                                       'box'])


class NpChargeAnalysis:
    """analysign the charge of the nanoparticle"""

    info_msg: str = 'Messege from NpChargeAnalysis:\n'

    cla_arr: np.ndarray
    input_config: "ComputeConfigurations"
    data_arrays: "DataArrays"

    def __init__(self,
                 cla_arr: np.ndarray,
                 input_config: "ComputeConfigurations",
                 log: logger.logging.Logger
                 ) -> None:
        self.data_arrays = ParseDataFiles(input_config, log).data_arrays
        self.input_config = input_config
        self.cla_arr = cla_arr


class ParseDataFiles:
    """rad and parse files here"""

    info_msg: str = 'Messegs from ParseDataFiles:\n'

    input_config: "ComputeConfigurations"
    data_arrays: "DataArrays"

    def __init__(self,
                 input_config: "ComputeConfigurations",
                 log: logger.logging.Logger
                 ) -> None:
        self.input_config = input_config
        self.data_arrays = self.initiate_data(log)

    def initiate_data(self,
                      log: logger.logging.Logger
                      ) -> "DataArrays":
        """parsing data files"""
        contact_data: pd.DataFrame = self.load_contact_data(log)
        contact_radius: np.ndarray = \
            self.parse_contact_data(contact_data, 'contact_radius', log)

        np_com_df: pd.DataFrame = self.load_np_com_data(log)
        np_com: np.ndarray = self.parse_gmx_xvg(np_com_df)

        rdf_df: pd.DataFrame = self.load_rdf_data(log)
        rdf: pd.DataFrame = self.parse_rdf_cdf_xvg(rdf_df)

        cdf_df: pd.DataFrame = self.load_cdf_data(log)
        cdf: pd.DataFrame = self.parse_rdf_cdf_xvg(cdf_df)

        box_df: pd.DataFrame = self.load_box_data(log)
        box: np.ndarray = self.parse_gmx_xvg(box_df)

        return DataArrays(contact_radius, np_com, rdf, cdf, box)

    def load_contact_data(self, log: logger.logging.Logger) -> pd.DataFrame:
        """Load and return the contact data from XVG file."""
        return xvg.XvgParser(self.input_config.f_contact, log).xvg_df

    def load_np_com_data(self, log: logger.logging.Logger) -> pd.DataFrame:
        """Load and return the NP center of mass data from XVG file."""
        return xvg.XvgParser(self.input_config.f_coord, log).xvg_df

    def load_box_data(self, log: logger.logging.Logger) -> pd.DataFrame:
        """Load and return the box dimension data from XVG file."""
        return xvg.XvgParser(self.input_config.f_box, log).xvg_df

    def load_rdf_data(self, log: logger.logging.Logger) -> pd.DataFrame:
        """Load and return the box dimension data from XVG file."""
        return xvg.XvgParser(self.input_config.f_rdf, log, x_type=float).xvg_df

    def load_cdf_data(self, log: logger.logging.Logger) -> pd.DataFrame:
        """Load and return the box dimension data from XVG file."""
        return xvg.XvgParser(self.input_config.f_cdf, log, x_type=float).xvg_df

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
    def parse_rdf_cdf_xvg(df_i: pd.DataFrame
                          ) -> np.ndarray:
        """parse the rdf and cdf by converting nm to angestrom"""
        unit_nm_to_angestrom: float = 10
        df_i.iloc[:, 0] *= unit_nm_to_angestrom
        return df_i.to_numpy()


if __name__ == '__main__':
    pass
