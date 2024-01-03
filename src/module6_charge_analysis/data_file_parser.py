"""
DataFileParser.py

This script contains the ParseDataFiles class, which is designed for
the efficient parsing and processing of various data files used in
nanoparticle analysis. The class handles the reading and extraction of
data from files such as radial distribution function (rdf), cumulative
distribution function (cdf), contact information, nanoparticle
coordinates, and box dimensions. It plays a crucial role in preparing
and structuring this data for further computational analysis.

Classes:
- ParseDataFiles: A class that facilitates the loading and parsing of
different data files. It integrates functionalities to parse contact
data, nanoparticle center of mass coordinates, rdf, cdf, and box
dimension data, converting and structuring them into appropriate
formats for analysis.
- FileConfigurations: A dataclass that holds the filenames for the
different data files required in the analysis.
- ComputeConfigurations: A dataclass that extends FileConfigurations,
providing additional configuration settings for the computations.

The script is structured to support easy integration with other modules
in the analysis pipeline, ensuring seamless data flow and processing.

Note: This script assumes the availability of specific data file formats
(like XVG) and utilizes specific libraries for data handling and
computations, such as pandas, numpy, and custom parsing utilities.

doc by ChatGpt
Jan 3, 2023
Saeed
"""

import typing
from dataclasses import dataclass

import numpy as np
import pandas as pd

from common import logger
from common import xvg_to_dataframe as xvg
from common.colors_text import TextColor as bcolors


UNIT_NM_TO_ANGSTROM: float = 10.0  # Conversion factor from nanom to angstroms


@dataclass
class FileConfigurations:
    """names of the input files"""
    f_contact: str = 'contact.xvg'
    f_coord: str = 'coord.xvg'
    f_box: str = 'box.xvg'


@dataclass
class ComputeConfigurations(FileConfigurations):
    """configure input for the calculations"""


class DataArrays(typing.NamedTuple):
    """Set the arrays from input files"""
    contact_radius: np.ndarray
    np_com: np.ndarray
    rdf: np.ndarray
    cdf: np.ndarray
    box: np.ndarray


class ParseDataFiles:
    """rad and parse files here"""

    # pylint: disable=too-few-public-methods

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
        contact_data: pd.DataFrame = \
            self._load_xvg_data(self.input_config.f_contact, log)
        contact_radius: np.ndarray = \
            self._parse_contact_data(contact_data, 'contact_radius', log)

        np_com_df: pd.DataFrame = \
            self._load_xvg_data(self.input_config.f_coord, log)
        np_com: np.ndarray = self._parse_gmx_coordinates(np_com_df)

        if hasattr(self.input_config, 'f_rdf'):
            rdf_df: pd.DataFrame = \
                self._load_xvg_data(self.input_config.f_rdf, log, x_type=float)
            rdf: pd.DataFrame = self._parse_gmx_rdf_cdf(rdf_df)
        else:
            rdf = pd.DataFrame()

        if hasattr(self.input_config, 'f_cdf'):
            cdf_df: pd.DataFrame = \
                self._load_xvg_data(self.input_config.f_cdf, log, x_type=float)
            cdf: pd.DataFrame = self._parse_gmx_rdf_cdf(cdf_df)
        else:
            cdf = pd.DataFrame()

        box_df: pd.DataFrame = \
            self._load_xvg_data(self.input_config.f_box, log)
        box: np.ndarray = self._parse_gmx_coordinates(box_df)

        return DataArrays(contact_radius, np_com, rdf, cdf, box)

    def _load_xvg_data(self,
                       fname: str,
                       log: logger.logging.Logger,
                       x_type: type = int
                       ) -> pd.DataFrame:
        """Load and return the contact data from XVG file."""
        return xvg.XvgParser(fname, log, x_type).xvg_df

    @staticmethod
    def _parse_contact_data(contact_data: pd.DataFrame,
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
    def _parse_gmx_coordinates(np_com_df: pd.DataFrame
                               ) -> np.ndarray:
        """return the nanoparticle center of mass as an array"""
        return np_com_df.iloc[:, 1:4].to_numpy() * UNIT_NM_TO_ANGSTROM

    @staticmethod
    def _parse_gmx_rdf_cdf(df_i: pd.DataFrame
                           ) -> np.ndarray:
        """parse the rdf and cdf by converting nm to angestrom"""
        df_i.iloc[:, 0] *= UNIT_NM_TO_ANGSTROM
        return df_i.to_numpy()
