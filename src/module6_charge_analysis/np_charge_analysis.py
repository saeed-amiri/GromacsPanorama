"""
Analysis the charge of the nanoparticle
1- Charge in total
2- Partial charge at the interface
"""

import typing

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import matplotlib.pylab as plt

from common import logger
from common import static_info as stinfo
from common import xvg_to_dataframe as xvg
from common.colors_text import TextColor as bcolors

from module6_charge_analysis import gmx_rdf_cdf_plotter

if typing.TYPE_CHECKING:
    from module6_charge_analysis.charge_analysis_interface_np import \
        ComputeConfigurations


UNIT_NM_TO_ANGSTROM: float = 10.0  # Conversion factor from nanom to angstroms


class DataArrays(typing.NamedTuple):
    """Set the arrays from input files"""
    contact_radius: np.ndarray
    np_com: np.ndarray
    rdf: np.ndarray
    cdf: np.ndarray
    box: np.ndarray


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
        self.inital_plots(log)
        self.initiate_computation(log)

    def inital_plots(self,
                     log: logger.logging.Logger
                     ) -> None:
        """plot the main rdf and cdf from gromacs"""
        rdf_df: pd.DataFrame = \
            xvg.XvgParser(self.input_config.f_rdf, log, x_type=float).xvg_df
        gmx_rdf_cdf_plotter.PlotGmxRdfCdf(df_in=rdf_df, df_type='rdf', log=log)
    
    def initiate_computation(self,
                             log: logger.logging.Logger
                             ) -> None:
        """Start calulation"""
        box_half: np.float64 = self.get_box_max_half(self.data_arrays.box)
        self.analys_rdf(self.data_arrays.rdf, box_half)

    def get_box_max_half(self,
                         box: np.ndarray
                         ) -> np.float64:
        """find the maximums of the box size"""
        return np.max(box) / 2.0

    def analys_rdf(self,
                   rdf: np.ndarray,
                   box_half: np.float64  # Max of the box in all axis, half
                   ) -> None:
        """analysing rdf to find extremums"""
        np_radius: float = stinfo.np_info['radius']
        filtered_indices = np.where(rdf[:, 0] < box_half)[0]
        filtered_rdf = rdf[filtered_indices]
        dr = np.diff(filtered_rdf[:, 0])
        grad_rdf = np.diff(filtered_rdf[:, 1]) / dr

        # Identify peaks and valleys
        peaks, _ = find_peaks(filtered_rdf[:, 1])
        valleys, _ = find_peaks(-filtered_rdf[:, 1])

        # Get the exact r values for peaks and valleys
        peak_positions = filtered_rdf[:, 0][peaks]
        valley_positions = filtered_rdf[:, 0][valleys]


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

        rdf_df: pd.DataFrame = \
            self._load_xvg_data(self.input_config.f_rdf, log, x_type=float)
        rdf: pd.DataFrame = self._parse_gmx_rdf_cdf(rdf_df)

        cdf_df: pd.DataFrame = \
            self._load_xvg_data(self.input_config.f_cdf, log, x_type=float)
        cdf: pd.DataFrame = self._parse_gmx_rdf_cdf(cdf_df)

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


if __name__ == '__main__':
    pass
