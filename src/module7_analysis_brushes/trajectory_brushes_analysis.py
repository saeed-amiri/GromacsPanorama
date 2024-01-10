"""
Brush Analysis Module

This module is designed for analyzing surfactant-rich systems,
specifically focusing on the behavior of octadecylamines (ODA) at the
interface of water and decane layers in the absence of nanoparticles.

Key Analysis Features:
    1. Surface Morphology: Examines the structural characteristics of
        the interface.
    2. ODA Distribution: Quantifies the number of ODA molecules at the
        interface and within the water phase.
    3. Order Parameters: Calculates the order parameters of ODA at the
        interface to assess alignment and organization.

Additional Analysis (if applicable):
    - Micelle Analysis: Investigates the formation and characteristics
        of ODA micelles in the water phase.

Input Data:
    - com_pickle: A serialized file representing the center of mass
     for all residues in the system. It is structured as an array with
     the following format:
     | time | NP_x | NP_y | NP_z | res1_x | res1_y | res1_z | ...
     | resN_x | resN_y | resN_z | odn1_x | odn1_y | odn1_z | ...
     | odnN_z |

    This data is loaded using the GetCom class from the
  common.com_file_parser module.
Jan 10 2023
Saeed
"""

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from common import logger
from common.com_file_parser import GetCom
from common import xvg_to_dataframe as xvg
from common.colors_text import TextColor as bcolors


@dataclass
class InputConfigus:
    """set the input files names"""
    f_box: str = 'box.xvg'


@dataclass
class ComputationConfigs:
    """to set the computation parameters and selections"""
    file_configs: "InputConfigus" = InputConfigus()
    unit_nm_to_angestrom: float = 10


class BrushesAnalysis:
    """analysing brushes systems"""

    info_msg: str = 'Messages from BrushesAnalysis:\n'
    compute_configs: "ComputationConfigs"
    parsed_com: "GetCom"
    box: np.ndarray

    def __init__(self,
                 fname: str,  # Name of the com_pickle file
                 log: logger.logging.Logger,
                 compute_configs: "ComputationConfigs" = ComputationConfigs()
                 ) -> None:
        self.parsed_com = GetCom(fname)
        self.compute_configs = compute_configs
        self.initiate(log)

    def initiate(self,
                 log: logger.logging.Logger
                 ) -> None:
        """initiate computations"""
        box: pd.DataFrame = self._load_xvg_data(
            self.compute_configs.file_configs.f_box, log)
        self.box = self._parse_gmx_coordinates(
            box, self.compute_configs.unit_nm_to_angestrom)

    def _load_xvg_data(self,
                       fname: str,
                       log: logger.logging.Logger,
                       x_type: type = int
                       ) -> pd.DataFrame:
        """Load and return the contact data from XVG file."""
        return xvg.XvgParser(fname, log, x_type).xvg_df

    @staticmethod
    def _parse_gmx_coordinates(df_in: pd.DataFrame,
                               nm_angestrom: float
                               ) -> np.ndarray:
        """return the nanoparticle center of mass as an array"""
        return df_in.iloc[:, 1:4].to_numpy() * nm_angestrom


if __name__ == '__main__':
    BrushesAnalysis(sys.argv[1], log=logger.setup_logger('brushes.log'))
