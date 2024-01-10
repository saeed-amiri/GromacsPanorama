"""
Brush Analysis Module

This module is designed for analyzing the behavior of octadecylamines
(ODA) at the interface of water and decane layers in surfactant-rich
systems without nanoparticles.

Key Analysis Features:
    1. Surface Morphology: Examines the interface's structural
        characteristics.
    2. ODA Distribution: Quantifies ODA molecules at the interface and
        in the water phase.
    3. Order Parameters: Calculates ODA order parameters at the
        interface for organization assessment.

Additional Analysis (if applicable):
    - Micelle Analysis: Investigates ODA micelle formation in the
        water phase.

Input Data:
    - com_pickle: Serialized file representing the center of mass for
        all system residues.

Authors: Saeed
Date: Jan 10, 2023
"""


import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from common import logger
from common.com_file_parser import GetCom
from common import xvg_to_dataframe as xvg
from common.colors_text import TextColor as bcolors

from module7_analysis_brushes.get_surface import GetSurface


@dataclass
class InputConfig:
    """set the input files names"""
    f_box: str = 'box.xvg'


@dataclass
class ComputationConfig:
    """to set the computation parameters and selections"""
    file_configs: "InputConfig" = InputConfig()
    unit_nm_to_angestrom: float = 10


class BrushAnalysis:
    """analysing brushes systems"""

    info_msg: str = 'Messages from BrushAnalysis:\n'
    compute_configs: "ComputationConfig"
    parsed_com: "GetCom"
    box: np.ndarray
    surface_waters: dict[int, np.ndarray]

    def __init__(self,
                 fname: str,  # Name of the com_pickle file
                 log: logger.logging.Logger,
                 compute_configs: "ComputationConfig" = ComputationConfig()
                 ) -> None:
        self.parsed_com = GetCom(fname)
        self.info_msg += f'\tReading com file: `{fname}`\n'
        self.compute_configs = compute_configs
        self.initiate(log)
        self.write_msg(log)

    def initiate(self,
                 log: logger.logging.Logger
                 ) -> None:
        """initiate computations"""
        self.set_box(log)
        self.get_interface(log)

    def set_box(self,
                log: logger.logging.Logger
                ) -> None:
        """set the box info as an attribute"""
        box_df: pd.DataFrame = self._load_xvg_data(
            fname := self.compute_configs.file_configs.f_box, log)
        self.box = self._parse_gmx_coordinates(
            box_df, self.compute_configs.unit_nm_to_angestrom)
        self.info_msg += f'\tReading box file: `{fname}`\n'

    def get_interface(self,
                      log: logger.logging.Logger
                      ) -> None:
        """analysis the interface by finding the water surface"""
        self.surface_waters = GetSurface(self.parsed_com.split_arr_dict['SOL'],
                                         self.parsed_com.box_dims,
                                         log).surface_waters

    def _load_xvg_data(self,
                       fname: str,
                       log: logger.logging.Logger,
                       x_type: type = int
                       ) -> pd.DataFrame:
        """Load and return the contact data from XVG file."""
        return xvg.XvgParser(fname, log, x_type).xvg_df

    @staticmethod
    def _parse_gmx_coordinates(df_in: pd.DataFrame,
                               conversion_factor: float
                               ) -> np.ndarray:
        """return the nanoparticle center of mass as an array"""
        return df_in.iloc[:, 1:4].to_numpy() * conversion_factor

    def write_msg(self,
                  log: logger.logging.Logger  # To log
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


def main():
    """main to call the main class"""
    log = logger.setup_logger('brushes.log')
    com_filename = sys.argv[1] if len(sys.argv) > 1 else "com_pickle"
    BrushAnalysis(com_filename, log)


if __name__ == '__main__':
    main()
