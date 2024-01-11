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
from module7_analysis_brushes.analysis_oda import AnalysisSurfactant


@dataclass
class InputConfig:
    """set the input files names"""
    f_box: str = 'box.xvg'
    f_surface_locz: str = 'contact.xvg'


@dataclass
class ParameterConfig:
    """constant values and other parameters"""
    unit_nm_to_angstrom: float = 10


@dataclass
class ComputationConfig(InputConfig, ParameterConfig):
    """to set the computation parameters and selections"""
    compute_surface: bool = False


class BrushAnalysis:
    """analysing brushes systems"""

    info_msg: str = 'Messages from BrushAnalysis:\n'
    compute_configs: "ComputationConfig"
    parsed_com: "GetCom"
    box: np.ndarray
    surface_locz: np.ndarray

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
        self.analysis_surfactants(log)

    def set_box(self,
                log: logger.logging.Logger
                ) -> None:
        """set the box info as an attribute"""
        self.box = self._load_xvg_data(
            fname := self.compute_configs.f_box, log, conversion=True)
        self.info_msg += f'\tReading box file: `{fname}`\n'

    def get_interface(self,
                      log: logger.logging.Logger
                      ) -> None:
        """analysis the interface by finding the water surface"""
        if self.compute_configs.compute_surface:
            self.surface_locz = \
                GetSurface(self.parsed_com.split_arr_dict['SOL'],
                           self.parsed_com.box_dims,
                           log).locz_arr
        else:
            self.surface_locz = self._load_xvg_data(
                self.compute_configs.f_surface_locz,
                log)['interface_z'].to_numpy()
            log.warning(
                msg := '\tThe interface location is reading from file!\n')
            print(f'{bcolors.CAUTION}{msg}{bcolors.ENDC}')

    def analysis_surfactants(self,
                             log: logger.logging.Logger
                             ) -> None:
        """Finding the number of the oda at the interface, in the water
        phase, also the order parameters and so on"""
        AnalysisSurfactant(
            oda_arr=self.parsed_com.split_arr_dict['ODN'],
            amino_arr=self.parsed_com.split_arr_dict['AMINO_ODN'],
            interface_z=self.surface_locz,
            log=log)

    def _load_xvg_data(self,
                       fname: str,
                       log: logger.logging.Logger,
                       conversion: bool = False,
                       x_type: type = int
                       ) -> pd.DataFrame:
        """Load and return the data from an XVG file."""
        try:
            data_df = xvg.XvgParser(fname, log, x_type).xvg_df
            if conversion:
                return data_df.iloc[:, 1:4].to_numpy() * \
                    self.compute_configs.unit_nm_to_angstrom
            return data_df
        except (ValueError, FileExistsError, FileNotFoundError) as err:
            log.error(f'Error loading {fname}: {err}')
            sys.exit(
                f'{bcolors.FAIL}Error loading {fname}: {err}{bcolors.ENDC}')

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
