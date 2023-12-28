"""
charge_analysis_interface_np.py

This script is designed to perform charge analysis on nanoparticles or
interfaces. It reads relevant data and computes charge distributions,
identifying key charge-related properties and interactions. This
analysis aids in understanding the electrostatic characteristics of
nanoparticles and interfaces in various environments.

Functions:
- read_data: Loads and preprocesses the input data.
- compute_charge_distribution: Calculates the charge distribution on
    the nanoparticle or interface.
- analyze_interactions: Analyzes electrostatic interactions based on
    charge distributions.
- generate_report: Summarizes the findings in a structured report.

Usage:
Run this script with the required data files to obtain a detailed
analysis of charges on nanoparticles or interfaces.

Saeed Amiri
Date: 27 Dec 2023
"""

import sys
from dataclasses import dataclass

import numpy as np

from common import logger
from common.com_file_parser import GetCom
from common.colors_text import TextColor as bcolors

from module6_charge_analysis import np_charge_analysis


@dataclass
class FileConfigurations:
    """names of the input files"""
    f_rdf: str = 'cla_rdf.xvg'
    f_cdf: str = 'cla_cdf.xvg'
    f_contact: str = 'contact.xvg'
    f_coord: str = 'coord.xvg'
    f_box: str = 'box.xvg'


@dataclass
class ComputeConfigurations(FileConfigurations):
    """configure input for the calculations"""


class ComputeCharges:
    """claclulate charges around nanoparticle (total or whole
    Also if needed for other situations
    """

    info_msg: str = "Messeges from ComputeCharges:\n"

    def __init__(self,
                 fname: str,  # Name of the com file
                 log: logger.logging.Logger,
                 config: "ComputeConfigurations" = ComputeConfigurations()
                 ) -> None:
        self.config = config
        self.parsed_com = GetCom(fname)
        self.write_msg(log)
        self.initiate_np_charge_analysis(log)

    def initiate_np_charge_analysis(self,
                                    log: logger.logging.Logger
                                    ) -> None:
        """
        Analysing the charge of the nanoparticle during simulations
        """
        cla_arr: np.ndarray = self.parsed_com.split_arr_dict['CLA']
        np_charge_analysis.NpChargeAnalysis(cla_arr, self.config, log)

    def write_msg(self,
                  log: logger.logging.Logger  # To log
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == "__main__":
    LOG = logger.setup_logger('charge_analysis.log')
    try:
        ComputeCharges(fname=sys.argv[1], log=LOG)
    except IndexError:
        LOG.error("No command line argument provided for the filename.")
        sys.exit(1)
    else:
        LOG.error("Failed to initialize charge analysis")
        sys.exit(1)
