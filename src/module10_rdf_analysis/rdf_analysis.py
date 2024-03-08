"""
RDF Analysis for all the residues in the system.
Rdf is mainly set from the nanoparticle point of view. Since GROMACS
cannot compute the Rdf from the NP COM, MDAnalysis won't give the Rdf
without normalization, and I don't want to go through their sources,
I will compute the Rdf myself. For this, I can use the COM of residues
(computed by module 2) or the trajectory read by MDAnalysis. I will
use the COM of the residues for the start.

The RDF is calculated as follows:
    1. Read the COM of the residues
    2. Read the COM of the NP
    3. Calculate the distance between the COM of the residues and the
    COM of the NP
    4. Calculate the RDF by using the correct volume system
March 7 2024
Saeed
"""

import os
import sys
import typing
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from common import logger, xvg_to_dataframe, my_tools, com_file_parser
from common.colors_text import TextColor as bcolors


@dataclass
class COMFileConfig:
    """Configuration for the RDF analysis"""
    # Input files
    residue_com_fname: str = "pickle_com"
    interface_info: str = "contact.xvg"
    box_size_fname: str = "box.xvg"
    np_com_fname: str = "coord.xvg"


@dataclass
class TrrFileConfig:
    """Configuration for the RDF analysis"""
    # Input files
    trr_fname: str = "traj.trr"
    top_fname: str = "topol.top"


@dataclass
class OutFileConfig:
    """Output files"""
    rdf_fout: str = "rdf.xvg"
    rdf_column: str = "RDF"


@dataclass
class ParameterConfig:
    """Parameters for the RDF analysis"""
    bin_width: float = 0.1
    max_distance: float = 10.0
    min_distance: float = 0.0


@dataclass
class AllConfig(ParameterConfig, OutFileConfig):
    """All the configurations for the RDF analysis
    compute_style: str = "COM" or "TRR"
    """
    compute_style: str = "COM"

    com_file_config: COMFileConfig = field(init=False)
    trr_file_config: TrrFileConfig = field(init=False)

    def __post_init__(self):
        if self.compute_style not in ["COM", "TRR"]:
            raise ValueError(f"compute_style should be either 'COM' or 'TRR' "
                             f"but it is {self.compute_style}")
        if self.compute_style == "COM":
            com_file_config = COMFileConfig()
        if self.compute_style == "TRR":
            trr_file_config = TrrFileConfig()


class RdfAnalysis:
    """compute RDF for the system based on the configuration"""

    info_msg: str = 'Message from RdfAnalysis:\n'
    config: AllConfig

    def __init__(self,
                 log: logger.logging.Logger,
                 config: AllConfig = AllConfig()
                 ) -> None:
        self.config = config
        self.initiate(log)
        self._write_msg(log)

    def initiate(self, log: logger.logging.Logger) -> None:
        """initiate the calculations"""
        file_config: typing.Union[COMFileConfig, TrrFileConfig]
        if self.config.compute_style == "COM":
            file_config = self.config.com_file_config
        if self.config.compute_style == "TRR":
            file_config = self.config.trr_file_config


    def _write_msg(self, log: logger.logging.Logger) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == "__main__":
    RdfAnalysis(logger.setup_logger(log_name="rdf_analysis.log"))
