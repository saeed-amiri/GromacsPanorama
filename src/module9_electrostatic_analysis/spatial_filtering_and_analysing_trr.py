"""
Dynamic Trajectory Filtering and Analysis Script

This script performs dynamic selection of atoms within a specified
radius from the center of mass (COM) of a defined target (e.g., a
nanoparticle) in molecular dynamics (MD) simulations. For each frame
of the input trajectory, it identifies atoms (and their respective
esidues) within the defined radius, analyzes specified properties of
this dynamic selection, and writes a new trajectory file containing
only the selected atoms for further analysis.

The script is designed to facilitate the study of dynamic interactions
and the local environment around specified targets in MD simulations,
providing insights into spatial and temporal variations in properties
such as density, coordination number, or other custom analyses.

Usage:
  python spatial_filtering_and_analysing_trr.py  <input_trajectory>
The tpr file and gro file names will be created from the name of the
input trr file.
Some of the functionality is similar as in pqr_from_pdb_gro.py

Arguments that set by the script:
    topology_file : Topology file (e.g., .gro, .pdb) corresponding to
        the input trajectory.
    output_trajectory: Output trajectory file name for the filtered
        selection.
    radius: Radius (in Ångströms) for dynamic selection around the
        target's COM.
    statistics output: Output file name for numbers of residues and
        charges of each of them.
    charge density output: Output file name for the charge density and
        potential of the final system

Options:
    selection_string : MDAnalysis-compatible selection string for
        defining the target (default: 'COR_APT').

Features:
    - Dynamic selection based on spatial criteria, adjustable for each
        frame of the trajectory.
    - Analysis of selected properties for the dynamically selected
        atoms/residues.
    - Generation of a new, filtered trajectory file for targeted
        further analysis.

Example:
    python spatial_filtering_and_analysing_trr.py simulation.trr

Opt. by ChatGpt.
21 Feb 2023
Saeed
"""
import os
import sys
import typing
import string
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from common import logger, itp_to_df, pdb_to_df, gro_to_df, my_tools
from common.colors_text import TextColor as bcolors

from module9_electrostatic_analysis import parse_charmm_data, \
    force_field_path_configure


@dataclass
class InFileConfig:
    """Set the name of the input files"""
    # Name of the input file, set by user:
    traj_fname: str = field(init=False)

    # ForceField file to get the charges of the nanoparticle, since
    # they are different depend on the contact angle:
    itp_file: str = 'APT_COR.itp'  # FF of nanoparticle

    accebtable_file_type: list[str] = \
        field(default_factory=lambda: ['trr', 'xtc'])


@dataclass
class OutFileConfig:
    """set the names of the output files"""
    filtered_traj: str = field(init=False)  # Will set based on input
    fout_stats: str = 'stats.xvg'
    fout_charge_density: str = 'charge_density.xvg'


@dataclass
class FFTypeConfig:
    """set the name of each file with their name of the data"""
    ff_type_dict: dict[str, str] = field(
        default_factory=lambda: {
            "SOL": 'TIP3',
            "CLA": 'CLA',
            "D10": 'D10_charmm',
            "ODN": 'ODAp_charmm',
            'APT': 'np_info',
            'COR': 'np_info'
        })


@dataclass
class NumerInResidue:
    """Number of atoms in each residues
    The number of atoms in APT is not known since it changes based on
    contact angle"""
    res_number_dict: dict[str, int] = field(
        default_factory=lambda: {
            "SOL": 3,
            "CLA": 1,
            'POT': 1,
            "D10": 32,
            "ODN": 59,
            'COR': 4356,
            'APT': 0
        })


@dataclass
class AllConfig(InFileConfig,
                OutFileConfig,
                FFTypeConfig,
                NumerInResidue):
    """set all the configurations and parameters"""
    stern_radius: float = 30  # In Ångströms


class TrrFilterAnalysis:
    """get the trajectory and do the analysis"""

    info_msg: str = 'Message from TrrFilterAnalysis:\n'
    configs: AllConfig

    def __init__(self,
                 traj_fname: str,
                 log: logger.logging.Logger,
                 configs: AllConfig = AllConfig()
                 ) -> None:
        self.configs = configs
        self.configs.traj_fname = traj_fname
        self.initiate(log)
        self.write_msg(log)

    def initiate(self,
                 log: logger.logging.Logger
                 ) -> None:
        """setting the names, reading files, filttering traj file and
        analaysing and writting output"""
        self.set_check_in_files(log)

    def set_check_in_files(self,
                           log: logger.logging.Logger
                           ) -> None:
        """set the names and check if they exist"""
        root_name: str = self.configs.traj_fname.split('.', -1)[0]
        tpr: str = f'{root_name}.tpr'
        gro: str = f'{root_name}.gro'
        if (if_tpr := my_tools.check_file_exist(tpr, log, False)) is None:
            self.info_msg += f'\tThe topology file is set to `{tpr}`\n'
        elif if_tpr is False:
            if my_tools.check_file_exist(gro, log, False) is None:
                self.info_msg += f'\tThe topology file is set to `{tpr}`\n'
            else:
                log.error(msg := f'\tError! `{gro}` or `{tpr}` not exist!\n')
                sys.exit(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')

    def write_msg(self,
                  log: logger.logging.Logger  # To log
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{TrrFilterAnalysis.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    TrrFilterAnalysis(sys.argv[1], logger.setup_logger('trr_charge.log'))
