"""
Computing the order parameter from the trajectory file.
The script reads the trajectory file frame by frame with MDAnalysis
and computes the order parameter for them.
The script computes the order paramters:
    - For water and oil molecules, it should be able to compute along
        the z-axis.
    - For the surfactant molecules, it should be able to compute along
        the y-axis, also only at the main interface; by the main
        interface, we mean the interface between the water and oil
        molecules where the must of surfactant molecules are located,
        and in case there is a nanoparticle in the system, the
        interface where the nanoparticle is located.

The steps will be as follows:
    - Read the trajectory file
    - Read the topology file
    - Select the atoms for the order parameter calculation
    - Compute the order parameter
    - Save the order parameter to a xvg file

Since we already have computed the interface location, we can use that
to compute the order parameter only at the interface.

input files:
    - trajectory file (centered on the NP if there is an NP in the system)
    - topology file
    - interface location file (contact.xvg)
output files:
    - order parameter xvg file along the z-axis
    - in the case of the surfactant molecules, we save a data file
        based on the grids of the interface, looking at the order
        parameter the value and save the average value of the order
        and contour plot of the order parameter.
        This will help us see the order parameter distribution in the
        case of an NP and see how the order parameter changes around
        the NP.
The prodidure is very similar to the computation of the rdf computation
with cosidering the actual volume in:
    read_volume_rdf_mdanalysis.py

12 April 2024
Saeed
Opt by VSCode CoPilot
"""

import os
import sys
import typing
from enum import Enum
from dataclasses import dataclass, field

import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import MDAnalysis as mda

from common import logger, xvg_to_dataframe, my_tools, cpuconfig
from common.colors_text import TextColor as bcolors


class ResidueName(Enum):
    """Residue names for the molecules in the trajectory"""
    # pylint: disable=invalid-name
    SOL = 'WATER'
    D10 = 'OIL'
    SURFACTANT = 'ODN'


@dataclass
class OrderParameter:
    """Order parameter dataclass"""
    resideu_name: ResidueName
    atom_selection: str
    order_parameter_avg: float
    order_parameter_std: float
    order_parameter_data: np.ndarray


@dataclass
class Interface:
    """Interface dataclass"""
    interface_location: float
    interface_location_std: float
    interface_location_data: np.ndarray


@dataclass
class InputFiles:
    """Input files dataclass"""
    trajectory_file: str
    topology_file: str
    interface_location_file: str = 'contact.xvg'
    box_file: str = 'box.xvg'


@dataclass
class ResiduesTails:
    """Residues tails atoms to compute the order parameter
    For water use average location of the hydrogen atoms as one tail
    """
    # pylint: disable=invalid-name
    SOL: list[str] = field(default_factory=lambda: (['OH2', 'H1', 'H2']))
    D10: list[str] = field(default_factory=lambda: (['C1', 'C9']))
    SURFACTANT: list[str] = field(default_factory=lambda: (['C1', 'NH2']))
