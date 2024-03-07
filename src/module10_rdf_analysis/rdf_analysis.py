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


@dataclass
class FileConfig:
    """Configuration for the RDF analysis"""
    # Input files
    residue_com_fname: str = "pickle_com"
    interface_info: str = "contact.xvg"
    box_size_fname: str = "box.xvg"
    np_com_fname: str = "coord.xvg"
    # Output files
    rdf_fout: str = "rdf.xvg"
    rdf_column: str = "RDF"


@dataclass
class ParameterConfig:
    """Parameters for the RDF analysis"""
    bin_width: float = 0.1
    max_distance: float = 10.0
    min_distance: float = 0.0


@dataclass
class AllConfig(FileConfig, ParameterConfig):
    """All the configurations for the RDF analysis"""


if __name__ == "__main__":
    pass
