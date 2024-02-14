"""
The `apbs_electrostatic_computation.py` module automates the preparation
and processing of biomolecular data for electrostatic computations
using the Adaptive Poisson-Boltzmann Solver (APBS). This module
integrates several steps essential for the accurate modeling of
biomolecular electrostatics, streamlining the workflow from raw data
extraction to ready-to-use input for APBS.

Workflow Overview:
    1. Extracts data from Gromacs trajectories using Visual Molecular
    Dynamics (VMD), enabling the analysis of molecular dynamics
    simulations.
    2. Converts the extracted data into PQR format, incorporating
    force field information to assign appropriate radii and charges to
    atoms. This step is crucial for accurately representing molecular
    structures in electrostatic calculations.
    3. Prepares the resulting PQR files for input into APBS,
    facilitating the solving of equations of continuum electrostatics
    for large biomolecular assemblages.
Opt. by ChatGpt
14 Feb 2024
Saeed
"""

import sys
import typing
from dataclasses import dataclass, field

from common import logger, itp_to_df, pdb_to_df, gro_to_df, my_tools
from common.colors_text import TextColor as bcolors
