"""
Computing Rdf by MDAnalysis module
"""

import os
import sys
import typing
from dataclasses import dataclass, field

import MDAnalysis as mda

from common import logger
from common.colors_text import TextColor as bcolors


@dataclass
class GroupConfig:
    """set the configurations for the rdf
    sel_type -> str: type of the selection, is it residue or atom
    sel_names -> list[str]: names of the selection groups
    sel_pos -> str: If use the ceter of mass (COM) of the group or their
        poistions
    """
    ref_group: dict[str, typing.Any] = field(default_factory=lambda: ({
        'sel_type': 'residue',
        'sel_names': ['COR'],
        'sel_pos': 'COM'
    }))

    traget_group: dict[str, typing.Any] = field(default_factory=lambda: ({
        'sel_type': 'residue',
        'sel_names': ['CLA'],
        'sel_pos': 'position'
    }))


@dataclass
class ParamConfig:
    """set the parameters for the rdf computations with MDA
    MDA:
        "The RDF is limited to a spherical shell around each atom by
        range. Note that the range is defined around each atom, rather
        than the center-of-mass of the entire group.
        If density=True, the final RDF is over the average density of
        the selected atoms in the trajectory box, making it comparable
        to the output of rdf.InterRDF. If density=False, the density
        is not taken into account. This can make it difficult to
        compare RDFs between AtomGroups that contain different numbers
        of atoms."
    """
    n_bins: int = 75  # Default value in MDA
    dist_range: tuple[float, float] = field(default_factory=lambda: ((
        0, 10
    )))
    density: bool = True
