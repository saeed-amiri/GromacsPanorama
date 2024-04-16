"""
# config_classes_trr.py
This module contains several dataclasses and an Enum used for
configuration and data storage in a molecular dynamics simulation
analysis.

## Enum: ResidueName
This Enum represents the names of the residues in the trajectory. It
has three members: `OIL`, `WATER`, and `SURFACTANT`.

## Dataclass: OrderParameterConfig
This dataclass stores configuration and results for the order parameter
computation. It has five fields: `resideu_name`, `atom_selection`,
`order_parameter_avg`, `order_parameter_std`, and `order_parameter_data`.

## Dataclass: InterfaceConfig
This dataclass stores configuration and results for the interface
location computation. It has three fields: `interface_location`,
`interface_location_std`, and `interface_location_data`.

## Dataclass: InputFiles
This dataclass stores the names and paths of the input files for the
simulation. It has five fields: `box_file`, `tpr_file`, `trajectory_file`,
`interface_location_file`, and `path_name`. The `__post_init__`
method is used to join the `path_name` with the file names.

## Dataclass: ResiduesTails
This dataclass stores the atom names for the head and tail of each
residue type. It has three fields: `OIL`, `WATER`, and `SURFACTANT`.
Each field is a dictionary with `head` and `tail` keys.

## Usage:
This module is used to store and manage configuration data and results
for a molecular dynamics simulation analysis. The dataclasses are used
to store data in a structured way, and the Enum is used to represent
the names of the residues in a type-safe way.
"""

import os
import typing

from enum import Enum
from dataclasses import dataclass, field

import numpy as np


class ResidueName(Enum):
    """Residue names for the residues in the trajectory"""
    # pylint: disable=invalid-name
    OIL = 'D10'
    WATER = 'SOL'
    SURFACTANT = 'ODN'


@dataclass
class OrderParameterConfig:
    """Order parameter dataclass"""
    resideu_name: typing.Union[None, ResidueName] = field(init=False)
    atom_selection: typing.Union[None, str, list[str]] = field(init=False)
    order_parameter_avg: float = field(init=False)
    order_parameter_std: float = field(init=False)
    order_parameter_data: np.ndarray = field(init=False)


@dataclass
class InterfaceConfig:
    """Interface configuration dataclass"""
    interface_location: float = 0.0
    interface_location_std: float = 0.0
    interface_location_data: np.ndarray = np.array([])


@dataclass
class InputFiles:
    """Input files dataclass"""
    box_file: str = 'box.xvg'
    tpr_file: str = field(init=False)
    trajectory_file: str = field(init=False)
    interface_location_file: str = 'contact.xvg'
    path_name: str = '/scratch/saeed/GÖHBP/PRE_DFG_7May24/single_np/15Oda/data'

    def __post_init__(self) -> None:
        """Post init function"""
        self.interface_location_file = \
            os.path.join(self.path_name, self.interface_location_file)
        self.box_file = os.path.join(self.path_name, self.box_file)


@dataclass
class ResiduesTails:
    """Residues tails atoms to compute the order parameter
    For water use average location of the hydrogen atoms as one tail
    """
    # pylint: disable=invalid-name
    OIL: dict[str, typing.Union[str, list[str]]] = \
        field(default_factory=lambda: ({'head': 'C1',
                                        'tail': 'C9'}))
    WATER: dict[str, typing.Union[str, list[str]]] = \
        field(default_factory=lambda: ({'head': 'OH2',
                                        'tail': ['H1', 'H2']}))
    SURFACTANT: dict[str, typing.Union[str, list[str]]] = \
        field(default_factory=lambda: ({'head': 'CT3',
                                        'tail': 'NH2'}))


@dataclass
class AllConfig(ResiduesTails):
    """Order parameter options dataclass"""
    residues_tails: ResiduesTails = field(default_factory=ResiduesTails)
    interface: InterfaceConfig = field(default_factory=InterfaceConfig)
    order_parameter: OrderParameterConfig = \
        field(default_factory=OrderParameterConfig)
    input_files: InputFiles = field(default_factory=InputFiles)

    # The residue names to compute the order parameter
    # use the ResidueName enum names
    selected_res: ResidueName = ResidueName.SURFACTANT
    # number of the cores for multiprocessing computation
    n_cores: int = field(init=False)
