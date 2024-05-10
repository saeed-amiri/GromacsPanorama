"""
# atom_selection.py

This module contains the `AtomSelection` class which is used for
selecting specific atoms in a molecular dynamics simulation for the
purpose of computing angles.

## Class: AtomSelection

### Attributes:
- `configs`: An instance of `AllConfig` class which holds all the
    configuration data.
- `universe`: An instance of `mda.coordinates.TRR.TRRReader` which
    represents the molecular dynamics universe.
- `tails_position`: A dictionary that stores the positions of tail
    atoms for all frames.
- `heads_position`: A dictionary that stores the positions of head
    atoms for all frames.
- `info_msg`: A string used to store messages from the `AtomSelection`
    class.

### Methods:
- `__init__(self, universe, configs, log)`: Initializes the
    `AtomSelection` class with a universe, configurations, and a logger.
- `get_atoms(self, tail_atoms, head_atoms)`: Returns the positions of
    the tail and head atoms for all frames.
- `get_tail_indices(self)`: Returns the tail atoms from the universe
    based on the selection string.
- `_get_tail_atoms_selection_str(self)`: Returns the selection string
    for tail atoms.
- `get_head_indices(self)`: Returns the head atoms from the universe
    based on the selection string.
- `_get_head_atoms_selection_str(self)`: Returns the selection string
    for head atoms.

## Usage:

This module is used for selecting specific atoms in a molecular
dynamics simulation. The `AtomSelection` class is initialized with a
universe, configurations, and a logger. The positions of the tail and
head atoms for all frames can be obtained using the `get_atoms` method.
The tail and head atoms can be selected from the universe using the
`get_tail_indices` and `get_head_indices` methods respectively.
"""


import numpy as np

import MDAnalysis as mda

from common import logger
from common.colors_text import TextColor as bcolors

from module8_analysis_order_parameter.config_classes_trr import AllConfig


class AtomSelection:
    """Atom selection for the copmuting the angles"""

    configs: AllConfig
    universe: "mda.coordinates.TRR.TRRReader"
    tails_position: dict[int, np.ndarray]
    heads_position: dict[int, np.ndarray]
    info_msg: str = 'Message from AtomSelection:\n'

    def __init__(self,
                 universe: "mda.coordinates.TRR.TRRReader",
                 configs: AllConfig,
                 log: logger.logging.Logger
                 ) -> None:
        self.universe = universe
        self.configs = configs
        tail_atoms: "mda.core.groups.AtomGroup" = self.get_tail_indices()
        head_atoms: "mda.core.groups.AtomGroup" = self.get_head_indices()

        self.tails_position, self.heads_position = \
            self.get_atoms(tail_atoms, head_atoms)
        self.write_msg(log)

    def get_atoms(self,
                  tail_atoms: "mda.core.groups.AtomGroup",
                  head_atoms: "mda.core.groups.AtomGroup"
                  ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        """Get the atoms positions"""
        # Initialize a dictionary to store positions for all frames
        tails_positions: dict[int, np.ndarray] = {}
        heads_positions: dict[int, np.ndarray] = {}

        # Loop over all frames
        for tstep in self.universe.trajectory:
            # Store positions for the current frame
            tails_positions[tstep.frame] = tail_atoms.positions
            heads_positions[tstep.frame] = head_atoms.positions

        return tails_positions, heads_positions

    def get_tail_indices(self) -> "mda.core.groups.AtomGroup":
        """Get the tail atoms"""
        selection_str: str = self._get_tail_atoms_selection_str()
        return self.universe.select_atoms(selection_str)

    def _get_tail_atoms_selection_str(self) -> str:
        """Get the tail atoms selection string"""
        resname: str = self.configs.selected_res.value
        selected_res: str = self.configs.selected_res.name
        tails_atoms: str = getattr(self.configs.residues_tails,
                                   selected_res,
                                   {'tail': None})['tail']
        selection_str: str = f'resname {resname} and name {tails_atoms}'
        self.info_msg += f'\tTail selection string: `{selection_str}`\n'
        return selection_str

    def get_head_indices(self) -> "mda.core.groups.AtomGroup":
        """Get the head atoms"""
        selection_str: str = self._get_head_atoms_selection_str()
        return self.universe.select_atoms(selection_str)

    def _get_head_atoms_selection_str(self) -> str:
        """Get the head atoms selection string"""
        resname: str = self.configs.selected_res.value
        selected_res: str = self.configs.selected_res.name
        head_atoms: str = getattr(self.configs.residues_tails,
                                  selected_res,
                                  {'head': 0})['head']
        selection_str: str = f'resname {resname} and name {head_atoms}'
        self.info_msg += f'\tHead selection string: `{selection_str}`\n'
        return selection_str

    def write_msg(self,
                  log: logger.logging.Logger
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{AtomSelection.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)
