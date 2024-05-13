"""
# angle_computation_trr.py
This module contains the `AngleProjectionComputation` class which is
used for computing the angle projection of vectors along all axes in
a molecular dynamics simulation.

## Class: AngleProjectionComputation
### Attributes:
- `info_msg`: A string used to store messages from the
    `AngleProjectionComputation` class.
- `configs`: An instance of `AllConfig` class which holds all the
    configuration data.
- `tail_with_angle`: A list of numpy arrays that store the tail
    coordinates and their angle projections along all axes.

### Methods:
- `__init__(self, log, universe, configs)`: Initializes the
    `AngleProjectionComputation` class with a logger, a universe, and
    configurations.
- `compute_angle_projection(self, log)`: Computes the angle projection
    of the vectors along all axes and returns a list of tail coordinates
    with their angle projections.
- `compute_head_tail_vectors(self, tails_positions, heads_positions,
    log)`: Computes the head and tail vectors for each frame.
- `compute_angles(self, unit_vec_dict)`: Computes the angles for each
    unit vector.

## Usage:
This module is used for computing the angle projection of vectors along
all axes in a molecular dynamics simulation. The
`AngleProjectionComputation` class is initialized with a logger, a
universe, and configurations. The angle projections are computed using
the `compute_angle_projection` method. The head and tail vectors for
each frame are computed using the `compute_head_tail_vectors` method,
and the angles for each unit vector are computed using the
`compute_angles` method.
"""


import typing
import numpy as np

from common import logger
from common.colors_text import TextColor as bcolors

from module11_order_parameter_from_trr.config_classes_trr import AllConfig
from module11_order_parameter_from_trr.atom_selection_trr import AtomSelection

if typing.TYPE_CHECKING:
    import MDAnalysis as mda


class AngleProjectionComputation:
    """doing the math for computing the angle projection of the vectors
    along all the axes and return a list of head cordinated with angle
    projections along all the axes in the form of:
        list[[x, y, z, angle_x, angle_y, angle_z], ...]
    """

    info_msg: str = 'Message from AngleProjectionComputation:\n'
    configs: AllConfig
    tail_with_angle: list[np.ndarray]

    def __init__(self,
                 log: logger.logging.Logger,
                 universe: "mda.coordinates.TRR.TRRReader",
                 configs: AllConfig
                 ) -> None:
        self.configs = configs
        self.universe = universe
        self.tail_with_angle = self.compute_angle_projection(log)
        self.info_msg += '\tThe angle along all the axes are computed\n'
        self.write_msg(log)

    def compute_angle_projection(self,
                                 log: logger.logging.Logger
                                 ) -> list[np.ndarray]:
        """Compute the anlge projection of the vectors along all the axes"""

        atoms_selection: AtomSelection = AtomSelection(
            self.universe, self.configs, log)
        tails_position: dict[int, np.ndarray] = atoms_selection.tails_position
        heads_position: dict[int, np.ndarray] = atoms_selection.heads_position
        unit_vec_dict: dict[int, np.ndarray] = \
            self.compute_head_tail_vectors(tails_position,
                                           heads_position,
                                           log)
        angels_dict: dict[int, np.ndarray] = \
            self.compute_angles(unit_vec_dict)
        tail_with_angle: list[np.ndarray] = \
            self.appned_angles_to_tails(tails_position, angels_dict)
        return tail_with_angle

    def compute_head_tail_vectors(self,
                                  tails_positions: dict[int, np.ndarray],
                                  heads_positions: dict[int, np.ndarray],
                                  log: logger.logging.Logger
                                  ) -> dict[int, np.ndarray]:
        """Compute the head and tail vectors
        The vectors are a vector from the tail to the head
        for each frame there are N vector, where N is the number of
        molecules in the system or the length of the tail_ or
        head_positions
        """
        unit_vec_dict: dict[int, np.ndarray] = {}
        for frame in tails_positions.keys():
            tail_positions: np.ndarray = tails_positions[frame]
            head_positions: np.ndarray = heads_positions[frame]
            unit_vec_dict[frame] = self.get_head_tail_vector(
                tail_positions, head_positions, log)
        return unit_vec_dict

    def compute_angles(self,
                       unit_vec_dict: dict[int, np.ndarray],
                       ) -> dict[int, np.ndarray]:
        """Compute the angles between the vectors and the all axes"""
        angles_dict: dict[int, np.ndarray] = {}
        for frame, vec in unit_vec_dict.items():
            angles_dict[frame] = self.compute_angles_for_frame(vec)
        return angles_dict

    def appned_angles_to_tails(self,
                               tails_positions: dict[int, np.ndarray],
                               angels_dict: dict[int, np.ndarray]
                               ) -> list[np.ndarray]:
        """Append the angles to the tail positions"""
        tail_angles: list[np.ndarray] = []
        for frame, tails in tails_positions.items():
            tail_angles.append(np.hstack((tails, angels_dict[frame])))
        return tail_angles

    def compute_angles_for_frame(self,
                                 vec: np.ndarray
                                 ) -> np.ndarray:
        """Compute the angles for a frame"""
        angles_vec: np.ndarray = np.zeros((vec.shape[0], 3))
        for i, vec_i in enumerate(vec):
            angles_vec[i] = self.compute_angles_for_vector(vec_i)
        return angles_vec

    def compute_angles_for_vector(self,
                                  vec: np.ndarray
                                  ) -> np.ndarray:
        """Compute the angles of the normelaized vector with the all axes"""
        vec = vec / np.linalg.norm(vec)
        angles: np.ndarray = np.zeros(3)
        for i in range(3):
            angles[i] = np.arccos(np.dot(vec, np.eye(3)[i]))
        return angles

    def get_head_tail_vector(self,
                             tail_positions: np.ndarray,
                             head_positions: np.ndarray,
                             log: logger.logging.Logger
                             ) -> np.ndarray:
        """
        Compute the head and tail vectors and return the vector
        """
        if tail_positions.shape != head_positions.shape:
            log.error(msg :=
                      'head_ and tail_positions must have the same shape')
            raise ValueError(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')

        return head_positions - tail_positions

    def write_msg(self,
                  log: logger.logging.Logger
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{AngleProjectionComputation.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)
