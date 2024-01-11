"""
Analyzes octadecylamine (ODA) molecules in surfactant-rich systems.
This method focuses on examining ODA molecules both at the water-decane
interface and within the water phase. It involves calculating the
distribution and order parameters of ODAs. These parameters provide
insights into the structural organization and behavior of ODA molecules
in different regions of the system.

Inputs:
    1. Array of the center of mass of ODA molecules: Provides spatial
        information of ODA molecules in the system.
    2. Array of the center of mass of the ODA's amino groups: Used to
        analyze the orientation and distribution of the functional
        groups of ODA.
    3. Array of the z-component of the location of the interface:
        Helps in distinguishing between ODAs at the interface and those
        in the water phase.

The method involves:
    - Identifying the location of ODA molecules relative to the
        interface.
    - Calculating the number of ODA molecules at the interface and in
        the water phase.
    - Assessing the order parameters of ODA at the interface and, if
        applicable, in the water phase for understanding molecular
        organization.
"""

from dataclasses import dataclass

import numpy as np

from common import logger


@dataclass
class ParamConfig:
    """constants values and other parameters"""


@dataclass
class ComputationConfig(ParamConfig):
    """set all the configurations"""


class AnalysisSurfactant:
    """computation!"""

    info_msg: str = 'Message from AnalysisSurfactant:\n'

    def __init__(self,
                 oda_arr: np.ndarray,  # COM of the oda residues
                 amino_arr: np.ndarray,  # COM of the amino group on oda
                 interface_z: np.ndarray,  # Average loc of the interface
                 log: logger.logging.Logger,
                 compute_config: "ComputationConfig" = ComputationConfig()
                 ) -> None:
        pass


if __name__ == "__main__":
    print('\nThis scripts is run within trajectory_brushes_analysis\n')
