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

import multiprocessing
from dataclasses import dataclass

import numpy as np

from common import logger
from common import cpuconfig
from common.colors_text import TextColor as bcolors


@dataclass
class ParamConfig:
    """constants values and other parameters"""
    interface_thickness: float = 10  # in Angstrom, to serch for the ODA
    interface_avg_nr_frames: int = 100  # Nr of frames to get interface avg


@dataclass
class ComputationConfig(ParamConfig):
    """set all the configurations"""


class AnalysisSurfactant:
    """ Class to analyze ODA molecules in surfactant-rich systems. """

    info_msg: str = 'Message from AnalysisSurfactant:\n'

    compute_config: "ComputationConfig"
    interface_z: np.ndarray

    def __init__(self,
                 oda_arr: np.ndarray,  # COM of the oda residues
                 amino_arr: np.ndarray,  # COM of the amino group on oda
                 interface_z: np.ndarray,  # Average loc of the interface
                 log: logger.logging.Logger,
                 compute_config: "ComputationConfig" = ComputationConfig()
                 ) -> None:

        # pylint: disable=too-many-arguments

        self.compute_config = compute_config
        self.interface_z = interface_z
        self.initiate(oda_arr[:-2], amino_arr[:-2], log)
        self._write_msg(log)

    def initiate(self,
                 oda_arr: np.ndarray,  # COM of the oda residues
                 amino_arr: np.ndarray,  # COM of the amino group on oda
                 log: logger.logging.Logger
                 ) -> None:
        """initialization of the calculations"""
        interface_oda_ind: dict[int, np.ndarray]
        interface_oda_nr: dict[int, int]
        interface_oda_ind, interface_oda_nr = \
            self.get_interface_oda_inidcies(amino_arr, log)

    def get_interface_oda_inidcies(self,
                                   amino_arr: np.ndarray,
                                   log: logger.logging.Logger
                                   ) -> tuple[dict[int, np.ndarray],
                                              dict[int, int]]:
        """find the indicies of the oda at interface at each frame
        using the amino com since they are more charctristic in the oda
        return the indices of the oda at the interface and number of
        them at each frame
        """
        cpu_info = cpuconfig.ConfigCpuNr(log)
        n_cores: int = min(cpu_info.cores_nr, amino_arr.shape[0])

        interface_oda_ind: dict[int, np.ndarray] = {}
        interface_oda_nr: dict[int, int] = {}
        interface_bounds: tuple[np.float64, np.float64] = \
            self._get_interface_bounds()

        with multiprocessing.Pool(processes=n_cores) as pool:
            results = pool.starmap(
                self._process_single_frame, [
                    (interface_bounds, i_frame, frame) for i_frame, frame
                    in enumerate(amino_arr)])
        for i_frame, result in enumerate(results):
            interface_oda_ind[i_frame] = result
            interface_oda_nr[i_frame] = len(result)
        return interface_oda_ind, interface_oda_nr

    def _process_single_frame(self,
                              interface_bounds: tuple[np.float64, np.float64],
                              i_frame: int,  # Index of the frame
                              frame: np.ndarray  # One frame of the amino com
                              ) -> np.ndarray:
        """process a single frame and find the index of the interfaces
        oda
        i_frame is useful for debugging
        """
        # pylint: disable=unused-argument

        xyz_i: np.ndarray = frame.reshape(-1, 3)
        ind_in_interface: np.ndarray = \
            np.where((xyz_i[:, 2] > interface_bounds[0]) &
                     (xyz_i[:, 2] < interface_bounds[1]))[0]
        return ind_in_interface

    def _get_interface_bounds(self) -> tuple[np.float64, np.float64]:
        """compute the upper and lower of the interface"""
        inface_std: np.float64 = np.std(self.interface_z)
        inface_ave: np.float64 = np.mean(
            self.interface_z[:self.compute_config.interface_avg_nr_frames])
        inface_bounds: tuple[np.float64, np.float64] = (
            inface_ave - inface_std - self.compute_config.interface_thickness,
            inface_ave + inface_std + self.compute_config.interface_thickness,
        )
        self.info_msg += ('\tThe average interface from first '
                          f'`{self.compute_config.interface_avg_nr_frames}` '
                          f'frames is `{inface_ave:.3f}\n'
                          f'\tThe bound set to `({inface_bounds[0]:.3f}, '
                          f'{inface_bounds[1]:.3f})`\n'
                          )
        return inface_bounds

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{AnalysisSurfactant.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == "__main__":
    print('\nThis scripts is run within trajectory_brushes_analysis\n')
