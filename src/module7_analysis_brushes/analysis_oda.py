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

import sys
import multiprocessing
from dataclasses import dataclass

import numpy as np
import pandas as pd

from common import logger
from common import cpuconfig
from common import file_writer
from common.colors_text import TextColor as bcolors


@dataclass
class ParamConfig:
    """constants values and other parameters"""
    interface_thickness: float = 10  # in Angstrom, to serch for the ODA
    interface_avg_nr_frames: int = 100  # Nr of frames to get interface avg


@dataclass
class OrderParameterConfig:
    """parametrs and other configuration for computing order parmaeter"""
    director_ax: str = 'z'


@dataclass
class ComputationConfig(ParamConfig):
    """set all the configurations"""
    orderp_config: "OrderParameterConfig" = OrderParameterConfig()


class AnalysisSurfactant:
    """ Class to analyze ODA molecules in surfactant-rich systems. """

    info_msg: str = 'Message from AnalysisSurfactant:\n'

    compute_config: "ComputationConfig"
    interface_z: np.ndarray
    order_parameters: np.ndarray

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
        self.order_parameters = \
            self.initiate(oda_arr[:-2], amino_arr[:-2], log)
        self._write_msg(log)

    def initiate(self,
                 oda_arr: np.ndarray,  # COM of the oda residues
                 amino_arr: np.ndarray,  # COM of the amino group on oda
                 log: logger.logging.Logger
                 ) -> np.ndarray:
        """initialization of the calculations"""
        interface_oda_ind: dict[int, np.ndarray]
        interface_oda_nr: dict[int, int]
        order_parameter: np.ndarray
        interface_oda_ind, interface_oda_nr = \
            self.get_interface_oda_inidcies(amino_arr, log)
        order_parameter = self.compute_interface_oda_order_parameter(
            amino_arr, oda_arr, interface_oda_ind, log)
        data_df: pd.DataFrame = self.make_df(interface_oda_nr, order_parameter)
        file_writer.write_xvg(data_df, log, fname := 'order_parameter.xvg')
        self.info_msg += f'\tThe dataframe saved to `{fname}`\n'
        return order_parameter

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

    def compute_interface_oda_order_parameter(self,
                                              amino_arr: np.ndarray,
                                              oda_arr: np.ndarray,
                                              interface_oda_ind: dict[
                                               int, np.ndarray],
                                              log: logger.logging.Logger
                                              ) -> np.ndarray:
        """compute the order parameter for the oda at interface"""
        order_parameter = ComputeOrderParameter(
            head_arr=amino_arr,
            tail_arr=oda_arr,
            indicies=interface_oda_ind,
            compute_config=self.compute_config.orderp_config,
            log=log
            ).order_parameters
        self.info_msg += (
            f'\tMean of order parameter: `{np.mean(order_parameter):.3f}`\n'
            f'\tStd of order parameter: `{np.std(order_parameter):.3f}\n')
        return order_parameter

    def make_df(self,
                interface_oda_nr: dict[int, int],
                order_parameter: np.ndarray
                ) -> pd.DataFrame:
        """prepare dataframe to write the data to xvg file"""
        columns: list[str] = ['nr_interface_oda', 'order_parameter']
        df_i: pd.DataFrame = pd.DataFrame(columns=columns)
        df_i['nr_interface_oda'] = interface_oda_nr.values()
        df_i['order_parameter'] = order_parameter.tolist()
        return df_i

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{AnalysisSurfactant.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


class ComputeOrderParameter:
    """compute the order parameter from inputs
    inputs:
        1- Head information (coordinates, in com_pickle format)
        2- Tail information (coordinates, in com_pickle format)
        4- Indicies of the selected Oda at each frame
        3- The director axis (optional, default: z)
    """
    # pylint: disable=too-few-public-methods

    config: "OrderParameterConfig"
    order_parameters: np.ndarray
    director_ax: np.ndarray

    def __init__(self,
                 head_arr: np.ndarray,
                 tail_arr: np.ndarray,
                 indicies: dict[int, np.ndarray],
                 compute_config: "OrderParameterConfig",
                 log: logger.logging.Logger
                 ) -> None:
        # pylint: disable=too-many-arguments

        self.config = compute_config
        self.order_parameters = \
            self._initiate(head_arr, tail_arr, indicies, log)

    def _initiate(self,
                  head_arr: np.ndarray,
                  tail_arr: np.ndarray,
                  indicies: dict[int, np.ndarray],
                  log: logger.logging.Logger
                  ) -> np.ndarray:
        """finding the order parameters here"""

        if len(head_arr) != len(tail_arr):
            log.error(
                msg := '\n\tThere is problem in length of tails and heads\n')
            sys.exit(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')

        cpu_info = cpuconfig.ConfigCpuNr(log)
        n_cores: int = min(cpu_info.cores_nr, len(head_arr))

        if self.config.director_ax == 'z':
            self.director_ax = np.array([0, 0, 1])
        with multiprocessing.Pool(processes=n_cores) as pool:
            results = pool.starmap(
                self._process_single_frame, [
                    (head, tail, indicies, i_frame, log) for
                    i_frame, (head, tail) in enumerate(zip(head_arr, tail_arr))
                    ])
        return np.array(results)

    def _process_single_frame(self,
                              head: np.ndarray,
                              tail: np.ndarray,
                              indicies: dict[int, np.ndarray],
                              i_frame: int,
                              log: logger.logging.Logger
                              ) -> np.float64:
        """compute order [arameter or each frame
        The indices of interface oda are used to calculate it
        """
        # pylint: disable=too-many-arguments

        interface_indices: np.ndarray = indicies[i_frame]
        try:
            interface_head: np.ndarray = head.reshape(-1, 3)[interface_indices]
            interface_tail: np.ndarray = tail.reshape(-1, 3)[interface_indices]
        except ValueError:
            log.error(msg := '\tThere is a problem in reshaping arrays!\n')
            sys.exit(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')

        if interface_head.shape[1] != 3:
            log.error(msg := "\tWrong nr of elements after reshape head\n")
            raise ValueError(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')

        if interface_tail.shape[1] != 3:
            log.error(msg := "\tWrong nr of elements after reshape tail\n")
            raise ValueError(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')

        head_tail_vec: np.ndarray = interface_head - interface_tail
        try:
            normalized_vectors: np.ndarray = head_tail_vec / np.linalg.norm(
                head_tail_vec, axis=1)[:, np.newaxis]
        except ZeroDivisionError:
            log.error(msg := "\tThere is problem in getting normalized vec\n")
            sys.exit(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')

        if normalized_vectors.ndim != 2 or normalized_vectors.shape[1] != 3:
            log.error(msg := ("\tNormalized_vectors must be a 2D array with "
                              "shape (-1, 3)\n"))
            raise ValueError(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')

        if self.director_ax.ndim != 1 or len(self.director_ax) != 3:
            log.error(msg := ("\t`self.director_ax` must be a 1D array with "
                              "3 elements\n"))
            raise ValueError(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')

        cos_theta = np.dot(normalized_vectors, self.director_ax)
        if not np.all((-1 <= cos_theta) & (cos_theta <= 1)):
            log.error(msg := ("\tComputed cos_theta values fall outside "
                              "the expected range of -1 to 1\n"))
            raise ValueError(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')

        order_params: np.ndarray = 0.5 * (3 * cos_theta**2 - 1)
        return np.mean(order_params)


if __name__ == "__main__":
    print('\nThis scripts is run within trajectory_brushes_analysis\n')
