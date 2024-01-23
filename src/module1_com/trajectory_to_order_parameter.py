"""
Calculating Order Parameter for All Residues in a System

This script utilizes `trajectory_residue_extractor.py` for determining
the order parameter of every residue in a given system. It involves
reading the system's trajectory data, calculating the order parameter
for each residue, and recording these values for subsequent analysis.
This analysis includes examining how the order parameter of residues
varies over time and in different spatial directions.

Key Points:

1. Utilization of Unwrapped Trajectory:
   Similar to the approach in 'com_pickle', this script requires the
   use of an unwrapped trajectory to ensure accurate calculations.

2. Data to Save:
   Simply saving the index of residues is insufficient for later
   identification of their spatial locations. Therefore, it's crucial
   to store the 3D coordinates (XYZ) along with each residue's order
   parameter. However, recalculating the center of mass (COM) for each
   residue would be computationally expensive.

3. Leveraging com_pickle:
   To optimize the process, the script will utilize 'com_pickle',
   which already contains the COM and index of each residue in the
   system. This allows for the direct calculation of the order
   parameter for each residue based on their respective residue index,
   without the need for additional COM computations.

4. Comprehensive Order Parameter Calculation:
   Rather than limiting the calculation to the order parameter in the
   z-direction, this script will compute the full tensor of the order
   parameter, providing a more detailed and comprehensive understanding
   of the residues' orientation.

This methodological approach ensures a balance between computational
efficiency and the thoroughness of the analysis.
Opt. by ChatGpt4
Saeed
23Jan2024
"""

import sys
from datetime import datetime
from dataclasses import dataclass

import numpy as np
from module1_com.trajectory_residue_extractor import GetResidues

from common import logger
from common import static_info as stinfo
from common.cpuconfig import ConfigCpuNr
from common.colors_text import TextColor as bcolors


@dataclass
class FileConfigur:
    """to set input files"""


@dataclass
class ParameterConfigur:
    """set the parameters for the computations"""


@dataclass
class AllConfigur(FileConfigur, ParameterConfigur):
    """set all the configurations"""


class ComputeOrderParameter:
    """compute order parameters for each residue"""

    info_msg: str = 'Message from ComputeOrderParameter:\n'

    get_residues: GetResidues  # Type of the info

    def __init__(self,
                 fname: str,  # Name of the trajectory file
                 log: logger.logging.Logger
                 ) -> None:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(current_time)
        self._initiate_data(fname, log)
        self._initiate_cpu(log)
        self._initiate_calc(log)
        self._write_msg(log)

    def _initiate_data(self,
                       fname: str,  # Name of the trajectory files
                       log: logger.logging.Logger
                       ) -> None:
        """
        This function Call GetResidues class and get the data from it.
        """
        self.get_residues = GetResidues(fname, log)
        self.n_frames = self.get_residues.trr_info.num_dict['n_frames']

    def _initiate_calc(self,
                       log: logger.logging.Logger
                       ) -> None:
        """
        First divide the list, than brodcast between processes.
        Get the lists the list contains timesteps.
        The number of sublist is equal to number of cores, than each
        sublist will be send to one core.
        The total trajectory will br brodcast to all the processors

        Args:
            None

        Returns:
            None

        Notes:
            - The `n_frames` should be equal or bigger than n_process,
              otherwise it will reduced to n_frames
            - u_traj: <class 'MDAnalysis.coordinates.TRR.TRRReader'>
            - chunk_tsteps: list[np.ndarray]]
        """
        data: np.ndarray = np.arange(self.n_frames)
        chunk_tsteps: list[np.ndarray] = self.get_chunk_lists(data)
        np_res_ind: list[int] = self.get_np_residues()
        sol_residues: dict[str, list[int]] = \
            self.get_solution_residues(stinfo.np_info['solution_residues'])
        residues_index_dict: dict[int, int] = \
            self.mk_residues_dict(sol_residues)
        u_traj = self.get_residues.trr_info.u_traj
        com_arr: np.ndarray = \
            self.mk_allocation(self.n_frames,
                               self.get_residues.nr_sol_res)
        odn_nr: int = self.get_residues.top.mols_num['ODN']

    def get_chunk_lists(self,
                        data: np.ndarray  # Range of the time steps
                        ) -> list[np.ndarray]:
        """prepare chunk_tstep based on the numbers of frames"""
        # determine the size of each sub-task
        ave, res = divmod(data.size, self.n_cores)
        counts: list[int]  # Length of each array in the list
        counts = [ave + 1 if p < res else ave for p in range(self.n_cores)]

        # determine the starting and ending indices of each sub-task
        starts: list[int]  # Start of each list of ranges
        ends: list[int]  # Ends of each list of ranges
        starts = [sum(counts[: p]) for p in range(self.n_cores)]
        ends = [sum(counts[: p+1]) for p in range(self.n_cores)]

        # converts data into a list of arrays
        chunk_tstep = [data[starts[p]: ends[p]].astype(np.int32)
                       for p in range(self.n_cores)]
        return chunk_tstep

    def get_np_residues(self) -> list[int]:
        """
        return list of the integer of the residues in the NP
        """
        np_res_ind: list[int] = []  # All the index in the NP
        try:
            for item in stinfo.np_info['np_residues']:
                np_res_ind.extend(
                    self.get_residues.trr_info.residues_indx[item])
        except KeyError:
            pass
        return np_res_ind

    def get_solution_residues(self,
                              res_group: list[str]
                              ) -> dict[str, list[int]]:
        """
        Return the dict of the residues in the solution with
        dropping the NP residues
        """
        sol_dict: dict[str, list[int]] = {}  # All the residues in solution
        for k, val in self.get_residues.trr_info.residues_indx.items():
            if k in res_group:
                sol_dict[k] = val
        return sol_dict

    @staticmethod
    def mk_residues_dict(sol_residues: dict[str, list[int]]
                         ) -> dict[int, int]:
        """
        Make a dict for indexing all the residues. Not always residues
        indexed from zero and/or are numberd sequently.

        Args:
            sol_residues of index for each residue in the solution
        Return:
            new indexing for each residues
            Since in the recived method of this return, the result could
            be None, the type is Union
            Key: The residue index in the main data (traj from MDAnalysis)
            Value: The new orderd indices
        Notes:
            Since we already have 4 elements before the these resideus,
            numbering will start from 4
        """
        all_residues: list[int] = \
            [item for sublist in sol_residues.values() for item in sublist]
        sorted_residues: list[int] = sorted(all_residues)
        residues_index_dict: dict[int, int] = {}
        if residues_index_dict is not None:
            for i, res in enumerate(sorted_residues):
                residues_index_dict[res] = i * 3 + 4
        return residues_index_dict

    @staticmethod
    def mk_allocation(n_frames: int,  # Number of frames
                      nr_residues: int  # Numbers of residues' indices
                      ) -> np.ndarray:
        """
        Allocate arrays for saving all the info.

        Parameters:
        - sol_residues: Residues in solution.

        Returns:
        - Initialized array.
            Columns are as follow:
            each atom has xyz, the center of mass also has xyx, and one
            for labeling the name of the residues, for example SOL will
            be 1

        The indexing method is updated, now every index getting a defiend
        index which is started from 4. See: mk_residues_dict
        number of row will be:
        number of frames + 2
        The extra rows are for the type of the residue at -1 and the
        orginal ids of the residues in the traj file

        number of the columns:
        n_residues: number of the residues in solution, without residues
        in NP
        n_ODA: number oda residues
        NP_com: Center of mass of the nanoparticle
        than:
        timeframe + NP_com + nr_residues:  xyz
             1    +   3    +  nr_residues * 3

        """

        rows: int = n_frames + 2  # Number of rows, 2 for name and index of res
        columns: int = 1 + 3 + nr_residues * 3
        return np.zeros((rows, columns))

    def _initiate_cpu(self,
                      log: logger.logging.Logger
                      ) -> None:
        """
        Return the number of core for run based on the data and the machine
        """
        cpu_info = ConfigCpuNr(log)
        self.n_cores: int = min(cpu_info.cores_nr, self.n_frames)
        self.info_msg += f'\tThe numbers of using cores: {self.n_cores}\n'

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{ComputeOrderParameter.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    ComputeOrderParameter(sys.argv[1],
                          log=logger.setup_logger("all_order_parameter.log"))
