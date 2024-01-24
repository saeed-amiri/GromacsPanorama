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
import typing
import multiprocessing
from datetime import datetime
from dataclasses import dataclass, field

import pickle
import numpy as np
from module1_com.trajectory_residue_extractor import GetResidues

from common import logger, my_tools
from common import static_info as stinfo
from common.cpuconfig import ConfigCpuNr
from common.colors_text import TextColor as bcolors


@dataclass
class FileConfig:
    """to set input files"""
    pickle_fout: str = 'order_parameter_pickle'


@dataclass
class OrderParameterConfig:
    """set the parameters for the computations"""
    director_z: np.ndarray = np.array([0, 0, 1])
    director_y: np.ndarray = np.array([0, 1, 0])
    director_x: np.ndarray = np.array([1, 0, 0])


@dataclass
class OdnConfig:
    """parameters and parameters of ODA (in the system is named ODN)"""
    head: str = 'CT3'  # Name of the head C atom
    tail: str = 'NH2'  # Name of the tail N atom


@dataclass
class DecaneConfig:
    """parameters and parameters of Decane"""
    head: str = 'C1'  # Name of the head C atom
    tail: str = 'C9'  # Name of the tail C atom


@dataclass
class WaterConfig:
    """parameters and parameters of Water (SOL in the system)"""
    head: str = 'OH2'  # Name of the head C atom
    tail_1: str = 'H1'  # Name of the tail C atom
    tail_2: str = 'H2'  # Name of the tail C atom


@dataclass
class AllConfigs(FileConfig, OrderParameterConfig):
    """set all the configurations"""
    odn_config: OdnConfig = field(default_factory=OdnConfig)
    decane_config: DecaneConfig = field(default_factory=DecaneConfig)
    sol_config: WaterConfig = field(default_factory=WaterConfig)


class ComputeOrderParameter:
    """compute order parameters for each residue"""

    info_msg: str = 'Message from ComputeOrderParameter:\n'

    configs: AllConfigs
    get_residues: GetResidues  # Info from the trajectory file

    def __init__(self,
                 fname: str,  # Name of the trajectory file
                 log: logger.logging.Logger,
                 configs: AllConfigs = AllConfigs()
                 ) -> None:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(current_time)
        self.configs = configs
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
        # np_res_ind: list[int] = self.get_np_residues()
        sol_residues: dict[str, list[int]] = \
            self.get_solution_residues(stinfo.np_info['solution_residues'])
        residues_index_dict: dict[int, int] = \
            self.mk_residues_dict(sol_residues)
        u_traj = self.get_residues.trr_info.u_traj
        order_parameters_arr: np.ndarray = \
            self.mk_allocation(self.n_frames,
                               self.get_residues.nr_sol_res)
        _, com_col = np.shape(order_parameters_arr)
        args = \
            [(chunk[:1], u_traj, com_col, sol_residues,
              residues_index_dict, log) for chunk in chunk_tsteps]
        with multiprocessing.Pool(processes=self.n_cores) as pool:
            results = pool.starmap(self.process_trj, args)
        # Merge the results
        recvdata: np.ndarray = np.vstack(results)
        tmp_arr: np.ndarray = self.set_residue_ind(
            order_parameters_arr, recvdata, residues_index_dict)
        order_parameters_arr = \
            self.set_residue_type(tmp_arr, sol_residues).copy()
        self.pickle_arr(order_parameters_arr, log)

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(current_time)
        self.info_msg += f'\n\tEnd at: {current_time}\n'

    def pickle_arr(self,
                   order_parameters_arr: np.ndarray,  # Array of pickle
                   log: logger.logging.Logger  # Name of the log file
                   ) -> None:
        """
        check the if the previus similar file exsitance the pickle
        data into a file
        """
        fname: str  # Name of the file to pickle to
        fname = my_tools.check_file_reanme(
            fout := self.configs.pickle_fout, log)
        self.info_msg += f'\tThe pickle out file is saved as `{fout}`\n'
        with open(fname, 'wb') as f_arr:
            pickle.dump(order_parameters_arr, f_arr)

    def process_trj(self,
                    tsteps: np.ndarray,  # Frames' indices
                    u_traj,  # Trajectory
                    com_col: int,  # Number of the columns
                    sol_residues: dict[str, list[int]],
                    residues_index_dict: dict[int, int],
                    log: logger.logging.Logger
                    ) -> np.ndarray:
        """Get atoms in the timestep"""
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals

        # head_pos, tail_pos are two np.ndarray to define the director
        # and mostly makes sense for the ODA atoms
        # and for ODA head_pos is C, since it suposed to be in higher
        # position then the tail which is N of the amino group

        config: typing.Union[OdnConfig, DecaneConfig]
        chunk_size: int = len(tsteps)
        my_data: np.ndarray = np.empty((chunk_size, com_col))
        for row, i in enumerate(tsteps):
            ind = int(i)
            frame = u_traj.trajectory[ind]
            atoms_position: np.ndarray = frame.positions
            for k, val in sol_residues.items():
                print(f'\ttimestep {ind}  -> getting residues: {k}')
                for item in val:
                    element = residues_index_dict[item]
                    single_flag: bool = False  # If the residue is one atom
                    if k in ('D10', 'ODN'):
                        if k == 'D10':
                            config = self.configs.decane_config
                        elif k == 'ODN':
                            config = self.configs.odn_config
                        head_pos, tail_pos = \
                            self.get_terminal_atoms(
                                atoms_position, item, config)
                    elif k == 'SOL':
                        head_pos, tail_pos = self.get_water_terminal_atoms(
                            atoms_position, item, self.configs.sol_config)
                    else:
                        single_flag = True
                        # head_pos = tail_pos = np.zeros((3,))
                    if not single_flag:
                        order_parameters = \
                            self.compute_order_parameter(head_pos,
                                                         tail_pos,
                                                         log)
                    else:
                        order_parameters = np.zeros((3,))
                    my_data[row][element:element+3] = order_parameters
            my_data[row, 0] = ind
            my_data[row, 1:4] = np.zeros((3,))

        return my_data

    def compute_order_parameter(self,
                                head_pos: np.ndarray,
                                tail_pos: np.ndarray,
                                log: logger.logging.Logger
                                ) -> np.ndarray:
        """compute the order parameter for given atoms"""
        head_tail_vec: np.ndarray = head_pos - tail_pos
        try:
            # Normalizing vectors
            norms = np.linalg.norm(head_tail_vec, axis=1)
            if np.any(norms == 0):
                log.error(
                    msg := "Zero length vector encountered in normalization.")
                raise ValueError(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')

            normalized_vectors = head_tail_vec / norms[:, np.newaxis]
            # Calculating dot products for x, y, z directions
            cos_theta = {
                'z': np.dot(normalized_vectors, self.configs.director_z),
                'y': np.dot(normalized_vectors, self.configs.director_y),
                'x': np.dot(normalized_vectors, self.configs.director_x)
            }

            # Calculating order parameters for x, y, z directions
            order_params = np.array([
                0.5 * (3 * cos_theta_value**2 - 1) for cos_theta_value
                in cos_theta.values()])
            return order_params.flatten()

        except ValueError as err:
            log.error(
                f"\tThere is a problem in getting normalized vector: {err}")
            sys.exit(f'{bcolors.FAIL}{err}{bcolors.ENDC}')

    def get_terminal_atoms(self,
                           all_atoms: np.ndarray,  # All the atoms pos
                           ind: int,  # Index of the residue,
                           config: typing.Union[OdnConfig, DecaneConfig]
                           ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculating the order parameter of the ODN

        Here, only atoms in NH2 and and first C on the chain (oppoiste
        side of the amino group) are used
        i_atoms:
            <Atom 523033: NH2 of type N of resname ODN, resid 176274
             and segid SYSTEM>,
             and segid SYSTEM>
        """
        i_residue = \
            self.get_residues.trr_info.u_traj.select_atoms(f'resnum {ind}')
        tail_atom = \
            i_residue.select_atoms(f'name {config.tail}')
        head_atom = \
            i_residue.select_atoms(f'name {config.head}')
        tail_positions: np.ndarray = all_atoms[tail_atom.indices]
        head_positions: np.ndarray = all_atoms[head_atom.indices]
        return tail_positions, head_positions

    def get_water_terminal_atoms(self,
                                 all_atoms: np.ndarray,
                                 ind: int,
                                 config: WaterConfig
                                 ) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieve terminal atoms for a specified water molecule.

        This method considers water molecules as fixed residues,
        i.e., the H-O bonds and H-O-H angles are constant. It
        identifies the oxygen atom as the 'head' and calculates the
        'tail' as the center of mass of the two hydrogen atoms.
        The terms 'head' and 'tail' are used for consistency, although
        they are less conventional for symmetric molecules like water.

        The 'head' is the position of the oxygen atom, and the 'tail'
        is computed as the average position of the two hydrogen atoms,
        representing the center of mass.

        Args:
            all_atoms (np.ndarray): Array containing positions of all
            atoms in the system.
            ind (int): Index of the specific water molecule.
            config (WaterConfig): Configuration parameters for water
                molecules, including atom names for the 'head' (oxygen)
                and 'tails' (hydrogens).

        Returns:
            np.ndarray: A tuple containing two numpy arrays. The first
                array represents the position of the 'head' (oxygen
                atom), and the second array is the calculated position
                of the 'tail' (center of mass of hydrogens).
        """
        i_residue = \
            self.get_residues.trr_info.u_traj.select_atoms(f'resnum {ind}')
        head_atom = \
            i_residue.select_atoms(f'name {config.head}')
        tail_atom_1 = \
            i_residue.select_atoms(f'name {config.tail_1}')
        tail_atom_2 = \
            i_residue.select_atoms(f'name {config.tail_2}')
        head_positions: np.ndarray = all_atoms[head_atom.indices]
        tail_positions_1: np.ndarray = all_atoms[tail_atom_1.indices]
        tail_positions_2: np.ndarray = all_atoms[tail_atom_2.indices]
        tail_positions = (tail_positions_1 + tail_positions_2) / 2
        return tail_positions, head_positions

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

    def set_residue_type(self,
                         order_parameter_arr: np.ndarray,  # set type
                         sol_residues: dict[str, list[int]]
                         ) -> np.ndarray:
        """
        I need to assign types to all residues and place them in the
        final row of the array.
        Args:
            order_parameter_arr: Filled array with information about res
                    and real index
            sol_residues: key: Name of the residue
                          Value: Residues belongs to the Key
        Return:
            Updated order_parameter_arr with type of each residue in
                the row belowthem.
        """
        reverse_mapping = {}
        for key, value_list in sol_residues.items():
            for num in value_list:
                reverse_mapping[num] = key
        for ind in range(order_parameter_arr.shape[1]):
            try:
                res_ind = int(order_parameter_arr[-2, ind])
                res_name = reverse_mapping.get(res_ind)
                order_parameter_arr[-1, ind] = stinfo.reidues_id[res_name]
            except KeyError:
                pass
        return order_parameter_arr

    @staticmethod
    def set_residue_ind(order_parameter_arr: np.ndarray,  # The final array
                        recvdata: np.ndarray,  # Info about time frames
                        residues_index_dict: dict[int, int]
                        ) -> np.ndarray:
        """
        Set the original residues' indices to the order_parameter_arr[-2]
        Set the type of residues' indices to the order_parameter_arr[-1]
        """
        # Copy data to the final array
        for row in recvdata:
            tstep = int(row[0])
            order_parameter_arr[tstep] = row.copy()

        # setting the index of NP and ODA Amino heads
        order_parameter_arr[-2, 1:4] = [-1, -1, -1]
        for res_ind, col_in_arr in residues_index_dict.items():
            ind = int(res_ind)
            order_parameter_arr[-2][col_in_arr:col_in_arr+3] = \
                np.array([ind, ind, ind]).copy()
        return order_parameter_arr

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
