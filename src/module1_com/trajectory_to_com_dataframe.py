#! /home/saeed/.conda/envs/analysing/bin/python
"""
Using multiprocessing to get the COM of the residues!
"""

import sys
import pickle
import multiprocessing
from datetime import datetime
import numpy as np

from module1_com.trajectory_residue_extractor import GetResidues

from common import logger
from common import my_tools
from common import static_info as stinfo
from common.cpuconfig import ConfigCpuNr
from common.colors_text import TextColor as bcolors



class CalculateCom:
    """
    Calculating the center of mass (COM) of the residues.

    Attributes:
        info_msg (str): Messages from CalculateCom for logging.
        get_residues (GetResidues or None): Objects of the GetResidues
        class.

    Args:
        fname (str): Name of the trajectory files.
        log (logging.Logger or None): Logger for logging messages
        (optional).

    Note:
        The class calculates the center of mass for each frame in the
        trajectory
        and stores the data in a 2D array with columns representing
        time,
        center of mass of the nanoparticle, and center of mass for each
        residue.

        The array layout is as follows:
        | time | NP_x | NP_y | NP_z | res1_x | res1_y | res1_z | ... |
         resN_x | resN_y | resN_z | odn1_x| odn1_y| odn1_z| ... odnN_z|

    Example:
        CalculateCom(fname="trajectory.trr", log=my_logger)
    """

    info_msg: str = 'Messages from CalculateCom:\n'  # To log
    get_residues: GetResidues  # Type of the info
    n_frames: int  # Number of the frame in the trajectory
    n_cores: int  # Number of the cores which system will run with it

    def __init__(self,
                 fname: str,  # Name of the trajectory files
                 log: logger.logging.Logger
                 ) -> None:
        """
        Initialize CalculateCom and perform calculations.

        Args:
            fname (str): Name of the trajectory files.
            log (logging.Logger or None, optional): Logger for logging
            messages.
        """
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

    def _initiate_cpu(self,
                      log: logger.logging.Logger
                      ) -> None:
        """
        Return the number of core for run based on the data and the machine
        """
        cpu_info = ConfigCpuNr(log)
        self.n_cores: int = min(cpu_info.cores_nr, self.n_frames)
        self.info_msg += f'\tThe numbers of using cores: {self.n_cores}\n'

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
                               self.get_residues.nr_sol_res,
                               odn_nr := self.get_residues.top.mols_num['ODN'])
        _, com_col = np.shape(com_arr)
        amino_odn_index: dict[int, int] = \
            self.set_amino_odn_index(com_arr, sol_residues['ODN'])

        args = \
            [(chunk, u_traj, np_res_ind, com_col, sol_residues,
              amino_odn_index, residues_index_dict) for chunk in chunk_tsteps]
        with multiprocessing.Pool(processes=self.n_cores) as pool:
            results = pool.starmap(self.process_trj, args)
        # Merge the results
        recvdata: np.ndarray = np.vstack(results)
        tmp_arr: np.ndarray = self.set_residue_ind(
            com_arr, recvdata, residues_index_dict, odn_nr)
        com_arr = self.set_residue_type(tmp_arr, sol_residues).copy()

        self.pickle_arr(com_arr, log)
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(current_time)

    def pickle_arr(self,
                   com_arr: np.ndarray,  # Array of the center of mass
                   log: logger.logging.Logger  # Name of the log file
                   ) -> None:
        """
        check the if the previus similar file exsitance the pickle
        data into a file
        """
        fname: str  # Name of the file to pickle to
        fname = my_tools.check_file_reanme(stinfo.files['com_pickle'], log)
        with open(fname, 'wb') as f_arr:
            pickle.dump(com_arr, f_arr)

    @staticmethod
    def set_residue_ind(com_arr: np.ndarray,  # The final array
                        recvdata: np.ndarray,  # Info about time frames
                        residues_index_dict: dict[int, int],
                        odn_nr: int  # Number of ODA in the system
                        ) -> np.ndarray:
        """
        Set the original residues' indices to the com_arr[-2]
        Set the type of residues' indices to the com_arr[-1]
        """
        # Copy data to the final array
        for row in recvdata:
            tstep = int(row[0])
            com_arr[tstep] = row.copy()

        # setting the index of NP and ODA Amino heads
        com_arr[-2, 1:4] = [-1, -1, -1]
        com_arr[-2, -odn_nr:] = np.arange(-1, -(odn_nr+1), -1)
        for res_ind, col_in_arr in residues_index_dict.items():
            ind = int(res_ind)
            com_arr[-2][col_in_arr:col_in_arr+3] = \
                np.array([ind, ind, ind]).copy()
        return com_arr

    def set_residue_type(self,
                         com_arr: np.ndarray,  # Updated array to set the type
                         sol_residues: dict[str, list[int]]
                         ) -> np.ndarray:
        """
        I need to assign types to all residues and place them in the
        final row of the array.
        Args:
            com_arr: Filled array with information about COM and real
                     index
            sol_residues: key: Name of the residue
                          Value: Residues belongs to the Key
        Return:
            Updated com_arr with type of each residue in the row below
            them.
        """
        reverse_mapping = {}
        for key, value_list in sol_residues.items():
            for num in value_list:
                reverse_mapping[num] = key
        for ind in range(com_arr.shape[1]):
            try:
                res_ind = int(com_arr[-2, ind])
                res_name = reverse_mapping.get(res_ind)
                com_arr[-1, ind] = stinfo.reidues_id[res_name]
            except KeyError:
                pass
        return com_arr

    def process_trj(self,
                    tsteps: np.ndarray,  # Frames' indices
                    u_traj,  # Trajectory
                    np_res_ind: list[int],  # NP residue id
                    com_col: int,  # Number of the columns
                    sol_residues: dict[str, list[int]],
                    amino_odn_index: dict[int, int],
                    residues_index_dict: dict[int, int]
                    ) -> np.ndarray:
        """Get atoms in the timestep"""
        chunk_size: int = len(tsteps)
        my_data: np.ndarray = np.empty((chunk_size, com_col))
        for row, i in enumerate(tsteps):
            ind = int(i)
            frame = u_traj.trajectory[ind]
            atoms_position: np.ndarray = frame.positions
            for k, val in sol_residues.items():
                print(f'\ttimestep {ind}  -> getting residues: {k}')
                for item in val:
                    com = self.get_com_all(atoms_position, item)
                    if com is None:
                        continue  # Skip if com is None

                    wrap_com = self.wrap_position(com, frame.dimensions)

                    element = residues_index_dict[item]
                    my_data[row][element:element+3] = wrap_com

                    if k == 'ODN':
                        if amino_odn_index is not None:
                            amin = self.get_odn_amino_com(atoms_position,
                                                          item)
                            amino_ind: int = amino_odn_index[item]
                            wrap_amin = \
                                self.wrap_position(amin, frame.dimensions)
                            my_data[row][amino_ind:amino_ind+3] = wrap_amin
            np_com = self.get_np_com(atoms_position, np_res_ind, u_traj)
            # Update my_data with ind and np_com values
            my_data[row, 0] = ind
            my_data[row, 1:4] = np_com
        return my_data

    def get_com_all(self,
                    all_atoms: np.ndarray,  # All the atoms position
                    ind: int  # Index of the residue
                    ) -> np.ndarray:
        """
        return all the residues' center of mass

        Algorithem:
            Atoms belong to the residue `ind` is save to:
                i_residue: <class 'MDAnalysis.core.groups.AtomGroup'>
                e.g.:
                <AtomGroup [<Atom 1: OH2 of type O of resname SOL,
                             resid 1803 and segid SYSTEM>,
                            <Atom 2: H1 of type H of resname SOL,
                             resid 1803 and segid SYSTEM>,
                            <Atom 3: H2 of type H of resname SOL,
                             resid 1803 and segid SYSTEM>]>

            Indicis of these atoms are:
                atom_indices: <class 'numpy.ndarray'>
                e.g.: [0 1 2]

            Positions of all the atoms in the residues are:
                atom_positions: <class 'numpy.ndarray'>
                e.g.:
                     [[186.44977   20.240488  14.459618]
                     [185.91147   20.554382  13.733033]
                     [186.8721    21.027851  14.803037]]

            Masses of these atoms are also found:
                atom_masses: <class 'numpy.ndarray'>
                e.g.: [15.999  1.008  1.008]
        Return:
            By getting weightesd average of these atoms the center of
            mass is returend
        """
        i_residue = \
            self.get_residues.trr_info.u_traj.select_atoms(f'resnum {ind}')
        atom_indices = i_residue.indices
        atom_positions = all_atoms[atom_indices]
        atom_masses = i_residue.masses
        return np.average(atom_positions, weights=atom_masses, axis=0)

    def get_odn_amino_com(self,
                          all_atoms: np.ndarray,  # All the atoms poistion
                          ind: int  # Index of the residue
                          ) -> np.ndarray:
        """
        Calculating the center of mass of the amino group in the ODN
        See the doc of get_com_all
        Here, only atoms in NH2 and HC (3 H atoms)
        i_atoms:
            <Atom 523033: NH2 of type N of resname ODN, resid 176274
             and segid SYSTEM>,
            <Atom 523034: HC1 of type H of resname ODN, resid 176274
             and segid SYSTEM>,
            <Atom 523035: HC2 of type H of resname ODN, resid 176274
             and segid SYSTEM>,
            <Atom 523036: HC3 of type H of resname ODN, resid 176274
             and segid SYSTEM>
        """
        i_residue = \
            self.get_residues.trr_info.u_traj.select_atoms(f'resnum {ind}')
        i_atoms = i_residue.select_atoms('name NH2 HC*')
        atom_indices = i_atoms.indices
        atom_positions = all_atoms[atom_indices]
        atom_masses = i_atoms.masses
        return np.average(atom_positions, weights=atom_masses, axis=0)

    def get_np_com(self,
                   all_atoms: np.ndarray,  # Atoms positions
                   np_res_ind: list[int],  # Residues in NP
                   u_traj
                   ) -> np.ndarray:
        """get the COM for each time step"""
        i_com: list[np.ndarray] = []  # Arrays contains center of masses
        total_mass: float = 0  # Total mass of each residue in the NP
        if np_res_ind:
            for i in np_res_ind:
                com: np.ndarray  # Center of mass of the residue i
                tmp_mass: float  # Mass of the residue
                com, tmp_mass = self.get_np_com_tstep(i, all_atoms, u_traj)
                total_mass += tmp_mass
                i_com.append(com)
                step_com = np.vstack(i_com)
            return np.sum(step_com, axis=0) / total_mass
        return np.array([0.0, 0.0, 0.0])

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

    def get_np_com_tstep(self,
                         res_ind: int,  # index of the residue
                         all_atoms: np.ndarray,  # Atoms positions
                         u_traj
                         ) -> tuple[np.ndarray, float]:
        """
        calculate the center of mass of each time step for NP
        """
        i_residue = \
            u_traj.select_atoms(f'resnum {res_ind}')
        atom_indices = i_residue.indices
        atom_positions = all_atoms[atom_indices]
        atom_masses = i_residue.masses
        tmp_mass = np.sum(atom_masses)
        com = np.average(atom_positions, weights=atom_masses,
                         axis=0) * tmp_mass
        return com, tmp_mass

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
                      nr_residues: int,  # Numbers of residues' indices
                      n_oda: int  # Number of ODA in the system
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
        timeframe + NP_com + nr_residues:  xyz + n_oda * xyz
             1    +   3    +  nr_residues * 3  +  n_oda * 3

        """

        rows: int = n_frames + 2  # Number of rows, 2 for name and index of res
        columns: int = 1 + 3 + nr_residues * 3 + n_oda * 3
        return np.zeros((rows, columns))

    @staticmethod
    def set_amino_odn_index(com_arr,  # The array to set all com in it
                            odn_residues: list[int]  # Indices of ODN residues
                            ) -> dict[int, int]:
        """
        Set (or find!) the indices for the COM of ODN amino group in
        the com_arr
        In the alocation of com_arr, nr_odn columns are added for the
        com of the amino groups of the ODN. Since the COM of all ODN
        are also set in the com_arr the extra indices should carefuly
        be setted.
        The indices of the ODN could be not sequal!
        Returns:
            key: 0 - nr_odn
            value: the column in the com_arr (the final arry)
        """
        sorted_odn_residues: list[int] = sorted(odn_residues, reverse=True)
        last_column: int = np.shape(com_arr)[1]
        odn_amino_indices: dict[int, int] = {}
        for i, odn in enumerate(sorted_odn_residues):
            odn_amino_indices[odn] = int(last_column - (i+1) * 3)
        return odn_amino_indices

    @staticmethod
    def wrap_position(pos: np.ndarray,  # The center of mass
                      vec: np.ndarray  # Box vectors
                      ) -> np.ndarray:
        """
        Wraped the position to the box for unwraped trr.
        """
        if pos is not None:
            for i in range(3):
                pos[i] -= np.floor(pos[i]/vec[i])*vec[i]
            return pos
        return np.zeros(3)

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{CalculateCom.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    SIZE = 96
    CalculateCom(fname=sys.argv[1], log=logger.setup_logger('get_frames.log'))
