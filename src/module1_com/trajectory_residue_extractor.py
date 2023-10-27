#!/usr/bin/env python3
"""
Trajectory Analysis with Multiproccesing module

This script calculates the center of mass (COM) of residues in a mol-
ecular dynamics (MD) trajectory.

The script consists of two main classes:
    1. GetResidues: Responsible for reading and processing the trajec-
       tory file, setting indices for residues based on their types,
       and storing the data for further calculation.
    2. CalculateCom: Performs the center of mass calculation for each
       frame in the trajectory using MPI parallelization.

The main steps in the script are as follows:
    1. Read the trajectory file and topology information using MDAnal-
       ysis.
    2. Preprocess the data to obtain residues in the solution and the
       nanoparticle (NP).
    3. Set indices for each residue based on its type.
    4. Utilize MPI to parallelize the computation and distribute the
       data among processes.
    5. Calculate the center of mass for each frame in the trajectory
       for different types of residues.
    6. Gather the results and combine them into a final array.
"""

import sys
import typing
import common.logger as logger
import common.static_info as stinfo
import common.get_topo as topo
from common.colors_text import TextColor as bcolors
from get_trajectory import GetInfo


class GetResidues:
    """
    Get the residues, based on the description on the doc.
    """

    info_msg: str = 'Messages from GetResidues:\n'  # To log
    # The following will set in _initiate_reading
    top: topo.ReadTop  # Topology file
    trr_info: GetInfo  # All the info from trajectory
    sol_res: dict[int, int]  # Residues with their type as an integer
    np_res: dict[int, int]  # Residues with their type as an integer
    # The number of resdiues are set in _initiate_data
    nr_sol_res: int  # Number of residues in solution (without NP)
    nr_np_res: int  # Number of residues in NP
    max_res: int  # Maxmum index of the residues in solution
    min_res: int  # Minimum index of the residues in solution

    def __init__(self,
                 fname: str,  # Name of the trajectory file
                 log: logger.logging.Logger
                 ) -> None:
        """
        Initialize GetResidues class.

        Args:
            fname (str): Name of the trajectory file.
            log (Union[Logger, None]): Logger object for logging messages.
        """
        self._initiate_reading(fname, log)
        self.sol_res, self. np_res = self._initiate_data(log)
        self.__write_msg(log)

    def _initiate_reading(self,
                          fname: str,  # Name of the trajectory file
                          log: logger.logging.Logger
                          ) -> None:
        """
        Call the other modules and read the files
        Top contains info from forcefield parameters files, the numbers
        of the residues.
        ttr_info contains trajectory read by MDAnalysis.
        Args:
            fname (str): Name of the trajectory file.
            log (Union[Logger, None]): Logger object for logging messages.
        """
        if log is not None:
            self.top = topo.ReadTop(log)
            self.trr_info = GetInfo(fname, log=log)

    def _initiate_data(self,
                       log: logger.logging.Logger
                       ) -> tuple[dict[int, int], dict[int, int]]:
        """
        Initialize data to obtain the center of mass of residues.
        MDAnalysis fails to manage indices, so NP (COR & APT) and
        solution indices are saved and set separately.

        Algorithm:
            Retrieve all residues for nanoparticles and solution, then
            set an index for each based on type defined in static_info.

        Returns:
            Tuple of dictionaries for each set with residue index as
            keys and the residues' type as values.

        """
        sol_res_tmp: dict[str, list[int]]
        np_res_tmp: dict[str, list[int]]
        sol_res_tmp, self.nr_sol_res, self.max_res, self.min_res = \
            self.get_residues(stinfo.np_info["solution_residues"], log)
        np_res_tmp, self.nr_np_res, _, _ = \
            self.get_residues(stinfo.np_info["np_residues"], log)
        sol_res = self.set_residues_index(sol_res_tmp)
        np_res = self.set_residues_index(np_res_tmp)
        return sol_res, np_res

    @staticmethod
    def set_residues_index(all_res_tmp: dict[str, list[int]]  # Name&index
                           ) -> dict[int, int]:
        """
        Set the type of the each residue as an index and number of
        residues
        """
        return \
            {res: stinfo.reidues_id[k] for k, val in all_res_tmp.items()
             for res in val}

    def get_residues(self,
                     res_name: list[str],  # Name of the residues
                     log: logger.logging.Logger
                     ) -> tuple[dict[str, list[int]], int, int, int]:
        """
        Return the dict of the residues in the solution with
        dropping the NP residues
        """
        self.info_msg += '\tGetting the residues:\n'
        all_res_dict: dict[str, list[int]] = {}  # All the residues in solution
        all_res_dict = \
            {k: val for k, val in self.trr_info.residues_indx.items()
             if k in res_name}
        if self.check_similar_items(all_res_dict):
            msg: str = (f'{self.__class__.__name__}:\n'
                        '\tError: There is a douplicate index in the '
                        'residues. Most modifying the code!\n')
            log.error(msg)
            sys.exit(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')

        nr_residues = int(sum(len(lst) for lst in all_res_dict.values()))
        max_res = int(max(max(lst) for lst in all_res_dict.values()))
        min_res = int(min(min(lst) for lst in all_res_dict.values()))
        self.info_msg += \
            (f'\tThe number of the read residues for {res_name} is:\n'
             f'\t\t`{nr_residues}`\n'
             f'\t\tThe max & min of indices are: `{max_res}` and'
             f' `{min_res}`\n')
        return all_res_dict, nr_residues, max_res, min_res

    @staticmethod
    def check_similar_items(dic: dict[str, list[int]]) -> bool:
        """
        Create an empty set to store all unique elements
        """
        unique_elements: set[typing.Any] = set()

        # Iterate over the lists in the dictionary values
        for lst in dic.values():
            # Convert the list to a set to remove duplicates
            unique_set = set(lst)

            # Check if there are any common elements with the previous sets
            if unique_set & unique_elements:
                return True
            # Update the set of unique elements
            unique_elements |= unique_set

        # If no common elements were found, return False
        return False

    def __write_msg(self,
                    log: logger.logging.Logger  # To log
                    ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{GetResidues.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        if log is not None:
            log.info(self.info_msg)


if __name__ == '__main__':
    GetResidues(fname=sys.argv[1],
                log=logger.setup_logger('get_residue.log'))
