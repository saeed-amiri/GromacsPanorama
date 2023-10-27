"""The script is designed to read the trajectory files and extract
essential information from them, including the number of molecules,
frames, and atoms. """

import sys
import json
import typing
import MDAnalysis as mda
import common.logger as logger
import common.my_tools as my_tools
from common.colors_text import TextColor as bcolors


class GetInfo:
    """read trr files and return info from it"""

    info_msg: str = 'Message:\n'  # To log info

    trr: str  # Name of the trajectory file
    gro: str  # Name of the gro file (final step of trai from GROMACS)
    u_traj: mda.Universe  # Trajectory read by MDAnalysis
    residues_indx: dict[str, list[int]]  # Info from the traj and gro
    num_dict: dict[str, typing.Any]  # Info from the traj and gro

    def __init__(self,
                 fname: str,  # Name of the trr file
                 log: logger.logging.Logger  # Name of the log file
                 ) -> None:
        # Check the file
        my_tools.check_file_exist(fname, log)
        self.trr = fname
        self.gro = self.check_gro(log)  # Toopology file
        self.info_msg += f'\tThe trajectory file is : `{self.trr}`\n'
        self.info_msg += f'\tThe gro file is : `{self.gro}`\n'
        self.u_traj = self.read_traj()
        self.residues_indx, self.num_dict = self.__get_info()
        self.__write_msg(log)

    def __get_info(self) -> tuple[dict[str, list[int]], dict[str, typing.Any]]:
        """get all the info from the input files"""
        residues_indx: dict[str, list[int]] = self.__get_residues()
        num_dict: dict[str, typing.Any] = self.__get_nums()
        return residues_indx, num_dict

    def __get_nums(self) -> dict[str, int]:
        """
        Return a dictionary with various numerical information from
        the trajectory.

        The returned dictionary contains:
        - n_atoms: Number of atoms.
        - total_mass: Total mass.
        - n_frames: Number of frames.
        - totaltime: Total time.

        Returns:
        - Dictionary with numerical information.
        """
        num_dict: dict[str, typing.Any] = \
            {
            'n_atoms': int(self.u_traj.atoms.n_atoms),
            'total_mass': self.u_traj.atoms.total_mass,
            'n_frames': int(self.u_traj.trajectory.n_frames),
            'totaltime': int(self.u_traj.trajectory.totaltime),
            'dt': int(self.u_traj.trajectory.dt)
            }
        self.info_msg += '\tInformation in the trajectory file are:\n'
        _json_data = \
            json.dumps({k: str(v) for k, v in num_dict.items()}, indent=8)
        self.info_msg += f'\t{_json_data}\n'
        return num_dict

    def __get_residues(self) -> dict:
        """
        Get the indices of all residues in the trajectory.

        Returns:
        - Dictionary with residue indices.
        """
        residue_indices: dict[str, list[int]] = {}

        # Iterate over the residues
        for residue in self.u_traj.residues:
            residue_name = residue.resname

            # Get the index of the current residue
            residue_index = residue.resnum

            # Add the index to the dictionary
            if residue_name not in residue_indices:
                residue_indices[residue_name] = []

            residue_indices[residue_name].append(residue_index)
        return residue_indices

    def read_traj(self) -> mda.Universe:
        """
        Read the trajectory and topology files.

        Returns:
        - mda.Universe instance representing the trajectory.
        """
        self.info_msg += f'\tTrajectory file `{self.trr}` and topology file:'
        self.info_msg += f' `{self.gro}` are read by MDAnalysis\n'
        return mda.Universe(self.gro, self.trr)

    def check_gro(self,
                  log: logger.logging.Logger  # Name of the log file
                  ) -> str:
        """
        Check if the gro file exists and return its name.

        Returns:
        - Name of the gro file.
        """
        tmp: str = self.trr.split('.')[0]
        gro_file: str = f'{tmp}.gro'
        my_tools.check_file_exist(gro_file, log)
        return gro_file

    def __write_msg(self,
                    log: logger.logging.Logger,  # To log info in it
                    ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{GetInfo.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    trr = GetInfo(sys.argv[1], log=logger.setup_logger('get_traj_log'))
