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

from module1_com.trajectory_residue_extractor import GetResidues

from common import logger
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
