"""
Read the com file wrote by trajectory_to_com_dataframe
read the com and return split them into np.ndarray into dict
From legecy script:
After running get_frames.py, the script reads the pickle file and
stores its contents in an array. This data will then be used for both
plotting and analyzing purposes.
The file has the following format:
elements:
The center of the mass of all residue is calculated, then it is wrapped
back into the box, and the center of mass of the NP at that time is
subtracted from it.
Number of columns is 3(N+1) + 1
    N: number of the residues,
    and one for the center of mass of the NP at the time
    and last row for saving the label of each residue
date: NN
Update:
  Reading updated com_pickle from get_frame_mpi.py Jul 21 2023
    The array layout is as follows:
        | time | NP_x | NP_y | NP_z | res1_x | res1_y | res1_z | ... |
         resN_x | resN_y | resN_z | odn1_x| odn1_y| odn1_z| ... odnN_z|
    number of row is:
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
    The data can be split based on the index in the last row. The index
    of the ODN heads is either 0 or the index of ODN defined in the
    stinfo. If they are zero, it is straightforward forward. If not,
    the data of the ODN should be split in half.
"""

import sys
from common import logger
from common.com_file_parser import GetCom
from module3_analysis_aqua.aquatic_analysis import AnalysisAqua


class ComAnalysis:
    """call all the other scripts and analyze them"""
    def __init__(self,
                 log: logger.logging.Logger,
                 fname: str  # Name of the com file
                 ) -> None:
        parsed_com = GetCom(fname)
        self.initiate(parsed_com, log)

    def initiate(self,
                 parsed_com: "GetCom",
                 log: logger.logging.Logger
                 ) -> None:
        """first analyze water, to get the interface and other
        properties"""
        AnalysisAqua(parsed_com, log)


if __name__ == "__main__":
    ComAnalysis(fname=sys.argv[1], log=logger.setup_logger('com_analysis.log'))
