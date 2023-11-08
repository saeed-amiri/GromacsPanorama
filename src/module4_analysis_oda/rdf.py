"""
calculating rdf from nanoparticle segment center.
The nanoparticle com for this situation (x, y, z_interface)
The z valu is only used for droping the oda which are not at interface
"""

import sys
import numpy as np
import pandas as pd

from common import logger
from common import xvg_to_dataframe
from common import my_tools
from common.colors_text import TextColor as bcolors


class RdfClculation:
    """calculate radial distribution function"""

    inf_msg: str = '\tMessage from RdfCalculation:\n'

    def __init__(self,
                 amino_arr: np.ndarray,  # amino head com of the oda
                 box_dims: dict[str, float],  # Dimension of the Box
                 log: logger.logging.Logger
                 ) -> None:
        self.np_com: np.ndarray = self._get_np_gmx(log)
        self.initiate(amino_arr, box_dims, log)

    def initiate(self,
                 amino_arr: np.ndarray,  # amino head com of the oda
                 box_dims: dict[str, float],  # Dimension of the Box
                 log: logger.logging.Logger
                 ) -> None:
        """initiate the calculations"""
        self.get_contact_info(log)

    def get_contact_info(self,
                         log: logger.logging.Logger
                         ) -> pd.DataFrame:
        """
        read the dataframe made by aqua analysing named "contact.info"
        """
        my_tools.check_file_exist(fname := 'contact.info', log)
        return pd.read_csv(fname, sep=' ')

    def _get_np_gmx(self,
                    log: logger.logging.Logger
                    ) -> np.ndarray:
        """geeting the COM of the nanoparticles from gmx traj
        the file name is coord.xvg
        convert AA to nm
        """
        xvg_df: pd.Dataframe = \
            xvg_to_dataframe.XvgParser('coord.xvg', log).xvg_df
        xvg_arr: np.ndarray = xvg_df.iloc[:, 1:4].to_numpy()
        return xvg_arr * 10


if __name__ == '__main__':
    pass
