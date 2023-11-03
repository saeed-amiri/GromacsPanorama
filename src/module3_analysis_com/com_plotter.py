"""
get an array and an index and plot xy, xz, yz of the
com.
the array is in the format of the com:

    x0 y0 z0 x1 y1 z1 ...

"""

import numpy as np
from common import logger


class ComPlotter:
    """plot the center of mass"""

    def __init__(self,
                 com_arr: np.ndarray,  # The array to plot
                 index: int,  # Time frame to plot
                 log: logger.logging.Logger,
                 to_png: bool = False,  # If want to save as png
                 out_file: str = 'com.png',  # name of the png file, if to_png
                 to_xyz: bool = False,  # If want to save the traj to xyz file
                 xyz_file: str = 'com.xyz'  # name of the png file, if to_xyz
                 ) -> None:
        xyz_zrr: np.ndarray = self._parse_arr(com_arr, index)

    def _parse_arr(self,
                   com_arr: np.ndarray,  # The array to plot
                   index: int  # Time frame to plot
                   ) -> np.ndarray:
        """return the data for plotting in one array"""
        return com_arr[index, :].reshape(-1, 3)


if __name__ == '__mian__':
    pass
