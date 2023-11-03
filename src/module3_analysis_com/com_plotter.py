"""
get an array and an index and plot xy, xz, yz of the
com.
the array is in the format of the com:

    x0 y0 z0 x1 y1 z1 ...

"""

import numpy as np
import matplotlib.pyplot as plt
from common import logger
from common import plot_tools
from common.colors_text import TextColor as bcolors


class ComPlotter:
    """plot the center of mass"""

    info_msg: str = 'Messages from ComPlotter:\n'  # To log

    def __init__(self,
                 com_arr: np.ndarray,  # The array to plot
                 index: int,  # Time frame to plot
                 log: logger.logging.Logger,
                 to_png: bool = False,  # If want to save as png
                 out_suffix: str = 'com.png',  # Suffix for fout, if to_png
                 to_xyz: bool = False,  # If want to save the traj to xyz file
                 xyz_file: str = 'com.xyz'  # name of the png file, if to_xyz
                 ) -> None:
        self.info_msg += f'mkaing graph for frame: `{index}`'
        xyz_arr: np.ndarray = self._parse_arr(com_arr, index)
        self.make_graph(xyz_arr, to_png, out_suffix)
        self._write_msg(log)

    def _parse_arr(self,
                   com_arr: np.ndarray,  # The array to plot
                   index: int  # Time frame to plot
                   ) -> np.ndarray:
        """return the data for plotting in one array"""
        return com_arr[index, :].reshape(-1, 3)

    def make_graph(self,
                   xyz_arr: np.ndarray,
                   to_png: bool = False,  # If want to save as png
                   out_suffix: str = 'com.png',  # Suffix for fout,if to_png
                   ) -> None:
        """plot and, if, save it"""
        fig_i, axes = \
            plot_tools.mk_canvas(x_range=(0, 240), ncols=3)
        axes[0].scatter(xyz_arr[:, 0], xyz_arr[:, 1])
        axes[1].scatter(xyz_arr[:, 0], xyz_arr[:, 2])
        axes[2].scatter(xyz_arr[:, 1], xyz_arr[:, 2])
        plt.show()

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{ComPlotter.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__mian__':
    pass
