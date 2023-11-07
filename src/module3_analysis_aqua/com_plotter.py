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
from common import static_info as stinfo
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
                 xyz_suufix: str = 'com.xyz',  # Suffix to xyz file, if to_xyz
                 atom: str = "O"
                 ) -> None:
        self.info_msg += f'\tmkaing graph for frame: `{index}`\n'
        xyz_arr: np.ndarray = self._parse_arr(com_arr, index)
        self.make_graph(xyz_arr, index, to_png, out_suffix)
        if to_xyz:
            self.write_xyz_file(xyz_arr, index, atom, xyz_suufix)
        self._write_msg(log)

    def _parse_arr(self,
                   com_arr: np.ndarray,  # The array to plot
                   index: int  # Time frame to plot
                   ) -> np.ndarray:
        """return the data for plotting in one array"""
        return com_arr[index, :].reshape(-1, 3)

    def make_graph(self,
                   xyz_arr: np.ndarray,
                   index: int,
                   to_png: bool = False,  # If want to save as png
                   out_suffix: str = 'com.png',  # Suffix for fout,if to_png
                   ) -> None:
        """plot and, if, save it"""
        fig_i, axes = plot_tools.mk_canvas(
            x_range=(0, 200), ncols=3, width_ratio=4)

        labels = ['xy', 'xz', 'yz']
        data = [xyz_arr[:, [0, 1]], xyz_arr[:, [0, 2]], xyz_arr[:, [1, 2]]]

        for i, ax_i in enumerate(axes):
            ax_i.scatter(
                data[i][:, 0],
                data[i][:, 1],
                label=f'{labels[i]},frame:{index}')
            ax_i.set_xlabel('x [nm]' if i != 2 else 'y [nm]')
            ax_i.set_ylabel('y [nm]' if i == 0 else 'z [nm]')
            ax_i.grid(axis='both', alpha=0.5, color='grey')

        if to_png:
            plot_tools.save_close_fig(
                fig_i, axes, fname := f"{index}_{out_suffix}")
            self.info_msg += f'\tThe out picture is: {fname}\n'
        else:
            plt.show()

    def write_xyz_file(self,
                       xyz_arr: np.ndarray,
                       index: int,
                       atom: str,
                       xyz_suffix: str,
                       ) -> None:
        """Write xyz_arr to an XYZ format file."""
        with open(f'{index}_{xyz_suffix}', 'w', encoding='utf8') as f_r:
            num_atoms, _ = xyz_arr.shape
            f_r.write(f"{num_atoms}\n\n")
            for i in range(num_atoms):
                f_r.write(f"{atom} "
                          f"{xyz_arr[i, 0]} {xyz_arr[i, 1]} {xyz_arr[i, 2]}\n")

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{ComPlotter.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__mian__':
    pass
