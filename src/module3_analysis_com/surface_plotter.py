"""
plot surface water, in different styles
"""

import random
import typing
import numpy as np
import matplotlib.pylab as plt

from common import logger
from common import plot_tools
from common.colors_text import TextColor as bcolors


class SurfPlotter:
    """
    plot the surface of the water
    main input should has type of dict[int, np.ndarry]
    it should support few type of plotting
    """

    info_msg: str = 'Messages from SurfPlotter:\n'  # To log

    def __init__(self,
                 surf_dict: dict[int, np.ndarray],
                 log: logger.logging.Logger,
                 indices: typing.Optional[list[int]] = None,
                 fout_suffix: str = 'surface.png',
                 nr_fout: int = 5  # Numbers pics in case the list is empty
                 ) -> None:
        selected_frames: dict[int, np.ndarray] = \
            self.get_selected_frames(surf_dict, indices, nr_fout)
        self.plot_surface(selected_frames, fout_suffix)
        self._write_msg(log)

    def get_selected_frames(self,
                            surf_dict: dict[int, np.ndarray],
                            indices: typing.Optional[list[int]],
                            nr_fout: int
                            ) -> dict[int, np.ndarray]:
        """return the numbers of the wanted frames data"""
        if indices is None:
            indices = random.sample(list(surf_dict.keys()), nr_fout)
        self.info_msg += f'\tThe selected indices are:\n\t\t{indices}\n'
        return {key: surf_dict[key] for key in indices}

    def plot_surface(self,
                     selected_frames: dict[int, np.ndarray],
                     fout_suffix: str
                     ) -> None:
        """plot the surface"""
        for frame, value in selected_frames.items():
            fig_i, ax_i = \
                plot_tools.mk_canvas((0, 200), height_ratio=(5**0.5 - 1))
            scatter = ax_i.scatter(value[:, 0], value[:, 1], c=value[:, 2],
                                   s=15, label=f'frame: {frame}')
            cbar = plt.colorbar(scatter)
            cbar.set_label('Z-Coordinate [A]')
            ax_i.set_xlabel('X_Coordinate [A]')
            ax_i.set_ylabel('Y_Coordinate [A]')
            plt.axis('equal')
            plot_tools.save_close_fig(
                fig_i, ax_i, fname=f'{frame}_{fout_suffix}', legend=False)

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{SurfPlotter.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == "__main__":
    pass
