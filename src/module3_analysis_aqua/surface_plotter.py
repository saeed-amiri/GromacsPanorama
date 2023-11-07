"""
plot surface water, in different styles
"""

import random
import typing
import numpy as np
import matplotlib.pylab as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Circle

from common import logger
from common import plot_tools
from common import static_info as stinfo
from common.colors_text import TextColor as bcolors

if typing.TYPE_CHECKING:
    import matplotlib


class SurfPlotter:
    """
    plot the surface of the water
    main input should has type of dict[int, np.ndarry]
    it should support few type of plotting
    """

    info_msg: str = 'Messages from SurfPlotter:\n'  # To log

    def __init__(self,
                 surf_dict: dict[int, np.ndarray],
                 np_com: np.ndarray,  # Com of the Np
                 box_dims: dict[str, float],
                 log: logger.logging.Logger,
                 indices: typing.Optional[list[int]] = None,
                 fout_suffix: str = 'surface.png',
                 nr_fout: int = 5  # Numbers pics in case the list is empty
                 ) -> None:
        self.np_com: np.ndarray = np_com
        selected_frames: dict[int, np.ndarray] = \
            self.get_selected_frames(surf_dict, indices, nr_fout)
        self.plot_surface(selected_frames, box_dims, fout_suffix)
        self._write_msg(log)
        self.selected_frames: list[int] = list(selected_frames.keys())

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
                     box_dims: dict[str, float],
                     fout_suffix: str
                     ) -> None:
        """plot the surface"""
        self.info_msg += f'\tThe suffix of the files is: `{fout_suffix}`\n'
        for frame, value in selected_frames.items():
            fig_i, ax_i = \
                plot_tools.mk_canvas((0, 200), height_ratio=5**0.5-1)
            scatter = ax_i.scatter(value[:, 0], value[:, 1], c=value[:, 2],
                                   s=15, label=f'frame: {frame}')
            # 'left', 'bottom', 'width', 'height'
            cbar_ax = fig_i.add_axes([.80, 0.15, 0.25, 0.7])
            cbar = fig_i.colorbar(scatter, cax=cbar_ax)
            desired_num_ticks = 5
            cbar.ax.yaxis.set_major_locator(MaxNLocator(desired_num_ticks))

            cbar.set_label('Z-Coordinate [A]', rotation=90)
            ax_i.set_xlabel('X_Coordinate [A]')
            ax_i.set_ylabel('Y_Coordinate [A]')
            self.add_circle_to_axis(ax_i,
                                    origin=self.np_com[frame][:2],
                                    radius=stinfo.np_info['radius'])
            ax_i.set_xlim(box_dims['x_lo'] - 7, box_dims['x_hi'] + 7)
            ax_i.set_ylim(box_dims['y_lo'] - 7, box_dims['y_hi'] + 7)
            plt.gca().set_aspect('equal')
            plot_tools.save_close_fig(
                fig_i, ax_i, fname=f'{frame}_{fout_suffix}', legend=False)

    def add_circle_to_axis(self,
                           ax_i: "matplotlib.axes._subplots.AxesSubplot",
                           origin: tuple[float, float],
                           radius: float,
                           color: str = 'red'
                           ) -> "matplotlib.axes._subplots.AxesSubplot":
        """
        Add a circle with the specified origin and radius to the
        given axis.
        """
        circle = Circle(origin, radius, fill=False, color=color, ls='--')
        ax_i.add_patch(circle)
        return ax_i

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{SurfPlotter.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == "__main__":
    pass
