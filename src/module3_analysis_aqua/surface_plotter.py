"""
plot surface water, in different styles
"""

import random
import typing
from dataclasses import dataclass
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Circle
import matplotlib.pylab as plt
import numpy as np

from common import logger
from common import plot_tools
from common import static_info as stinfo
from common.colors_text import TextColor as bcolors

if typing.TYPE_CHECKING:
    import matplotlib


@dataclass
class PlotConfig:
    """set the output configurations"""
    fout_suffix: str = 'surface.png'
    nr_fout: int = 5  # Numbers pics in case the list is empty
    indices: typing.Optional[list[int]] = None
    show_ticks: bool = False
    show_labels: bool = False
    show_legend: bool = False


@dataclass
class DataConfig:
    """set up the data"""
    surf_dict: dict[int, np.ndarray]
    np_com: np.ndarray  # Com of the Np
    box_dims: dict[str, float]


class SurfPlotter:
    """
    plot the surface of the water
    main input should has type of dict[int, np.ndarry]
    it should support few type of plotting
    """

    info_msg: str = 'Messages from SurfPlotter:\n'  # To log

    def __init__(self,
                 log: logger.logging.Logger,
                 data_config: "DataConfig",
                 plot_config: "PlotConfig" = PlotConfig,
                 ) -> None:
        self.np_com: np.ndarray = data_config.np_com
        selected_frames: dict[int, np.ndarray] = self.get_selected_frames(
            data_config.surf_dict, plot_config.indices, plot_config.nr_fout)
        self.plot_surface(
            selected_frames, data_config.box_dims, plot_config)
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
                     plot_config: "PlotConfig"
                     ) -> None:
        """plot the surface"""
        self.info_msg += \
            f'\tThe suffix of the files is: `{plot_config.fout_suffix}`\n'
        for frame, value in selected_frames.items():
            fig_i, ax_i = \
                plot_tools.mk_canvas((0, 20), height_ratio=5**0.5-1)
            if plot_config.show_legend:
                scatter = ax_i.scatter(value[:, 0] / 10.0,  # Convert to nm
                                       value[:, 1] / 10.0,  # Convert to nm
                                       c=value[:, 2] / 10.0,  # Convert to nm
                                       s=15,
                                       label=f'frame: {frame}')
            else:
                scatter = ax_i.scatter(value[:, 0] / 10.0,
                                       value[:, 1] / 10.0,
                                       c=value[:, 2] / 10.0,
                                       s=15)

            # 'left', 'bottom', 'width', 'height'
            cbar_ax = fig_i.add_axes([.80, 0.15, 0.1, 0.5])
            cbar = fig_i.colorbar(scatter, cax=cbar_ax)
            desired_num_ticks = 5
            cbar.ax.yaxis.set_major_locator(MaxNLocator(desired_num_ticks))

            cbar.set_label('Z-Coordinate [nm]', rotation=90)
            if plot_config.show_labels:
                ax_i.set_xlabel('X_Coordinate [nm]')
                ax_i.set_ylabel('Y_Coordinate [nm]')
            else:
                ax_i.set_xlabel('')
                ax_i.set_ylabel('')
            if not plot_config.show_ticks:
                ax_i.set_xticks([])
                ax_i.set_yticks([])
            ax_i.spines['bottom'].set_visible(False)
            ax_i.spines['left'].set_visible(False)
            ax_i.spines['top'].set_visible(False)
            ax_i.spines['right'].set_visible(False)
            self.add_circle_to_axis(ax_i,
                                    origin=self.np_com[frame][:2] / 10.0,
                                    radius=stinfo.np_info['radius'] / 10.0)
            ax_i.set_xlim(box_dims['x_lo'] / 10.0 - 0.7,
                          box_dims['x_hi'] / 10.0 + 0.7)
            ax_i.set_ylim(box_dims['y_lo'] / 10.0 - 0.7,
                          box_dims['y_hi'] / 10.0 + 0.7)
            plt.gca().set_aspect('equal')
            plot_tools.save_close_fig(
                fig_i,
                ax_i,
                fname=f'{frame}_{plot_config.fout_suffix}',
                legend=False)

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
