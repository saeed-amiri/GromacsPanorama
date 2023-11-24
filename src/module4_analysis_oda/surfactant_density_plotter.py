"""
plot the density of the ODA from SurfactantDensityAroundNanoparticle.
"""

import typing
from dataclasses import dataclass
from collections import namedtuple

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.patches import Circle

from common import logger
from common import plot_tools
from common import static_info as stinfo
from common.colors_text import TextColor as bcolors

if typing.TYPE_CHECKING:
    from module4_analysis_oda.oda_density_around_np \
        import SurfactantDensityAroundNanoparticle


@dataclass
class PlotConfig:
    """set the parameters for the the plot"""
    heatmap_suffix: str = 'heatmap.png'
    graph_suffix: str = 'desnty.png'


# Define a named tuple for plot parameters
HeatPlotParams = \
    namedtuple('HeatPlotParams', 'fig ax radial_distances theta density_grid')


class SurfactantDensityPlotter:
    """plot the desinty of the oda around the NP"""

    info_msg: str = 'Message from SurfactantDensityPlotter:\n'
    density: dict[float, list[float]]
    ave_density: dict[float, float]
    contact_data: pd.DataFrame

    def __init__(self,
                 density_obj: "SurfactantDensityAroundNanoparticle",
                 log: logger.logging.Logger,
                 plot_config: "PlotConfig" = PlotConfig()
                 ) -> None:
        self.density = density_obj.density_per_region
        self.ave_density = density_obj.avg_density_per_region
        self.contact_data = density_obj.contact_data
        self.plot_config = plot_config
        self._initialize_plotting()
        self.write_msg(log)

    def _initialize_plotting(self) -> None:
        self.plot_density_heatmap()

    def plot_density_heatmap(self) -> None:
        """self explanetory"""
        ax_i: plt.axes
        fig_i: plt.figure
        theta: np.ndarray
        density_grid: np.ndarray
        radial_distances: np.ndarray

        fig_i, ax_i = self._setup_plot()
        radial_distances, theta, density_grid = self._create_density_grid()
        plot_params = \
            HeatPlotParams(fig_i, ax_i, radial_distances, theta, density_grid)
        self._plot_and_save_heatmap(plot_params)

    def _create_density_grid(self) -> tuple[np.ndarray, ...]:
        """Create a grid in polar coordinates with interpolated densities."""
        radii = np.array(list(self.ave_density.keys()))
        densities = np.array(list(self.ave_density.values()))
        # Create a grid in polar coordinates
        radial_distances, theta = \
            np.meshgrid(radii, np.linspace(0, 2 * np.pi, len(radii)))
        # Interpolate densities onto the grid
        density_grid = np.tile(densities, (len(radii), 1))
        return radial_distances, theta, density_grid

    def _setup_plot(self) -> tuple[plt.figure, plt.axes]:
        """Set up the polar plot."""
        fig_i, ax_i = plt.subplots(
            subplot_kw={'projection': 'polar'}, figsize=(8, 8))
        ax_i.set_yticklabels([])
        ax_i.set_xticklabels([])
        ax_i.grid(False)
        return fig_i, ax_i

    def _plot_and_save_heatmap(self,
                               plot_params: "HeatPlotParams"
                               ) -> None:
        """Plot the heatmap and save the figure."""
        ax_i: plt.axes = plot_params.ax
        fig_i: plt.figure = plot_params.fig
        cbar = ax_i.pcolormesh(plot_params.theta,
                               plot_params.radial_distances,
                               plot_params.density_grid,
                               shading='auto',
                               cmap='Greys'
                               )
        ax_i = self._add_heatmap_grid(ax_i)
        ax_i = self._add_np_radii(ax_i)
        plt.colorbar(cbar, ax=ax_i, label='Average Density')
        plt.show()
        plot_tools.save_close_fig(
            fig_i, ax_i, fout := self.plot_config.graph_suffix)
        self.info_msg += \
            f'\tThe heatmap of the density is saved as {fout}\n'

    def _add_np_radii(self,
                      ax_i: plt.axes
                      ) -> plt.axes:
        """attach circle denoting the np"""
        contact_radius: float = self._get_avg_contact_raduis()
        np_radius: float = stinfo.np_info['radius']
        self.info_msg += (
            f'\tThe radius of the nanoparticle was set to `{np_radius:.3f}`\n'
            f'\tThe average contact radius is {contact_radius:.3f}\n')
        ax_i = self._add_heatmap_circle(ax_i, contact_radius)
        ax_i = self._add_heatmap_circle(ax_i, np_radius)
        return ax_i

    def _get_avg_contact_raduis(self) -> float:
        """self explanatory"""
        return self.contact_data.loc[:, 'contact_radius'].mean()

    @staticmethod
    def _add_heatmap_grid(ax_i: plt.axes
                          ) -> plt.axes:
        """self explanatory"""
        # Customizing grid lines
        ax_i.yaxis.grid(True, color='green', linestyle='dashed', linewidth=0.5)
        ax_i.xaxis.grid(True, color='green', linestyle='dashed', linewidth=0.5)

        # Positioning grid lines and ticks to the top and right
        ax_i.xaxis.tick_top()
        ax_i.yaxis.tick_right()
        ax_i.xaxis.set_label_position('top')
        ax_i.yaxis.set_label_position('right')
        return ax_i

    @staticmethod
    def _add_heatmap_circle(ax_i: plt.axes,
                            radius: float,
                            origin: tuple[float, float] = (0, 0),
                            color: str = 'red',
                            line_style: str = '--'
                            ) -> plt.axes:
        """add circle for representing the nanoparticle"""
        circle = Circle(origin,
                        radius,
                        transform=ax_i.transData._b,
                        color=color,
                        ls=line_style,
                        fill=False)
        ax_i.add_patch(circle)
        return ax_i

    def write_msg(self,
                  log: logger.logging.Logger  # To log
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == "__main__":
    print(f'{bcolors.CAUTION}\tThis script runs within '
          f'trajectory_oda_analysis.py{bcolors.ENDC}')
