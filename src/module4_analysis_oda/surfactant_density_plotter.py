"""
plot the density of the ODA from SurfactantDensityAroundNanoparticle.
"""

import typing
from dataclasses import dataclass
from collections import namedtuple

import numpy as np
import matplotlib.pylab as plt

from common import logger
from common import plot_tools
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

    def __init__(self,
                 density_obj: "SurfactantDensityAroundNanoparticle",
                 log: logger.logging.Logger,
                 plot_config: "PlotConfig" = PlotConfig()
                 ) -> None:
        self.density = density_obj.density_per_region
        self.ave_density = density_obj.avg_density_per_region
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
        plt.colorbar(cbar, ax=ax_i, label='Average Density')
        plt.show()
        plot_tools.save_close_fig(
            fig_i, ax_i, fout := self.plot_config.graph_suffix)
        self.info_msg += \
            f'\tThe heatmap of the density is saved as {fout}\n'

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
