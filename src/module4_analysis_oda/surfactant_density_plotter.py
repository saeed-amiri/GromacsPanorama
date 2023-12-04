"""
plot the density of the ODA from SurfactantDensityAroundNanoparticle.
"""

import typing
from dataclasses import dataclass, field
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


# Define a named tuple for plot parameters
HeatMapPlottingData = namedtuple(
    'HeatMapPlottingData', 'fig ax radial_distances theta density_grid')


@dataclass
class DensityHeatMapConfig:
    """Configuration parameters for heatmap plotting.

    Attributes:
        heatmap_suffix (str): Filename suffix for the saved heatmap image.
        heatmap_color (str): Color scheme used for the heatmap.
        show_grid (bool): To indicate whether a grid is shown on the heatmap.
        cbar_label (str): Label for the heatmap's color bar.
    """
    heatmap_suffix: str = 'heatmap.png'
    heatmap_color: str = 'Greys'
    show_grid: bool = False
    cbar_label: str = 'Average Density'


@dataclass
class Rdf2dHeatMapConfig:
    """Configuration parameters for heatmap plotting of 2d rdf.

    Attributes:
        heatmap_suffix (str): Filename suffix for the saved heatmap image.
        heatmap_color (str): Color scheme used for the heatmap.
        show_grid (bool): To indicate whether a grid is shown on the heatmap.
        cbar_label (str): Label for the heatmap's color bar.
    """
    heatmap_suffix: str = 'rdf2dheatmap.png'
    heatmap_color: str = 'Greys'
    show_grid: bool = False
    cbar_label: str = 'g(r)'


@dataclass
class FittedsRdf2dHeatMapConfig:
    """Configuration parameters for heatmap plotting of fitted 2d rdf.

    Attributes:
        heatmap_suffix (str): Filename suffix for the saved heatmap image.
        heatmap_color (str): Color scheme used for the heatmap.
        show_grid (bool): To indicate whether a grid is shown on the heatmap.
        cbar_label (str): Label for the heatmap's color bar.
    """
    heatmap_suffix: str = 'fittedRdf2dheatmap.png'
    heatmap_color: str = 'Greys'
    show_grid: bool = False
    cbar_label: str = r'$g_{fitted}(r)$'


@dataclass
class SmoothedRdf2dHeatMapConfig:
    """Configuration parameters for heatmap plotting of smoothed 2d rdf.

    Attributes:
        heatmap_suffix (str): Filename suffix for the saved heatmap image.
        heatmap_color (str): Color scheme used for the heatmap.
        show_grid (bool): To indicate whether a grid is shown on the heatmap.
        cbar_label (str): Label for the heatmap's color bar.
    """
    heatmap_suffix: str = 'smoothedRdf2dheatmap.png'
    heatmap_color: str = 'Greys'
    show_grid: bool = False
    cbar_label: str = r'$g_{smoothed}(r)$'


@dataclass
class DensityGraphConfig:
    """set the parameters for the graph"""
    graph_suffix: str = 'desnty.png'
    graph_legend: str = 'density'
    xlabel: str = 'Distance from Nanoparticle (units)'
    ylabel: str = 'Average Density (units)'
    title: str = 'ODA Density vs Distance from NP'
    graph_style: dict = field(default_factory=lambda: {
        'legend': 'density',
        'color': 'k',
        'marker': 'o',
        'linestyle': '-',
        'markersize': 5
    })


@dataclass
class Rdf2dGraphConfig:
    """set the parameters for the graph"""
    graph_suffix: str = 'rdf_2d.png'
    graph_legend: str = 'g(r)'
    xlabel: str = 'Distance from Nanoparticle [A]'
    ylabel: str = 'g(r)'
    title: str = 'Rdf vs Distance from NP'
    graph_style: dict = field(default_factory=lambda: {
        'legend': 'g(r)',
        'color': 'k',
        'marker': 'o',
        'linestyle': '-',
        'markersize': 5
    })


@dataclass
class FittedRdf2dGraphConfig:
    """set the parameters for the graph"""
    graph_suffix: str = 'fitted_rdf_2d.png'
    graph_legend: str = r'$g_{fitted}(r)$'
    graph_legend_2: str = 'g(r)'
    xlabel: str = 'Distance from Nanoparticle [A]'
    ylabel: str = 'g(r)'
    title: str = 'fitted Rdf vs Distance from NP'

    graph_style: dict = field(default_factory=lambda: {
        'legend': 'g(r)',
        'color': 'r',
        'marker': 'o',
        'linestyle': '-',
        'markersize': 1,
        'unfit_markersize': 5
    })


@dataclass
class SmoothedRdf2dGraphConfig:
    """set the parameters for the graph"""
    graph_suffix: str = 'smoothed_rdf_2d.png'
    graph_legend: str = r'$g_{smoothed}(r)$'
    graph_legend_2: str = 'g(r)'
    xlabel: str = 'Distance from Nanoparticle [A]'
    ylabel: str = 'g(r)'
    title: str = 'smoothed Rdf vs Distance from NP'

    graph_style: dict = field(default_factory=lambda: {
        'legend': 'g(r)',
        'color': 'r',
        'marker': 'o',
        'linestyle': '-',
        'markersize': 1,
        'unfit_markersize': 5
    })


@dataclass
class GrpahsConfig:
    """all the graphs configurations"""
    graph_config:  "DensityGraphConfig" = DensityGraphConfig()
    rdf_config: "Rdf2dGraphConfig" = Rdf2dGraphConfig()
    fitted_rdf_config: "FittedRdf2dGraphConfig" = FittedRdf2dGraphConfig()
    smoothed_rdf_config: "SmoothedRdf2dGraphConfig" = \
        SmoothedRdf2dGraphConfig()


class SurfactantDensityPlotter:
    """plot the desinty of the oda around the NP"""

    info_msg: str = 'Message from SurfactantDensityPlotter:\n'
    density: dict[float, list[float]]
    ave_density: dict[float, float]
    contact_data: pd.DataFrame
    box: np.ndarray  # Size of the box at each frame (from gromacs)

    def __init__(self,
                 density_obj: "SurfactantDensityAroundNanoparticle",
                 log: logger.logging.Logger,
                 graphs_config: "GrpahsConfig" = GrpahsConfig()
                 ) -> None:

        self.density = density_obj.density_per_region
        self.ave_density = density_obj.avg_density_per_region
        self.rdf_2d = density_obj.rdf_2d
        self.fitted_rdf = density_obj.fitted_rdf
        self.smoothed_rdf = density_obj.smoothed_rdf

        self.contact_data = density_obj.contact_data
        self.box = density_obj.box

        self.graph_configs = graphs_config

        self._initialize_plotting(log)
        self.write_msg(log)

    def _initialize_plotting(self,
                             log: logger.logging.Logger
                             ) -> None:
        HeatmapPlotter(ave_density=self.ave_density,
                       contact_data=self.contact_data,
                       config=DensityHeatMapConfig(),
                       log=log)
        HeatmapPlotter(ave_density=self.rdf_2d,
                       contact_data=self.contact_data,
                       config=Rdf2dHeatMapConfig(),
                       log=log)
        HeatmapPlotter(ave_density=self.fitted_rdf,
                       contact_data=self.contact_data,
                       config=FittedsRdf2dHeatMapConfig(),
                       log=log)
        HeatmapPlotter(ave_density=self.smoothed_rdf,
                       contact_data=self.contact_data,
                       config=SmoothedRdf2dHeatMapConfig(),
                       log=log)
        self.plot_density_graph(self.graph_configs.graph_config)
        self.plot_2d_rdf(self.graph_configs.rdf_config)
        self.plot_fitted_or_smoothed_rdf(self.fitted_rdf,
                                         'fitted',
                                         self.graph_configs.fitted_rdf_config
                                         )
        self.plot_fitted_or_smoothed_rdf(self.smoothed_rdf,
                                         'smoothed',
                                         self.graph_configs.smoothed_rdf_config
                                         )

    def plot_density_graph(self,
                           config: "DensityGraphConfig") -> None:
        """Plot a simple graph of density vs distance."""
        # Extracting radii and average densities
        self._plot_graphes(self.ave_density, config)

    def plot_2d_rdf(self,
                    config: "Rdf2dGraphConfig"
                    ) -> None:
        """Plot a simple graph of 2d rdf vs distance."""
        self._plot_graphes(self.rdf_2d, config)

    def plot_fitted_or_smoothed_rdf(self,
                                    rdf: dict[float, float],
                                    style: str,
                                    config: typing.Union[
                                        "FittedRdf2dGraphConfig",
                                        "SmoothedRdf2dGraphConfig"]
                                    ) -> None:
        """plot the fitted graph alongside main data"""
        fig_i, ax_i = self._plot_graphes(rdf, config, return_ax=True)
        radii = np.array(list(self.rdf_2d.keys()))
        densities = np.array(list(self.rdf_2d.values()))
        ax_i.plot(radii,
                  densities,
                  marker=config.graph_style['marker'],
                  linestyle=config.graph_style['linestyle'],
                  color='k',
                  label=config.graph_legend_2,
                  markersize=config.graph_style['unfit_markersize'],
                  zorder=1)
        plot_tools.save_close_fig(
            fig_i, ax_i, config.graph_suffix, loc='upper left')
        self.info_msg += \
            f'\tThe `{style}` graph saved: `{config.graph_suffix}`\n'

    def _plot_graphes(self,
                      data: dict[float, float],
                      config: typing.Union["Rdf2dGraphConfig",
                                           "DensityGraphConfig",
                                           "FittedRdf2dGraphConfig",
                                           "SmoothedRdf2dGraphConfig"],
                      return_ax: bool = False
                      ) -> tuple[plt.figure, plt.axes]:
        """plot graphs"""
        radii = np.array(list(data.keys()))
        densities = np.array(list(data.values()))
        ax_i: plt.axes
        fig_i: plt.figure
        fig_i, ax_i = plot_tools.mk_canvas((np.min(radii), np.max(radii)),
                                           height_ratio=5**0.5-1)
        ax_i.plot(radii,
                  densities,
                  marker=config.graph_style['marker'],
                  linestyle=config.graph_style['linestyle'],
                  color=config.graph_style['color'],
                  label=config.graph_legend,
                  markersize=config.graph_style['markersize'])
        ax_i.set_xlabel(config.xlabel)
        ax_i.set_ylabel(config.ylabel)
        ax_i.set_title(config.title)
        # Set grid for primary axis
        ax_i.grid(True, linestyle='--', color='gray', alpha=0.5)

        if not return_ax:
            plot_tools.save_close_fig(
                fig_i, ax_i, config.graph_suffix, loc='upper left')
            self.info_msg += \
                f'\tThe density graph saved: `{config.graph_suffix}`\n'
        return fig_i, ax_i

    def write_msg(self,
                  log: logger.logging.Logger  # To log
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


class HeatmapPlotter:
    """Class for plotting heatmap of surfactant density."""

    info_msg: str = "Messages from HeatmapPlotter:\n"

    def __init__(self,
                 ave_density: dict[float, float],
                 contact_data: pd.DataFrame,
                 config: typing.Union["DensityHeatMapConfig",
                                      "Rdf2dHeatMapConfig",
                                      "FittedsRdf2dHeatMapConfig",
                                      "SmoothedRdf2dHeatMapConfig"],
                 log: logger.logging.Logger
                 ) -> None:
        self.ave_density = ave_density
        self.config = config
        self.contact_data = contact_data
        self.plot_density_heatmap()
        self._write_msg(log)

    def plot_density_heatmap(self) -> None:
        """Plot and save the density heatmap."""
        fig_i, ax_i = self.setup_plot()
        radial_distances, theta, density_grid = self.create_density_grid()
        plot_params = HeatMapPlottingData(
            fig_i, ax_i, radial_distances, theta, density_grid)
        ax_i = self.plot_heatmap(plot_params)
        self._save_close_fig(
            fig_i, ax_i, fout := self.config.heatmap_suffix, legend=False)
        self.info_msg += f'\tThe heatmap of the density is saved as {fout}\n'

    def create_density_grid(self) -> tuple[np.ndarray, ...]:
        """Create a grid in polar coordinates with interpolated densities."""
        radii = np.array(list(self.ave_density.keys()))
        densities = np.array(list(self.ave_density.values()))
        # Create a grid in polar coordinates
        radial_distances, theta = \
            np.meshgrid(radii, np.linspace(0, 2 * np.pi, len(radii)))
        # Interpolate densities onto the grid
        density_grid = np.tile(densities, (len(radii), 1))
        return radial_distances, theta, density_grid

    def setup_plot(self) -> tuple[plt.figure, plt.axes]:
        """Set up the polar plot."""
        fig_i, ax_i = plt.subplots(
            subplot_kw={'projection': 'polar'}, figsize=(8, 8))
        ax_i.set_yticklabels([])
        ax_i.set_xticklabels([])
        ax_i.grid(False)
        return fig_i, ax_i

    def plot_heatmap(self,
                     plot_params: "HeatMapPlottingData"
                     ) -> plt.axes:
        """Plot the heatmap and save the figure."""
        ax_i: plt.axes = plot_params.ax
        cbar = ax_i.pcolormesh(plot_params.theta,
                               plot_params.radial_distances,
                               plot_params.density_grid,
                               shading='auto',
                               cmap=self.config.heatmap_color
                               )
        if self.config.show_grid:
            ax_i = self._add_heatmap_grid(ax_i)
        ax_i, contact_radius, np_radius = self._add_np_radii(ax_i)
        ax_i = self._add_radius_arrows(ax_i, contact_radius, np_radius)
        cbar = plt.colorbar(cbar, ax=ax_i)
        cbar.ax.tick_params(labelsize=13)  # Adjust tick label font size
        cbar.set_label(label=self.config.cbar_label, fontsize=13)
        return ax_i

    def _add_np_radii(self,
                      ax_i: plt.axes
                      ) -> tuple[plt.axes, float, float]:
        """attach circle denoting the np"""
        contact_radius: float = self._get_avg_contact_raduis()
        np_radius: float = stinfo.np_info['radius']
        self.info_msg += (
            f'\tThe radius of the nanoparticle was set to `{np_radius:.3f}`\n'
            f'\tThe average contact radius is {contact_radius:.3f}\n')
        ax_i = self._add_heatmap_circle(ax_i, contact_radius)
        ax_i = self._add_heatmap_circle(ax_i, np_radius, color='blue')
        return ax_i, contact_radius, np_radius

    def _add_radius_arrows(self,
                           ax_i: plt.axes,
                           contact_radius: float,
                           np_radius: float
                           ) -> plt.axes:
        """self explanatory"""
        self._add_polar_arrow(
            ax_i, length=contact_radius, theta=np.pi/2, color='red')
        ax_i = \
            self._add_radii_label(ax_i,
                                  label=(
                                    rf'$r_{{c, avg}}$={contact_radius:.2f}'),
                                  location=(1, 1),
                                  color='red')
        self._add_polar_arrow(ax_i, length=np_radius, theta=0, color='blue')
        ax_i = self._add_radii_label(ax_i,
                                     label=rf'$a$={np_radius:.2f}',
                                     location=(1, 0.95),
                                     color='blue')
        return ax_i

    @staticmethod
    def _add_polar_arrow(ax_i: plt.axes,
                         length: float,  # Radial direction of the arrow
                         theta: float,  # Angle of the arrow's location
                         dtheta: float = 0,  # Angular direction of the arrow
                         color: str = 'red'
                         ) -> None:
        """Add an arrow to the polar plot."""
        # The quiver function adds the arrow to the plot
        r_loc: float = 0  # Radius of the arrow's location
        ax_i.quiver(theta,
                    r_loc,
                    dtheta,
                    length,
                    angles='xy',
                    scale_units='xy',
                    scale=1,
                    width=0.005,
                    color=color)

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

    @staticmethod
    def _add_radii_label(ax_i: plt.axes,
                         label: str,
                         location: tuple[float, float],
                         color: str,
                         fontsize: float = 12
                         ) -> plt.axes:
        """add text for the label of the radiii"""
        ax_i.text(location[0], location[1],
                  label,
                  horizontalalignment='right',
                  verticalalignment='center',
                  color=color,
                  fontsize=fontsize,
                  transform=ax_i.transAxes)
        return ax_i

    @staticmethod
    def _save_close_fig(fig: plt.figure,  # The figure to save,
                        axs: plt.axes,  # Axes to plot
                        fname: str,  # Name of the output for the fig
                        loc: str = 'upper right',  # Location of the legend
                        transparent=False,
                        legend=True
                        ) -> None:
        """
        Cannot use the plot_tools
        Save the figure and close it.

        This method saves the given figure and closes it after saving.
        """
        if legend:
            for ax_j in np.ravel(axs):
                legend = ax_j.legend(loc=loc)
        fig.savefig(fname,
                    dpi=300,
                    pad_inches=0.1,
                    edgecolor='auto',
                    bbox_inches='tight',
                    transparent=transparent
                    )
        plt.close(fig)

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == "__main__":
    print(f'{bcolors.CAUTION}\tThis script runs within '
          f'trajectory_oda_analysis.py{bcolors.ENDC}')
