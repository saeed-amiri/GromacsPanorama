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
class BaseHeatMapConfig:
    """base configuration for the heatmap plots"""
    cbar_label: str
    heatmap_suffix: str
    circles_configs: dict[str, list[str]]
    show_grid: bool = False
    heatmap_color: str = 'Greys'


@dataclass
class DensityHeatMapConfig(BaseHeatMapConfig):
    """
    Configuration parameters for heatmap plotting of density.
    """
    heatmap_suffix: str = 'heatmap.png'
    cbar_label: str = 'Average Density'
    circles_configs: dict[str, list[str]] = field(default_factory=lambda: {
        'list': ['contact_radius', 'np_radius'],
        'color': ['k', 'b'],
        'linestyle': ['--', ':']})


@dataclass
class Rdf2dHeatMapConfig(BaseHeatMapConfig):
    """
    Configuration parameters for heatmap plotting of 2d rdf.
    """
    heatmap_suffix: str = 'rdf2dheatmap.png'
    cbar_label: str = 'g(r)'
    circles_configs: dict[str, list[str]] = field(default_factory=lambda: {
        'list': ['contact_radius', 'np_radius'],
        'color': ['k', 'b'],
        'linestyle': ['--', ':']})


@dataclass
class FittedsRdf2dHeatMapConfig(BaseHeatMapConfig):
    """
    Configuration parameters for heatmap plotting of fitted 2d rdf.
    """
    heatmap_suffix: str = 'fittedRdf2dheatmap.png'
    cbar_label: str = r'$g_{fitted}(r)$'
    circles_configs: dict[str, list[str]] = field(default_factory=lambda: {
        'list': ['contact_radius', 'turn_points'],
        'color': ['k', 'b', 'g', 'r'],
        'linestyle': ['-', '--', ':', ':']})


@dataclass
class SmoothedRdf2dHeatMapConfig(BaseHeatMapConfig):
    """
    Configuration parameters for heatmap plotting of smoothed 2d rdf.
    """
    heatmap_suffix: str = 'smoothedRdf2dheatmap.png'
    cbar_label: str = r'$g_{smoothed}(r)$'
    circles_configs: dict[str, list[str]] = field(default_factory=lambda: {
        'list': ['contact_radius', 'np_radius'],
        'color': ['k', 'b'],
        'linestyle': ['--', ':']})


@dataclass
class BaseGraphConfig:
    """Configurations for the simple graphs"""
    graph_suffix: str
    graph_legend: str
    title: str
    ylabel: str
    xlabel: str = 'Distance from Nanoparticle [A]'
    graph_style: dict = field(default_factory=lambda: {
        'legend': 'density',
        'color': 'k',
        'marker': 'o',
        'linestyle': '-',
        'markersize': 5,
        '2nd_marksize': 1
    })
    graph_2nd_legend: str = 'g(r)'


@dataclass
class DensityGraphConfig(BaseGraphConfig):
    """set the parameters for the graph"""
    graph_suffix: str = 'desnty.png'
    graph_legend: str = 'density'
    ylabel: str = 'Average Density'
    title: str = 'ODA Density vs Distance from NP'


@dataclass
class Rdf2dGraphConfig(BaseGraphConfig):
    """set the parameters for the graph"""
    graph_suffix: str = 'rdf_2d.png'
    graph_legend: str = 'g(r)'
    ylabel: str = 'g(r)'
    title: str = 'Rdf vs Distance from NP'


@dataclass
class FittedRdf2dGraphConfig(BaseGraphConfig):
    """set the parameters for the graph"""
    graph_suffix: str = 'fitted_rdf_2d.png'
    graph_legend: str = 'g(r)'
    graph_2nd_legend: str = r'$g_{fitted}(r)$'
    ylabel: str = 'g(r)'
    title: str = 'fitted Rdf vs Distance from NP'
    graph_style: dict = field(default_factory=lambda: {
        'legend': 'density',
        'color': 'k',
        'marker': 'o',
        'linestyle': '-',
        'markersize': 4,
        '2nd_marksize': 1
    })


@dataclass
class SmoothedRdf2dGraphConfig(BaseGraphConfig):
    """set the parameters for the graph"""
    graph_suffix: str = 'smoothed_rdf_2d.png'
    graph_legend: str = 'g(r)'
    graph_2nd_legend: str = r'$g_{smoothed}(r)$'
    ylabel: str = 'g(r)'
    title: str = 'smoothed Rdf vs Distance from NP'


@dataclass
class GraphsConfigs:
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
    rdf_2d: dict[float, float]
    fitted_rdf: dict[float, float]
    smoothed_rdf: dict[float, float]
    time_dependent_rdf: dict[int, dict[float, float]]
    time_dependent_ave_density: dict[int, dict[float, float]]
    contact_data: pd.DataFrame
    midpoint: float
    first_turn: float
    second_turn: float
    box: np.ndarray  # Size of the box at each frame (from gromacs)

    def __init__(self,
                 density_obj: "SurfactantDensityAroundNanoparticle",
                 log: logger.logging.Logger,
                 graphs_config: "GraphsConfigs" = GraphsConfigs()
                 ) -> None:

        self.density = density_obj.density_per_region
        self.ave_density = density_obj.avg_density_per_region
        self.rdf_2d = density_obj.rdf_2d
        self.fitted_rdf = density_obj.fitted_rdf
        self.smoothed_rdf = density_obj.smoothed_rdf
        self.time_dependent_rdf = density_obj.time_dependent_rdf
        self.time_dependent_ave = density_obj.time_dependent_ave_density
        self.midpoint = density_obj.midpoint
        self.first_turn = density_obj.first_turn
        self.second_turn = density_obj.second_turn
        self.contact_data = density_obj.contact_data
        self.box = density_obj.box

        self.graph_configs = graphs_config

        self._initialize_plotting(log)
        self.write_msg(log)

    def _initialize_plotting(self,
                             log: logger.logging.Logger
                             ) -> None:
        HeatmapPlotter(ref_density=self.ave_density,
                       contact_data=self.contact_data,
                       config=DensityHeatMapConfig(),
                       log=log)
        HeatmapPlotter(ref_density=self.rdf_2d,
                       contact_data=self.contact_data,
                       config=Rdf2dHeatMapConfig(),
                       log=log)
        HeatmapPlotter(ref_density=self.fitted_rdf,
                       contact_data=self.contact_data,
                       config=FittedsRdf2dHeatMapConfig(),
                       log=log,
                       fitted_turn_points={
                        'midpoint': self.midpoint,
                        '1st': self.first_turn,
                        '2nd': self.second_turn})
        HeatmapPlotter(ref_density=self.smoothed_rdf,
                       contact_data=self.contact_data,
                       config=SmoothedRdf2dHeatMapConfig(),
                       log=log)
        # DensityTimePlotter(density=self.density, log=log)
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
        TimeDependentPlotter(
            self.time_dependent_rdf, self.time_dependent_ave, log)

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
        fig_i, ax_i = self._plot_graphes(self.rdf_2d, config, return_ax=True)
        radii = np.array(list(rdf.keys()))
        densities = np.array(list(rdf.values()))
        ax_i.plot(radii,
                  densities,
                  marker=config.graph_style['marker'],
                  linestyle=config.graph_style['linestyle'],
                  color='r',
                  label=config.graph_2nd_legend,
                  markersize=config.graph_style['2nd_marksize'],
                  zorder=1)
        if style == 'fitted':
            ax_i = self._add_vline(
                ax_i, self.first_turn, legend='1st', lstyle=':', color='g')
            ax_i = self._add_vline(ax_i, self.midpoint, lstyle='--', color='b')
            ax_i = self._add_vline(
                ax_i, self.second_turn, legend='2nd', lstyle=':', color='r')
        plot_tools.save_close_fig(
            fig_i, ax_i, config.graph_suffix, loc='lower right')
        self.info_msg += \
            f'\tThe `{style}` graph saved: `{config.graph_suffix}`\n'

    @staticmethod
    def _add_vline(ax_i: plt.axes,
                   x_loc: float,
                   legend: str = 'c',
                   lstyle: str = '--',
                   color: str = 'k'
                   ) -> plt.axes:
        """add vline to the axes"""

        ylims: tuple[float, float] = ax_i.get_ylim()
        ax_i.vlines(x=x_loc,
                    ymin=ylims[0],
                    ymax=ylims[1],
                    ls=lstyle,
                    color=color,
                    label=f'{legend}={x_loc:.1f}')
        ax_i.set_ylim(ylims)
        return ax_i

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
                 ref_density: dict[float, float],
                 contact_data: pd.DataFrame,
                 config: typing.Union["DensityHeatMapConfig",
                                      "Rdf2dHeatMapConfig",
                                      "FittedsRdf2dHeatMapConfig",
                                      "SmoothedRdf2dHeatMapConfig"],
                 log: logger.logging.Logger,
                 fitted_turn_points:
                 typing.Union[None, dict[str, float]] = None
                 ) -> None:
        self.ref_density = ref_density
        self.config = config
        self.contact_data = contact_data
        self.turns_points = fitted_turn_points
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
        radii = np.array(list(self.ref_density.keys()))
        densities = np.array(list(self.ref_density.values()))
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
        np_radius: float = stinfo.np_info['radius']
        contact_radius: float = self._get_avg_contact_raduis()
        self.info_msg += (
            f'\tThe radius of the nanoparticle was set to `{np_radius:.3f}`\n'
            f'\tThe average contact radius is {contact_radius:.3f}\n')

        if 'contact_radius' in self.config.circles_configs['list']:
            ax_i = self._add_heatmap_circle(
                ax_i,
                contact_radius,
                color=self.config.circles_configs['color'][0],
                line_style=self.config.circles_configs['linestyle'][0]
                )
        if 'np_radius' in self.config.circles_configs['list']:
            ax_i = self._add_heatmap_circle(
                ax_i,
                np_radius,
                color=self.config.circles_configs['color'][1],
                line_style=self.config.circles_configs['linestyle'][1])
        if 'turn_points' in self.config.circles_configs['list']:
            if self.turns_points is not None:
                for i, point_i in enumerate(list(self.turns_points.values())):
                    ax_i = self._add_heatmap_circle(
                        ax_i,
                        point_i,
                        color=self.config.circles_configs['color'][i+1],
                        line_style=self.config.circles_configs[
                            'linestyle'][i+1])

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
        if 'np_radius' in self.config.circles_configs['list']:
            self._add_polar_arrow(
                ax_i, length=np_radius, theta=0, color='blue')
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


@dataclass
class TimeDependentPlotterConfig(BaseGraphConfig):
    """configurations for the time dependents plots"""
    graph_suffix: str = 'rdf_2d_time.png'
    graph_suffix2: str = 'ave_density_time.png'
    graph_legend: str = 'g(r,t)'
    graph_legend2: str = 'density(r,t)'
    ylabel: str = 'g(r)'
    ylabel2: str = 'density(r)'
    title: str = 'Fitted Rdf(t) vs Distance from NP'
    title2: str = 'Density vs Distance from NP'


class TimeDependentPlotter:
    """plot the time dependentes density and rdf"""

    info_msg: str = 'Messege from TimeDependentPlotter:\n'
    rdf: dict[int, dict[float, float]]
    ave_density: dict[int, dict[float, float]]

    def __init__(self,
                 time_dependent_rdf: dict[int, dict[float, float]],
                 time_dependent_ave_density: dict[int, dict[float, float]],
                 log: logger.logging.Logger,
                 config: "TimeDependentPlotterConfig" =
                 TimeDependentPlotterConfig()
                 ) -> None:
        self.config = config
        self.rdf = time_dependent_rdf
        self.ave_density = time_dependent_ave_density
        self.initiate_plot()

    def initiate_plot(self) -> None:
        """initiate plotting rdf and density"""
        fig_i, ax_i = self._plot_graph(self.rdf, label_prefix='Rdf')
        self.modify_rdf_plot(fig_i, ax_i)

    def modify_rdf_plot(self,
                        fig_i: plt.figure,
                        ax_i: plt.axes
                        ) -> None:
        """modify and save the rdf figure"""
        ax_i.set_xlabel(self.config.xlabel)
        ax_i.set_ylabel(self.config.ylabel)
        ax_i.set_title(self.config.title)
        plot_tools.save_close_fig(
            fig_i, ax_i, fname=self.config.graph_suffix, loc='lower right')

    def _plot_graph(self,
                    density: dict[int, dict[float, float]],
                    label_prefix: str
                    ) -> tuple[plt.figure, plt.axes]:
        """plot over time"""
        fig_i: plt.figure
        ax_i: plt.axes
        steps: list[int] = list(density.keys())
        steps_nr: int = len(steps)
        alpha_zero: float = 1/steps_nr
        radii = np.array(list(density[steps[-1]].keys()))
        fig_i, ax_i = \
            plot_tools.mk_canvas(x_range=(np.min(radii), np.max(radii)),
                                 height_ratio=5**0.5-1)
        for i, (_, frame) in enumerate(density.items(), start=1):
            densities = np.array(list(frame.values()))
            if i == steps_nr:
                color = 'r'
                alpha = 1.0
                label = label_prefix
            else:
                color = 'k'
                alpha = 1.0-i*alpha_zero
                label = None
                if i == 1:
                    label = f'{label_prefix}(t)'
            ax_i.plot(radii,
                      densities,
                      marker=self.config.graph_style['marker'],
                      linestyle=self.config.graph_style['linestyle'],
                      color=color,
                      label=label,
                      markersize=0,
                      alpha=alpha)
        ax_i.grid(True, linestyle='--', color='gray', alpha=0.5)
        return fig_i, ax_i


if __name__ == "__main__":
    print(f'{bcolors.CAUTION}\tThis script runs within '
          f'trajectory_oda_analysis.py{bcolors.ENDC}')
