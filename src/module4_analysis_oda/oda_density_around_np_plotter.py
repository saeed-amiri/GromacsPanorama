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
from common import plot_tools, elsevier_plot_tools
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
    # pylint: disable=too-many-instance-attributes
    cbar_label: str
    heatmap_suffix: str
    circles_configs: dict[str, list[str]]
    show_grid: bool = False
    heatmap_color: str = 'Greys'
    if_arrow: bool = False
    if_title: bool = False
    if_elsevier: bool = True
    show_oda_label: bool = True


@dataclass
class DensityHeatMapConfig(BaseHeatMapConfig):
    """
    Configuration parameters for heatmap plotting of density.
    """
    heatmap_suffix: str = f'heatmap.{elsevier_plot_tools.IMG_FORMAT}'
    cbar_label: str = 'Average Density'
    circles_configs: dict[str, list[str]] = field(default_factory=lambda: {
        'list': ['contact_radius', 'np_radius'],
        'color': ['k', 'darkred'],
        'linestyle': ['-', '--', '-.', ':']})


@dataclass
class Rdf2dHeatMapConfig(BaseHeatMapConfig):
    """
    Configuration parameters for heatmap plotting of 2d rdf.
    """
    heatmap_suffix: str = f'rdf2dheatmap.{elsevier_plot_tools.IMG_FORMAT}'
    cbar_label: str = r'$g^\star(r^\star)$, a. u.'
    circles_configs: dict[str, list[str]] = field(default_factory=lambda: {
        'list': ['contact_radius', 'np_radius'],
        'color': ['k', 'darkred'],
        'linestyle': ['-', '--', '-.', ':']})


@dataclass
class FittedsRdf2dHeatMapConfig(BaseHeatMapConfig):
    """
    Configuration parameters for heatmap plotting of fitted 2d rdf.
    """
    heatmap_suffix: str = \
        f'fittedRdf2dheatmap.{elsevier_plot_tools.IMG_FORMAT}'
    cbar_label: str = r'$g^\star_{fitted}(r^\star)$, a.u.'
    circles_configs: dict[str, list[str]] = field(default_factory=lambda: {
        'list': ['contact_radius', 'turn_points'],
        'color': ['k', 'darkred', 'g', 'b'],
        'linestyle': [':', '--', '-.', ':'],
        'turn_style': ['--', ':', '-.']})


@dataclass
class SmoothedRdf2dHeatMapConfig(BaseHeatMapConfig):
    """
    Configuration parameters for heatmap plotting of smoothed 2d rdf.
    """
    heatmap_suffix: str = \
        f'smoothedRdf2dheatmap.{elsevier_plot_tools.IMG_FORMAT}'
    cbar_label: str = r'$g^\star_{smoothed}(r^\star)$'
    circles_configs: dict[str, list[str]] = field(default_factory=lambda: {
        'list': ['contact_radius', 'np_radius'],
        'color': ['k', 'b'],
        'linestyle': ['-', '--', ':', ':']})


@dataclass
class BaseGraphConfig:
    """Configurations for the simple graphs"""
    # pylint: disable=too-many-instance-attributes
    graph_suffix: str
    graph_legend: str
    title: str
    ylabel: str
    xlabel: str = r'$r^\star$ [nm]'
    graph_style: dict = field(default_factory=lambda: {
        'legend': 'density',
        'color': 'k',
        'marker': 'o',
        'linestyle': '-',
        'markersize': 2,
        '2nd_marksize': 1
    })
    graph_2nd_legend: str = r'$g^\star(r^\star)$, a. u.'
    if_elsevier: bool = True
    show_oda_label: bool = True
    show_mirror_axis: bool = False


@dataclass
class DensityGraphConfig(BaseGraphConfig):
    """set the parameters for the graph"""
    graph_suffix: str = f'desnty.{elsevier_plot_tools.IMG_FORMAT}'
    graph_legend: str = 'density'
    ylabel: str = 'Average Density'
    title: str = 'ODA Density vs Distance from NP'


@dataclass
class Rdf2dGraphConfig(BaseGraphConfig):
    """set the parameters for the graph"""
    graph_suffix: str = f'rdf_2d.{elsevier_plot_tools.IMG_FORMAT}'
    graph_legend: str = r'$g^\star(r^\star)$'
    ylabel: str = r'$g^\star(r^\star)$, a. u.  '
    title: str = 'Rdf vs Distance from NP'


@dataclass
class FittedRdf2dGraphConfig(BaseGraphConfig):
    """set the parameters for the graph"""
    graph_suffix: str = f'fitted_rdf_2d.{elsevier_plot_tools.IMG_FORMAT}'
    graph_legend: str = r'$g^\star(r^\star)$'
    graph_2nd_legend: str = r'$g^\star_{fitted}(r^\star)$'
    ylabel: str = r'$g^\star(r^\star)$, a. u.'
    title: str = 'fitted Rdf vs Distance from NP'
    graph_style: dict = field(default_factory=lambda: {
        'legend': 'density',
        'color': 'k',
        'marker': 'o',
        'linestyle': '-',
        'markersize': 2,
        '2nd_marksize': 1
    })


@dataclass
class SmoothedRdf2dGraphConfig(BaseGraphConfig):
    """set the parameters for the graph"""
    graph_suffix: str = f'smoothed_rdf_2d.{elsevier_plot_tools.IMG_FORMAT}'
    graph_legend: str = r'$g^\star(r^\star)$'
    graph_2nd_legend: str = r'$g^\star_{smoothed}(r^\star)$'
    ylabel: str = r'$g^\star(r^\star)$'
    title: str = 'smoothed Rdf vs Distance from NP'


@dataclass
class GraphsConfigs:
    """all the graphs configurations"""
    graph_config:  "DensityGraphConfig" = \
        field(default_factory=DensityGraphConfig)
    rdf_config: "Rdf2dGraphConfig" = field(default_factory=Rdf2dGraphConfig)
    fitted_rdf_config: "FittedRdf2dGraphConfig" = \
        field(default_factory=FittedRdf2dGraphConfig)
    smoothed_rdf_config: "SmoothedRdf2dGraphConfig" = \
        field(default_factory=SmoothedRdf2dGraphConfig)
    if_elsevier: bool = True


class SurfactantDensityPlotter:
    """plot the desinty of the oda around the NP"""
    # pylint: disable=too-many-instance-attributes

    info_msg: str = 'Message from SurfactantDensityPlotter:\n'
    residue: str  # Name of the residue

    density: dict[float, list[float]]

    ave_density: dict[float, float]
    rdf_2d: dict[float, float]
    fitted_rdf: dict[float, float]
    smoothed_rdf: dict[float, float]

    time_dependent_rdf: dict[int, dict[float, float]]
    time_dependent_ave_density: dict[int, dict[float, float]]

    contact_data: pd.DataFrame
    box: np.ndarray  # Size of the box at each frame (from gromacs)

    angstrom_to_nm: float
    midpoint: float
    first_turn: float
    second_turn: float

    def __init__(self,
                 density_obj: "SurfactantDensityAroundNanoparticle",
                 log: logger.logging.Logger,
                 graphs_config: "GraphsConfigs" = GraphsConfigs(),
                 residue: str = 'ODA'
                 ) -> None:

        self.angstrom_to_nm = 0.1
        self.contact_data = density_obj.contact_data
        self.box = density_obj.box
        self.residue = residue

        self.density = density_obj.density_per_region
        self.ave_density = density_obj.avg_density_per_region
        self.rdf_2d = density_obj.rdf_2d
        self.smoothed_rdf = density_obj.smoothed_rdf

        if residue == 'AMINO_ODN':
            self.time_dependent_rdf = density_obj.time_dependent_rdf
            self.fitted_rdf = density_obj.fitted_rdf
            self.midpoint = density_obj.midpoint * self.angstrom_to_nm
            self.first_turn = density_obj.first_turn * self.angstrom_to_nm
            self.second_turn = density_obj.second_turn * self.angstrom_to_nm
            self.time_dependent_ave = density_obj.time_dependent_ave_density

        self.graph_configs = graphs_config

        self._initialize_plotting(log)
        self.write_msg(log)

    def _initialize_plotting(self,
                             log: logger.logging.Logger,
                             residue: str = 'AMINO_ODN'
                             ) -> None:
        HeatmapPlotter(ref_density=self.ave_density,
                       contact_data=self.contact_data,
                       config=DensityHeatMapConfig(),
                       log=log,
                       residue=self.residue)
        HeatmapPlotter(ref_density=self.rdf_2d,
                       contact_data=self.contact_data,
                       config=Rdf2dHeatMapConfig(),
                       log=log,
                       residue=self.residue)
        HeatmapPlotter(ref_density=self.smoothed_rdf,
                       contact_data=self.contact_data,
                       config=SmoothedRdf2dHeatMapConfig(),
                       log=log,
                       residue=self.residue)
        if residue == 'AMINO_ODN' and hasattr(self, 'fitted_rdf'):
            HeatmapPlotter(ref_density=self.fitted_rdf,
                           contact_data=self.contact_data,
                           config=FittedsRdf2dHeatMapConfig(),
                           log=log,
                           residue=self.residue,
                           fitted_turn_points={
                            'midpoint': self.midpoint,
                            '1st': self.first_turn,
                            '2nd': self.second_turn})
            self.plot_fitted_or_smoothed_rdf(
                self.fitted_rdf,
                'fitted',
                self.graph_configs.fitted_rdf_config)
            # TimeDependentPlotter(
            #    # self.time_dependent_rdf, self.time_dependent_ave, log)
        # DensityTimePlotter(density=self.density, log=log)
        self.plot_density_graph(self.graph_configs.graph_config)
        self.plot_2d_rdf(self.graph_configs.rdf_config)
        self.plot_2d_rdf_bpm(self.graph_configs.rdf_config)
        self.plot_2d_rdf_paper(self.graph_configs.rdf_config)
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

    def plot_2d_rdf_bpm(self,
                        config: "Rdf2dGraphConfig"
                        ) -> None:
        """
        Plot a simple graph of 2d rdf vs distance, for the presntation
        at the BPM.
        """
        # Save the current rcParams
        original_rc_params = plt.rcParams.copy()
        # Set the rcParams for the BPM
        # Set local font size for this method
        _config = config
        _config.if_elsevier = False
        _config.graph_style['markersize'] = 3
        _config.graph_style['linestyle'] = ':'
        _config.graph_style['color'] = 'k'
        fig_i, ax_i = self._plot_graphes(self.rdf_2d, _config, return_ax=True)
        golden_ratio: float = (1 + 5 ** 0.5) / 2
        hight: float = 2.35
        width: float = hight * golden_ratio
        fig_i.set_size_inches(width, hight)
        ax_i.set_xlabel(ax_i.get_xlabel(), fontsize=14)
        ax_i.set_ylabel(ax_i.get_ylabel(), fontsize=14)

        if self.residue == 'AMINO_ODN':
            ax_i.set_yticks([0.0, 0.5, 1.0])
        ax_i.tick_params(axis='x', labelsize=14)  # X-ticks
        ax_i.tick_params(axis='y', labelsize=14)  # Y-ticks

        contact_radius: float = \
            self.contact_data.loc[:, 'contact_radius'].mean() * \
            self.angstrom_to_nm
        self._add_vline(ax_i,
                        contact_radius,
                        lstyle='--',
                        color='darkred',
                        legend='r$^\star_c$')

        fout: str = f'{self.residue}_rdf_2d_bpm.png'
        plot_tools.save_close_fig(fig_i,
                                  ax_i,
                                  fout,
                                  loc='lower right',
                                  legend_font_size=13,
                                  )
        self.info_msg += \
            f'\tThe rdf graph saved: for bpm style as `{fout}`\n'
        # Restore the original rcParams
        plt.rcParams.update(original_rc_params)

    def plot_2d_rdf_paper(self,
                        config: "Rdf2dGraphConfig"
                        ) -> None:
        """
        Plot a simple graph of 2d rdf vs distance, for the presntation
        at the BPM.
        """
        # Save the current rcParams
        original_rc_params = plt.rcParams.copy()
        # Set local font size for this method
        _config = config
        _config.if_elsevier = True
        _config.graph_style['markersize'] = 2
        _config.graph_style['linestyle'] = ':'
        _config.graph_style['color'] = 'k'
        _config.graph_style['linewidth'] = 0.1
        _config.graph_legend = r'0.11 ODA/nm$^2$'
        r_debye: float = 5.23
        fig_i, ax_i = self._plot_graphes(self.rdf_2d, _config, return_ax=True)
        ax_i.set_xlabel(ax_i.get_xlabel(),
                        fontsize=elsevier_plot_tools.FONT_SIZE_PT)
        ax_i.set_ylabel(ax_i.get_ylabel(),
                        fontsize=elsevier_plot_tools.FONT_SIZE_PT)

        if self.residue == 'AMINO_ODN':
            ax_i.set_yticks([0.0, 0.5, 1.0])
        ax_i.tick_params(axis='x',
                         labelsize=elsevier_plot_tools.FONT_SIZE_PT)  # X-ticks
        ax_i.tick_params(axis='y',
                         labelsize=elsevier_plot_tools.FONT_SIZE_PT)  # Y-ticks

        contact_radius: float = \
            self.contact_data.loc[:, 'contact_radius'].mean() * \
            self.angstrom_to_nm
        self._add_vline(ax_i,
                        contact_radius,
                        lstyle='-',
                        color='gray',
                        legend='r$^\star_c$')
        self._add_vline(ax_i,
                        r_debye,
                        lstyle='--',
                        color='gray',
                        legend='r$^\star_d$')

        ax_i.grid(True, linestyle='--', color='gray', alpha=0.5)

        fout: str = f'{self.residue}_rdf_2d_paper.jpg'
        plot_tools.save_close_fig(fig_i,
                                  ax_i,
                                  fout,
                                  loc='upper left',
                                  )
        self.info_msg += \
            f'\tThe rdf graph saved: for bpm style as `{fout}`\n'
        # Restore the original rcParams
        plt.rcParams.update(original_rc_params)

    def plot_fitted_or_smoothed_rdf(self,
                                    rdf: dict[float, float],
                                    style: str,
                                    config: typing.Union[
                                        "FittedRdf2dGraphConfig",
                                        "SmoothedRdf2dGraphConfig"]
                                    ) -> None:
        """plot the fitted graph alongside main data"""
        fig_i, ax_i = self._plot_graphes(self.rdf_2d, config, return_ax=True)
        contact_radius: float = \
            self.contact_data.loc[:, 'contact_radius'].mean() * \
            self.angstrom_to_nm
        radii = np.array(list(rdf.keys())) * self.angstrom_to_nm
        densities = np.array(list(rdf.values()))
        ax_i.plot(radii,
                  densities,
                  linestyle=config.graph_style['linestyle'],
                  linewidth=1,
                  color='darkred',
                  label=config.graph_2nd_legend,
                  zorder=1)
        if style == 'fitted':
            ymax = ax_i.get_ylim()[1]
            ax_i = self._add_vline(ax_i,
                                   contact_radius,
                                   legend=r'r$^\star_c$',
                                   lstyle=':',
                                   color='k',
                                   y_cut=1.05)
            ax_i = self._add_vline(ax_i,
                                   self.first_turn,
                                   legend='a',
                                   lstyle=':',
                                   color='darkred',
                                   y_cut=ymax)
            ax_i = self._add_vline(ax_i,
                                   self.midpoint,
                                   legend='b',
                                   lstyle='--',
                                   color='darkred')
            ax_i = self._add_vline(ax_i,
                                   self.second_turn,
                                   legend='c',
                                   lstyle='-.',
                                   color='darkred')
            ax_i.text(-0.09,
                      1,
                      'a)',
                      ha='right',
                      va='top',
                      transform=ax_i.transAxes,
                      fontsize=elsevier_plot_tools.LABEL_FONT_SIZE_PT)
            yticks = [0, 0.5, 1.0]
            ax_i.set_yticks(yticks)
        if config.show_oda_label:
            if not config.show_mirror_axis:
                height: float = 1.0
            else:
                height = 0.98
            ax_i.text(0.28,
                      height,
                      r'0.03 ODA/nm$^2$',
                      ha='right',
                      va='top',
                      transform=ax_i.transAxes,
                      fontsize=elsevier_plot_tools.FONT_SIZE_PT)

        fout: str = f'{self.residue}_{config.graph_suffix}'
        if not config.show_mirror_axis:
            elsevier_plot_tools.remove_mirror_axes(ax_i)
        plot_tools.save_close_fig(fig_i, ax_i, fout, loc='lower right')
        self.info_msg += f'\tThe `{style}` graph saved: `{fout}`\n'

    @staticmethod
    def _add_vline(ax_i: plt.axes,
                   x_loc: float,
                   legend: str = 'b',
                   lstyle: str = '--',
                   color: str = 'k',
                   y_cut: typing.Union[float, None] = None
                   ) -> plt.axes:
        """add vline to the axes"""

        ylims: tuple[float, float] = ax_i.get_ylim()
        if y_cut is not None:
            ylims = (ylims[0], y_cut)
        ax_i.vlines(x=x_loc,
                    ymin=ylims[0],
                    ymax=ylims[1],
                    ls=lstyle,
                    lw=1,
                    color=color,
                    label=f'{legend}={x_loc:.2f}')
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
        radii = np.array(list(data.keys())) * self.angstrom_to_nm
        densities = np.array(list(data.values()))
        ax_i: plt.axes
        fig_i: plt.figure
        if config.if_elsevier:
            fig_i, ax_i = elsevier_plot_tools.mk_canvas(
                size_type='single_column')
            font_size: int = elsevier_plot_tools.FONT_SIZE_PT
        else:
            fig_i, ax_i = plot_tools.mk_canvas(
                (np.min(radii), np.max(radii)),
                height_ratio=(5 ** 0.5 - 1) * 1.5)
            font_size: int = 18
            ax_i.grid(True, linestyle='--', color='gray', alpha=0.5)
        if not config.if_elsevier:
            ax_i.plot(radii,
                      densities,
                      marker=config.graph_style['marker'],
                      linestyle=config.graph_style['linestyle'],
                      color=config.graph_style['color'],
                      label=config.graph_legend,
                      markersize=config.graph_style['markersize'])
        else:
            ax_i.scatter(radii,
                         densities,
                         color='k',
                         s=2,
                         label=config.graph_legend)
        ax_i.set_xlabel(config.xlabel, fontsize=font_size)
        ax_i.set_ylabel(config.ylabel, fontsize=font_size)
        if hasattr(config, 'if_title'):
            if config.title:
                ax_i.set_title(config.title)
        # Set grid for primary axis

        if not return_ax:
            fout: str = f'{self.residue}_{config.graph_suffix}'
            plot_tools.save_close_fig(
                fig_i, ax_i, fout, loc='upper left')
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
    angstrom_to_nm: float

    def __init__(self,
                 ref_density: dict[float, float],
                 contact_data: pd.DataFrame,
                 config: typing.Union["DensityHeatMapConfig",
                                      "Rdf2dHeatMapConfig",
                                      "FittedsRdf2dHeatMapConfig",
                                      "SmoothedRdf2dHeatMapConfig"],
                 log: logger.logging.Logger,
                 fitted_turn_points:
                 typing.Union[None, dict[str, float]] = None,
                 residue: str = 'ODA'
                 ) -> None:
        # pylint: disable=too-many-arguments
        self.angstrom_to_nm = 0.1
        self.residue = residue
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
        fout: str = f'{self.residue}_{self.config.heatmap_suffix}'
        self._save_close_fig(fig_i, ax_i, fout, legend=False)
        self.info_msg += f'\tThe heatmap of the density is saved as {fout}\n'

    def create_density_grid(self) -> tuple[np.ndarray, ...]:
        """Create a grid in polar coordinates with interpolated densities."""
        radii = np.array(list(self.ref_density.keys())) * self.angstrom_to_nm
        densities = np.array(list(self.ref_density.values()))
        # Create a grid in polar coordinates
        radial_distances, theta = \
            np.meshgrid(radii, np.linspace(0, 2 * np.pi, len(radii)))
        # Interpolate densities onto the grid
        density_grid = np.tile(densities, (len(radii), 1))
        return radial_distances, theta, density_grid

    def setup_plot(self) -> tuple[plt.figure, plt.axes]:
        """Set up the polar plot."""
        if self.config.if_elsevier:
            figsize = elsevier_plot_tools.set_figure_size('single_column')
        else:
            figsize = plot_tools.set_sizes(width=stinfo.plot['width'],
                                           height_ratio=(5 ** 0.5 - 1) * 1.5)
        fig_i, ax_i = plt.subplots(
            subplot_kw={'projection': 'polar'}, figsize=figsize)
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
        if self.config.if_arrow:
            ax_i = self._add_radius_arrows(ax_i, contact_radius, np_radius)
        cbar = plt.colorbar(cbar, ax=ax_i, shrink=1)
        cbar.set_ticks([0, 0.5, 1])
        cbar.ax.tick_params(labelsize=elsevier_plot_tools.FONT_SIZE_PT)
        cbar.set_label(label=self.config.cbar_label,
                       fontsize=elsevier_plot_tools.FONT_SIZE_PT-1)
        ax_i.text(-0.44860,
                  1,
                  'b)',
                  ha='right',
                  va='top',
                  transform=ax_i.transAxes,
                  fontsize=elsevier_plot_tools.LABEL_FONT_SIZE_PT)
        if self.config.show_oda_label:
            ax_i.text(0.08,
                      0.98,
                      r'0.03 ODA/nm$^2$',
                      ha='right',
                      va='top',
                      transform=ax_i.transAxes,
                      fontsize=elsevier_plot_tools.FONT_SIZE_PT)
        return ax_i

    def _add_np_radii(self,
                      ax_i: plt.axes
                      ) -> tuple[plt.axes, float, float]:
        """attach circle denoting the np"""
        np_radius: float = stinfo.np_info['radius'] * self.angstrom_to_nm
        contact_radius: float = self._get_avg_contact_raduis() * \
            self.angstrom_to_nm
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
                        color=self.config.circles_configs['color'][1],
                        line_style=self.config.circles_configs[
                            'turn_style'][i])

        return ax_i, contact_radius, np_radius

    def _add_radius_arrows(self,
                           ax_i: plt.axes,
                           contact_radius: float,
                           np_radius: float
                           ) -> plt.axes:
        """self explanatory"""
        self._add_polar_arrow(
            ax_i, length=contact_radius, theta=np.pi/2, color='darkred')
        ax_i = self._add_radii_label(
            ax_i,
            label=rf'$r_{{c, avg}}$={contact_radius:.2f}',
            location=(1, 1),
            color='darkred')

        if self.config.if_label_arrow:
            ax_i = \
                self._add_radii_label(
                    ax_i,
                    label=(rf'$r_{{c, avg}}$={contact_radius:.2f}'),
                    location=(1, 1),
                    color='darkred')

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
                         color: str = 'darkred'
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
                            color: str = 'darkred',
                            line_style: str = '--'
                            ) -> plt.axes:
        """add circle for representing the nanoparticle"""
        # pylint: disable=protected-access
        circle = Circle(origin,
                        radius,
                        transform=ax_i.transData._b,
                        color=color,
                        ls=line_style,
                        lw=0.75,
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
        # pylint: disable=too-many-arguments
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
    # pylint: disable=too-many-instance-attributes
    """configurations for the time dependents plots"""
    graph_suffix: str = f'ODA_rdf_2d_time.{elsevier_plot_tools.IMG_FORMAT}'
    graph_suffix2: str = \
        f'ODA_ave_density_time.{elsevier_plot_tools.IMG_FORMAT}'
    graph_legend: str = 'g(r,t)'
    graph_legend2: str = 'density(r,t)'
    ylabel: str = r'$g^\star(r^\star)$'
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
        # pylint: disable=unused-argument
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
        ax_i.set_xlabel(self.config.xlabel, fontsize=18)
        ax_i.set_ylabel(self.config.ylabel, fontsize=18)
        ax_i.set_title(self.config.title)
        plot_tools.save_close_fig(
            fig_i, ax_i, fname=self.config.graph_suffix, loc='lower right')

    def _plot_graph(self,
                    density: dict[int, dict[float, float]],
                    label_prefix: str
                    ) -> tuple[plt.figure, plt.axes]:
        """plot over time"""
        # pylint: disable=too-many-locals
        fig_i: plt.figure
        ax_i: plt.axes
        steps: list[int] = list(density.keys())
        steps_nr: int = len(steps)
        alpha_zero: float = 1/steps_nr
        radii = np.array(list(density[steps[-1]].keys()))
        fig_i, ax_i = \
            plot_tools.mk_canvas(x_range=(np.min(radii), np.max(radii)),
                                 height_ratio=(5 ** 0.5 - 1) * 1.5)
        for i, (_, frame) in enumerate(density.items(), start=1):
            densities = np.array(list(frame.values()))
            if i == steps_nr:
                color = 'darkred'
                alpha = 1.0
                label = label_prefix
                lwidth = 1.0
            else:
                color = 'k'
                alpha = 1.0-i*alpha_zero
                label = None
                lwidth = i*alpha_zero
                if i == 1:
                    label = f'{label_prefix}(t)'
            ax_i.plot(radii,
                      densities,
                      marker=self.config.graph_style['marker'],
                      linestyle=self.config.graph_style['linestyle'],
                      color=color,
                      label=label,
                      markersize=0,
                      alpha=alpha,
                      lw=lwidth)
        ax_i.grid(True, linestyle='--', color='gray', alpha=0.5)
        return fig_i, ax_i


if __name__ == "__main__":
    print(f'{bcolors.CAUTION}\tThis script runs within '
          f'trajectory_oda_analysis.py{bcolors.ENDC}')
