"""plots the densities of a residue around the nanoparticle calculated
from densities_around_np.py
"""

import typing
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pylab as plt

from common import logger
from common import plot_tools
from common.colors_text import TextColor as bcolors

if typing.TYPE_CHECKING:
    from module6_charge_analysis.densities_around_np import Densities


@dataclass
class BaseGraphConfig:
    """Basic configurations for plot and saving"""
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
class RdfGraphConfigs(BaseGraphConfig):
    """update the base for rdf plotting"""
    graph_suffix: str = 'rdf.png'
    graph_legend: str = 'rdf'
    ylabel: str = 'g(r)'
    title: str = 'Rdf vs Distance from NP'


@dataclass
class AvgDensityGraphConfigs(BaseGraphConfig):
    """update the base for rdf plotting"""
    graph_suffix: str = 'avg_density.png'
    graph_legend: str = 'average density'
    ylabel: str = 'density(r)'
    title: str = 'Rdf vs Distance from NP'


@dataclass
class AllGraphConfigs:
    """set all the graphes configurations"""
    rdf_configs: "RdfGraphConfigs" = RdfGraphConfigs()
    avg_configs: "AvgDensityGraphConfigs" = AvgDensityGraphConfigs()


class ResidueDensityPlotter:
    """plot the denisty related properties calculated for the residue
    around the nanoparticle"""

    info_msg: str = 'Messages from ResidueDensityPlotter:\n'

    densities: "Densities"
    plot_configs: "AllGraphConfigs"
    res_name: str

    def __init__(self,
                 densities: "Densities",
                 log: logger.logging.Logger,
                 plot_configs: "AllGraphConfigs" = AllGraphConfigs()
                 ) -> None:
        self.densities = densities
        self.res_name = densities.res_name
        self.plot_configs = plot_configs
        self._initialize_plotting()
        self.write_msg(log)

    def _initialize_plotting(self) -> None:
        """initialize plotting for each density"""
        self.plot_avg_density()
        self.plot_rdf()

    def plot_avg_density(self) -> None:
        """plot the average density"""
        self._plot_graphes(self.densities.avg_density_per_region,
                           self.plot_configs.avg_configs)

    def plot_rdf(self) -> None:
        """plot the rdf"""
        self._plot_graphes(self.densities.rdf, self.plot_configs.rdf_configs)

    def _plot_graphes(self,
                      data: dict[float, float],
                      configs: typing.Union["RdfGraphConfigs",
                                            "AvgDensityGraphConfigs"],
                      return_ax: bool = False
                      ) -> tuple[plt.figure, plt.axes]:
        """plot graphs"""
        radii = np.array(list(data.keys()))
        dens_values = np.array(list(data.values()))
        ax_i: plt.axes
        fig_i: plt.figure
        fig_i, ax_i = plot_tools.mk_canvas((np.min(radii), np.max(radii)),
                                           height_ratio=5**0.5-1)
        ax_i.plot(radii,
                  dens_values,
                  marker=configs.graph_style['marker'],
                  linestyle=configs.graph_style['linestyle'],
                  color=configs.graph_style['color'],
                  label=configs.graph_legend,
                  markersize=configs.graph_style['markersize'])
        ax_i.set_xlabel(configs.xlabel)
        ax_i.set_ylabel(configs.ylabel)
        ax_i.set_title(configs.title)
        # Set grid for primary axis
        ax_i.grid(True, linestyle='--', color='gray', alpha=0.5)

        if not return_ax:
            fout: str = f'{self.res_name}_{configs.graph_suffix}'
            plot_tools.save_close_fig(
                fig_i, ax_i, fout, loc='upper left')
            self.info_msg += \
                f'\tThe density graph saved: `{configs.graph_suffix}`\n'
        return fig_i, ax_i

    def write_msg(self,
                  log: logger.logging.Logger  # To log
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKGREEN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)

if __name__ == "__main__":
    print("This script is called within charge_analysis_interface_np.py")
