"""plots the densities of a residue around the nanoparticle calculated
from densities_around_np.py
"""

import typing
from dataclasses import dataclass, field

from common import logger

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
    graph_suffix: str = 'rdf.png'
    graph_legend: str = 'rdf'
    ylabel: str = 'g(r)'
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

    denisties: "Densities"
    plot_configs: "AllGraphConfigs"

    def __init__(self,
                 denisties: "Densities",
                 log: logger.logging.Logger,
                 plot_configs: "AllGraphConfigs" = AllGraphConfigs()
                 ) -> None:
        self.densities = denisties
        self.plot_configs = plot_configs


if __name__ == "__main__":
    print("This script is called within charge_analysis_interface_np.py")
