"""
plot the density of the ODA from SurfactantDensityAroundNanoparticle.
"""

import typing
from dataclasses import dataclass

from common import logger
from common.colors_text import TextColor as bcolors

if typing.TYPE_CHECKING:
    from module4_analysis_oda.oda_density_around_np \
        import SurfactantDensityAroundNanoparticle


@dataclass
class PlotConfig:
    """set the parameters for the the plot"""
    heatmap_suffix: str = 'heatmap.png'
    graph_suffix: str = 'desnty.png'


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


if __name__ == "__main__":
    print(f'{bcolors.CAUTION}\tThis script runs within '
          f'trajectory_oda_analysis.py{bcolors.ENDC}')
