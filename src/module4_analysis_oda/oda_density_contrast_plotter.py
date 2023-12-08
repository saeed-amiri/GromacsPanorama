"""
plot the density contrast of the ODA from SurfactantsLocalizedDensityContrast.
"""

import typing
from dataclasses import dataclass

import matplotlib.pylab as plt

from common import logger
from common import plot_tools
from common.colors_text import TextColor as bcolors

if typing.TYPE_CHECKING:
    from module4_analysis_oda.oda_density_contrast \
        import SurfactantsLocalizedDensityContrast, \
        InputFilesConfig, \
        NumberDensity


@dataclass
class BaseConfig:
    """base configur for plotting"""
    fig_suffix: str = 'contrast.png'
    height_ratio: float = (5**0.5-1)*1.8


@dataclass
class GraphConfing(BaseConfig):
    """configuration for the class"""


class SurfactantContrastDensityPlotter:
    """plot the density contrast"""

    info_msg: str = 'Messege from SurfactantContrastDensityPlotter:\n'

    number_density: "NumberDensity"
    fig_config: "GraphConfing"

    def __init__(self,
                 number_density: "NumberDensity",
                 log: logger.logging.Logger,
                 fig_config: "GraphConfing" = GraphConfing()
                 ) -> None:
        self.number_density = number_density
        self.fig_config = fig_config
        self._initialize_plotting()
        self.write_msg(log)

    def _initialize_plotting(self) -> None:
        """start plotting"""
        self.plot_contrast_graph()

    def plot_contrast_graph(self) -> None:
        """plot the contrast graphs"""
        xrange: tuple[float, float] = \
            (0, self.number_density.nr_oda_in_zone.shape[0])
        ax_i: plt.axes
        fig_i: plt.figure
        fig_i, ax_i = plot_tools.mk_canvas(
            xrange, height_ratio=self.fig_config.height_ratio)
        ax_i.plot(self.number_density.dens_oda_out_zone, label='out')
        ax_i.plot(self.number_density.dens_oda_in_zone, label='in')
        plt.legend()
        plt.show()

    def write_msg(self,
                  log: logger.logging.Logger  # To log
                  ) -> None:
        """write and log messages"""
        message: str = f'{self.__class__.__name__}:\n\t{self.info_msg}'
        print(f'{bcolors.OKGREEN}{message}{bcolors.ENDC}')
        log.info(message)
