"""
Plot the potential of the average of the layers of the box.
The potential in each grid of the system that computed by APBS.
"""

import typing

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from common import logger
from common import elsevier_plot_tools
from common.colors_text import TextColor as bcolors


class PlotPotentialLayerConfig:
    """set the name of the input files"""


class PlotPotentialLayer:
    """
    Plot the potential of the average of the layers of the box.
    The potential in each grid of the system that computed by APBS.
    """
    info_msg: str = 'Message from PlotPotentialLayer:\n'
    configs: PlotPotentialLayerConfig

    def __init__(self,
                 cut_radii: list[np.ndarray],
                 cut_radial_average: list[np.ndarray],
                 radii_list: list[np.ndarray],
                 radial_average_list: list[np.ndarray],
                 sphere_grid_range: np.ndarray,
                 log: logger.logging.Logger,
                 configs: PlotPotentialLayerConfig = PlotPotentialLayerConfig()
                 ) -> None:
        """write and log messages"""


if __name__ == '__main__':
    print(f'{bcolors.WARNING}\n\tThis script is not intended to be run '
          f'independently.{bcolors.ENDC}\n')
