"""
Plot combination data
plot the selcted Boltzman factor for ODA ans the 2d RDF of the ODA at
the interface
"""
from typing import Dict, Union, Tuple, List
from dataclasses import field
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from common import logger
from common import xvg_to_dataframe
from common import elsevier_plot_tools
from common.colors_text import TextColor as bcolors


@dataclass
class FileConfig:
    """set all the configs and parameters
    """
    rdf_file: Dict[str, str | int] = field(default_factory=lambda: {
        'fname': '15_oda_densities.xvg',
        'column_rdf': 'fitted_rdf',
        })
    boltzman_file: Dict[str, str | int] = field(default_factory=lambda: {
        'fname': 'boltzman_distribution.xvg',
        'column_boltzman': 90,
        })


class PlotBolzmannRdfConfiguratio:
    """
    set all the configs and parameters and properties for the plot
    """
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=invalid-name
    @property
    def PLOT_PROPERTIES(self) -> Dict[str, Union[str, int, float]]:
        """set the plot properties"""
        return {'y_label': None,
                'xlabel': 'r (nm)',
                'title': 'Boltzman factor and RDF',
                'x_lims': [-0.5, 10.5],
                'y_lims': [-0.1, 1.1],
                'x_ticks': [0, 2, 4, 6, 8, 10],
                'y_ticks': [0, 0.5, 1],
                'legend_loc': 'upper right',
                }

    @property
    def RDF_LS(self) -> str:
        """set the line style for RDF"""
        return '-'

    @property
    def RDF_COLOR(self) -> str:
        """set the color for RDF"""
        return 'darkblue'

    @property
    def RDF_LABEL(self) -> str:
        """set the label for RDF"""
        return 'RDF'

    @property
    def BOLTZMAN_LS(self) -> str:
        """set the line style for Boltzman factor"""
        return ':'

    @property
    def BOLTZMAN_COLOR(self) -> str:
        """set the color for Boltzman factor"""
        return 'darkred'

    @property
    def BOLTZMAN_LABEL(self) -> str:
        """set the label for Boltzman factor"""
        return 'Boltzman factor'


class PlotBolzmannRdf:
    """
    Read the files and plot the data
    """
    __solts__ = ['radii', 'rdf_data', 'boltzman_data', 'config']
    radii: np.ndarray
    rdf_data: np.ndarray
    boltzman_data: np.ndarray
    config: FileConfig
