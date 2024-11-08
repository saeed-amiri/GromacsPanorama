"""
To plot the average Boltzman distribution for different ODA concentrations,
and compare them together.
The same layers which are plotted in the rdf_boltzmann plots should be
used here.
"""

from dataclasses import field
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import logger
from common import plot_tools
from common import xvg_to_dataframe
from common import elsevier_plot_tools
from common.colors_text import TextColor as bcolors


@dataclass
class FileConfig:
    """set the names of the files and the layers to get average from.
    defines dict of info for files.
    """
    boltzmann_files: list[dict[str, str | int | list[int]]] = \
        field(default_factory=[
            {'fname': '5_boltzman_distribution.xvg',
             'nr_oda': 5,
             'layers': [90, 91, 92, 93],
             },
            {'fname': '10_boltzman_distribution.xvg',
             'nr_oda': 10,
             'layers': [90, 91, 92, 93]
             },
            ])


@dataclass
class PlotConfig:
    """Set the plot configurations"""
    title: str = 'Avg. Boltzman distribution for different ODA concentrations'
    xlabel: str = r'$r^*$ [nm]'
    ylabel: str = 'a.u.'
    legend: str = r'ODA/nm$^2$'
    fig_name: str = 'average_boltzman_distribution.png'
    save_fig: bool = True
