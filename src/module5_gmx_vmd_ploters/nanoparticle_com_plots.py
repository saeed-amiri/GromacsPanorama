"""
plotting the behaviour of the nanoparticle center of mass (COM) in the
any of the three dimensions (x, y, z) as a function of time.
inputs:
    A dict of xvg files containing the COM of the nanoparticle in the
    x, y, and z dimensions. The name of the files should contains the
    number of the ODA in system, e.g., 'coord_15Oda.xvg'.
    Output:
        A plot with the COM of the nanoparticle in the selected dimension
        as a function of time.
    Since the location may differs, the initial position of the COM is
    set to zero, hense, the ylabel is the distance from the initial.
26 April 2024
Saeed
"""

import typing
from enum import Enum
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import logger, xvg_to_dataframe, elsevier_plot_tools
from common.colors_text import TextColor as bcolors


@dataclass
class BasePlotConfig:
    """
    Base class for the graph configuration
    """
    linewidth: float = 1.0
    linecolotrs: list[str] = field(default_factory=lambda: [
        'black', 'blue', 'green', 'red'])
    line_styles: list[str] = field(default_factory=lambda: [
        '-', ':', '--', '-.'])
    xlabel: str = 'Time (ns)'
    ylabel: str = r'$\Delta$' + ' '  # The axis will be add later


@dataclass
class DataConfig:
    """set the name of the files and labels"""
    xvg_files: dict[str, str] = field(default_factory=lambda: {
        'coord_15Oda.xvg': '15Oda',
        'coord_200Oda.xvg': '200Oda',
    })


class Direction(Enum):
    """Direction of the nanoparticle COM
    Since in the data files first dolumn is the time, the direction
    labeled from 1.
    """
    X = 1
    Y = 2
    Z = 3

@dataclass
class AllConfig(BasePlotConfig, DataConfig):
    """All configurations"""
    direction: Direction = Direction.X
    output_file: str = 'np_com.png'

    def __post_init__(self) -> None:
        """Post init function"""
        self.ylabel += f'{self.direction.name} [nm]'


if __name__ == '__main__':
    pass
