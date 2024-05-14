"""
Configure classes for ploting lab data
This plot was seperated, incase later need to to some analysis besides
plotting.
InputFiles:
    data_file: str
    The data is taken from ppt send by the lab

"""

import sys

from dataclasses import dataclass, field


@dataclass
class InputFiles:
    """Input files dataclass"""
    data_file: str = field(init=False)

    def __post_init__(self) -> None:
        """Post init function"""
        self.data_file = sys.argv[1]


@dataclass
class PlotConfig:
    """Plot configuration dataclass"""


@dataclass
class AnalysisConfig(PlotConfig, InputFiles):
    """Analysis configuration dataclass"""
