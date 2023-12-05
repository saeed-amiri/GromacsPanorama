"""must be updated
for PRE
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

import common.logger as logger
import common.static_info as stinfo
import common.xvg_to_dataframe as xvg
import common.plot_tools as plot_tools
from common.colors_text import TextColor as bcolors


class PlotXvg(xvg.XvgParser):
    """plot xvg here"""
    def __init__(self,
                 fname: str,
                 log: logger.logging.Logger
                 ) -> None:
        super().__init__(fname, log)


if __name__ == "__main__":
    PlotXvg(sys.argv[1], log=logger.setup_logger('plot_xvg.log'))
