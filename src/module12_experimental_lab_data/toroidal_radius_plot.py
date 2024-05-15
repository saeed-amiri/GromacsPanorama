"""
Ploting the toroidal radius of the exlusion zone based on the
concertration of salt and surfactant.
The figure is two panels, one for the salt and the other for the surfactant
But, here we add an extra panel for the toroidal radius scheme.
The data is read from the 'data.xvg' file and plotted.
"""

import pandas as pd

import matplotlib.pyplot as plt

from module12_experimental_lab_data.config_classes import AllConfig
from common.colors_text import TextColor as bcolors
from common import logger, elsevier_plot_tools


class ToroidalRadiusPlot:
    """plot the toroidal radius"""

    info_msg: str = 'Message from ToroidalRadiusPlot:\n'
    data: pd.DataFrame

    def __init__(self,
                 log: logger.logging.Logger,
                 data: pd.DataFrame,
                 config: AllConfig = AllConfig()
                 ) -> None:
        self.config = config
        self.write_msg(log)

    def write_msg(self,
                  log: logger.logging.Logger
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{ToroidalRadiusPlot.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    print(f'\n{bcolors.WARNING}This script is not meant to be run '
          f'directly, it called by `plot_lab_data.py`{bcolors.ENDC}\n')
