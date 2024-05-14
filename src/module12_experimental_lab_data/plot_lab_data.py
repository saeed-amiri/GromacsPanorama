"""
plotting the lab data in the elsevier style, similar to the other plots
from the simulation data
Data is read from 'data.xvg' file and plotted
Maybe I'll try with multicolumns plots
The first figure:
    It's 2 by 2 plots, on y axis is the contact angle and coverages and
    on the x axis is the concentration of the surfactant (ODA)
    and each panel is for different salt concentration
    for this grpah I use the style of the fig.4 and 5 from the paper:
    https://doi.org/10.1021/acs.langmuir.1c00559
    from M.M.

"""

import pandas as pd

from module12_experimental_lab_data.plot_ca_coverage import PlotCaCoverage
from module12_experimental_lab_data.config_classes import AllConfig
from module12_experimental_lab_data.read_data import ReadData
from common.colors_text import TextColor as bcolors
from common import logger


class PlotLabData:
    """plot the several plots here"""

    info_msg: str = 'Message from PlotLabData:\n'
    data: pd.DataFrame

    def __init__(self,
                 log: logger.logging.Logger,
                 config: AllConfig = AllConfig()
                 ) -> None:
        self.config = config
        data = ReadData(log, self.config).data
        self.plot_data(log, data)
        self.write_msg(log)

    def plot_data(self,
                  log: logger.logging.Logger,
                  data: pd.DataFrame
                  ) -> None:
        """plot the data"""
        PlotCaCoverage(log, data, self.config)

    def write_msg(self,
                  log: logger.logging.Logger
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{PlotLabData.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    # Read the input file
    PlotLabData(logger.setup_logger('plot_lab_data.log'))
