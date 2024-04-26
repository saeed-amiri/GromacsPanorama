"""
plotting the contact inforamtion of the nanoparticle with the interface
of the water and oil interface
inputs:
    A dict of xvg files containing the contact information of the
    nanoparticle.
    The contact_info file contins:
        "contact_radius"
        "contact_angles"
        "interface_z"
        "np_com_z"
    Output:
        A plot in png format
    The selection of the contact information is done by the user
26 April 2024
Saeed
"""

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
    xlabel: str = 'Time [ns]'
    ylabel: str = r'$\Delta X$'


@dataclass
class DataConfig:
    """set the name of the files and labels"""
    xvg_files: dict[str, str] = field(default_factory=lambda: {
        'contact_15Oda.xvg': '15Oda',
    })


class Selection(Enum):
    """Selection of the contact information"""
    CONTACT_RADIUS = 'contact_radius'
    CONTACT_ANGLES = 'contact_angles'
    INTERFACE_Z = 'interface_z'
    NP_COM_Z = 'np_com_z'


@dataclass
class AllConfig(BasePlotConfig, DataConfig):
    """All configurations"""
    selection: list[Selection] = field(default_factory=lambda: [
        Selection.CONTACT_RADIUS,
        Selection.CONTACT_ANGLES])
    output_file: str = 'np_contact_info.png'


class PlotNpContactInfo:
    """load the data and plot the contact information of the
    nanoparticle"""
    info_msg: str = 'Message from PlotNpContactInfo:\n'
    configs: AllConfig

    def __init__(self,
                 log: logger.logging.Logger,
                 configs: AllConfig = AllConfig()
                 ) -> None:
        self.configs = configs
        self.generate_plot(log)
        self.write_msg(log)

    def generate_plot(self,
                      log: logger.logging.Logger
                      ) -> None:
        """load data and generate the plot"""
        data: dict[str, pd.DataFrame] = self._load_data(log)
        self._log_avg_std(data)

    def _load_data(self,
                   log: logger.logging.Logger
                   ) -> dict[str, pd.DataFrame]:
        """load the data and retrun in a dict"""
        data: dict[str, pd.DataFrame] = {}
        for file, label in self.configs.xvg_files.items():
            data[label] = xvg_to_dataframe.XvgParser(file, log).xvg_df
        return data

    def _log_avg_std(self,
                     data: dict[str, pd.DataFrame]
                     ) -> None:
        """log the average and std of the data"""
        for label, df_i in data.items():
            self.info_msg += f'{label}:\n'
            for selection in self.configs.selection:
                self._log_avg_std_label(selection.value, df_i[selection.value])

    def _log_avg_std_label(self,
                           selection: str,
                           df_i: pd.DataFrame
                           ) -> None:
        self.info_msg += (f'\t{selection}:\n'
                          f'\t\tAverage: {df_i.mean():.3f}\n'
                          f'\t\tStd: {df_i.std():.3f}\n')

    def write_msg(self,
                  log: logger.logging.Logger
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    PlotNpContactInfo(logger.setup_logger('np_contact_info.log'))
