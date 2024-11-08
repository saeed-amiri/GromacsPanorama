"""
To plot the average Boltzman distribution for different ODA concentrations,
and compare them together.
The same layers which are plotted in the rdf_boltzmann plots should be
used here.
"""

import sys
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
        field(default_factory=lambda: [
            {'fname': '5_boltzmann_distribution.xvg',
             'nr_oda': 5,
             'layers': [90, 91, 92, 93],
             },
            {'fname': '10_boltzmann_distribution.xvg',
             'nr_oda': 10,
             'layers': [90, 91, 92, 93]
             },
            ])


@dataclass
class PlotConfig:
    """Set the plot configurations"""
    title: str = 'Avg. boltzmann distribution for different ODA concentrations'
    xlabel: str = r'$r^*$ [nm]'
    ylabel: str = 'a.u.'
    legend: str = r'ODA/nm$^2$'
    fig_name: str = 'average_boltzmann_distribution.png'
    save_fig: bool = True


class AverageBoltzmanPlot:
    """plot the comparing graph"""

    __solts__ = ['file_config',
                 'plot_config'
                 'info_msg',
                 ]
    file_config: "FileConfig"
    plot_config: "PlotConfig"
    info_msg: str

    def __init__(self,
                 log: logger.logging.Logger,
                 file_config: "FileConfig" = FileConfig(),
                 plot_config: "PlotConfig" = PlotConfig(),
                 ) -> None:
        self.info_msg = 'Message from AverageBoltzmanPlot:\n'
        self.file_config = file_config
        self.plot_config = plot_config
        data: dict[str, pd.DataFrame]
        radii: np.ndarray
        data, radii = self.process_files(log)
        avg_data: dict[str, np.ndarray] = self.get_average_boltzmann(data)

    def process_files(self,
                      log: logger.logging.Logger
                      ) -> tuple[dict[str, pd.DataFrame], np.ndarray]:
        """process the files"""
        data: dict[str, pd.DataFrame] = {}
        radii: np.ndarray = np.array([])
        for i, file in enumerate(self.file_config.boltzmann_files):
            try:
                df_i: pd.DataFrame = \
                    xvg_to_dataframe.XvgParser(file['fname'], log).xvg_df
                df_i.columns = df_i.columns.astype(str)
                data[str(file['nr_oda'])] = \
                    df_i[[str(layer) for layer in file['layers']]]
                if i == 0:
                    radii: np.ndarray = np.asanyarray(df_i['r_nm'])
            except Exception as e:
                log.error(msg := (f'{bcolors.FAIL}\tError:{bcolors.ENDC} '
                                  f'{e}\n'
                                  f'Could not process file {file["fname"]}\n'))
                self.info_msg += msg
                if i == 0:
                    sys.exit(1)
                continue
        return data, radii

    def get_average_boltzmann(self,
                              data: dict[str, pd.DataFrame]
                              ) -> dict[str, np.ndarray]:
        """get the average boltzmann distribution"""
        avg_data: dict[str, np.ndarray] = {}
        for key, df in data.items():
            avg_data[key] = np.asanyarray(df.iloc[1:].mean(axis=1))
        return avg_data


if __name__ == '__main__':
    AverageBoltzmanPlot(log=logger.setup_logger('compare_boltzmann_plot.log'))
