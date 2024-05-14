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
        'black', 'royalblue', 'darkgreen', 'darkred'])
    line_styles: list[str] = field(default_factory=lambda: [
        ':', '-', '--', '-.'])
    xlabel: str = 'Time [ns]'
    ylabel: str = r'$\Delta$' + ' '  # The axis will be add later
    show_mirror_axis: bool = False


@dataclass
class DataConfig:
    """set the name of the files and labels"""
    xvg_files: dict[str, str] = field(default_factory=lambda: {
        'coord_15Oda.xvg': r'0.03 ODA/nm$^2$',
        'coord_200Oda.xvg': r'0.42 ODA/nm$^2$',
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
    direction: Direction = Direction.Z
    output_file: str = f'np_com.{elsevier_plot_tools.IMG_FORMAT}'
    if_multi_label: bool = True

    def __post_init__(self) -> None:
        """Post init function"""
        self.ylabel += f'{self.direction.name.lower()} [nm]'


class PlotNpCom:
    """Plot the nanoparticle COM in the selected dimension"""

    info_msg: str = 'Message from PlotNpCom:\n'
    configs: AllConfig

    def __init__(self,
                 log: logger.logging.Logger,
                 configs: AllConfig = AllConfig()
                 ) -> None:
        self.configs = configs
        self.generate_com_plot(log)
        self.write_msg(log)

    def generate_com_plot(self,
                          log: logger.logging.Logger
                          ) -> None:
        """Plot the nanoparticle COM"""
        time: np.ndarray
        data: dict[str, np.ndarray]
        data, time = self._load_data(log)
        self._log_avg_std(data)
        self._plot_com(data, time)

    def _load_data(self,
                   log: logger.logging.Logger
                   ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Load the data"""
        data: dict[str, np.ndarray] = {}
        for i, (file, label) in enumerate(self.configs.xvg_files.items()):
            df_i: pd.DataFrame = xvg_to_dataframe.XvgParser(file, log).xvg_df
            if i == 0:
                time: np.ndarray = df_i.iloc[:, 0].values / 1000.0
            data[label] = \
                df_i.iloc[:, self.configs.direction.value].values
        return data, time

    def _plot_com(self,
                  data: dict[str, np.ndarray],
                  time: np.ndarray
                  ) -> None:
        """Plot the nanoparticle COM"""
        fig_i: plt.Figure
        ax_i: plt.Axes

        fig_i, ax_i = elsevier_plot_tools.mk_canvas(size_type='single_column')
        for i, (label, data_i) in enumerate(data.items()):
            ax_i.plot(time,
                      data_i - data_i[0],
                      label=label,
                      linewidth=self.configs.linewidth,
                      color=self.configs.linecolotrs[i],
                      linestyle=self.configs.line_styles[i])

        ax_i.set_xlabel(self.configs.xlabel)
        ax_i.set_ylabel(self.configs.ylabel)
        self._add_multi_label(ax_i)

        if not self.configs.show_mirror_axis:
            elsevier_plot_tools.remove_mirror_axes(ax_i)
        elsevier_plot_tools.save_close_fig(fig_i,
                                           fname  := self.configs.output_file,
                                           loc='lower left',
                                           horizontal_legend=True)
        self.info_msg += f'The plot is saved as `{fname}`\n'

    def _add_multi_label(self,
                         ax_i: plt.Axes
                         ) -> None:
        """Add a label to the plot"""
        if self.configs.if_multi_label:
            ax_i.text(-0.085,
                      1,
                      'a)',
                      ha='right',
                      va='top',
                      transform=ax_i.transAxes,
                      fontsize=elsevier_plot_tools.LABEL_FONT_SIZE_PT)

    def _log_avg_std(self,
                     data: dict[str, np.ndarray]
                     ) -> None:
        """Log the average and standard deviation"""
        for label, data_i in data.items():
            self.info_msg += (f'\t{label}:\n'
                              f'\t\tAverage: {np.mean(data_i):.2f}\n'
                              f'\t\tStandard Deviation: {np.std(data_i):.2f}\n'
                              )

    def write_msg(self,
                  log: logger.logging.Logger
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    PlotNpCom(log=logger.setup_logger(log_name='np_com.log'))
