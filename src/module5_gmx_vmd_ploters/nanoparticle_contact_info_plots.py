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
        'red', 'green', 'black', 'blue'])
    line_styles: list[str] = field(default_factory=lambda: [
        '-', '--', ':', '-.'])
    line_labels: list[str] = field(init=False)
    xlabel: str = 'Time [ns]'
    ylabel: str = 'X'


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


class LineLabels(Enum):
    """Line labels"""
    CONTACT_RADIUS = r'$r^\star$ [nm]'
    CONTACT_ANGLES = 'CA [deg]'
    INTERFACE_Z = 'Interface Z'
    NP_COM_Z = 'NP COM Z'


@dataclass
class AllConfig(BasePlotConfig, DataConfig):
    """All configurations"""
    selection: list[Selection] = field(default_factory=lambda: [
        Selection.CONTACT_RADIUS,
        Selection.CONTACT_ANGLES])
    if_multi_label: bool = True
    output_file: str = 'np_contact_info.png'

    def __post_init__(self) -> None:
        """Post init function"""
        selcted: list[str] = [sel.name for sel in self.selection]
        self.line_labels = [LineLabels[sel].value for sel in selcted]


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
        self._plot_data(data)

    def _plot_data(self,
                   data: dict[str, pd.DataFrame]
                   ) -> None:
        """plot the data"""
        fig_i: plt.Figure
        ax_i: plt.Axes
        fig_i, ax_i = elsevier_plot_tools.mk_canvas(size_type='single_column')
        for i, selection in enumerate(self.configs.selection):
            self._plot_data_label(selection.value, data, ax_i, i)
        self._add_multi_label(ax_i)
        ax_i.set_xlabel(self.configs.xlabel)
        ax_i.set_ylabel(self.configs.ylabel)
        elsevier_plot_tools.save_close_fig(
            fig_i, fname := self.configs.output_file, loc='lower left')
        self.info_msg += f'The plot is saved to {fname}\n'

    def _plot_data_label(self,
                         selection: str,
                         data: dict[str, pd.DataFrame],
                         ax_i: plt.Axes,
                         i: int
                         ) -> None:
        for _, df_i in data.items():
            ax_i.plot(df_i.iloc[:, 0] / 10.0,  # Convert to nm
                      df_i[selection],
                      label=self.configs.line_labels[i],
                      linestyle=self.configs.line_styles[i],
                      color=self.configs.linecolotrs[i],
                      linewidth=self.configs.linewidth)

    def _add_multi_label(self,
                         ax_i: plt.Axes
                         ) -> None:
        """Add a label to the plot"""
        if self.configs.if_multi_label:
            ax_i.text(-0.075,
                      1,
                      'b)',
                      ha='right',
                      va='top',
                      transform=ax_i.transAxes,
                      fontsize=elsevier_plot_tools.LABEL_FONT_SIZE_PT)

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
