"""
plots the 2d rdf with the Boltzmann distribution for each of the oda
concentrations
There are other scripts which are doing somehow the same, but I want
all of them to be in one place
"""

import numpy as np
import pandas as pd

from common import logger
from common import xvg_to_dataframe

from module10_rdf_analysis.config import StatisticsConfig
from module10_rdf_analysis.statistic_analysis_2drdf_plots import PlotStatistics


class Rdf2dWithBoltzmann:
    """
    read and plot the 2d rdf with the Boltzmann distribution
    """

    __slots__ = ['config', 'info_msg']

    def __init__(self,
                 rdf_x: pd.DataFrame,
                 rdf_data: pd.DataFrame,
                 rdf_fit_data: pd.DataFrame,
                 log: logger.logging.Logger,
                 config: StatisticsConfig
                 ) -> None:
        self.info_msg: str = "Message from Rdf2dWithBoltzmann:\n"
        self.config = config
        boltzmann_data: dict[str, pd.DataFrame] = \
            self.process_boltzmann_data(log)
        self.plot_data(rdf_x, rdf_data, rdf_fit_data, boltzmann_data, log)

    def process_boltzmann_data(self,
                               log: logger.logging.Logger
                               ) -> dict[str, pd.DataFrame]:
        """
        Read the boltzmann distribution
        """
        boltz_cfg: StatisticsConfig = self.config.files.boltzmann

        # Read the all boltzmann data
        boltzmann_data: tuple[dict[str, pd.Series],
                              dict[str, pd.DataFrame]] = \
            self.extract_boltzmann_data(log, boltz_cfg)
        boltzmann_x_dict, boltzmann_y_dict = boltzmann_data

        # Truncate the Boltzmann distribution
        truct_boltzmann_data: tuple[dict[str, pd.Series],
                                    dict[str, pd.DataFrame]] = \
            self.truncate_boltzmann_x(
                boltzmann_x_dict, boltzmann_y_dict, boltz_cfg)
        truct_boltzmann_x_dict, truct_boltzmann_y_dict = truct_boltzmann_data

        # Calculate the average Boltzmann distribution
        boltzmann_y_mean_dict: dict[str, pd.Series] = \
            self.calculate_average_boltzmann(truct_boltzmann_y_dict)

        # Create a new DataFrame with the Boltzmann distribution
        processed_boltzmann_data: dict[str, pd.DataFrame] = \
            self.create_boltzmann_dataframe(
                truct_boltzmann_x_dict, boltzmann_y_mean_dict)

        return processed_boltzmann_data

    def plot_data(self,
                  rdf_x: pd.DataFrame,
                  rdf_data: pd.DataFrame,
                  rdf_fit_data: pd.DataFrame,
                  boltzmann_data: dict[str, pd.DataFrame],
                  log: logger.logging.Logger
                  ) -> None:
        """
        Plot the 2d rdf with the Boltzmann distribution
        """

    @staticmethod
    def extract_boltzmann_data(log: logger.logging.Logger,
                               boltz_cfg: StatisticsConfig
                               ) -> tuple[dict[str, pd.Series],
                                          dict[str, pd.DataFrame]]:
        """
        Read the 2d rdf data
        """
        boltzmann_x_dict: dict[str, pd.Series] = {}
        boltzmann_y_dict: dict[str, pd.DataFrame] = {}
        for oda, fname in boltz_cfg.file_names.items():
            df_i: pd.DataFrame = \
             xvg_to_dataframe.XvgParser(fname, log, x_type=float).xvg_df
            boltzmann_x: pd.Series = df_i[boltz_cfg.xdata]
            boltzmann_y: pd.DataFrame = \
                df_i[list(map(str, boltz_cfg.ydata[oda]))]
            boltzmann_x_dict[oda] = boltzmann_x
            boltzmann_y_dict[oda] = boltzmann_y
        return boltzmann_x_dict, boltzmann_y_dict

    @staticmethod
    def truncate_boltzmann_x(boltzmann_x_dict: dict[str, pd.Series],
                             boltzmann_y_dict: dict[str, pd.DataFrame],
                             boltz_cfg: StatisticsConfig
                             ) -> tuple[dict[str, pd.Series],
                                        dict[str, pd.DataFrame]]:
        """
        Cut the Boltzmann distribution
        """
        cut_boltzmann_x_dict: dict[str, pd.Series] = {}
        cut_boltzmann_y_dict: dict[str, pd.DataFrame] = {}
        for oda, boltzmann_x in boltzmann_x_dict.items():
            cut_boltzmann_x: pd.Series = boltzmann_x[
                boltzmann_x <= boltz_cfg.r_cut]
            cut_boltzmann_x_dict[oda] = cut_boltzmann_x
            cut_boltzmann_y: pd.DataFrame = boltzmann_y_dict[oda][
                boltzmann_x <= boltz_cfg.r_cut]
            cut_boltzmann_y_dict[oda] = cut_boltzmann_y
        return cut_boltzmann_x_dict, cut_boltzmann_y_dict

    @staticmethod
    def calculate_average_boltzmann(boltzmann_y_dict: dict[str, pd.DataFrame],
                                    ) -> dict[str, pd.Series]:
        """
        Average the Boltzmann distribution
        """
        boltzmann_y_mean_dict: dict[str, pd.Series] = {}
        for oda, boltzmann_y in boltzmann_y_dict.items():
            boltzmann_y_mean: pd.Series = boltzmann_y.mean(axis=1)
            boltzmann_y_mean_dict[oda] = boltzmann_y_mean
            print(len(boltzmann_y_mean))
        return boltzmann_y_mean_dict

    @staticmethod
    def create_boltzmann_dataframe(cut_boltzmann_x_dict: dict[str, pd.Series],
                                   boltzmann_y_mean_dict: dict[str, pd.Series],
                                   ) -> dict[str, pd.DataFrame]:
        """
        make a new DataFrame with the Boltzmann distribution for each
        of the oda concentrations
        """
        boltzmann_data: dict[str, pd.DataFrame] = {}
        for oda, boltzmann_x in cut_boltzmann_x_dict.items():
            boltzmann_data[oda] = pd.concat(
                [boltzmann_x, boltzmann_y_mean_dict[oda]], axis=1)
        return boltzmann_data
