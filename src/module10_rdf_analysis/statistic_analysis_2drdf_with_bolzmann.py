"""
plots the 2d rdf with the Boltzmann distribution for each of the oda
concentrations
There are other scripts which are doing somehow the same, but I want
all of them to be in one place
"""

import pandas as pd

from common import logger
from common import xvg_to_dataframe

from module10_rdf_analysis.config import StatisticsConfig
from module10_rdf_analysis.statistic_analysis_2drdf_boltzmann_plots import \
    PlotRdfBoltzmann


class Rdf2dWithBoltzmann:
    """
    read and plot the 2d rdf with the Boltzmann distribution
    """
    # pylint: disable=too-many-arguments

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
        boltzmann_data: pd.DataFrame = self.process_boltzmann_data(log)
        vlines_data: pd.DataFrame = self.get_radii_vlines(log)
        self.plot_data(
            rdf_x, rdf_data, rdf_fit_data, boltzmann_data, vlines_data, log)

    def process_boltzmann_data(self,
                               log: logger.logging.Logger
                               ) -> pd.DataFrame:
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

        boltzmann_y_normalized_dict: dict[str, pd.Series] = \
            self.normalize_boltzmann(boltzmann_y_mean_dict)

        # Create a new DataFrame with the Boltzmann distribution
        processed_boltzmann_data: pd.DataFrame = \
            self.create_boltzmann_dataframe(
                truct_boltzmann_x_dict, boltzmann_y_normalized_dict)

        return processed_boltzmann_data

    def get_radii_vlines(self,
                         log: logger.logging.Logger
                         ) -> pd.DataFrame:
        """
        Get the computed contact radius and r of half max from the
        wriiten files
        """
        contact_r_file: str = self.config.files.contact.radii_out_fname
        r_half_max_file: str = self.config.files.fit_parameters.out_fname
        contact_r_df: pd.DataFrame = xvg_to_dataframe.XvgParser(
            contact_r_file, log, x_type=float).xvg_df
        r_half_max_df: pd.DataFrame = xvg_to_dataframe.XvgParser(
            r_half_max_file, log, x_type=float).xvg_df
        vlines_data: dict[str, tuple[float, float]] = {}
        for idx, radius in contact_r_df.iterrows():
            oda: str = radius['ODA']
            contact_r: float = radius['mean']
            r_half_max: float = r_half_max_df.at[idx, 'mean']
            vlines_data[str(int(oda))] = (contact_r, r_half_max)
        df_vlines_data: pd.DataFrame = pd.DataFrame(vlines_data)
        df_vlines_data.index = pd.Index(['contact_r', 'r_half_max'])
        return df_vlines_data

    def plot_data(self,
                  rdf_x: pd.DataFrame,
                  rdf_data: pd.DataFrame,
                  rdf_fit_data: pd.DataFrame,
                  boltzmann_data: dict[str, pd.DataFrame],
                  vlines_data: pd.DataFrame,
                  log: logger.logging.Logger
                  ) -> None:
        """
        Plot the 2d rdf with the Boltzmann distribution
        """
        # pylint: disable=too-many-arguments
        PlotRdfBoltzmann(rdf_x,
                         rdf_data,
                         rdf_fit_data,
                         boltzmann_data,
                         vlines_data,
                         log,
                         self.config.plots.boltzmann_rdf)

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
        return boltzmann_y_mean_dict

    @staticmethod
    def normalize_boltzmann(boltzmann_y_mean_dict: dict[str, pd.Series],
                            ) -> dict[str, pd.Series]:
        """
        Normalize the Boltzmann distribution
        """
        boltzmann_y_normalized_dict: dict[str, pd.Series] = {}
        for oda, boltzmann_y in boltzmann_y_mean_dict.items():
            y_copy: pd.Series = boltzmann_y.copy()
            # sort the values and get average of the last 10 bigest values
            y_copy.sort_values(ascending=False, inplace=True)
            aveg_max: float = y_copy[:7].mean()
            boltzmann_y_normalized_dict[oda] = boltzmann_y / aveg_max
        return boltzmann_y_normalized_dict

    @staticmethod
    def create_boltzmann_dataframe(cut_boltzmann_x_dict: dict[str, pd.Series],
                                   boltzmann_y_mean_dict: dict[str, pd.Series],
                                   ) -> pd.DataFrame:
        """
        make a new DataFrame with the Boltzmann distribution for each
        of the oda concentrations
        """
        # check if all the xdata are similar
        for boltzmann_x in cut_boltzmann_x_dict.values():
            assert boltzmann_x.equals(list(cut_boltzmann_x_dict.values())[0])
        data_columns: list[str] = \
            ['r_nm'] + list(boltzmann_y_mean_dict.keys())
        boltzmann_data_df: pd.DataFrame = pd.DataFrame(columns=data_columns)
        boltzmann_data_df['r_nm'] = \
            list(cut_boltzmann_x_dict.values())[0] * 0.1  # nm
        for oda, boltzmann_y in boltzmann_y_mean_dict.items():
            boltzmann_data_df[oda] = boltzmann_y
        return boltzmann_data_df
