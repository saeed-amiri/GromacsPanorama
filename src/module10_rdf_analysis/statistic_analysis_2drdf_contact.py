"""
Read and analyze the contact data, which contain info about the
interface and contact between the surfactants and the nanoparticle.

"""

import numpy as np
import pandas as pd

from common import logger
from common import file_writer
from common import xvg_to_dataframe
from common.colors_text import TextColor as bcolors

from module10_rdf_analysis.config import StatisticsConfig
from module10_rdf_analysis.statistic_analysis_2drdf_plots import PlotStatistics
from module10_rdf_analysis.statistic_analysis_2drdf_tools import \
    bootstrap_turn_points


class ContactAnalysis:
    """
    Read and analyze the contact data
    """

    __slots__ = ['config', 'info_msg']

    def __init__(self,
                 config: StatisticsConfig,
                 log: logger.logging.Logger
                 ) -> None:
        self.info_msg: str = "Message from ContactAnalysis:\n"
        self.config = config
        self.process_contact(log)
        self._write_msg(log)

    def process_contact(self,
                        log: logger.logging.Logger
                        ) -> None:
        """
        Read the contact data
        """
        df_radii: pd.DataFrame
        df_angles: pd.DataFrame
        df_radii, df_angles = \
            self.read_contact_data(log, self.config.files.contact)
        self.stats_radii(df_radii, log)
        self.stats_angles(df_angles, log)

    def read_contact_data(self,
                          log: logger.logging.Logger,
                          f_config: StatisticsConfig
                          ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Read the contact data
        """
        contact_radii: dict[str, pd.Series] = {}
        contact_angles: dict[str, pd.Series] = {}
        for oda, fname in f_config.contact_files.items():
            df_i: pd.DataFrame = \
                xvg_to_dataframe.XvgParser(fname, log, x_type=int).xvg_df
            contact_radii[str(oda)] = df_i[f_config.contact_radius] * 0.1  # nm
            contact_angles[str(oda)] = df_i[f_config.contact_angle]
        df_radii: pd.DataFrame = pd.concat(contact_radii, axis=1)
        df_angles: pd.DataFrame = pd.concat(contact_angles, axis=1)
        return df_radii, df_angles

    def stats_radii(self,
                    df_radii: pd.DataFrame,
                    log: logger.logging.Logger
                    ) -> None:
        """
        Analyze the contact data
        """
        radii_stats_dict: dict[str, tuple[np.float64, ...]] = {}
        df_radii = self.clean_data(df_radii)
        for oda, radii in df_radii.items():
            radii_stas, msg = bootstrap_turn_points(oda, radii.values)
            radii_stats_dict[str(oda)] = radii_stas
            self.info_msg += msg
        radii_stats_df: pd.DataFrame = self.make_df(radii_stats_dict)
        PlotStatistics(
            radii_stats_df, log, self.config.plots.contact_radii, True)
        extra_comments: str = ('Stats of the contact radius data:\n'
                               f' {self.__module__}')
        file_writer.write_xvg(radii_stats_df,
                              log,
                              fname=self.config.files.contact.radii_out_fname,
                              extra_comments=extra_comments,
                              xaxis_label='ODA',
                              yaxis_label='Turn Points',
                              title='Turn Points of the 2D RDF data')

    def stats_angles(self,
                     df_angles: pd.DataFrame,
                     log: logger.logging.Logger
                     ) -> None:
        """
        Analyze the contact data
        """
        angles_stats_dict: dict[str, tuple[np.float64, ...]] = {}
        df_angles = self.clean_data(df_angles)
        for oda, angles in df_angles.items():
            radii_stas, msg = bootstrap_turn_points(oda, angles.values)
            angles_stats_dict[str(oda)] = radii_stas
            self.info_msg += msg
        angles_stats_df: pd.DataFrame = self.make_df(angles_stats_dict)
        PlotStatistics(
            angles_stats_df, log, self.config.plots.contact_angles, True)
        extra_comments: str = ('Stats of the contact angles data:\n'
                               f' {self.__module__}')
        file_writer.write_xvg(angles_stats_df,
                              log,
                              fname=self.config.files.contact.angles_out_fname,
                              extra_comments=extra_comments,
                              xaxis_label='ODA',
                              yaxis_label='Angles',
                              title='Angles of the 2D RDF data')

    def make_df(self,
                stats_dict: dict[str, tuple[np.float64, ...]]
                ) -> pd.DataFrame:
        """
        Make a DataFrame from the stats
        """
        stats_df: pd.DataFrame = pd.DataFrame(stats_dict).T
        stats_df.columns = [
            'mean', 'normal_std_err', 'std_err', 'ci_low', 'ci_high']
        stats_df.index = stats_df.index.astype(int)
        return stats_df

    def clean_data(self,
                   df_radii: pd.DataFrame,
                   ) -> pd.DataFrame:
        """
        Clean the data
        """
        # if any raw contains Nan values, drop it
        df_radii.dropna(inplace=True)
        return df_radii

    def _write_msg(self, log: logger.logging.Logger) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)
