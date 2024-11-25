"""
Computing the exclusion zone from varius of inputs:
1- Computing the exclusion zone c from all fit params and subtract the
    contact radius from it.
2- Computing the exclusion zone c from the mean fit param and subtract
    the contact radius from it.
3- Computing directly for all the frames and get the mean and std of it.

"""
import numpy as np
import pandas as pd

from common import logger
from common import file_writer
from common import xvg_to_dataframe
from common.colors_text import TextColor as bcolors

from module10_rdf_analysis.config import StatisticsConfig
from module10_rdf_analysis.statistic_analysis_2drdf_plots import PlotStatistics


class ComputeExcludedAreas:
    """
    Read and plot the exclusion zone!
    """

    __slots__ = ['config', 'info_msg']

    def __init__(self,
                 config: StatisticsConfig,
                 log: logger.logging.Logger
                 ) -> None:
        self.info_msg: str = "Message from ComputeExcludedAreas:\n"
        self.config = config
        self.process_ex_zone(log)
        self._write_msg(log)

    def process_ex_zone(self,
                        log: logger.logging.Logger
                        ) -> None:
        """
        Read the exclusion zone
        """
        r_half_max_file: str = self.config.files.fit_parameters.out_fname
        contact_file: str = self.config.files.contact.radii_out_fname
        df_r_half_max: pd.DataFrame = xvg_to_dataframe.XvgParser(
            r_half_max_file, log, x_type=float).xvg_df
        df_contact: pd.DataFrame = xvg_to_dataframe.XvgParser(
            contact_file, log, x_type=float).xvg_df
        r_half_max_arr: np.ndarray = df_r_half_max['mean'].to_numpy()
        contact_arr: np.ndarray = df_contact['mean'].to_numpy()
        ex_zone: np.ndarray = r_half_max_arr - contact_arr
        err: np.ndarray = self.estimate_error(df_contact, df_r_half_max)
        ex_zone_df: pd.DataFrame = \
            self.make_df(ex_zone, err, df_contact['ODA'])
        PlotStatistics(
            ex_zone_df, log, self.config.plots.ex_zone, err_plot=True)
        combined_df: pd.DataFrame = ex_zone_df.copy()
        combined_df['contact_radius'] = contact_arr
        combined_df['contact_radius_err'] = \
            df_contact['normal_std_err'].to_numpy()
        combined_df['r_half_max'] = r_half_max_arr
        combined_df['r_half_max_err'] = \
            df_r_half_max['normal_std_err'].to_numpy()

        PlotStatistics(
            combined_df, log, self.config.plots.ex_zone_paper, paper_plot=True)
        self.write_ex_zone(ex_zone_df, log)

    def estimate_error(self,
                       df_contact: pd.DataFrame,
                       df_r_half_max: pd.DataFrame,
                       ) -> np.ndarray:
        """
        Estimate the error
        Since the two data are not independent, we need to estimate
        the error by computing the covariance of the two data.
        The error is estimated as:
          err^2 = err1^2 + err2^2 - 2*err1*err2*cov
          using numpy.cov to compute the covariance.
          I think they have positive correlation.
        """
        contact_err: np.ndarray = df_contact['normal_std_err'].to_numpy()
        r_half_max_err: np.ndarray = df_r_half_max['normal_std_err'].to_numpy()
        cov_matrix: np.ndarray = np.cov(contact_err, r_half_max_err)
        err: np.ndarray = np.sqrt(
            contact_err**2 + r_half_max_err**2
            - 2*contact_err*r_half_max_err*cov_matrix[0, 1])
        self.info_msg += (f"\tError: {err}\n"
                          f"\tCovariance matrix: {cov_matrix}\n")
        return err

    def make_df(self,
                ex_zone: np.ndarray,
                err: np.ndarray,
                oda: pd.Series
                ) -> pd.DataFrame:
        """
        Convert the dictionary to a DataFrame
        """
        df_ex_zone: pd.DataFrame = pd.DataFrame(
            data={'ODA': oda, 'ex_zone': ex_zone, 'err': err})
        # set the ODA as the index
        df_ex_zone.set_index('ODA', inplace=True)
        return df_ex_zone

    def write_ex_zone(self,
                      ex_zone_df: pd.DataFrame,
                      log: logger.logging.Logger
                      ) -> None:
        """
        Write the exclusion zone to a file
        """
        extra_comments: str = (
            'Exclusion zone computed from the fit parameters'
            f' and the contact radius, by: {self.__module__}')
        file_writer.write_xvg(
            ex_zone_df,
            log,
            fname=self.config.files.ex_zone.out_fname,
            extra_comments=extra_comments,
            xaxis_label='ODA',
            yaxis_label='Exclusion zone (nm)',
        )

    def _write_msg(self, log: logger.logging.Logger) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)
