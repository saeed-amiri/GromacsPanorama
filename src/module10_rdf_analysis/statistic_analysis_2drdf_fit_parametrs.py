"""
Read the extracted from the log files which are contain info about the
parameters in the fit function:
    g_modified_guess: float =
        2 * abs(b_init_guess) * g_init_guess / (1 + g_init_guess)
    return response_infinite +
        (self.config.response_zero - response_infinite) /
        (1+(x_data/c_init_guess)**b_init_guess) ** g_modified_guess
where:
    b: is the slope of the curve at the inflection point
    c: is the inflection point
    g: is the slope of the curve at the inflection point
    response_infinite: is the response at infinite x values
    response_zero: is the response at zero x values
    x_data: is the x values

    c: is the inflection point
    it is in Angstrom, and cannot be less than contact_radius and cannot be
    bigger than half of the box size in x or y directions
    wSSE: is the weighted sum of squared errors:
    There is no defined values for wSSE, but it should be as small as possible
    also very big values are not good

    We use this two values and their limitaios to clean the data
"""

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt

from common import logger
from common import file_writer
from common import xvg_to_dataframe
from common.colors_text import TextColor as bcolors

from module10_rdf_analysis.config import StatisticsConfig
from module10_rdf_analysis.statistic_analysis_2drdf_plots import PlotStatistics
from module10_rdf_analysis.statistic_analysis_2drdf_tools import \
    bootstrap_turn_points


class FitParameters:
    """
    Read and plot the fitted parameters
    """

    __slots__ = ['config', 'info_msg']

    def __init__(self,
                 config: StatisticsConfig,
                 log: logger.logging.Logger
                 ) -> None:
        self.info_msg: str = "Message from FitParameters:\n"
        self.config = config
        self.process_fit_parameters(log)
        self.write_msg(log)

    def process_fit_parameters(self,
                               log: logger.logging.Logger
                               ) -> None:
        """
        Read the fitted parameters
        """
        fit_params: dict[str, tuple[np.float64, np.float64, np.float64]] = \
            self.read_fit_parameters(log)
        df_stats: pd.DataFrame = self.make_df(fit_params)
        self.plot_data(df_stats, log)
        self.write_xvg(df_stats, log)

    def read_fit_parameters(self,
                            log: logger.logging.Logger
                            ) -> dict[str, tuple[np.float64,
                                                 np.float64,
                                                 np.float64]]:
        """
        Read the fit parameters
        """
        fit_params: dict[str, tuple[np.float64, np.float64, np.float64]] = {}
        for oda, fname in self.config.files.fit_parameters.file_names.items():
            df_i: pd.DataFrame = xvg_to_dataframe.XvgParser(
                fname, log, x_type=int).xvg_df
            df_i = self.clean_data(
                 df_i, oda, self.config.files.fit_parameters)
            mean_estimate, normal_std_err, std_err, _, _ = \
                self.get_r_half_max_columns_average(df_i, oda)
            fit_params[oda] = (mean_estimate * 0.1,  # nm
                               normal_std_err * 0.1,  # nm
                               std_err * 0.1  # nm
                               )
        return fit_params

    def make_df(self,
                fit_params: dict[str, tuple[np.float64,
                                            np.float64,
                                            np.float64]]
                ) -> pd.DataFrame:
        """
        Convert the dictionary to a DataFrame
        """
        # Convert the dictionary to a DataFrame
        return pd.DataFrame.from_dict(
            fit_params,
            orient='index', columns=['mean', 'normal_std_err', 'std_err'])

    def clean_data(self,
                   df_i: pd.DataFrame,
                   oda: str,
                   param: StatisticsConfig
                   ) -> pd.DataFrame:
        """
        Clean the data
        The available statistics for the fits are:
        WSSE: is the weighted sum of squared errors
        p_value: is the p-value of the fit (probability that the null
            hypothesis) it should be as close to 1 as possible
        R_squared: is the coefficient of determination, it should be as close
            to 1 as possible
        RMSE: is the root mean squared error, it should be as small as possible
        MAE: is the mean absolute error, it should be as small as possible

        The fitted parameters are:
        c: is the inflection point, it is in Angstrom, and cannot be
            less than contact_radius and cannot be bigger than half of the
            box size in x or y directions
        b: is the slope of the curve at the inflection point
        g: is the slope ratio of the curve in reaching the plateaus
        d: is the response at infinite x values
        """
        df_copy = df_i.copy()

        # drop NaN values
        df_copy = df_copy.dropna()
        # remove outliers based on the statistics
        df_copy = \
            df_copy[(df_copy[param.mae_column] < param.mae_max) &
                    (df_copy[param.rmse_column] < param.rmse_max) &
                    (df_copy[param.wsse_column] < param.wsse_max) &
                    (df_copy[param.p_value_column] > param.p_value_max) &
                    (df_copy[param.r_squared_column] > param.r_squared_max)
                    ]
        # remove outliers of r_half_max_column
        max_r_posible = \
            max(float(param.box_size['x']),
                float(param.box_size['y'])) / 2

        min_r_posible = float(param.min_contact_radius[oda])
        df_copy = df_copy[(df_copy[param.r_half_max_column] > min_r_posible) &
                          (df_copy[param.r_half_max_column] < max_r_posible)]
        # remove outliers of param.r_half_max_column
        df_copy = df_copy[
            (np.abs(stats.zscore(df_copy[param.r_half_max_column])) < 3)]

        return df_copy

    def get_r_half_max_columns_average(self,
                                       df_i: pd.DataFrame,
                                       oda: str
                                       ) -> tuple[np.float64, ...]:
        """
        Get the average of the r_half_max_column
        """
        r_half_max_column = self.config.files.fit_parameters.r_half_max_column
        c_arr = df_i[r_half_max_column].values
        boot_stats, msg = bootstrap_turn_points(oda, c_arr)
        self.info_msg += msg
        return boot_stats

    def plot_data(self,
                  df_stats: pd.DataFrame,
                  log: logger.logging.Logger
                  ) -> None:
        """
        Plot the data
        """
        PlotStatistics(df_stats, log, self.config.plots.fit_parameters, True)

    def write_xvg(self,
                  df_stats: pd.DataFrame,
                  log: logger.logging.Logger
                  ) -> None:
        """
        Write the data to an xvg file
        """
        file_writer.write_xvg(
            df_stats,
            log,
            self.config.files.fit_parameters.out_fname,
            extra_comments="Stats of the fitted parameters from all:\n",
            xaxis_label='ODA',
            yaxis_label='varies [nm]',
            title='computed c_mean of the fitted parameters')

    def write_msg(self,
                  log: logger.logging.Logger  # To log
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKGREEN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)
