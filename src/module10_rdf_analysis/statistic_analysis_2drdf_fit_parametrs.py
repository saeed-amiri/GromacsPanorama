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

    def process_fit_parameters(self,
                               log: logger.logging.Logger
                               ) -> None:
        """
        Read the fitted parameters
        """
        self.read_fit_parameters(log)

    def read_fit_parameters(self,
                            log: logger.logging.Logger
                            ) -> None:
        """
        Read the fit parameters
        """
        fit_params: dict[str, pd.DataFrame] = {}
        for oda, fname in self.config.files.fit_parameters.file_names.items():
            df_i: pd.DataFrame = xvg_to_dataframe.XvgParser(
                fname, log, x_type=int).xvg_df
            df_i = self.clean_data(
                 df_i, oda, self.config.files.fit_parameters)

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

        # remove outliers of c_column
        max_r_posible = \
            max(float(param.box_size['x']),
                float(param.box_size['y'])) / 2

        min_r_posible = float(param.min_contact_radius[oda])
        df_copy = df_copy[(df_copy[param.c_column] > min_r_posible) &
                          (df_copy[param.c_column] < max_r_posible)]
        # remove outliers of param.c_column
        df_copy = df_copy[(np.abs(stats.zscore(df_copy[param.c_column])) < 1)]

        return df_copy