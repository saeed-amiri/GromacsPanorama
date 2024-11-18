"""
Read and plot the fitted parameters
About Bootstrap:
    - Bootstrapping is a resampling technique used to estimate the
    sampling distribution of a statistic (e.g., median, mean) by
    repeatedly sampling with replacement from the original data.
    - It allows us to assess the variability and compute confidence
    intervals for statistics without making strong assumptions about
    the data's underlying distribution.

"""

import numpy as np
import pandas as pd
from scipy import stats

from common import logger
from common import xvg_to_dataframe
from common.colors_text import TextColor as bcolors

from module10_rdf_analysis.config import StatisticsConfig
from module10_rdf_analysis.statistic_analysis_2drdf_plots import PlotStatistics


class AnalysisFitParameters:
    """
    Read and plot the fitted parameters
    """
    _slots__ = ['config', 'info_msg']

    def __init__(self,
                 config: StatisticsConfig,
                 log: logger.logging.Logger
                 ) -> None:
        self.info_msg: str = "Message from AnalysisFitParameters:\n"
        self.config = config
        self.process_midpoints(log)
        self._write_msg(log)

    def process_midpoints(self,
                          log: logger.logging.Logger
                          ) -> None:
        """
        Read the fitted parameters
        """
        df_stats: pd.DataFrame = \
            self._analysis_turn_points(log, self.config.files.turn_points)

        self.plot_data(pd.DataFrame(df_stats['mean']), log, self.config)

    def _analysis_turn_points(self,
                              log: logger.logging.Logger,
                              f_config: StatisticsConfig
                              ) -> pd.DataFrame:
        """
        Read the turn points
        """
        ydata: str = f_config.turn_points_ydata
        midpoints_stats: dict[str, tuple[np.float64, ...]] = {}
        for oda, fname in f_config.turn_points_files.items():
            df_i: pd.DataFrame = \
                xvg_to_dataframe.XvgParser(fname, log, x_type=int).xvg_df
            y_arr: np.ndarray = df_i[ydata].values * 0.1  # nm
            bootstrap: tuple[np.float64, ...] = \
                self._bootstrap_turn_points(oda, y_arr[1:])
            midpoints_stats[str(oda)] = bootstrap
        midpoints_df: pd.DataFrame = self._make_turn_points_df(midpoints_stats)
        return midpoints_df

    def _bootstrap_turn_points(self,
                               oda: int,
                               y_arr: np.ndarray
                               ) -> tuple[np.float64, np.float64, np.float64,
                                          np.float64, np.float64]:
        """
        Bootstrap the turn points
        """
        def mean_func(data, axis):
            return np.mean(data, axis=axis)

        # Perform bootstrapping
        res = stats.bootstrap(
            (y_arr,),
            statistic=mean_func,
            confidence_level=0.95,
            n_resamples=10000,
            method='percentile',
            random_state=0  # For reproducibility
        )

        # Extract results
        mean_estimate: np.float64 = mean_func(y_arr, axis=0)
        normal_std_err: np.float64 = \
            np.std(y_arr, ddof=1) / np.sqrt(len(y_arr))
        ci: np.float64 = res.confidence_interval
        std_err: np.float64 = res.standard_error

        # Display results
        self.info_msg += (
            f"\tTurn point: {oda}\n"
            f"\tMean estimate: {mean_estimate}\n"
            f"\t95% Confidence Interval for the mean: {ci.low} to {ci.high}\n"
            f"\tStandard Error of the mean: {std_err}\n\n")
        return mean_estimate, normal_std_err, std_err, ci.low, ci.high

    def _make_turn_points_df(self,
                             midpoints_stats: dict[str, tuple[np.float64, ...]]
                             ) -> pd.DataFrame:
        """
        Make a DataFrame of the turn points
        """
        df_i: pd.DataFrame = pd.DataFrame(midpoints_stats).T
        df_i.columns = [
            'mean', 'normal_std_err', 'std_err', 'ci_low', 'ci_high']
        # set type of index to int
        df_i.index = df_i.index.astype(int)

        return df_i

    def plot_data(self,
                  df_i: pd.DataFrame,
                  log: logger.logging.Logger,
                  config: StatisticsConfig
                  ) -> None:
        """
        Plot the fitted parameters
        """
        PlotStatistics(df_i, log, config.plots.turn_points)

    def _write_msg(self, log: logger.logging.Logger) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)

