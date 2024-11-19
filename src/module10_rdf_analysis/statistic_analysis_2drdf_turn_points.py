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
from common import file_writer
from common import xvg_to_dataframe
from common.colors_text import TextColor as bcolors

from module10_rdf_analysis.config import StatisticsConfig
from module10_rdf_analysis.statistic_analysis_2drdf_plots import PlotStatistics
from module10_rdf_analysis.statistic_analysis_2drdf_tools import \
    bootstrap_turn_points


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

        self.plot_data(df_stats, log, self.config.plots)
        self.write_midpoints(df_stats, log, self.config.files.turn_points)

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
            df_i = self._clean_data(df_i, f_config.min_contact_radius[oda])
            y_arr: np.ndarray = df_i[ydata].values * 0.1  # nm
            bootstrap: tuple[tuple[np.float64, ...], str] = \
                bootstrap_turn_points(oda, y_arr[1:])
            midpoints_stats[str(oda)] = bootstrap[0]
            self.info_msg += bootstrap[1]
        midpoints_df: pd.DataFrame = self._make_turn_points_df(midpoints_stats)
        return midpoints_df

    @staticmethod
    def _clean_data(df_i: pd.DataFrame,
                    min_contact_radius: float = 13.0
                    ) -> pd.DataFrame:
        """
        Clean the data
        There are some situations where the fitted parameters are not
        physically meaningful. For example, the first turn point should
        be greater than 17.5 (minumim possible contact radius), or it
        is bigger than twice of the minimum contact radius, and the
        second turn point should be greater than the midpoint.
        """
        df_i = df_i[df_i['second_turn'] != 0]
        df_i = df_i[df_i['second_turn'] > df_i['midpoint']]
        df_i = df_i[df_i['midpoint'] > df_i['first_turn']]
        df_i = df_i[df_i['first_turn'] > min_contact_radius]
        df_i = df_i[df_i['first_turn'] < 2 * min_contact_radius]
        df_i = df_i[(np.abs(stats.zscore(df_i['first_turn'])) < 3)]
        df_i = df_i[(np.abs(stats.zscore(df_i['midpoint'])) < 3)]
        df_i = df_i[(np.abs(stats.zscore(df_i['second_turn'])) < 3)]
        return df_i

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
        PlotStatistics(df_i, log, config.turn_points, err_plot=True)

    def write_midpoints(self,
                        df_i: pd.DataFrame,
                        log: logger.logging.Logger,
                        config: StatisticsConfig
                        ) -> None:
        """
        Write the turn points to a file
        """
        extra_comments: str = ('turn points of the 2D RDF data, for normalized'
                               ' data and fitted data, by:'
                               f' {self.__module__}')
        file_writer.write_xvg(df_i,
                              log,
                              fname=config.out_avg_fname,
                              extra_comments=extra_comments,
                              xaxis_label='ODA',
                              yaxis_label='Turn Points',
                              title='Turn Points of the 2D RDF data')

    def _write_msg(self, log: logger.logging.Logger) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)
