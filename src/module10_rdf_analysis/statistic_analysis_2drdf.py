"""
Statistical Analysis Overview:
This script provides statistical tools for analyzing and comparing data
distributions. Each method offers unique insights, as described below.

- **Mean**:
    The average value, providing a measure of central tendency in the
    data.

- **Median**:
    The middle value, less sensitive to outliers than the mean, showing
    the data's central position.

- **Standard Deviation**:
    Indicates the spread of data around the mean, with higher values
    indicating more variability.

- **Variance**:
    The square of the standard deviation, measuring data dispersion
    relative to the mean.

- **Confidence Interval**:
    Provides a range within which the true mean is likely to fall,
    offering a measure of uncertainty.

Each of these methods enables nuanced statistical analysis, helping to
uncover trends, differences, and patterns within data sets.
"""
# pylint: disable=unused-import

import hydra
from hydra.core.config_store import ConfigStore


from module10_rdf_analysis.config import StatisticsConfig
from module10_rdf_analysis.statistic_analysis_2drdf_read_data import \
    ProcessData
from module10_rdf_analysis.statistic_analysis_2drdf_median import \
    MedianAnalysis
from module10_rdf_analysis.statistic_analysis_2drdf_median_from_rdf import \
    CalculateMedian
from module10_rdf_analysis.statistic_analysis_2drdf_turn_points import \
    AnalysisFitParameters
from module10_rdf_analysis.statistic_analysis_2drdf_contact import \
    ContactAnalysis
from module10_rdf_analysis.statistic_analysis_2drdf_fit_parametrs import \
    FitParameters
from module10_rdf_analysis.statistic_analysis_2drdf_compute_ex_zone import \
    ComputeExcludedAreas
from module10_rdf_analysis.statistic_analysis_2drdf_with_bolzmann import \
    Rdf2dWithBoltzmann

from common import logger

conf_store = ConfigStore.instance()
conf_store.store(name="configs", node=StatisticsConfig)


@hydra.main(version_base=None,
            config_path="conf",
            config_name="config")
def main(cfg: StatisticsConfig) -> None:
    # pylint: disable=missing-function-docstring
    # pylint: disable=unused-variable
    log: logger.logging.Logger = \
        logger.setup_logger('statistic_analysis_2drdf.log')
    # read and process the data
    rdf_data: "ProcessData" = ProcessData(cfg, log)

    # apply normal statistics to the data
    CalculateMedian(rdf_data.xdata,
                    rdf_data.data,
                    rdf_data.fit_data,
                    log,
                    cfg)

    # read and process the fitted parameters
    fit_data: "AnalysisFitParameters" = AnalysisFitParameters(cfg, log)

    # read and process the contact data
    contact_data: "ContactAnalysis" = ContactAnalysis(cfg, log)

    # read and process the median data
    # median_data: "MedianAnalysis" = MedianAnalysis(cfg, log)

    # read and process the fit parameters
    fit_params: "FitParameters" = FitParameters(cfg, log)

    # compute the excluded areas
    ComputeExcludedAreas(cfg, log)

    # compute the median with Boltzmann
    Rdf2dWithBoltzmann(rdf_data.xdata,
                       rdf_data.data,
                       rdf_data.fit_data,
                       log,
                       cfg)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
