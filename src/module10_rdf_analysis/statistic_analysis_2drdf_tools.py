"""
Tools!
"""

import numpy as np
from scipy import stats


def bootstrap_turn_points(oda: int,
                          y_arr: np.ndarray
                          ) -> tuple[tuple[np.float64, np.float64, np.float64,
                                     np.float64, np.float64],
                                     str]:
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
    normal_std_err: np.float64 = np.std(y_arr, ddof=0)
    ci: np.float64 = res.confidence_interval
    std_err: np.float64 = res.standard_error

    # Display results
    info_msg = (
        f"\tData for: {oda} ODA:\n"
        f"\t\tMean estimate: {mean_estimate}\n"
        f"\t\t95% Confidence Interval for the mean: {ci.low} to {ci.high}\n"
        f"\t\tNormal Standard Error of the mean: {normal_std_err}\n"
        f"\t\tStandard Error of the mean: {std_err}\n\n")
    return (mean_estimate, normal_std_err, std_err, ci.low, ci.high), info_msg