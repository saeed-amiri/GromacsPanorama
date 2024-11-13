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

- **Skewness**:
    Measures the asymmetry of the data distribution; positive values
    indicate right skew, negative values indicate left skew.

- **Kurtosis**:
    Reflects the "tailedness" of the distribution; high kurtosis
    indicates more outliers, while low kurtosis suggests fewer.

- **Confidence Interval**:
    Provides a range within which the true mean is likely to fall,
    offering a measure of uncertainty.

- **T-test**:
    Compares the means of two groups to see if they are significantly
    different.

- **ANOVA**:
    Analyzes the variance across multiple groups to determine if there
    are any statistically significant differences between them.

- **Tukey's HSD**:
    A post-hoc test following ANOVA to identify which specific groups
    differ from each other.

- **Kruskal-Wallis**:
    A non-parametric test that evaluates differences between multiple
    groups without assuming normality.

- **Dunn's Test**:
    A post-hoc test following Kruskal-Wallis to pinpoint which groups
    differ.

- **Mann-Whitney U Test**:
    A non-parametric test to compare two independent groups, assessing
    differences in their distributions.

- **Wilcoxon Signed Rank Test**:
    A non-parametric test comparing two related samples, used for
    paired data.

- **Shapiro-Wilk Test**:
    Assesses if data follow a normal distribution, suitable for small
    sample sizes.

- **Anderson-Darling Test**:
    Tests for normality with an emphasis on the tails, helping to check
    distribution fit.

Each of these methods enables nuanced statistical analysis, helping to
uncover trends, differences, and patterns within data sets.
"""

import hydra
from hydra.core.config_store import ConfigStore


from module10_rdf_analysis.config import StatisticsConfig

conf_store = ConfigStore.instance()
conf_store.store(name="configs", node=StatisticsConfig)

@hydra.main(version_base=None,
            config_path="conf",
            config_name="config")
def main(cfg: StatisticsConfig) -> None:
    # pylint: disable=missing-function-docstring
    pass

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
