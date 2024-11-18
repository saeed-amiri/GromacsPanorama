"""
Set the types for the configuration parameters.
"""

from dataclasses import dataclass


@dataclass
class Files:
    rdf: str


@dataclass
class Plots:
    plots: str


@dataclass
class StatisticsConfig:
    files: Files
    plots: Plots
