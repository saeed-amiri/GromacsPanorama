"""
Plot the surface potential and the density of the system along the z-axis.
The goal is to compare the surface potential and the charge density
of the system and see where is the surface we selected for averaging
the surface potential.
It read the density of the systems in different xvg files and plot them.
The potential will be using the function:
    plot_debye_surface_potential
in the [...]_plots.py file.
"""

import typing

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from common import logger

from module9_electrostatic_analysis.apbs_analysis_average_pot_plots import \
    plot_debye_surface_potential
