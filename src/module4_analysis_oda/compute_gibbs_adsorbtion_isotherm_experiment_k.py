"""
Estimate the K in the adsorption equilibrium constant (K) and maximum
surface excess (Γ_max) in the Gibbs adsorption isotherm:
    C = Γ_max / (K * (Γ_max - Γ))
    C: concentration
    Γ: surface excess
    p: pressure
    Γ_max: maximum surface excess
    K: adsorption equilibrium constant

    Γ_max is estimated by the maximum surface excess in the experiment
    and the surface area of the adsorbent:
        Γ_max = 1 / (a_max * N_A)
        a_max: maximum area occupied by a molecule
        N_A: Avogadro's number

- Computing the surface excess for the different concentrations
    Γ = - (1/nRT) * (d tension / d (ln C))
    n: Number of species whose chemical potential changes with concentration.
    For a non-ionic surfactant: = 1
    For an ionic surfactant (protonated ODA), n=2 (considering the counterion).
    R: Gas constant (8.314 K/molT).
    T: Temperature in Kelvin.
- Using Langumir isotherm to estimate K

"""

from enum import Enum

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from common import logger
from common import elsevier_plot_tools
from common import xvg_to_dataframe


class ComputeGibbsAdsorbtionIsothermExperimentK:
    """
    Compute the Gibbs adsorption isotherm values (Gamma) for each
    concentration data point.
    """

    __slots__ = ["config"]

    def __init__(self,
                 config: dict,
                 constants: Enum,
                 log: logger.logging.Logger
                 ) -> None:
        self.config = config.inputs
        data: pd.DataFrame = self.get_data(config, log)
        data = self.compute_surface_excess_each_point(
            data, constants, log, config)
        self.plot_data(data, log, config)

    def get_data(self,
                 config: dict,
                 log: logger.logging.Logger
                 ) -> pd.DataFrame:
        """
        Get the data from the hydra (assuming self.config.joeri is a
        dict with C as keys and gamma as values)
        """
        if config.experiment == "joeri":
            data = pd.DataFrame.from_dict(self.config.joeri, orient='index')
            # Ensure data is sorted by concentration
            return data.sort_index()
        if config.experiment == "maas":
            xvg_file = self.config.maas.xvg_file
            data = \
                xvg_to_dataframe.XvgParser(xvg_file, log, x_type=float).xvg_df
            columns: list[str] = [self.config.maas.salt_column_name,
                                  self.config.maas.oda_column_name,
                                  self.config.maas.ift_column_name,
                                  ]
            return data[columns]
        log.error(msg := "\tExperiment not recognized\n")
        raise ValueError(msg)

    def compute_surface_excess_each_point(self,
                                          data: pd.DataFrame,
                                          constants: Enum,
                                          log: logger.logging.Logger,
                                          config: dict
                                          ) -> pd.DataFrame:
        """
        Compute the surface excess (Gamma) for each concentration dat
        point.
        This method uses finite differences to approximate
        d(gamma)/dln_concentration locally at each point.
        """
        if config.experiment == "joeri":
            # Convert index (C) to a column so we can easily handle it
            data["C"] = data.index * 1e-3  # Convert to mol/L
            data["ln_concentration"] = np.log(data["C"])

            # Extract arrays for convenience
            ln_concentration = data["ln_concentration"].values
            gamma_vals = data["gamma_np_mN/m"].values * 1e-3  # to N/m
        elif config.experiment == "maas":
            data["C"] = data[self.config.maas.oda_column_name] * 1e-3
            data["ln_concentration"] = np.log(data["C"])

            # Extract arrays for convenience
            ln_concentration = data["ln_concentration"].values
            gamma_vals = data[self.config.maas.ift_column_name].values

        # Prepare an array for d_gamma/d_ln_concentration
        d_gamma_d_ln_concentration = np.zeros(len(data))

        # Use central differences for interior points
        # For i = 1 to len(data)-2 (since we need i-1 and i+1)
        for i in range(1, len(data)-1):
            d_gamma_d_ln_concentration[i] = \
                (gamma_vals[i+1] - gamma_vals[i-1]) / \
                (ln_concentration[i+1] - ln_concentration[i-1])

        # Handle endpoints with one-sided differences if you wish:
        # If we have at least 2 points:
        if len(data) > 1:
            # Forward difference for first point
            d_gamma_d_ln_concentration[0] = \
                (gamma_vals[1] - gamma_vals[0]) / \
                (ln_concentration[1] - ln_concentration[0])
            # Backward difference for last point
            d_gamma_d_ln_concentration[-1] = \
                (gamma_vals[-1] - gamma_vals[-2]) / \
                (ln_concentration[-1] - ln_concentration[-2])
        else:
            # Only one data point means we cannot get a derivative
            d_gamma_d_ln_concentration[0] = np.nan

        # Compute Gamma for each point
        gamma_values = -(
            1 / (constants.n.value * constants.R.value * constants.T.value)
            ) * d_gamma_d_ln_concentration
        data["Gamma"] = gamma_values

        # Log or print the results
        log.info("Computed Gamma at each concentration point:")
        for c_val, gamma_val in zip(data["C"], data["Gamma"]):
            log.info(f"C: {c_val}, Gamma: {gamma_val}")

        return data

    def plot_data(self,
                  data: pd.DataFrame,
                  log: logger.logging.Logger,
                  config: dict
                  ) -> None:
        """
        Plot the data to visually inspect the relationship between
        concentration and surface tension.
        """
        figure: tuple[plt.Figure, plt.Axes] = elsevier_plot_tools.mk_canvas(
            "single_column", aspect_ratio=1)
        fig_i, ax_i = figure
        plt.xlabel("Concentration [mM]")
        plt.ylabel("Surface tension [mN/m]")
        if config.log_x:
            ax_i.set_xscale('log')
            data[self.config.maas.oda_column_name] = \
                self.handel_log_zero(
                    data[self.config.maas.oda_column_name])
        if config.log_y:
            ax_i.set_yscale('log')

        if config.experiment == "joeri":
            self.plot_joeri(data, ax_i)
        elif config.experiment == "maas":
            self.plot_maas(data, ax_i)
        # ax_i.set_xlim(left=1e-10)
        elsevier_plot_tools.save_close_fig(
            fig_i, 'tensio_exp.jpg', loc='upper right')

    def plot_joeri(self,
                   data: pd.DataFrame,
                   ax_i: plt.Axes
                   ) -> None:
        """
        Plot the data from the Joeri experiment.
        """
        ax_i.plot(data.index,
                  data["gamma_np_mN/m"],
                  'o:',
                  markersize=3,
                  color='black',
                  label='With NP')

        ax_i.plot(data.index,
                  data["gamma_no_np_mN/m"],
                  '^--',
                  markersize=3,
                  color='darkred',
                  label='No NP')

    def plot_maas(self,
                  data: pd.DataFrame,
                  ax_i: plt.Axes
                  ) -> None:
        """
        Plot the data from the Maas experiment.
        """
        salt_values: list[float] = \
            data[self.config.maas.salt_column_name].unique()
        colors: list[str] = list(elsevier_plot_tools.CLEAR_COLOR_GRADIENT)
        colors = colors[::-1]
        markers: list[str] = list(elsevier_plot_tools.MARKER_STYLES)
        for i, salt_value in enumerate(salt_values):
            df_i = data[data[self.config.maas.salt_column_name] == salt_value]
            ax_i.plot(df_i[self.config.maas.oda_column_name],
                      df_i[self.config.maas.ift_column_name],
                      markers[i],
                      ls=':',
                      color=colors[i],
                      markersize=3,
                      label=f"NaCl: {salt_value:.2f} mM",
                      )
    @staticmethod
    def handel_log_zero(data: pd.Series
                        ) -> pd.Series:
        """
        Handle zero values in the data when plotting in log scale.
        """
        data[data == 0] = data[data != 0].min()/10
        return data
