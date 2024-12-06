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

from enum import EnumType

import numpy as np
import pandas as pd

from common import logger


class ComputeGibbsAdsorbtionIsothermExperimentK:
    """
    Get data from the hydra and compute the Gibbs adsorption isotherm
    """

    __slots__ = ["info_msg", "experiment_data", "a_max", "config"]

    def __init__(self,
                 config: dict,
                 constants: EnumType,
                 log: logger.logging.Logger
                 ) -> None:
        self.config = config
        data: pd.DataFrame = self.get_data()

    def get_data(self) -> pd.DataFrame:
        """
        Get the data from the hydra
        """
        return pd.DataFrame.from_dict(self.config.joeri, orient='index')
