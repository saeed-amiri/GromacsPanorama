"""
Computing the Gibbs adsorption isotherm to compute the concentration
of ODA in the box based on the nominal concentration of ODA at the interface
and the computed interfacial tension.
The Gibbs adsorption isotherm is given by:

Surfactant Adsorption to Different Fluid Interfaces
https://pubs.acs.org/doi/10.1021/acs.langmuir.1c00668

"Effect of Oil on Area Per Surfactant When surfactants adsorb to
interfaces, they have to replace an oil molecule from the interfaces and
create a contact structure with the oil phase. At nonpolar oil and air,
this process happens fast and extensively because no hydrophilic
interactions with the aqueous phase are formed. In contrast, polar oil
molecules interact with the aqueous phase by hydrogen bonds and polar−π
interactions. (7) The increased affinity to the interface results in a
competition between the polar oil molecules and the surfactant, hence
lowering interfacial excess concentrations. The maximum interfacial
excess concentration Γ∞ is the area-related maximum concentration of
surfactants at the interface. From the fits in Figure 1, we calculated
Γ∞ with the Gibbs adsorption isotherm (eq 2), in which T is the
temperature, R is the ideal gas constant, and m is the ion parameter
(for ionic surfactants, m = 2, and for non-ionic surfactants, m = 1)."

Gamma =
    frac{1}{text{RTm}} left(frac{partial gamma}{partial c} right)_{T,P}
where:
    Gamma: Gibbs adsorption
    R: Gas constant (8.314 J/(mol K))
    T: Temperature (K)
    m: 2 for ionic surfactants and 1 for non-ionic surfactants
    gamma: Interfacial tension (mN/m)
    c: Concentration of ODA in the box

Inputs:
    A xvg file containing the interfacial tension of the system and the
    nominal concentration of ODA at the interface.
Outputs:
    A xvg file containing the Gibbs adsorption isotherm, the concentration
    of ODA in the box, and the interfacial tension.
Nov 4, 2024

The concentration of the ODA based on the numbers of the oil molecules
and the number of ODA molecules in the box is also going to be computed.
Nov 29, 2024

Saeed
"""
# pylint: disable=import-error
from dataclasses import dataclass
from enum import Enum

import hydra
from hydra.core.config_store import ConfigStore

import numpy as np
import pandas as pd

from common import logger
from common import xvg_to_dataframe


# Constants
class Constants(Enum):
    """Physical constants"""
    # pylint: disable=invalid-name
    R: float = 8.314  # J/(mol K)
    m: float = 2  # unitless
    T: float = 298.15  # K
    NA: float = 6.022e23  # Avogadro's number


# Dataclass
@dataclass
class Config:
    """Configuration for the ComputeIsotherm class"""
    inputs: str
    output_file: str = "gibbs_adsorption_isotherm.xvg"


class ComputeOdaConcentration:
    """
    compute the concentration of ODA in the box based on the number of
    oil molecules and the number of ODA molecules in the box
    """
    __slots__ = ['config', 'info_msg']

    def __init__(self,
                 config: Config,
                 log: logger.logging.Logger
                 ) -> None:
        self.info_msg: str = "Message from ComputeIsotherm:\n"
        self.config = config
        self.compute_oda_concentration()

    def compute_oda_concentration(self) -> None:
        """
        Compute the concentration of ODA in the box
        """


class GetTension:
    """
    Read tension files
    """

    __slots__ = ['config', 'info_msg']

    def __init__(self,
                 config: Config,
                 log: logger.logging.Logger
                 ) -> None:
        self.info_msg: str = "Message from GetTension:\n"
        self.config = config.inputs.tension_files
        self.read_tension(log)

    def read_tension(self,
                     log: logger.logging.Logger
                     ) -> pd.DataFrame:
        """
        Read the tension file
        """
        tension_dict: dict[str, pd.Series] = {}
        for oda, fname in self.config.items():
            tension_i: pd.DataFrame = xvg_to_dataframe.XvgParser(
                fname, log, x_type=float).xvg_df
            tension_dict[str(oda)] = tension_i['Surf_SurfTen']
        return pd.DataFrame(tension_dict)


conf_store = ConfigStore.instance()
conf_store.store(name="configs", node=Config)


@hydra.main(version_base=None,
            config_path="conf",
            config_name="config")
def main(cfg: Config) -> None:
    # pylint: disable=missing-function-docstring
    log: logger.logging.Logger = logger.setup_logger(
        'compute_gibbs_adsorption_isotherm.log')
    GetTension(cfg, log)


if __name__ == "__main__":
    main()
