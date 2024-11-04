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
Saeed
"""
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from common import logger
from common import xvg_to_dataframe
from common.colors_text import TextColor as bcolors


# Constants
class Constants(Enum):
    """Physical constants"""
    # pylint: disable=invalid-name
    R = 8.314  # J/(mol K)
    m = 2  # unitless
    T = 298.15  # K


# Dataclass
@dataclass
class Config:
    """Configuration for the ComputeIsotherm class"""
    input_file: str = "data/interfacial_tension.xvg"
    output_file: str = "data/gibbs_adsorption_isotherm.xvg"


class ComputeIsotherm:
    """Reading data and computing the Gibbs adsorption isotherm"""
    info_msg: str = "Message from ComputeIsotherm:\n"

    def __init__(self,
                 config: Config,
                 log: logger.logging.Logger
                 ) -> None:
        self.config = config
