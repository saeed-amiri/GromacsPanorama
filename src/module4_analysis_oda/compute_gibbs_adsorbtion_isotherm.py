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
import json
import random
import typing
from collections import Counter

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
    CR: float = 20.0  # conversion rate from mN/m to J/m^2


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

    __slots__ = ['config', 'nr_frames', 'info_msg']

    def __init__(self,
                 config: Config,
                 log: logger.logging.Logger
                 ) -> None:
        self.info_msg: str = "Message from GetTension:\n"
        self.config = config.inputs.tension_files
        tension_df: pd.DataFrame = self.read_tension(log)

        self.analyze_tension(tension_df, log)
        self.log_msg(log)

    def read_tension(self,
                     log: logger.logging.Logger
                     ) -> pd.DataFrame:
        """
        Read the tension file
        """
        tension_dict: dict[str, pd.Series] = {}
        for i, (oda, fname) in enumerate(self.config.items()):
            xvg_data = xvg_to_dataframe.XvgParser(fname, log, x_type=float)
            if i == 0:
                self.nr_frames = xvg_data.nr_frames
            tension_i = xvg_data.xvg_df
            tension_dict[str(oda)] = \
                tension_i['Surf_SurfTen'] / Constants.CR.value  # mN/m
        return pd.DataFrame(tension_dict)

    def analyze_tension(self,
                        tension_df: pd.DataFrame,
                        log: logger.logging.Logger
                        ) -> None:
        """
        Analyze the tension data
        """
        self.perform_normal_bootstrap(tension_df)

    def perform_normal_bootstrap(self,
                                 tension_df: pd.DataFrame
                                 ) -> None:
        """do the sampling here"""
        all_stats: dict[str, dict[str, typing.Any]] = {}
        for oda, tension in tension_df.items():
            samples: pd.Series = \
                self.sample_randomly_with_replacement(tension)

            all_stats[oda] = self.calc_raw_stats(oda, samples, 'normal')
        df_stats = pd.DataFrame.from_dict(all_stats, orient='index')

    def sample_randomly_with_replacement(self,
                                         tension: pd.Series
                                         ) -> list[np.float64]:
        """Randomly Select With Replacement"""
        samples: list[np.float64] = []
        for _ in range(self.nr_frames):
            sample_i = random.choices(tension,
                                      k=self.nr_frames)
            samples.append(sum(sample_i)/self.nr_frames)
        return samples

    def calc_raw_stats(self,
                       oda: str,
                       samples: typing.Union[list[np.float64], list[float]],
                       style: str
                       ) -> dict[str, typing.Any]:
        """calculate std and averages"""
        raw_stats_dict: dict[str, typing.Any] = {}
        sample_arr: np.ndarray = np.array(samples)
        raw_stats_dict['std'] = np.std(sample_arr)
        raw_stats_dict['mean'] = np.mean(sample_arr)
        raw_stats_dict['mode'] = \
            self.calc_mode(sample_arr, raw_stats_dict['std'])
        if style == 'initial':
            boots = ''
        else:
            boots = ' bootstraping'
        self.info_msg += \
            (f'\tStats for `{style}`{boots}:'
             f'{json.dumps(raw_stats_dict, indent=8)}\n')
        return raw_stats_dict

    @staticmethod
    def calc_mode(samples: np.ndarray,
                  tolerance: np.float64
                  ) -> np.float64:
        """return mode for the sample"""
        # Round the data to the nearest multiple of the tolerance
        rounded_data = [round(x / tolerance) * tolerance for x in samples]
        # Use Counter to count occurrences of rounded values
        counts = Counter(rounded_data)
        max_count: float = max(counts.values())
        modes \
            = [value for value, count in counts.items() if count == max_count]
        return modes[0]

    def log_msg(self,
                log: logger.logging.Logger  # Name of the output file
                ) -> None:
        """writing and logging messages from methods"""
        log.info(self.info_msg)


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
    # pylint: disable=no-value-for-parameter
    main()
