"""
Analysis the charge of the nanoparticle
1- Charge in total
2- Partial charge at the interface
"""

import typing

import numpy as np

from common import logger


if typing.TYPE_CHECKING:
    from module6_charge_analysis.charge_analysis_interface_np import \
        ComputeConfigurations


class NpChargeAnalysis:
    """analysign the charge of the nanoparticle"""

    info_msg: str = 'Messege from NpChargeAnalysis:\n'

    cla_arr: np.ndarray
    config: "ComputeConfigurations"

    def __init__(self,
                 cla_arr: np.ndarray,
                 config: "ComputeConfigurations",
                 log: logger.logging.Logger
                 ) -> None:
        self.cla_arr = cla_arr
        self.config = config


if __name__ == '__main__':
    pass
