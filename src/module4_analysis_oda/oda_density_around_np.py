"""
For each defined region, calculate the number density of ODA molecules.
This is typically done by counting the number of ODA molecules in each
region and dividing by the volume of that region.
"""

import sys
from dataclasses import dataclass

from common import logger
from common import xvg_to_dataframe as xvg
from common.colors_text import TextColor as bcolors


@dataclass
class OdaInputConfig:
    """set the input for analysing"""
    contact_xvg: str = 'contact.xvg'
    np_coord_xvg: str = 'coord.xvg'


class SurfactantDensityAroundNanoparticle:
    """self explained"""
    info_msg: str = '\tMessage from SurfactantDensityAroundNanoparticle:\n'

    def __init__(self,
                 log: logger.logging.Logger
                 ) -> None:
        pass


if __name__ == "__main__":
    pass
