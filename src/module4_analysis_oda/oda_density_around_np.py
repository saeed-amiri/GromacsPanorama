"""
For each defined region, calculate the number density of ODA molecules.
This is typically done by counting the number of ODA molecules in each
region and dividing by the volume of that region.
"""

import sys
from dataclasses import dataclass

from common import logger
from common import xvg_to_dataframe as xvg
from common import my_tools
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
                 log: logger.logging.Logger,
                 input_config: "OdaInputConfig" = OdaInputConfig()
                 ) -> None:
        self._initiate(log, input_config)

    def _initiate(self,
                  log: logger.logging.Logger,
                  input_config: "OdaInputConfig"
                  ) -> None:
        """Initiate the calculation by checking necessary files."""
        self._check_files(log, input_config)

    @staticmethod
    def _check_files(log: logger.logging.Logger,
                     input_config: "OdaInputConfig"
                     ) -> None:
        my_tools.check_file_exist(input_config.contact_xvg, log)
        my_tools.check_file_exist(input_config.np_coord_xvg, log)


if __name__ == "__main__":
    pass
