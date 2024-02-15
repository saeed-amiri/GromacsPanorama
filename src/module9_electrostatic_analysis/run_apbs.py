"""
Run apbs module by preparing the input file and exacute them with
commands
"""

import os
import sys
import typing
import subprocess
from dataclasses import dataclass, field

from common import logger
from common.colors_text import TextColor as bcolors


@dataclass
class ApbsBaseFile:
    """template of the apbs input"""
    template = '''
        read
            mol pqr &IN_STRUCTURE
        end
        elec
            mg-auto
            dime 289 161 161
            cglen 220.5665 108.7578 98.2056
            fglen 149.745 83.9752 77.768
            cgcent mol 1
            fgcent mol 1
            mol 1
            lpbe
            bcfl sdh
            pdie 2.0
            sdie 78.54
            srfm smol
            chgm spl2
            sdens 10.0
            srad 1.4
            swin 0.3
            temp @TEMP
            calcenergy total
            calcforce no
            write pot dx &IN_STRUCTURE
        end
        quit
    '''


@dataclass
class ParameterConfig:
    """set the parameters for the modifying the tcl file"""
    key_values: dict[str, str] = field(default_factory=lambda: ({
        'TEMP': '298.15'}))


@dataclass
class AllConfig(ApbsBaseFile, ParameterConfig):
    """set the all the parameters and configurations"""
    fout: str = 'apbs.in'


class ExecuteApbs:
    """
    upda the apbs input files and run them
    """

    info_msg: str = 'Message from ExecuteApbs:\n'
    configs: AllConfig

    def __init__(self,
                 src: list[str],  # List of the pqr files
                 log: logger.logging.Logger,
                 configs: AllConfig = AllConfig()
                 ) -> None:
        self.configs = configs
        self.write_log_msg(log)

    def write_log_msg(self,
                      log: logger.logging.Logger  # Name of the output file
                      ) -> None:
        """writing and logging messages from methods"""
        log.info(self.info_msg)
        print(f'{bcolors.OKGREEN}{self.__module__}:\n'
              f'\t{self.info_msg}\n{bcolors.ENDC}')


if __name__ == '__main__':
    ExecuteApbs(src=sys.argv[1:], log=logger.setup_logger('run_apbs.log'))
