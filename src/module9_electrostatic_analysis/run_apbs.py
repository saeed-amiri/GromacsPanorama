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
    apbs_in = '''
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
class EnvironmentConfig:
    """the modules to load to run the apbs command
    Apbs is built on the local machine in:
        /usr/opt/apbs/3.4.0/
    """
    env_modules: str = '''
        module load intel-compilers/2022.0.1
        module load impi/2021.5.0
        module load imkl/2022.0.1
        export OMP_NUM_THREADS=1
        export PATH=$PATH:/usr/opt/apbs/3.4.0/bin/
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/opt/apbs/3.4.0/lib/
    '''


@dataclass
class ParameterConfig:
    """set the parameters for the modifying the tcl file"""
    key_values: dict[str, str] = field(default_factory=lambda: ({
        'TEMP': '298.15'}))


@dataclass
class AllConfig(ApbsBaseFile, ParameterConfig, EnvironmentConfig):
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
        apbs_in: list[str] = self.prepare_input(src)
        self.execute_apbs(apbs_in, log)
        self.write_log_msg(log)

    def prepare_input(self,
                      src: list[str]
                      ) -> None:
        """parapre the input files for the APBS to run
        The prameters with @ will be replaced and name of the files
        with & will be set from the src
        """
        in_list: list[str] = []
        apbs_in: str = self._set_parameter()
        for item in src:
            in_i = apbs_in.replace('&IN_STRUCTURE', item)
            in_list.append(in_i)
        return apbs_in

    def execute_apbs(self,
                     apbs_in: list[str],
                     log: logger.logging.Logger
                     ) -> typing.Union[str, None]:
        """(re)write the input file and run the apbs command"""
        self._set_environment()

    def _set_environment(self) -> None:
        """Load the modules needed for running APBS."""
        commands: list[str] = [cmd.strip() for cmd in
            self.configs.env_modules.strip().split('\n') if cmd]
        combined_command = '; '.join(commands)

        # Execute the combined command in a shell
        subprocess.run(combined_command,
                       shell=True,
                       check=True,
                       capture_output=True,
                       text=True,
                       executable='/bin/bash')

    def _set_parameter(self) -> str:
        """set the parameters in the input template"""
        template: str = self.configs.apbs_in
        for key, value in self.configs.key_values.items():
            placeholder = f"@{key}"
            template = template.replace(placeholder, value)
            self.info_msg += f'\t`{placeholder}` is replaced by `{value}`\n'
        return template

    def write_log_msg(self,
                      log: logger.logging.Logger  # Name of the output file
                      ) -> None:
        """writing and logging messages from methods"""
        log.info(self.info_msg)
        print(f'{bcolors.OKGREEN}{self.__module__}:\n'
              f'\t{self.info_msg}\n{bcolors.ENDC}')


if __name__ == '__main__':
    ExecuteApbs(src=sys.argv[1:], log=logger.setup_logger('run_apbs.log'))
