"""
Set the path for the force fields files in each system
"""

import os
import socket
from dataclasses import dataclass, field

import common.logger as logger
from common.colors_text import TextColor as bcolors


@dataclass
class FileConfig:
    """set the path and files names"""
    _myscript_path: str = '/MyScripts/GromacsPanorama/data/force_field'
    _local_parent: str = '/scratch/saeed'
    _server_parent: str = '/scratch/projects/hbp00076'
    local_path: str = os.path.join(_local_parent, _myscript_path)
    server_path: str = os.path.join(_server_parent, _myscript_path)

    file_names: list[str] = field(default_factory=lambda: [
        'charmm36_silica.itp', 'CLA.itp', 'POT.itp', 'TIP3.itp'])


@dataclass
class MachinName:
    """set the machine names"""
    local_host: str = 'hmigws03'  # Name of the host in the office
    server_front_host: list[str] = ['glogin', 'blogin']  # Front names in HLRN
    # Name of the goettingen of HLRN
    server_host_list: list[str] = ['gcn', 'gfn', 'gsn', 'bcn', 'bfn', 'bsn']


@dataclass
class AllConfig(FileConfig, MachinName):
    """set all the configurations"""


class ConfigCpuNr:
    """
    Find the path of the files
    """

    info_msg: str = 'message from ConfigCpuNr:\n'  # Meesage in methods to log
    configs: AllConfig

    def __init__(self,
                 log: logger.logging.Logger,
                 configs: AllConfig = AllConfig()
                 ) -> None:
        self.configs = configs
        self.hostname: str = self.get_hostname()
        self.cores_nr: int = self.set_ff_path()
        self.write_log_msg(log)

    def set_ff_path(self) -> str:
        """set the nmbers of the cores based on the hostname"""
        if self.hostname == self.configs.local_host:
            # In local machine
            ff_path = self.configs.local_path
        elif self.hostname[:6] in self.configs.server_front_host:
            # On frontend
            ff_path = self.configs.server_path
        elif self.hostname[:3] in self.configs.server_host_list:
            # On the backends
            ff_path = self.configs.server_path
        else:
            ff_path = self.configs.server_path
        self.info_msg += (f'\tThe path for force field is'
                          f' set to: `{ff_path}`\n')
        return ff_path

    def get_hostname(self) -> str:
        """Retrieve the hostname of the machine."""
        try:
            hostname = socket.gethostname()
            self.info_msg += f'\tHostname is `{hostname}`\n'
        except socket.error as err:
            raise RuntimeError("Failed to retrieve hostname.") from err
        return hostname

    def write_log_msg(self,
                      log: logger.logging.Logger  # Name of the output file
                      ) -> None:
        """writing and logging messages from methods"""
        log.info(self.info_msg)
        print(f'{bcolors.OKBLUE}{self.__module__}:\n'
              f'\t{self.info_msg}\n{bcolors.ENDC}')
