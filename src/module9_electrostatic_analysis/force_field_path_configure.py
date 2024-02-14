"""
Set the path for the force fields files in each system
"""

import os
import socket
import typing
from dataclasses import dataclass, field

from common import logger
from common.colors_text import TextColor as bcolors


@dataclass
class FileConfig:
    """set the path and files names"""
    local_path: str = \
        '/scratch/saeed/MyScripts/GromacsPanorama/data/force_field'
    server_path: str = \
        '/scratch/projects/hbp00076/MyScripts/GromacsPanorama/data/force_field'

    file_names: dict[str, typing.Any] = field(default_factory=lambda: {
        'all_atom_info': 'charmm36_silica.itp',
        'apbs_info': 'CHARMM.DAT',
        'charge_info': [
                        'CLA.itp',  # charge for Cl ion
                        'POT.itp',  # charge for Na ion
                        'TIP3.itp',  # charge for water atoms 
                        'D10_charmm.itp',  # charge for the oil (Decane)
                        'ODAp_charmm.itp'  # charges for the protonated ODA
                        ]})

    np_info: str = 'APT_COR.itp'  # charge for the COR and APT of the NP


@dataclass
class MachineName:
    """set the machine names"""
    local_host: str = 'hmigws03'  # Name of the host in the office
    # Front names in HLRN
    server_front_host: list[str] = field(
        default_factory=lambda: ['glogin', 'blogin'])
    # Name of the goettingen of HLRN
    server_host_list: list[str] = field(
        default_factory=lambda: ['gcn', 'gfn', 'gsn', 'bcn', 'bfn', 'bsn'])


@dataclass
class AllConfig(FileConfig, MachineName):
    """set all the configurations"""


class ConfigFFPath:
    """
    Find the path of the files
    """

    info_msg: str = 'message from ConfigFFPath:\n'  # Meesage in methods to log
    configs: AllConfig
    ff_files: dict[str, typing.Any]  # Path of the ff files

    def __init__(self,
                 log: logger.logging.Logger,
                 configs: AllConfig = AllConfig()
                 ) -> None:
        self.configs = configs
        self.hostname: str = self.get_hostname()
        ff_path: str = self.set_ff_path()
        self.set_file_names(ff_path)
        self.write_log_msg(log)

    def set_file_names(self,
                       ff_path: str
                       ) -> None:
        """set the list for the ff files' path"""
        self.ff_files = {}
        self.ff_files['all_atom_info'] = \
            os.path.join(ff_path, self.configs.file_names['all_atom_info'])
        self.ff_files['apbs_info'] = \
            os.path.join(ff_path, self.configs.file_names['apbs_info'])
        self.ff_files['charge_info'] = [
            os.path.join(ff_path, item) for item in
            self.configs.file_names['charge_info']]
        self.ff_files['np_info'] = self.configs.np_info
        self.info_msg += '\tThe path of the force field files are set to:\n'
        for item, path in self.ff_files.items():
            self.info_msg += f'\t`{item}`: {path}\n'

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
