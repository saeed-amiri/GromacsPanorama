"""
make the data from the pdb and itp

"""

import sys
import typing
from dataclasses import dataclass, field

import pandas as pd

from common import logger, itp_to_df, pdb_to_df, gro_to_df, my_tools
from common.colors_text import TextColor as bcolors

from module9_electrostatic_analysis import parse_charmm_data, \
    force_field_path_configure


@dataclass
class FileConfig:
    """Set the name of the input files"""
    struct_file: str = field(init=False)  # Structure file
    itp_file: str = 'APT_COR.itp'  # FF of nanoparticle
    ff_user: str = 'CHARMM.DAT'  # Radius of the atoms in CAHRMM
    pqr_file: str = field(init=False)  # The output file to write, ext.: pqr


@dataclass
class AllConfig(FileConfig):
    """set all the configs"""
    compute_radius: bool = True


class PdbToPqr:
    """
    preapre the file with positions, charges and radii
    """

    info_msg: str = 'Message from PdbToPqr:\n'
    configs: AllConfig

    def __init__(self,
                 struct_file: str,  # Name of the structure file
                 log: logger.logging.Logger,
                 configs: AllConfig = AllConfig()
                 ) -> None:
        configs.struct_file = struct_file
        self.configs = configs
        self.initiate(log)
        self.write_msg(log)

    def initiate(self,
                 log: logger.logging.Logger
                 ) -> pd.DataFrame:
        """get all the infos"""

        self.check_all_file(log)
        struct_file_type: str = self.get_structure_file_type()
        itp_atoms: pd.DataFrame = \
            itp_to_df.Itp(self.configs.itp_file, section='atoms').atoms
        if struct_file_type == 'pdb':
            structure: pd.DataFrame = \
                pdb_to_df.Pdb(self.configs.struct_file, log).pdb_df
        elif struct_file_type == 'gro':
            structure = \
                gro_to_df.ReadGro(self.configs.struct_file, log).gro_data
        ff_radius: pd.DataFrame
        if self.configs.compute_radius:
            ff_radius = self.compute_radius(log)
        else:
            ff_radius = parse_charmm_data.ParseData(
                self.configs.ff_user, log).radius_df

    def compute_radius(self,
                       log: logger.logging.Logger
                       ) -> pd.DataFrame:
        """reading the force filed files and compute the radius based
        on sigma"""
        ff_files: dict[str, typing.Any] = \
            force_field_path_configure.ConfigFFPath(log).ff_files
        all_atom_info = \
            itp_to_df.Itp(ff_files['all_atom_info'], 'atomtypes').atomtypes
        return self.set_radius(all_atom_info)

    @staticmethod
    def set_radius(all_atom_info: pd.DataFrame
                   ) -> pd.DataFrame:
        """Compute the radius based on the sigma vlaues
        sigma (in nm) = 1 * R_min (in nm) * 2^(-1/6)
        """
        radius = all_atom_info['sigma'] * 1**(1/6) / 2
        all_atom_info['radius'] = radius
        return all_atom_info

    def get_structure_file_type(self) -> str:
        """find the type of the input structure file"""
        return self.configs.struct_file.split('.', -1)[1]

    def check_all_file(self,
                       log: logger.logging.Logger
                       ) -> None:
        """check all the existence of the all files"""
        for file in [self.configs.itp_file,
                     self.configs.struct_file]:
            my_tools.check_file_exist(file, log)
        charmm: typing.Union[bool, None] = \
            my_tools.check_file_exist(self.configs.ff_user, log, if_exit=False)
        if charmm is False:
            self.configs.compute_radius = True
            self.info_msg += (f'\tWarning! `{self.configs.ff_user}` does not '
                              f'exsit!\n\tWill try to comput the radius\n')

    def write_msg(self,
                  log: logger.logging.Logger  # To log
                  ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{PdbToPqr.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    PdbToPqr(struct_file=sys.argv[1], log=logger.setup_logger('pdb2pqr.log'))
