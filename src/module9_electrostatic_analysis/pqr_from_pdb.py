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
    structure_files: list[str] = field(init=False)
    itp_file: str = 'APT_COR.itp'  # FF of nanoparticle
    ff_user: str = 'CHARMM.DAT'  # Radius of the atoms in CAHRMM
    pqr_file: str = field(init=False)  # The output file to write, ext.: pqr
    accebtable_file_type: list[str] = \
        field(default_factory=lambda: ['gro', 'pdb'])


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
                 structure_files: list[str],  # Structure files
                 log: logger.logging.Logger,
                 configs: AllConfig = AllConfig()
                 ) -> None:
        configs.structure_files = structure_files
        self.configs = configs
        self.initiate(log)
        # self.write_msg(log)

    def initiate(self,
                 log: logger.logging.Logger
                 ) -> pd.DataFrame:
        """get all the infos"""
        ReadInputStructureFile(log, self.configs.structure_files)
        # self.check_all_file(log)
        # structure_files_type: str = self.get_structure_file_type()
        # itp_atoms: pd.DataFrame = \
            # itp_to_df.Itp(self.configs.itp_file, section='atoms').atoms
        # if structure_files_type == 'pdb':
            # structure: pd.DataFrame = \
                # pdb_to_df.Pdb(self.configs.structure_files, log).pdb_df
        # elif structure_files_type == 'gro':
            # structure = \
                # gro_to_df.ReadGro(self.configs.structure_files, log).gro_data
        # ff_radius: pd.DataFrame
        # if self.configs.compute_radius:
            # ff_radius = self.compute_radius(log)
        # else:
            # ff_radius = parse_charmm_data.ParseData(
                # self.configs.ff_user, log).radius_df

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
        return self.configs.structure_files.split('.', -1)[1]

    def check_all_file(self,
                       log: logger.logging.Logger
                       ) -> None:
        """check all the existence of the all files"""
        for file in [self.configs.itp_file,
                     self.configs.structure_files]:
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


class ReadInputStructureFile:
    """reading all the input structure files and return them in dict
    with the name of the file as key"""
    # pylint: disable=too-few-public-methods

    info_msg: str = '\nMessage from ReadInputStructureFile:\n'
    file_type: str  # Type of extension of the files
    _configs: FileConfig

    def __init__(self,
                 log: logger.logging.Logger,
                 strucure_files: list[str],
                 configs: FileConfig = FileConfig()
                 ) -> None:
        self._configs = configs
        self._proccess_files(strucure_files, log)
        self._write_msg(log)

    def _proccess_files(self,
                        strucure_files: list[str],
                        log) -> dict[str, pd.DataFrame]:
        """read and return data about each structure file"""
        self.file_type = self._check_file_extension(strucure_files, log)
        self._read_files(strucure_files, log)

    def _read_files(self,
                    strucure_files: list[str],
                    log: logger.logging.Logger
                    ) -> dict[str, pd.DataFrame]:
        """read the files"""
        # if self.file_type == ''

    def _check_file_extension(self,
                              strucure_files: list[str],
                              log: logger.logging.Logger
                              ) -> str:
        """check the files' extension, they all should be same gro or
        pdb"""
        file_extension: list[str] = \
            [item.split('.', -1)[1] for item in strucure_files]
        if (l_list := len(set_ext := set(file_extension))) > 1:
            log.error(
                msg := (f'\tThere are `{l_list}` file types: '
                        f'`{set_ext}`! There should be one of: '
                        f'{self._configs.accebtable_file_type}\n'))
            sys.exit(f'\n\t{bcolors.FAIL}{msg}{bcolors.ENDC}\n')
        if (exten_type := list(set(file_extension))[0]) not in \
           self._configs.accebtable_file_type:
            log.error(
                msg := (f'\tThe file type: `{exten_type}` is not exceptable!'
                        'should be one of the\n'
                        f'{self._configs.accebtable_file_type}\n'))
            sys.exit(f'\n\t{bcolors.FAIL}{msg}{bcolors.ENDC}\n')

        self.info_msg += \
            f'\tReading `{len(strucure_files)}` files, type: `{exten_type}`\n'
        return exten_type

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{ReadInputStructureFile.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    PdbToPqr(
        structure_files=sys.argv[1:], log=logger.setup_logger('pdb2pqr.log'))
