"""
make the data from the pdb and itp

"""

import os
import sys
import typing
import string
from dataclasses import dataclass, field

import numpy as np
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
class FFTypeConfig:
    """set the name of each file with their name of the data"""
    ff_type_dict: dict[str, str] = field(
        default_factory=lambda: {
            "SOL": 'TIP3',
            "CLA": 'CLA',
            "D10": 'D10_charmm',
            "ODN": 'ODAp_charmm',
            'APT': 'np_info',
            'COR': 'np_info'
        })


@dataclass
class AllConfig(FileConfig, FFTypeConfig):
    """set all the configs"""
    compute_radius: bool = True


class PdbToPqr:
    """
    preapre the file with positions, charges and radii
    """

    info_msg: str = 'Message from PdbToPqr:\n'
    configs: AllConfig
    force_field: "ReadForceFieldFile"

    def __init__(self,
                 structure_files: list[str],  # Structure files
                 log: logger.logging.Logger,
                 configs: AllConfig = AllConfig()
                 ) -> None:
        configs.structure_files = structure_files
        self.configs = configs
        self.initiate(log)
        self.write_msg(log)

    def initiate(self,
                 log: logger.logging.Logger
                 ) -> pd.DataFrame:
        """get all the infos"""
        strcuture_data: dict[str, pd.DataFrame] = ReadInputStructureFile(
            log, self.configs.structure_files).structure_dict
        self.force_field = ReadForceFieldFile(log)
        self.ff_radius: pd.DataFrame = self.compute_radius()
        self.generate_pqr(strcuture_data)

    def compute_radius(self) -> pd.DataFrame:
        """compute the radius based on sigma"""
        ff_radius: pd.DataFrame = self.force_field.ff_sigma.copy()
        radius = ff_radius['sigma'] * 1**(1/6) / 2
        ff_radius['radius'] = radius
        return ff_radius

    def generate_pqr(self,
                     strcuture_data: dict[str, pd.DataFrame]
                     ) -> None:
        """generate the pqr data and write them"""
        for fname, struct in strcuture_data.items():
            df_i: pd.DataFrame = self.get_atom_type(struct)
            df_i = self.set_radius(df_i)
            df_i = self.set_charge(df_i)
            df_i = self.assign_chain_ids(df_i)
            df_i = self.mk_pqr_df(df_i)
            self.write_pqr(f'{fname}.pqr', df_i)

    def get_atom_type(self,
                      struct: pd.DataFrame,
                      ) -> pd.DataFrame:
        """get atom type for each of them in the strcuture dataframe"""
        df_i: pd.DataFrame = struct.copy()
        df_i['atom_type'] = ['' for _ in range(len(struct))]
        for index, item in df_i.iterrows():
            res_i = self.configs.ff_type_dict[item['residue_name']]
            atom_i = item['atom_name']
            ff_i = self.force_field.ff_charge[res_i]

            atom_type_series = ff_i[ff_i['atomname'] == atom_i]['atomtype']
            if not atom_type_series.empty:
                atom_type = atom_type_series.values[0]
                df_i.at[index, 'atom_type'] = atom_type
            else:
                df_i.at[index, 'atom_type'] = 'nan'
        return df_i

    def set_radius(self,
                   df_i: pd.DataFrame
                   ) -> pd.DataFrame:
        # Merge to add radius based on atom_type
        df_i = df_i.merge(self.ff_radius[['name', 'radius']],
                          how='left',
                          left_on='atom_type',
                          right_on='name')
        df_i.drop(columns=['name'], inplace=True)
        return df_i

    def set_charge(self,
                   df_i: pd.DataFrame
                   ) -> pd.DataFrame:
        """set charge values for the atoms"""
        # print(self.force_field.ff_charge)
        df_i['charge'] = -2.0
        for index, row in df_i.iterrows():
            res: str = row['residue_name']
            ff_df: pd.DataFrame = \
                self.force_field.ff_charge[self.configs.ff_type_dict[res]]
            atom_type: str = row['atom_type']
            charge: float = \
                ff_df[ff_df['atomtype'] == atom_type]['charge'].values[0]
            df_i.at[index, 'charge'] = charge
        return df_i

    def assign_chain_ids(self,
                         df_i: pd.DataFrame
                         ) -> pd.DataFrame:
        """Factorize the residue names to get a unique ID for each
        unique name"""
        residue_ids: np.ndarray
        unique_residues: pd.core.indexes.base.Index
        residue_ids, unique_residues = pd.factorize(df_i['residue_name'])
        alphabet: list[str] = list(string.ascii_uppercase)
        chain_ids: list[str] = \
            alphabet + [f'{i}{j}' for i in alphabet for j in alphabet]
        # Map from factorized IDs to chain IDs
        residue_id_to_chain_id: dict[int, str] = \
            {i: chain_ids[i] for i in range(len(unique_residues))}

        # Apply the mapping to the factorized IDs
        df_i['chain_id'] = [residue_id_to_chain_id[id] for id in residue_ids]

        return df_i

    @staticmethod
    def mk_pqr_df(pdb_with_charge_radii: pd.DataFrame
                  ) -> pd.DataFrame:
        """prepare df in the format of the pqr file"""
        columns: list[str] = ['atom_id',
                              'atom_name',
                              'residue_name',
                              'chain_id',
                              'residue_number',
                              'x', 'y', 'z', 'charge', 'radius']
        float_columns: list[str] = ['x', 'y', 'z', 'charge', 'radius']
        df_i: pd.DataFrame = pdb_with_charge_radii[columns].copy()
        df_i[float_columns] = df_i[float_columns].astype(float)
        return df_i

    @staticmethod
    def write_pqr(pqr_file_name: str,
                  pqr_df: pd.DataFrame
                  ) -> None:
        """writing the pqr to a file"""
        with open(pqr_file_name, 'w', encoding='utf8') as f_w:
            for _, row in pqr_df.iterrows():
                line = f"ATOM  {row['atom_id']:>5} " \
                       f"{row['atom_name']:<4} " \
                       f"{row['residue_name']:<3} " \
                       f"{row['chain_id']:>1} " \
                       f"{row['residue_number']:>5} " \
                       f"{row['x']:>8.3f}" \
                       f"{row['y']:>8.3f}" \
                       f"{row['z']:>8.3f} " \
                       f"{row['charge']:>7.4f} " \
                       f"{row['radius']:>6.4f}\n"
                f_w.write(line)
            f_w.write('TER\n')
            f_w.write('END\n')

    def check_ff_files(self,
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
    structure_dict: dict[str, pd.DataFrame]

    def __init__(self,
                 log: logger.logging.Logger,
                 strucure_files: list[str],
                 configs: FileConfig = FileConfig()
                 ) -> None:
        self._configs = configs
        self.structure_dict = self._proccess_files(strucure_files, log)
        self._write_msg(log)

    def _proccess_files(self,
                        strucure_files: list[str],
                        log) -> dict[str, pd.DataFrame]:
        """read and return data about each structure file"""
        strucure_files = list((dict.fromkeys(strucure_files, '')).keys())
        self.file_type = self._check_file_extension(strucure_files, log)
        return self._read_files(strucure_files, log)

    def _read_files(self,
                    strucure_files: list[str],
                    log: logger.logging.Logger
                    ) -> dict[str, pd.DataFrame]:
        """read the files"""
        structure_dict: dict[str, pd.DataFrame] = {}
        if self.file_type == 'gro':
            for struct_i in strucure_files:
                fname_i: str = struct_i.split('.', -1)[0]
                structure_dict[fname_i] = \
                    gro_to_df.ReadGro(struct_i, log).gro_data
        else:
            for struct_i in strucure_files:
                fname_i = struct_i.split('.', -1)[0]
                structure_dict[fname_i] = \
                    pdb_to_df.Pdb(struct_i, log).pdb_df
        return structure_dict

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


class ReadForceFieldFile:
    """reading the force field files (itp files) and return dataframe
    with names of the atoms and thier charge and radius"""
    #  pylint: disable=too-few-public-methods

    info_msg: str = 'Message from ReadForceFieldFile:\n'
    _configs: AllConfig
    ff_sigma: pd.DataFrame  # From main itp file for getting sigma
    ff_charge: dict[str, pd.DataFrame]   # From itp files to get charges
    apbs_charmm: pd.DataFrame  # Radius of atoms from APBS

    def __init__(self,
                 log: logger.logging.Logger,
                 configs: AllConfig = AllConfig()
                 ) -> None:
        self._configs = configs
        ff_files: dict[str, typing.Any] = \
            force_field_path_configure.ConfigFFPath(log).ff_files
        self._procces_files(ff_files, log)
        self._write_msg(log)

    def _procces_files(self,
                       ff_files: dict[str, typing.Any],
                       log: logger.logging.Logger
                       ) -> None:
        """prccess the force filed files"""
        self.check_ff_files(ff_files, log)
        self.ff_sigma = \
            self._read_main_force_field(ff_files['all_atom_info'])
        self.ff_charge = \
            self._read_charge_of_atoms(ff_files)
        self.apbs_charmm = \
            self._read_apbs_charmm(ff_files['apbs_info'], log)

    def check_ff_files(self,
                       ff_files: dict[str, typing.Any],
                       log: logger.logging.Logger
                       ) -> None:
        """check all the existence of the all files"""
        for key, value in ff_files.items():
            if isinstance(value, str):
                # Value is a single file path
                if my_tools.check_file_exist(value, log) is False:
                    self.info_msg += f"\tFile does not exist: {value}\n"
            elif isinstance(value, list):
                # Value is a list of file paths
                for file_path in value:
                    if my_tools.check_file_exist(file_path, log) is False:
                        self.info_msg += \
                            f"\tFile does not exist: {file_path}\n"
            else:
                self.info_msg += \
                    f"Unexpected value type for key {key}: {type(value)}"

    def _read_main_force_field(self,
                               ff_file: str
                               ) -> pd.DataFrame:
        """reading the main force file file: charmm36_silica.itp"""
        return itp_to_df.Itp(ff_file, section='atomtypes').atomtypes

    def _read_apbs_charmm(self,
                          ff_file: str,
                          log: logger.logging.Logger
                          ) -> pd.DataFrame:
        """read charmm file from apbs file"""
        try:
            return parse_charmm_data.ParseData(ff_file, log).radius_df
        except (FileNotFoundError, FileExistsError):
            self.info_msg += '\tCHARMM file from apbs is not found\n'
            return pd.DataFrame()

    def _read_charge_of_atoms(self,
                              ff_files: dict[str, typing.Any]
                              ) -> dict[str, pd.DataFrame]:
        """reading the itp files which contains the charge of each atom
        used in the simulations:
        These files are:
        key: charge_info:
            'CLA.itp': charge for Cl ion
            'POT.itp': charge for Na ion
            'TIP3.itp': charge for water atoms
            'D10_charmm.itp': charge for the oil (Decane)
            'ODAp_charmm.itp': charges for the protonated ODA
        key: np_info:
            'APT_COR.itp': charge for the COR and APT of the NP
        Tha charge files are constant and not system dependent but the
        np_info depends on the simulation
        """
        charge_dict: dict[str, pd.DataFrame] = {}
        for fpath in ff_files['charge_info']:
            key = os.path.basename(fpath).split('.')[0]
            charge_dict[key] = itp_to_df.Itp(fpath, section='atoms').atoms
        charge_dict['np_info'] = \
            itp_to_df.Itp(ff_files['np_info'], section='atoms').atoms
        return charge_dict

    def _write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{ReadForceFieldFile.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    PdbToPqr(
        structure_files=sys.argv[1:], log=logger.setup_logger('pdb2pqr.log'))
