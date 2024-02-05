"""
make the data from the pdb and itp

"""

import sys
from dataclasses import dataclass, field

import pandas as pd

from common import logger, itp_to_df, pdb_to_df, my_tools
from common.colors_text import TextColor as bcolors

from module9_electrostatic_analysis import parse_charmm_data


@dataclass
class FileConfig:
    """Set the name of the input files"""
    pdb_file: str = field(init=False)  # Structure file
    itp_file: str = 'APT_COR.itp'  # FF of nanoparticle
    ff_file: str = 'CHARMM.DAT'  # Radius of the atoms in CAHRMM
    pqr_file: str = field(init=False)  # The output file to write, ext.: pqr


@dataclass
class AllConfig(FileConfig):
    """set all the configs"""


class PdbToPqr:
    """
    preapre the file with positions, charges and radii
    """

    info_msg: str = 'Message from PdbToPqr:\n'
    configs: AllConfig

    def __init__(self,
                 pdb_file: str,  # Name of the structure file
                 log: logger.logging.Logger,
                 configs: AllConfig = AllConfig()
                 ) -> None:
        configs.pdb_file = pdb_file
        self.configs = configs
        self.initiate(log)
        self.write_msg(log)

    def initiate(self,
                 log: logger.logging.Logger
                 ) -> pd.DataFrame:
        """get all the infos"""

        self.check_all_file(log)

        itp: pd.DataFrame = itp_to_df.Itp(self.configs.itp_file).atoms
        pdb: pd.DataFrame = pdb_to_df.Pdb(self.configs.pdb_file, log).pdb_df
        force_field: pd.DataFrame = \
            parse_charmm_data.ParseData(self.configs.ff_file, log).radius_df

        pdb_with_charges: pd.DataFrame = self.get_charges(pdb, itp)
        pdb_with_charge_radii: pd.DataFrame = \
            self.set_radii(pdb_with_charges, force_field, itp)

        pdb_df: pd.DataFrame = self.add_chain_identifier(pdb_with_charge_radii)
        pqr_df: pd.DataFrame = self.mk_pqr_df(pdb_df)

        self._set_outfile_name()
        self.write_pqr(self.configs.pqr_file, pqr_df)

    def check_all_file(self,
                       log: logger.logging.Logger
                       ) -> None:
        """check all the existence of the all files"""
        for file in [self.configs.ff_file,
                     self.configs.itp_file,
                     self.configs.pdb_file]:
            my_tools.check_file_exist(file, log)

    def get_charges(self,
                    pdb: pd.DataFrame,
                    itp: pd.DataFrame
                    ) -> pd.DataFrame:
        """get charges of the atoms in the pdb file"""
        aptes_with_charges: pd.DataFrame = self.set_aptes_charges(pdb, itp)
        cores_with_charges: pd.DataFrame = self.set_cores_charges(pdb, itp)
        return pd.concat(
            [cores_with_charges, aptes_with_charges], axis=0, ignore_index=True
            )

    def set_radii(self,
                  pdb_with_charges: pd.DataFrame,
                  force_field: pd.DataFrame,
                  itp: pd.DataFrame
                  ) -> pd.DataFrame:
        """set the radii for all"""
        aptes_df: pd.DataFrame = \
            self.set_radii_for_aptes(pdb_with_charges, force_field)
        cores_df: pd.DataFrame = \
            self.set_radii_for_cores(pdb_with_charges, force_field, itp)
        return pd.concat([cores_df, aptes_df], axis=0, ignore_index=True)

    def _set_outfile_name(self) -> str:
        """set the name of the output based on the structure file"""
        struct_name: str = self.configs.pdb_file.split('.')[0]
        self.configs.pqr_file = f'{struct_name}.pqr'
        self.info_msg += \
            f'\tThe output will be save as: `{self.configs.pqr_file}`\n'

    @staticmethod
    def add_chain_identifier(pdb_df: pd.DataFrame
                             ) -> pd.DataFrame:
        """add the column"""
        chain_identifier_map = {
            'COR': 'A',
            'APT': 'B'
        }
        pdb_df['chain_id'] = \
            pdb_df['residue_name'].map(chain_identifier_map)
        return pdb_df

    @staticmethod
    def mk_pqr_df(pdb_with_charge_radii: pd.DataFrame
                  ) -> pd.DataFrame:
        """prepare df in the format of the pqr file"""
        columns: list[str] = ['records',
                              'atom_id',
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

    @staticmethod
    def set_radii_for_aptes(pdb_with_charges: pd.DataFrame,
                            force_field: pd.DataFrame
                            ) -> pd.DataFrame:
        """get the radius from the file"""
        aptes_radii: pd.DataFrame = \
            force_field[force_field['resname'] == 'APT'].copy()
        aptes_df = pd.DataFrame = \
            pdb_with_charges[pdb_with_charges['residue_name'] == 'APT'].copy()
        aptes_df = pd.merge(aptes_df,
                            aptes_radii[['atom_name', 'radius']],
                            on='atom_name',
                            how='left')
        return aptes_df

    @staticmethod
    def set_radii_for_cores(pdb_with_charges: pd.DataFrame,
                            force_field: pd.DataFrame,
                            itp: pd.DataFrame
                            ) -> pd.DataFrame:
        """get the radius from the file"""
        cores_radii: pd.DataFrame = \
            force_field[force_field['resname'] == 'COR'].copy()
        cores_df = pd.DataFrame = \
            pdb_with_charges[pdb_with_charges['residue_name'] == 'COR'].copy()
        cores_df['atomtype'] = itp[itp['resname'] == 'COR']['atomtype'].copy()
        cores_df = pd.merge(cores_df,
                            cores_radii[['atom_name', 'radius']],
                            left_on='atomtype',
                            right_on='atom_name',
                            how='left')
        cores_df['atom_name'] = cores_df['atom_name_x'].copy()
        return cores_df.drop(
            ['atomtype', 'atom_name_x', 'atom_name_y'], axis=1)

    def set_aptes_charges(self,
                          pdb: pd.DataFrame,
                          itp: pd.DataFrame
                          ) -> pd.DataFrame:
        """get charges for different aptes longs"""
        itp_df: pd.DataFrame = itp[itp['resname'] == 'APT']
        apt_df: pd.DataFrame = pdb[pdb['residue_name'] == 'APT']
        pro_charges: pd.DataFrame
        unpro_charges: pd.DataFrame
        pro_charges, unpro_charges = self.get_aptes_charges(itp_df)
        dfs_with_charges: list[pd.DataFrame] = []
        for _, group in apt_df.groupby('residue_number'):
            if len(group) == 13:
                dfs_with_charges.append(self.add_charge_column_to_df(
                    group, pro_charges))
            elif len(group) == 12:
                dfs_with_charges.append(self.add_charge_column_to_df(
                    group, unpro_charges))
        return pd.concat(dfs_with_charges, axis=0, ignore_index=True)

    def set_cores_charges(self,
                          pdb: pd.DataFrame,
                          itp: pd.DataFrame
                          ) -> pd.DataFrame:
        """set the charges for the core atoms of the nano particles
        since only the charges on the shell is important, only those
        are checked
        """
        itp_df: pd.DataFrame = itp[itp['resname'] == 'COR'].copy()
        cor_df: pd.DataFrame = pdb[pdb['residue_name'] == 'COR'].copy()
        if not cor_df['atom_name'].equals(itp_df['atomname']):
            sys.exit("The columns name are different!\n")
        cor_charges: pd.DataFrame = itp[['atomnr', 'charge']].copy()
        cor_charges['atomnr'] = cor_charges['atomnr'].astype(str)
        cor_df['atom_id'] = cor_df['atom_id'].astype(str)
        cor_df = pd.merge(cor_df,
                          cor_charges[['atomnr', 'charge']],
                          left_on='atom_id',
                          right_on='atomnr',
                          how='left')
        return cor_df.drop('atomnr', axis=1)

    @staticmethod
    def add_charge_column_to_df(main_df: pd.DataFrame,
                                charge_df: pd.DataFrame
                                ) -> pd.DataFrame:
        """add charge column to the dataframe"""
        tmp_df = main_df.copy()
        tmp_df = pd.merge(tmp_df,
                          charge_df[['atomname', 'charge']],
                          left_on='atom_name',
                          right_on='atomname',
                          how='left')
        return tmp_df.drop('atomname', axis=1)

    @staticmethod
    def get_aptes_charges(itp_df: pd.DataFrame
                          ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """return charges for pro- and unprotonated aptes"""
        pro_found: bool = False
        unpro_found: bool = False
        for _, group in itp_df.groupby('resnr'):
            if len(group) == 13 and not pro_found:
                pro_charges: pd.DataFrame = \
                    group[['atomname', 'charge']].copy()
                pro_found = True
            elif len(group) == 12 and not unpro_found:
                unpro_charges: pd.DataFrame = \
                    group[['atomname', 'charge']].copy()
                unpro_found = True
            if pro_found and unpro_found:
                break
        return pro_charges, unpro_charges

    def write_msg(self,
                   log: logger.logging.Logger  # To log
                   ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{PdbToPqr.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)

if __name__ == '__main__':
    PdbToPqr(pdb_file=sys.argv[1], log=logger.setup_logger('pdb2pqr.log'))
