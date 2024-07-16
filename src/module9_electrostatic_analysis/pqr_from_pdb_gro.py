"""
This script is designed to generate parameterized PQR files from
structural data files (PDB or GRO format) using specified force field
parameters and topology files (ITP format). It integrates structural
information with atomistic details such as atomic charges and radii
derived from force field data, facilitating the creation of PQR files
necessary for molecular simulations and electrostatic analysis.

Main Components:
    - FileConfig: Data class that specifies the names and types of
    input and output files, including structure files (PDB/GRO), force
    field ITP file, and the output PQR file.
    - FFTypeConfig: Data class defining the mapping between residue
    names and their corresponding force field types, supporting
    customized force field assignments.
    - AllConfig: Inherits from FileConfig and FFTypeConfig to
    consolidate all configuration settings into a single, accessible
    object.
    - StructureToPqr: The core class responsible for orchestrating the
    conversion process, including reading input files, applying force
    field parameters (charges and radii), and generating the output
    PQR file.
    - ReadInputStructureFile: Utility class for reading and processing
    structural input files (PDB/GRO), ensuring compatibility with the
    desired force field parameters.
    - ReadForceFieldFile: Handles the extraction of force field
    parameters from ITP files, including atomic charges and radii, and
    organizes them for easy access and application to the structural
    data.
    - Utility Functions: Includes functions for assigning chain IDs,
    computing atomic radii, setting charges, and formatting the final
    PQR file output.

Workflow:
    1. Initialize the script with a list of structure files and a
    logging mechanism.
    2. Configuration settings (file names, force field mappings) are
    defined and passed to the StructureToPqr class.
    3. Structural input files are read, and relevant atomistic data
    (atom types, positions) are extracted.
    4. Force field parameters (atomic charges and radii) are applied
    based on the atom types and residue names, utilizing mappings
    defined in FFTypeConfig and data extracted by ReadForceFieldFile.
    5. Additional properties (e.g., chain IDs) are assigned, and the
    data is formatted according to PQR file standards.
    6. The final PQR file is written out for each input structure,
    ready for use in molecular simulation or analysis tools.

Usage:
    This script is intended to be used as part of a molecular modeling
    or simulation workflow, where PQR files are needed for
    electrostatics calculations or other analyses. It requires a
    proper environment with necessary dependencies (Pandas, NumPy,
    etc.) and access to relevant input files (PDB/GRO and ITP).

Example:
    To run the script from the command line, provide the paths to the
    structure files as arguments:
`python pqr_from_pdb.py structure1.pdb structure2.gro`

Ensure that the force field ITP files and any additional configuration
settings are correctly specified within the script or passed as
parameters.
doc. by ChatGpt
Saeed
12 Feb 2024
"""

import re
import os
import sys
import typing
import string
import datetime
from collections import Counter
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import common.logger as logger
import common.my_tools as my_tools
import common.itp_to_df as itp_to_df
import common.pdb_to_df as pdb_to_df
import common.gro_to_df as gro_to_df
import common.cpuconfig as cpuconfig
import common.file_writer as file_writer

from common.colors_text import TextColor as bcolors

import module9_electrostatic_analysis.parse_charmm_data as \
    parse_charmm_data
import module9_electrostatic_analysis.force_field_path_configure \
    as force_field_path_configure


@dataclass
class FileConfig:
    """Set the name of the input files"""
    structure_files: list[str] = field(init=False)
    itp_file: str = 'APT_COR.itp'  # FF of nanoparticle
    ff_user: str = 'CHARMM.DAT'  # Radius of the atoms in CAHRMM
    pqr_file: str = field(init=False)  # The output file to write, ext.: pqr
    accebtable_file_type: list[str] = \
        field(default_factory=lambda: ['gro', 'pdb'])
    out_xvg_file: dict[str, str] = field(
        default_factory=lambda: {'charge': 'charge_df.xvg',
                                 'number': 'number_df.xvg'})


@dataclass
class FFTypeConfig:
    """set the name of each file with their name of the data"""
    ff_type_dict: dict[str, str] = field(
        default_factory=lambda: {
            "SOL": 'TIP3',
            "CLA": 'CLA',
            "POT": 'POT',
            "D10": 'D10_charmm',
            "ODN": 'ODAp_charmm',
            'APT': 'np_info',
            'COR': 'np_info'
        })


@dataclass
class NumerInResidue:
    """Number of atoms in each residues
    The number of atoms in APT is not known since it changes based on
    contact angle"""
    res_number_dict: dict[str, int] = field(
        default_factory=lambda: {
            "SOL": 3,
            "CLA": 1,
            'POT': 1,
            "D10": 32,
            "ODN": 59,
            'COR': 4356,
            'APT': 4726
        })


@dataclass
class AllConfig(FileConfig, FFTypeConfig, NumerInResidue):
    """set all the configs
    parallel_type: str = 'threading' or 'multiprocessing'
    """
    n_cores: int = 1
    compute_radius: bool = True
    write_debug: bool = False
    run_parallel: bool = True
    paralle_type: str = 'multiprocessing'


class StructureToPqr:
    """
    preapre the file with positions, charges and radii
    """

    info_msg: str = 'Message from StructureToPqr:\n'
    configs: AllConfig
    force_field: "ReadForceFieldFile"
    pqr_files_list: list[str]  # Name of the pqr files outputs

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
        init_time: datetime.datetime = datetime.datetime.now()
        self.info_msg += f'\tThe script is initiated at `{init_time}`\n'
        charges_df: pd.DataFrame
        numbers_df: pd.DataFrame
        strcuture_info: "ReadInputStructureFile" = ReadInputStructureFile(
            log, self.configs.structure_files)
        strcuture_data: dict[str, pd.DataFrame] = strcuture_info.structure_dict
        self.file_type: str = strcuture_info.file_type
        self.force_field = ReadForceFieldFile(log)
        self.ff_radius: pd.DataFrame = self.compute_radius()
        if self.configs.run_parallel:
            self.set_number_of_cores(log, len(strcuture_data))
            charges_df, numbers_df = \
                self.generate_pqr_parallel(strcuture_data, log)
        else:
            charges_df, numbers_df = self.generate_pqr(strcuture_data, log)
        self.write_df_to_xvg(charges_df, log, 'charge')
        self.write_df_to_xvg(numbers_df, log, 'number')

    def compute_radius(self) -> pd.DataFrame:
        """compute the radius based on sigma"""
        ff_radius: pd.DataFrame = self.force_field.ff_sigma.copy()
        radius = ff_radius['sigma'] * 2**(1/6) / 2.0
        ff_radius['radius'] = radius
        return ff_radius

    def set_number_of_cores(self,
                            log: logger.logging.Logger,
                            n_files: int
                            ) -> None:
        """set the number of cores"""
        self.configs.n_cores = cpuconfig.ConfigCpuNr(log).cores_nr
        if self.configs.n_cores > n_files:
            self.configs.n_cores = n_files
            self.info_msg += \
                f'\tThe number of cores was set to {self.configs.n_cores}\n'

    def generate_pqr(self,
                     strcuture_data: dict[str, pd.DataFrame],
                     log: logger.logging.Logger
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """generate the pqr data and write them"""
        charges_dfs: list[pd.DataFrame] = []
        numbers_dfs: list[pd.DataFrame] = []
        charge_df_i: pd.DataFrame
        number_df_i: pd.DataFrame

        for fname, struct in strcuture_data.items():
            charge_df_i, number_df_i = \
                self._process_structure((fname, struct, log))
            charges_dfs.append(charge_df_i)
            numbers_dfs.append(number_df_i)
            self.info_msg += f'\tA pqr file writen as `{fname}`\n'
            self.info_msg += '\t' + '-' * 75 + '\n'
        charges_df: pd.DataFrame = pd.concat(charges_dfs)
        numbers_df: pd.DataFrame = pd.concat(numbers_dfs)
        return charges_df, numbers_df

    def generate_pqr_parallel(self,
                              structure_data: dict[str, pd.DataFrame],
                              log: logger.logging.Logger
                              ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """generate the pqr data and write them in parallel"""
        # Prepare data for multiprocessing
        pool_data: list[typing.Any] = [(fname, struct, log) for
                                       fname, struct in structure_data.items()]

        if self.configs.paralle_type == 'threading':
            return self.generate_pqr_therad(pool_data)
        return self.generate_pqr_multiprocessing(pool_data)

    def generate_pqr_multiprocessing(self,
                                     pool_data: list[typing.Any],
                                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """generate the pqr data and write them in parallel"""

        with Pool(processes=self.configs.n_cores) as pool:
            results = pool.map(self._process_structure, pool_data)
        return self._collect_results(results)

    def generate_pqr_therad(self,
                            pool_data: list[typing.Any],
                            ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """generate the pqr data and write them in parallel using threads"""

        with ThreadPoolExecutor(max_workers=self.configs.n_cores) as executor:
            results = executor.map(self._process_structure, pool_data)
        return self._collect_results(results)

    def _collect_results(self,
                         results: list[tuple[pd.DataFrame, pd.DataFrame]]
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
        charges_dfs: list[pd.DataFrame] = []
        numbers_dfs: list[pd.DataFrame] = []
        for charge_df, number_df in results:
            charges_dfs.append(charge_df)
            numbers_dfs.append(number_df)

        charges_df = pd.concat(charges_dfs)
        numbers_df = pd.concat(numbers_dfs)
        return charges_df, numbers_df

    def _process_structure(self,
                           fname_struct: tuple[
                               str, pd.DataFrame, logger.logging.Logger],
                           ) -> pd.DataFrame:
        """process the structure data"""
        fname, struct, log = fname_struct
        self.count_residues(fname, struct)
        df_i: pd.DataFrame = self.get_atom_type(struct)
        df_i = self.set_radius(df_i)
        df_i = self.set_charge(df_i, fname, log)
        df_i = self.assign_chain_ids(df_i)
        df_i = self.mk_pqr_df(df_i)
        df_i = self.convert_nm_ang(self.file_type, df_i)
        pqr_fname = f'{fname}.pqr'
        self.write_pqr(pqr_fname, df_i)
        charge_df = self._report_residue_charge(pqr_fname, df_i)
        del df_i
        return charge_df

    def count_residues(self,
                       fname: str,
                       struct: pd.DataFrame
                       ) -> None:
        """count the number of the each residue in each input"""
        residues: list[str] = list(struct['residue_name'])
        counts: "Counter" = Counter(residues)
        msg: str = f'\tNumber of each resdiue and atoms in `{fname}`:\n'
        for item, value in counts.items():
            msg += f'\t\t{item}: {value} atoms'
            if item != 'APT':
                msg += f' -> {value/self.configs.res_number_dict[item]} res\n'
            else:
                df_apt: pd.DataFrame = struct[struct['residue_name'] == 'APT']
                res_nr: int = len(set(df_apt['residue_number']))
                h_charge_nr: int = \
                    len(df_apt[df_apt['atom_name'] == 'HN3']['atom_name'])
                msg += f' -> {res_nr} res\n\t\t\t{h_charge_nr} charged APT\n'
        self.info_msg += msg

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
        """Merge to add radius based on atom_type"""
        df_i = df_i.merge(self.ff_radius[['name', 'radius']],
                          how='left',
                          left_on='atom_type',
                          right_on='name')
        df_i.drop(columns=['name'], inplace=True)
        self.info_msg += \
            f'\tThe number of atoms of this portion is: `{len(df_i)}`\n'
        return df_i

    def set_charge(self,
                   df_i: pd.DataFrame,
                   fname: str,
                   log: logger.logging.Logger
                   ) -> pd.DataFrame:
        """set charge values for the atoms"""
        df_i['charge'] = 0.0
        df_np: pd.DataFrame = df_i[
            (df_i['residue_name'] == 'COR') | (df_i['residue_name'] == 'APT')]
        df_oda: pd.DataFrame = df_i[df_i['residue_name'] == 'ODN']
        df_solution: pd.DataFrame = df_i[~((df_i['residue_name'] == 'COR') |
                                           (df_i['residue_name'] == 'APT') |
                                           (df_i['residue_name'] == 'ODN'))]

        if not df_solution.empty:
            df_solution = self._set_solution_charge(df_solution)

        if not df_oda.empty:
            df_oda = self._set_oda_charge(df_oda)

        if not df_np.empty:
            df_np = self._set_np_charge(df_np, fname, log)

        df_recombined = pd.concat([df_np, df_oda, df_solution])
        df_recombined = df_recombined.sort_index()
        self.info_msg += ('\tThe total charge of this portion is: '
                          f'`{sum(df_recombined["charge"]):.3f}`\n')
        del df_i
        del df_np
        del df_oda
        del df_solution
        return df_recombined

    def _report_residue_charge(self,
                               fname: str,
                               df_recombined: pd.DataFrame
                               ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """log all the charges for each residues"""
        resdiue_list: list[str] = list(self.configs.res_number_dict.keys())

        charge_df: pd.DataFrame = \
            pd.DataFrame(columns=['frame'] + resdiue_list)
        numbers_df: pd.DataFrame = \
            pd.DataFrame(columns=['frame'] + resdiue_list)

        charge_df['frame'] = [ind := self.extract_number_from_filename(fname)]
        numbers_df['frame'] = [ind]

        self.info_msg += '\tThe charges in each residue:\n'

        for res in resdiue_list:
            df_i: pd.DataFrame = \
                df_recombined[df_recombined['residue_name'] == res]
            total_charge: float = sum(df_i['charge'])
            total_atoms: int = len(df_i)

            try:
                nr_res = total_atoms / self.configs.res_number_dict[res]
            except ZeroDivisionError:
                nr_res = 0

            charge_df[res] = [total_charge]
            numbers_df[res] = [nr_res]

            if self.configs.write_debug:
                df_i.to_csv(f'{res}_charge_debug', sep=' ')

            del df_i
        del df_recombined

        return charge_df, numbers_df

    @staticmethod
    def extract_number_from_filename(filename: str
                                     ) -> int:
        """extract the number from the filename"""
        match = re.search(r'(\d+)', filename)
        if match:
            return int(match.group(1))
        raise ValueError(f"No integer found in {filename}")

    def _set_oda_charge(self,
                        df_oda: pd.DataFrame
                        ) -> pd.DataFrame:
        """set the charges for ODA"""
        res_indices: list[int] = list(set(df_oda['residue_number']))
        ff_df: pd.DataFrame = \
            self.force_field.ff_charge[self.configs.ff_type_dict['ODN']]
        df_list: list[pd.DataFrame] = []
        for res in res_indices:
            df_i = df_oda[df_oda['residue_number'] == res].copy()
            df_i['charge'] = list(ff_df['charge'])
            df_list.append(df_i)
        df_updated: pd.DataFrame = pd.concat(df_list)
        del df_oda
        del df_list
        return df_updated.sort_index()

    def _set_solution_charge(self,
                             df_solution: pd.DataFrame
                             ) -> pd.DataFrame:
        """set the charges for the section without np"""
        for index, row in df_solution.iterrows():
            res: str = row['residue_name']
            ff_df: pd.DataFrame = \
                self.force_field.ff_charge[self.configs.ff_type_dict[res]]
            atom_type: str = row['atom_type']
            charge: float = \
                ff_df[ff_df['atomtype'] == atom_type]['charge'].values[0]
            df_solution.at[index, 'charge'] = float(charge)
        self.info_msg += ('\tTotal charge of the no_np section is: '
                          f'{sum(df_solution["charge"]):.3f}\n')
        return df_solution

    def _set_np_charge(self,
                       df_np: pd.DataFrame,
                       fname: str,
                       log: logger.logging.Logger
                       ) -> pd.DataFrame:
        """set the charges for the np section"""
        df_i: pd.DataFrame = df_np.copy()
        if len(df_np) == len(
           ff_df := self.force_field.ff_charge['np_info']):
            charge: np.ndarray = ff_df['charge'].values
            df_i['charge'] = charge

            atom_ff_name: np.ndarray = ff_df['atomname'].values
            atom_pos_name: np.ndarray = df_np['atom_name'].values
            # Check if atom names are the same
            self._non_equal_names_error(
                atom_ff_name, atom_pos_name, fname, log)

            self.info_msg += ('\tTotal charge of the np section is: '
                              f'{sum(df_i["charge"]):.3f}\n')

        else:
            log.error(msg := '\tError! There is problem in np data!\n')
            sys.exit(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')

        del ff_df
        del df_np

        return df_i

    def _non_equal_names_error(self,
                               atom_ff_name: np.ndarray,
                               atom_pos_name: np.ndarray,
                               fname: str,
                               log: logger.logging.Logger
                               ) -> None:
        """check if the names arent the same"""
        if not np.array_equal(atom_ff_name, atom_pos_name):
            diff_indices = [i for i, (a, b) in enumerate(
                zip(atom_ff_name, atom_pos_name)) if a != b]
            log.error(
                '\tError! The atom names are not the same at indices: '
                + ', '.join(map(str, diff_indices)))
            for i in diff_indices:
                print(
                    f'{bcolors.CAUTION}\tIndex {i}: ff_df atom name = '
                    f'{atom_ff_name[i]}, df_i atom name = '
                    f'{atom_pos_name[i]} {bcolors.ENDC}\n')
            sys.exit(
                f'{bcolors.FAIL}\n\tError! The atom names are not '
                f'the same in `{fname}` for APT_COR!{bcolors.ENDC}\n')

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
        del pdb_with_charge_radii
        return df_i

    @staticmethod
    def convert_nm_ang(file_type: str,
                       df_i: pd.DataFrame) -> pd.DataFrame:
        """convert the unit of data"""
        if file_type == 'gro':
            columns: list[str] = ['x', 'y', 'z', 'radius']
        else:
            columns = ['radius']
        df_i[columns] *= 10
        return df_i

    @staticmethod
    def write_pqr(pqr_file_name: str,
                  pqr_df: pd.DataFrame
                  ) -> None:
        """writing the pqr to a file"""
        with open(pqr_file_name, 'w', encoding='utf8') as f_w:
            print(f'{bcolors.OKGREEN}\tWriting the pqr file as '
                  f'`{pqr_file_name}`{bcolors.ENDC}')
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
        del pqr_df

    def write_df_to_xvg(self,
                        df_i: pd.DataFrame,
                        log: logger.logging.Logger,
                        df_type: str
                        ) -> None:
        """write the charge data to a xvg file"""
        # sort the dataframe based on the frame
        df_i = df_i.sort_values(by='frame')
        row_sums: pd.Series = df_i.drop(columns=['frame']).sum(axis=1)
        self._check_total_charge(row_sums)
        df_i[f'total_{df_type}'] = row_sums
        df_i = df_i.set_index('frame', drop=True)
        extra_comments: str = f'# {df_type} in each residue'
        file_writer.write_xvg(df_i,
                              log,
                              fname := self.configs.out_xvg_file[df_type],
                              extra_comments=extra_comments,
                              xaxis_label='Frame index',
                              yaxis_label=f'"{df_type}" in each residue',
                              title=f'{df_type} information')
        self.info_msg += \
            f'\tData was written to `{fname}`\n'
        del df_i

    @staticmethod
    def _check_total_charge(row_sums: pd.Series
                            ) -> None:
        """check the total charge"""
        all_whole = all(round(value, 3) % 1 == 0 for value in row_sums)
        if not all_whole:
            raise ValueError(
                f'\n{bcolors.FAIL}The total charge is not a whole number!'
                f'{bcolors.ENDC}')

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
        final_time: datetime.datetime = datetime.datetime.now()
        self.info_msg += f'\tThe script is finished at `{final_time}`\n'
        print(f'{bcolors.OKCYAN}{StructureToPqr.__name__}:\n'
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
        strucure_files = [
            file for file in strucure_files
            if my_tools.check_int_in_filename(os.path.basename(file))]
        if not strucure_files:
            log.error(msg := '\tNo structure files are provided!\n')
            sys.exit(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')

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
                fname_i: str = os.path.splitext(os.path.basename(struct_i))[0]
                structure_dict[fname_i] = \
                    gro_to_df.ReadGro(struct_i, log).gro_data
        else:
            for struct_i in strucure_files:
                fname_i = os.path.splitext(os.path.basename(struct_i))[0]
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
            [os.path.splitext(item)[1][1:] for item in strucure_files]
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
                        '\tshould be one of the\n'
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
    StructureToPqr(
        structure_files=sys.argv[1:], log=logger.setup_logger('pdb2pqr.log'))
