"""
Dynamic Trajectory Filtering and Analysis Script

This script performs dynamic selection of atoms within a specified
radius from the center of mass (COM) of a defined target (e.g., a
nanoparticle) in molecular dynamics (MD) simulations. For each frame
of the input trajectory, it identifies atoms (and their respective
esidues) within the defined radius, analyzes specified properties of
this dynamic selection, and writes a new trajectory file containing
only the selected atoms for further analysis.

The script is designed to facilitate the study of dynamic interactions
and the local environment around specified targets in MD simulations,
providing insights into spatial and temporal variations in properties
such as density, coordination number, or other custom analyses.

Usage:
  python spatial_filtering_and_analysing_trr.py  <input_trajectory>
The tpr file and gro file names will be created from the name of the
input trr file.
Some of the functionality is similar as in pqr_from_pdb_gro.py

Arguments that set by the script:
    topology_file : Topology file (e.g., .gro, .pdb) corresponding to
        the input trajectory.
    output_trajectory: Output trajectory file name for the filtered
        selection.
    radius: Radius (in Ångströms) for dynamic selection around the
        target's COM.
    statistics output: Output file name for numbers of residues and
        charges of each of them.
    charge density output: Output file name for the charge density and
        potential of the final system

Options:
    selection_string : MDAnalysis-compatible selection string for
        defining the target (default: 'COR_APT').

Features:
    - Dynamic selection based on spatial criteria, adjustable for each
        frame of the trajectory.
    - Analysis of selected properties for the dynamically selected
        atoms/residues.
    - Generation of a new, filtered trajectory file for targeted
        further analysis.

Example:
    python spatial_filtering_and_analysing_trr.py simulation.trr

Opt. by ChatGpt.
21 Feb 2023
Saeed
"""
import os
import sys
import typing
from datetime import datetime
from dataclasses import field
from dataclasses import dataclass
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd

import MDAnalysis as mda

from common import logger
from common import my_tools
from common import cpuconfig
from common import itp_to_df
from common.colors_text import TextColor as bcolors

from module9_electrostatic_analysis import force_field_path_configure


@dataclass
class InFileConfig:
    """Set the name of the input files"""
    # Name of the input file, set by user:
    trajectory: str = field(init=False)
    topology: str = field(init=False)

    # ForceField file to get the charges of the nanoparticle, since
    # they are different depend on the contact angle:
    itp_file: str = 'APT_COR.itp'  # FF of nanoparticle

    accebtable_file_type: list[str] = \
        field(default_factory=lambda: ['trr', 'xtc'])


@dataclass
class OutFileConfig:
    """set the names of the output files"""
    filtered_traj: str = field(init=False)  # Will set based on input
    fout_stats: str = 'stats.xvg'
    fout_charge_density: str = 'charge_density.xvg'


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
            'COR': 'np_info',
            'POT': 'POT'
        })


@dataclass
class NubmerInResidue:
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
            'APT': 0
        })


@dataclass
class ChainIdConfig:
    """set the name of each file with their name of the data"""
    chain_id_dict: dict[str, str] = field(
        default_factory=lambda: {
            "SOL": 'A',
            "D10": 'B',
            "CLA": 'C',
            "POT": 'D',
            "ODN": 'E',
            'APT': 'F',
            'COR': 'G'
        })


@dataclass
class GroupName:
    """set the names of atoms for the groups
    e.g., np_group: str = 'resname COR APT'
    """
    np_group: str = 'resname COR APT'


@dataclass
class DebugConfig:
    """set options for configurations of computaion"""
    filter_debug: dict[str, typing.Any] = field(default_factory=lambda: {
        'if': False,
        'suffix': '_filter_debug.pdb',
        'indices': [0, 1]
    })
    charge_debug: dict[str, typing.Any] = field(default_factory=lambda: {
        'if': False,
        'suffix': '_charge_debug_df',
        'indices': [0, 1]
    })
    counts_df_debug: dict[str, typing.Any] = field(default_factory=lambda: {
        'if': False,
        'suffix': '_final_debug_df',
    })


@dataclass
class AllConfig(InFileConfig,
                OutFileConfig,
                FFTypeConfig,
                NubmerInResidue,
                ChainIdConfig,
                GroupName,
                DebugConfig
                ):
    """set all the configurations and parameters"""
    stern_radius: float = 30  # In Ångströms
    if_parallel: bool = True


class TrrFilterAnalysis:
    """get the trajectory and do the analysis"""
    # pylint: disable=too-many-instance-attributes

    info_msg: str = 'Message from TrrFilterAnalysis:\n'
    configs: AllConfig
    force_field: "ReadForceFieldFile"
    ff_radius: pd.DataFrame
    num_frames: int
    com_list: list[np.ndarray]  # COM of the nanoparticle
    df_numbers: pd.DataFrame
    df_charges: pd.DataFrame

    def __init__(self,
                 trajectory: str,
                 log: logger.logging.Logger,
                 configs: AllConfig = AllConfig()
                 ) -> None:
        self.configs = configs
        self.configs.trajectory = trajectory
        self.initiate(log)
        self.write_msg(log)

    def initiate(self,
                 log: logger.logging.Logger
                 ) -> None:
        """setting the names, reading files, filttering traj file and
        analaysing and writting output"""

        sel_list: list["mda.core.groups.AtomGroup"]  # All atoms in radius

        self.set_check_in_files(log)
        self.force_field = ReadForceFieldFile(log)
        self.ff_radius: pd.DataFrame = self.compute_radius()
        self.com_list, sel_list = self.read_trajectory()
        if self.configs.if_parallel:
            self.df_numbers, self.df_charges = \
                self.analyzing_frames_parallel(sel_list, log)
        else:
            self.df_numbers, self.df_charges = \
                self.analyzing_frames(sel_list, log)
        if self.configs.counts_df_debug['if']:
            self.df_charges.to_csv(
                f'charges{self.configs.counts_df_debug["suffix"]}', sep=' ')
            self.df_numbers.to_csv(
                f'numbers{self.configs.counts_df_debug["suffix"]}', sep=' ')

    def set_check_in_files(self,
                           log: logger.logging.Logger
                           ) -> None:
        """set the names and check if they exist"""
        root_name: str = self.configs.trajectory.split('.', -1)[0]
        tpr: str = f'{root_name}.tpr'
        gro: str = f'{root_name}.gro'
        if (if_tpr := my_tools.check_file_exist(tpr, log, False)) is None:
            self.configs.topology = tpr
            self.info_msg += f'\tThe topology file is set to `{tpr}`\n'
        elif if_tpr is False:
            if my_tools.check_file_exist(gro, log, False) is None:
                self.info_msg += f'\tThe topology file is set to `{tpr}`\n'
                self.configs.topology = gro
            else:
                log.error(msg := f'\tError! `{gro}` or `{tpr}` not exist!\n')
                sys.exit(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')

    def compute_radius(self) -> pd.DataFrame:
        """Compute the radius based on sigma.

        Returns:
            pd.DataFrame: A DataFrame containing the computed radius values.
        """
        ff_radius: pd.DataFrame = self.force_field.ff_sigma.copy()
        radius = ff_radius['sigma'] * 2**(1/6) / 2
        ff_radius['radius'] = radius
        return ff_radius

    def read_trajectory(self
                        ) -> tuple[list[np.ndarray],
                                   list["mda.core.groups.AtomGroup"]]:
        """read the traj file"""
        # pylint: disable=unsubscriptable-object
        # pylint: disable=too-many-locals

        u_traj = \
            mda.Universe(self.configs.topology, self.configs.trajectory)
        nanoparticle = u_traj.select_atoms(self.configs.np_group)
        self.num_frames = u_traj.trajectory.n_frames
        com_list: list[np.ndarray] = []
        sel_list: list["mda.core.groups.AtomGroup"] = []

        for tstep in u_traj.trajectory:
            com = nanoparticle.center_of_mass()
            com_list.append(com)

            # Manual selection based on distance calculation
            all_atoms = u_traj.atoms.positions
            distances = np.linalg.norm(all_atoms - com, axis=1)
            within_radius_indices = \
                np.where(distances <= self.configs.stern_radius)[0]

            # Ensuring APT and COR residues are always included
            apt_cor_residues = \
                u_traj.select_atoms("resname APT or resname COR")
            apt_cor_indices = set(apt_cor_residues.indices)
            within_radius_set = set(within_radius_indices)
            combined_indices_set = within_radius_set.union(apt_cor_indices)

            combined_indices_list = sorted(list(combined_indices_set))
            combined_indices_array = np.array(combined_indices_list)

            selected_atoms = u_traj.atoms[combined_indices_array]
            sel_list.append(selected_atoms.residues.atoms)
            if self.configs.filter_debug['if']:
                time = tstep.time
                if time in self.configs.filter_debug['indices']:
                    fout: str = \
                        f'sel_{int(time)}{self.configs.filter_debug["suffix"]}'
                    with mda.Writer(fout, reindex=False, bonds=None) as f_w:
                        f_w.write(selected_atoms.residues.atoms)
                    self.info_msg += f'\t{fout} is written for debugging\n'
        return com_list, sel_list

    def analyzing_frames(self,
                         sel_list: list["mda.core.groups.AtomGroup"],
                         log: logger.logging.Logger
                         ) -> tuple[pd.DataFrame, ...]:
        """analaysing each frame by counting the number of atoms and
        residues"""
        count_nr_dir: list[dict[str, float]] = []
        count_q_dir: list[dict[str, float]] = []

        for frame_i, frame in enumerate(sel_list):
            count_nr, count_q = self.process_frame((frame_i, frame, log))
            count_nr_dir.append(count_nr)
            count_q_dir.append(count_q)

        df_nr: pd.DataFrame = pd.DataFrame.from_dict(count_nr_dir)
        df_q: pd.DataFrame = pd.DataFrame.from_dict(count_q_dir)
        self._write_xvg(df_nr, log, 'number')
        self._write_xvg(df_q, log, 'charge')
        return df_nr, df_q

    def analyzing_frames_parallel(self,
                                  sel_list: list["mda.core.groups.AtomGroup"],
                                  log: logger.logging.Logger
                                  ) -> tuple[pd.DataFrame, ...]:
        """analysing the frames in parallel"""
        n_cores: int = self._initiate_cpu(log)
        # Prepare data for multiprocessing
        frame_data_list = [(i, frame, log) for
                           i, frame in enumerate(sel_list)]

        # Initialize multiprocessing pool
        with Pool(processes=n_cores) as pool:
            results = pool.map(self.process_frame, frame_data_list)

        # Unpack results
        count_nr_dir = [result[0] for result in results]
        count_q_dir = [result[1] for result in results]

        # Convert results to DataFrames
        df_nr = pd.DataFrame.from_dict(count_nr_dir)
        df_q = pd.DataFrame.from_dict(count_q_dir)
        self._write_xvg(df_nr, log, 'number')
        self._write_xvg(df_q, log, 'charge')
        return df_nr, df_q

    def process_frame(self,
                      frame_data,
                      ) -> tuple[dict[str, float], ...]:
        """process single frame"""
        frame_i: int  # Index of the frame
        frame: "mda.core.groups.AtomGroup"
        log: logger.logging.Logger

        frame_i, frame, log = frame_data

        df_frame: pd.DataFrame
        df_frame = self._get_gro_df(frame)
        df_frame = self._assign_chain_ids(df_frame)
        df_frame = self._get_atom_type(df_frame)
        df_frame = self._set_radius(df_frame)
        df_frame = self._set_charge(frame_i, df_frame, log)
        count_nr: dict[str, float] = \
            self._count_residues(frame_i, df_frame, log)
        count_q: dict[str, float] = \
            self._report_residue_charge(frame_i, df_frame)

        return count_nr, count_q

    def _initiate_cpu(self,
                      log: logger.logging.Logger
                      ) -> int:
        """
        Return the number of cores to use based on the data and the machine
        """
        cpu_info = cpuconfig.ConfigCpuNr(log)
        n_cores: int = min(cpu_info.cores_nr, self.num_frames)
        self.info_msg += f'\tThe number of cores to use: {n_cores}\n'
        return n_cores

    def _get_gro_df(self,
                    frame: "mda.core.groups.AtomGroup"
                    ) -> pd.DataFrame:
        """put the frame data into gro format"""

        residue_indices: list[int] = [atom.resindex for atom in frame.atoms]
        residue_names: list[str] = [atom.resname for atom in frame.atoms]
        atom_names: list[str] = [atom.name for atom in frame.atoms]
        atom_ids: list[int] = [atom.id for atom in frame.atoms]
        positions: np.ndarray = frame.atoms.positions

        return pd.DataFrame({
            'residue_number': residue_indices,
            'residue_name': residue_names,
            'atom_name': atom_names,
            'atom_id': atom_ids,
            'x': positions[:, 0],
            'y': positions[:, 1],
            'z': positions[:, 2]
        })

    def _assign_chain_ids(self,
                          df_i: pd.DataFrame
                          ) -> pd.DataFrame:
        """Factorize the residue names to get a unique ID for each
        unique name"""
        chain_ids: list[str] = \
            [self.configs.chain_id_dict[item] for item in df_i['residue_name']]
        df_i['chain_id'] = chain_ids
        return df_i

    def _get_atom_type(self,
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

    def _set_radius(self,
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

    def _count_residues(self,
                        frame: int,
                        df_frame: pd.DataFrame,
                        log: logger.logging.Logger
                        ) -> dict[str, float]:
        """count the number of the each residue in each input"""
        nr_dict: dict[str, float] = {
            "frame": frame,
            "SOL": 0.0,
            "D10": 0.0,
            "CLA": 0.0,
            "POT": 0.0,
            "ODN": 0.0,
            'APT': 0.0,
            'COR': 0.0}
        residues: list[str] = list(df_frame['residue_name'])
        counts: "Counter" = Counter(residues)
        msg = '\tThe Number of atoms in each residue is:\n'
        for item, value in counts.items():
            msg += f'\t\t{item}: {value} atoms'
            if item != 'APT':
                nr_i: float = value/self.configs.res_number_dict[item]
                if nr_i != int(nr_i):
                    log.error(msg := (f'Error! The residue `{item}` is '
                                      f'not complete number {nr_i}!\n'))
                    sys.exit(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')

                nr_dict[item] = value/self.configs.res_number_dict[item]
                msg += f' -> {nr_dict[item]} res\n'
            else:
                df_apt: pd.DataFrame = df_frame[df_frame['atom_name'] == 'N']
                res_nr: int = len(set(df_apt['residue_number']))
                nr_dict[item] = res_nr
                h_charge_nr: int = \
                    len(df_frame[df_frame['atom_name'] == 'HN3']['atom_name'])
                msg += f' -> {res_nr} res\n\t\t\t{h_charge_nr} charged APT\n'
        self.info_msg += msg
        self.info_msg += '\t' + '-' * 75 + '\n'
        return nr_dict

    def _set_charge(self,
                    frame_i: int,
                    df_i: pd.DataFrame,
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
            df_np = self._set_np_charge(df_np, log)

        df_recombined = pd.concat([df_np, df_oda, df_solution])
        df_recombined = df_recombined.sort_index()
        self._report_residue_charge(frame_i, df_recombined)
        self.info_msg += ('\tThe total charge of this portion is: '
                          f'`{sum(df_recombined["charge"]):.3f}`\n')
        return df_recombined

    def _report_residue_charge(self,
                               frame_i: int,
                               df_recombined: pd.DataFrame
                               ) -> dict[str, float]:
        """log all the charges for each residues"""
        charge_dict: dict[str, float] = {
            "frame": frame_i,
            "total": 0.0,
            "q_density [e/nm2]": 0.0,
            "SOL": 0.0,
            "D10": 0.0,
            "CLA": 0.0,
            "POT": 0.0,
            "ODN": 0.0,
            'APT': 0.0,
            'COR': 0.0}
        self.info_msg += '\tThe charges in each residue:\n'
        total_q: float = 0.0
        for res in self.configs.res_number_dict.keys():
            df_i: pd.DataFrame = \
                df_recombined[df_recombined['residue_name'] == res]
            total_res_charge: float = sum(df_i['charge'])
            self.info_msg += f'\t\t{res}: {total_res_charge:.3f}\n'
            charge_dict[res] = round(total_res_charge, 2)
            total_q += total_res_charge
            if (self.configs.charge_debug['if'] and
               frame_i in self.configs.charge_debug['indices']):
                df_i.to_csv(
                    f'{res}_{frame_i}{self.configs.charge_debug["suffix"]}',
                    sep=' ')
            del df_i
        total_q = round(total_q, 2)
        charge_dict['total'] = total_q
        q_density: float = \
            total_q / (4 * np.pi * (self.configs.stern_radius/10)**2)
        charge_dict['q_density [e/nm2]'] = round(q_density, 4)
        return charge_dict

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
                       log: logger.logging.Logger
                       ) -> pd.DataFrame:
        """set the charges for the np section"""
        np_flag: bool = True
        if len(df_np) == len(
           ff_df := self.force_field.ff_charge['np_info']):
            for index, row in df_np.iterrows():
                if np_flag:
                    np_id_zero: int = int(row['atom_id'])
                    np_flag = False
                atom_id: int = int(row['atom_id'] - np_id_zero + 1)
                res_nr: int = row['residue_number']
                try:
                    charge = \
                        ff_df[(ff_df['atomnr'] == atom_id) &
                              (ff_df['resnr'] == res_nr)
                              ]['charge'].values[0]
                except IndexError:
                    charge = \
                        ff_df[
                            ff_df['atomnr'] == atom_id]['charge'].values[0]
                df_np.at[index, 'charge'] = float(charge)
            self.info_msg += ('\tTotal charge of the np section is: '
                              f'{sum(df_np["charge"]):.3f}\n')
        else:
            log.error(msg := '\tError! There is problem in np data!\n')
            sys.exit(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')
        return df_np

    def _write_xvg(self,
                   df_i: pd.DataFrame,
                   log: logger.logging.Logger,
                   fout_prefix: str
                   ) -> None:
        """write the rdf to a xvg file"""
        df_i.sort_values(by='frame', inplace=True)
        df_i.set_index('frame', inplace=True)
        fout: str = \
            f'{fout_prefix}_df'
        fout += ".xvg"
        extra_msg: list[str] = [
            f'# This {fout_prefix} is analyzed by using MDAnalysis',
            f'# Ran by {self.__module__}']
        my_tools.write_xvg(df_i,
                           log,
                           extra_msg,
                           fout,
                           write_index=True,
                           x_axis_label='Frame index',
                           y_axis_label=f'{fout_prefix}',
                           title=f'{fout_prefix}')
        self.info_msg += f'\tThe {fout_prefix} is saved as `{fout}`\n'

    def write_msg(self,
                  log: logger.logging.Logger  # To log
                  ) -> None:
        """write and log messages"""
        now = datetime.now()
        self.info_msg += \
            f'\tTime: {now.strftime("%Y-%m-%d %H:%M:%S")}\n'
        print(f'{bcolors.OKCYAN}{TrrFilterAnalysis.__name__}:\n'
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
        now = datetime.now()
        self.info_msg += \
            f'\tTime: {now.strftime("%Y-%m-%d %H:%M:%S")}\n'
        print(f'{bcolors.OKCYAN}{ReadForceFieldFile.__name__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    TrrFilterAnalysis(sys.argv[1], logger.setup_logger('trr_charge.log'))
