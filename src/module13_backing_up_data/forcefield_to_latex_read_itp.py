"""
Reading and processing the force field parameters from the ITP file.
"""

import typing
import pandas as pd

from common import logger
from common import itp_to_df

if typing.TYPE_CHECKING:
    import numpy as np
    from omegaconf import DictConfig


class ProccessForceField:
    """Reads and processes the force field parameters."""
    __slots__ = ['cfg', 'atoms_df', 'bonds_df', 'angles_df', 'dihedrals_df']

    atoms_df: pd.DataFrame
    bonds_df: pd.DataFrame
    angles_df: pd.DataFrame
    dihedrals_df: pd.DataFrame

    def __init__(self,
                 cfg: "DictConfig",
                 log: logger.logging.Logger
                 ) -> None:
        self.cfg = cfg
        self.process_itp(log)

    def process_itp(self,
                    log: logger.logging.Logger
                    ) -> None:
        """Reads the ITP file and processes it."""
        itp_files: pd.DataFrame = pd.DataFrame.from_dict(
            self.cfg.files.ff_path.itps, orient='index', columns=['path'])
        itps: dict[str, itp_to_df.Itp] = self.read_itp(itp_files)
        self.make_df_to_latex(itps, log)

    def read_itp(self,
                 itp_files: pd.DataFrame,
                 ) -> dict[str, itp_to_df.Itp]:
        """Reads the ITP file."""
        itps: dict[str, itp_to_df.Itp] = {}
        for itp in itp_files.itertuples():
            itp_df: itp_to_df.Itp = itp_to_df.Itp(fname=itp.path)
            itps[str(itp[0])] = itp_df
        return itps

    def make_df_to_latex(self,
                         itps: dict[str, itp_to_df.Itp],
                         log: logger.logging.Logger
                         ) -> None:
        """Writes the data to a LaTeX file."""
        charmm_atoms_types: pd.DataFrame = itps['charmm'].atomtypes
        charmm_bonds: pd.DataFrame = itps['charmm'].bondtypes
        charmm_angles: pd.DataFrame = itps['charmm'].angletypes
        charmm_dihedrals: pd.DataFrame = itps['charmm'].dihedraltypes
        atoms_df_list: list[pd.DataFrame] = []
        bonds_df_list: list[pd.DataFrame] = []
        angle_df_list: list[pd.DataFrame] = []
        dihedrals_df_list: list[pd.DataFrame] = []
        for res, itp in itps.items():
            if res == 'charmm':
                continue
            atoms_df_list.append(
                self.atoms_to_latex_df(itp, charmm_atoms_types, log))
            bonds_df_list.append(
                self.bonds_to_latex_df(res, itp, charmm_bonds, log))
            angle_df_list.append(
                self.angles_to_latex_df(res, itp, charmm_angles, log))
            # dihedrals_df_list.append(
                # self.dihedrals_to_latex_df(res, itp, charmm_dihedrals, log))
        self.atoms_df = pd.concat(atoms_df_list)
        self.bonds_df = pd.concat(bonds_df_list)
        self.angles_df = pd.concat(angle_df_list)

    def atoms_to_latex_df(self,
                          itp: itp_to_df.Itp,
                          charmm_atoms_types: pd.DataFrame,
                          log: logger.logging.Logger
                          ) -> pd.DataFrame:
        """Writes the atoms to a LaTeX file."""
        # Ensure the necessary columns are present in `itp.atoms`
        required_itp_cols: set[str] = \
            {'atomnr', 'atomname', 'resnr', 'resname', 'chargegrp',
             'atomtype', 'charge'}
        if not required_itp_cols.issubset(itp.atoms.columns):
            missing = required_itp_cols - set(itp.atoms.columns)
            log.error(
                msg := f"\tMissing columns in `itp.atoms`: {missing}\n"
                )
            raise ValueError(msg)

        # Ensure the necessary columns are present in `charmm_atoms_types`
        required_charmm_cols: set[str] = {'name', 'sigma', 'epsilon'}
        if not required_charmm_cols.issubset(charmm_atoms_types.columns):
            missing = required_charmm_cols - set(charmm_atoms_types.columns)
            log.error(
                msg := f"\tMissing in `charmm_atoms_types`: {missing}\n")
            raise ValueError(msg)

        df_atoms: pd.DataFrame = \
            itp.atoms.drop_duplicates(subset=['atomtype', 'charge'])
        df_c: pd.DataFrame = df_atoms.copy()
        df_c = df_c.drop(
            columns=['atomnr', 'atomname', 'resnr', 'chargegrp'])

        atomname: list[str] = list(df_atoms['atomtype'])

        sigma_dict: dict[str, float] = {}
        epsilon_dict: dict[str, float] = {}
        for name in atomname:
            charmm_param = charmm_atoms_types[
                charmm_atoms_types['name'] == name.upper()]
            sigma_dict[name] = charmm_param.iloc[0]['sigma']
            epsilon_dict[name] = charmm_param.iloc[0]['epsilon']

        df_c['sigma'] = df_c['atomtype'].map(sigma_dict)
        df_c['epsilon'] = df_c['atomtype'].map(epsilon_dict)

        return df_c.reset_index(drop=True)

    def bonds_to_latex_df(self,
                          res: str,
                          itp: itp_to_df.Itp,
                          charmm_bonds: pd.DataFrame,
                          log: logger.logging.Logger
                          ) -> pd.DataFrame:
        """Writes the bonds to a LaTeX file."""
        # pylint: disable=too-many-locals

        # Ensure the necessary columns are present in `itp.bonds`
        if itp.bonds is None or itp.atoms is None:
            log.warning("Missing bonds or atoms data.")
            return pd.DataFrame()

        # Map atomnr to atomtype for ai and aj in bonds
        atomtype_map = itp.atoms.set_index('atomnr')['atomtype']

        try:
            a_i_names: pd.Series = itp.bonds['ai'].map(atomtype_map)
            a_j_names: pd.Series = itp.bonds['aj'].map(atomtype_map)
        except KeyError as e:
            log.error(f"KeyError: {e}")
            missing_atoms = \
                set(itp.bonds['ai']).union(itp.bonds['aj']) - \
                set(atomtype_map.index)
            log.error(f"Missing atoms in atomtype map: {missing_atoms}")
            return pd.DataFrame()

        residue_col = pd.Series([res] * len(itp.bonds['ai']), name='residue')
        residue_col.index += 1

        # Combine the data into a DataFrame for LaTeX export
        blatex_df = pd.DataFrame({
            'typ': itp.bonds['typ'],
            'ai_type': a_i_names,
            'aj_type': a_j_names,
            'resname': residue_col
        })
        # Drop one of the rows of m and n if m(ai) == n(ai) and m(aj) == n(aj)
        blatex_df = blatex_df.drop_duplicates(
            subset=['ai_type', 'aj_type'])
        # get the k and r from the charmm_bonds when the ai_type and
        # aj_type are the same as ai and aj in blatex_df
        k_dict: dict[np.float64, np.float64] = {}
        r_dict: dict[np.float64, np.float64] = {}
        for i, row in blatex_df.iterrows():
            charmm_param = charmm_bonds[
                ((charmm_bonds['ai'] == row['ai_type']) &
                 (charmm_bonds['aj'] == row['aj_type'])) |
                ((charmm_bonds['ai'] == row['aj_type']) &
                 (charmm_bonds['aj'] == row['ai_type']))
                ]
            k_dict[i] = charmm_param.iloc[0]['k']
            r_dict[i] = charmm_param.iloc[0]['r']
        blatex_df['k'] = blatex_df.index.map(k_dict)
        blatex_df['r'] = blatex_df.index.map(r_dict)
        blatex_df['bondname'] = \
            blatex_df.apply(
                lambda row:
                f"{str(row['ai_type']).upper()}-{str(row['aj_type']).upper()}",
                axis=1)
        return blatex_df.reset_index(drop=True)

    def angles_to_latex_df(self,
                           res: str,
                           itp: itp_to_df.Itp,
                           charmm_angles: pd.DataFrame,
                           log: logger.logging.Logger
                           ) -> pd.DataFrame:
        """Writes the angles to a LaTeX file."""
        # pylint: disable=too-many-locals
        if itp.angles is None or itp.atoms is None:
            log.warning("Missing angles or atoms data.")
            return pd.DataFrame()
        # drop the duplicates if in rows m and n:
        #  m(ai) == n(ai) and m(aj) == n(aj) and m(ak) == n(ak)
        # or m(ai) == n(ak) and m(aj) == n(aj) and m(ak) == n(ai)
        atomtype_map = itp.atoms.set_index('atomnr')['atomtype']
        try:
            a_i_names: pd.Series = itp.angles['ai'].map(atomtype_map)
            a_j_names: pd.Series = itp.angles['aj'].map(atomtype_map)
            a_k_names: pd.Series = itp.angles['ak'].map(atomtype_map)
        except KeyError as e:
            log.error(f"KeyError: {e}")
            missing_atoms = \
                set(itp.angles['ai']).union(itp.angles['aj']).union(
                    itp.angles['ak']) - set(atomtype_map.index)
            log.error(f"Missing atoms in atomtype map: {missing_atoms}")
            return pd.DataFrame()
        residue_col = pd.Series([res] * len(itp.angles['ai']), name='residue')
        residue_col.index += 1
        # Combine the data into a DataFrame for LaTeX export
        anglatex_df = pd.DataFrame({
            'typ': itp.angles['typ'],
            'ai_type': a_i_names,
            'aj_type': a_j_names,
            'ak_type': a_k_names,
            'resname': residue_col
        })
        anglatex_df = anglatex_df.drop_duplicates(
            subset=['ai_type', 'aj_type', 'ak_type'])
        # get the k and theta from the charmm_angles when the ai_type, aj_type
        # and ak_type are the same as ai, aj and ak in anglatex_df
        k_dict: dict[np.float64, np.float64] = {}
        theta_dict: dict[np.float64, np.float64] = {}
        s_0_dict: dict[np.float64, np.float64] = {}
        cth_dict: dict[np.float64, np.float64] = {}
        for i, row in anglatex_df.iterrows():
            charmm_param = charmm_angles[
                ((charmm_angles['ai'] == row['ai_type']) &
                 (charmm_angles['aj'] == row['aj_type']) &
                 (charmm_angles['ak'] == row['ak_type'])) |
                ((charmm_angles['ai'] == row['ak_type']) &
                 (charmm_angles['aj'] == row['aj_type']) &
                 (charmm_angles['ak'] == row['ai_type']))
                ]
            k_dict[i] = charmm_param.iloc[0]['kub']
            theta_dict[i] = charmm_param.iloc[0]['theta']
            s_0_dict[i] = charmm_param.iloc[0]['s0']
            cth_dict[i] = charmm_param.iloc[0]['cth']
        anglatex_df['k'] = anglatex_df.index.map(k_dict)
        anglatex_df['theta'] = anglatex_df.index.map(theta_dict)
        anglatex_df['s_0'] = anglatex_df.index.map(s_0_dict)
        anglatex_df['cth'] = anglatex_df.index.map(cth_dict)
        anglatex_df['anglename'] = \
            anglatex_df.apply(
                lambda row:
                f"{str(row['ai_type']).upper()}-{str(row['aj_type']).upper()}"
                f"-{str(row['ak_type']).upper()}",
                axis=1)
        return anglatex_df.reset_index(drop=True)

    def dihedrals_to_latex_df(self,
                              res: str,
                              itp: itp_to_df.Itp,
                              charmm_dihedrals: pd.DataFrame,
                              log: logger.logging.Logger
                              ) -> pd.DataFrame:
        """Writes the dihedrals to a LaTeX file."""
        # pylint: disable=too-many-locals
        if itp.dihedrals is None or itp.atoms is None:
            log.warning("Missing dihedrals or atoms data.")
            return pd.DataFrame()
        # drop the duplicates if in rows m and n:
        #  m(ai) == n(ai) and m(aj) == n(aj) and m(ak) == n(ak) and
        #  m(ah) == n(ah)
        # or m(ai) == n(ah) and m(aj) == n(ak) and m(ak) == n(aj) and
        #  m(ah) == n(ai)
        atomtype_map = itp.atoms.set_index('atomnr')['atomtype']

        try:
            a_i_names: pd.Series = itp.dihedrals['ai'].map(atomtype_map)
            a_j_names: pd.Series = itp.dihedrals['aj'].map(atomtype_map)
            a_k_names: pd.Series = itp.dihedrals['ak'].map(atomtype_map)
            a_h_names: pd.Series = itp.dihedrals['ah'].map(atomtype_map)
        except KeyError as e:
            log.error(f"KeyError: {e}")
            missing_atoms = \
                set(itp.dihedrals['ai']).union(itp.dihedrals['aj']).union(
                    itp.dihedrals['ak']).union(itp.dihedrals['ah']) - \
                set(atomtype_map.index)
            log.error(f"Missing atoms in atomtype map: {missing_atoms}")
            return pd.DataFrame()
        residue_col = pd.Series(
            [res] * len(itp.dihedrals['ai']), name='residue')
        residue_col.index += 1
        # Combine the data into a DataFrame for LaTeX export
        dihlatex_df = pd.DataFrame({
            'typ': itp.dihedrals['typ'],
            'ai_type': a_i_names,
            'aj_type': a_j_names,
            'ak_type': a_k_names,
            'ah_type': a_h_names,
            'resname': residue_col
        })
        dihlatex_df = dihlatex_df.drop_duplicates(
            subset=['ai_type', 'aj_type', 'ak_type', 'ah_type'])
        # get the k and theta from the charmm_dihedrals when the ai_type,
        # aj_type, ak_type and ah_type are the same as ai, aj, ak and ah in
        # dihlatex_df
        # funct (int) phi0(float) cp(float) mult(int)
        func_dict: dict[np.float64, np.int64] = {}
        phi0_dict: dict[np.float64, np.float64] = {}
        cp_dict: dict[np.float64, np.float64] = {}
        mult_dict: dict[np.float64, np.int64] = {}
        for i, row in dihlatex_df.iterrows():
            charmm_param = charmm_dihedrals[
                ((charmm_dihedrals['ai'] == row['ai_type']) &
                 (charmm_dihedrals['aj'] == row['aj_type']) &
                 (charmm_dihedrals['ak'] == row['ak_type']) &
                 (charmm_dihedrals['ah'] == row['ah_type'])) |
                ((charmm_dihedrals['ai'] == row['ah_type']) &
                 (charmm_dihedrals['aj'] == row['ak_type']) &
                 (charmm_dihedrals['ak'] == row['aj_type']) &
                 (charmm_dihedrals['ah'] == row['ai_type']))
                ]
            if charmm_param.empty:
                continue
            func_dict[i] = charmm_param.iloc[0]['func']
            phi0_dict[i] = charmm_param.iloc[0]['phi0']
            cp_dict[i] = e.iloc[0]['cp']
            mult_dict[i] = charmm_param.iloc[0]['mult']
        if not func_dict:
            return pd.DataFrame()
        dihlatex_df['func'] = dihlatex_df.index.map(func_dict)
        dihlatex_df['phi0'] = dihlatex_df.index.map(phi0_dict)
        dihlatex_df['cp'] = dihlatex_df.index.map(cp_dict)
        dihlatex_df['mult'] = dihlatex_df.index.map(mult_dict)
        dihlatex_df['dihname'] = \
            dihlatex_df.apply(
                lambda row:
                f"{str(row['ai_type']).upper()}-{str(row['aj_type']).upper()}"
                f"-{str(row['ak_type']).upper()}-{str(row['ah_type']).upper()}"
                , axis=1)
        return dihlatex_df.reset_index(drop=True)
