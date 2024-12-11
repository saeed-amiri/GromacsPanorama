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
    __slots__ = ['cfg', 'atoms_df', 'bonds_df']

    atoms_df: pd.DataFrame
    bonds_df: pd.DataFrame

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
        atoms_df_list: list[pd.DataFrame] = []
        bonds_df_list: list[pd.DataFrame] = []
        for res, itp in itps.items():
            if res == 'charmm':
                continue
            atoms_df_list.append(
                self.atoms_to_latex_df(itp, charmm_atoms_types, log))
            bonds_df_list.append(
                self.bonds_to_latex_df(res, itp, charmm_bonds, log))
        self.atoms_df = pd.concat(atoms_df_list)
        self.bonds_df = pd.concat(bonds_df_list)

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
        bonds_latex_df = pd.DataFrame({
            'typ': itp.bonds['typ'],
            'ai_atomtype': a_i_names,
            'aj_atomtype': a_j_names,
            'residue': residue_col
        })
        # Drop one of the rows of m and n if m(ai) == n(ai) and m(aj) == n(aj)
        bonds_latex_df = bonds_latex_df.drop_duplicates(
            subset=['ai_atomtype', 'aj_atomtype'])
        # get the k and r from the charmm_bonds when the ai_atomtype and
        # aj_atomtype are the same as ai and aj in bonds_latex_df
        k_dict: dict[np.float64, np.float64] = {}
        r_dict: dict[np.float64, np.float64] = {}
        for i, row in bonds_latex_df.iterrows():
            charmm_param = charmm_bonds[
                ((charmm_bonds['ai'] == row['ai_atomtype']) &
                 (charmm_bonds['aj'] == row['aj_atomtype'])) |
                ((charmm_bonds['ai'] == row['aj_atomtype']) &
                 (charmm_bonds['aj'] == row['ai_atomtype']))
                ]
            k_dict[i] = charmm_param.iloc[0]['k']
            r_dict[i] = charmm_param.iloc[0]['r']
        bonds_latex_df['k'] = bonds_latex_df.index.map(k_dict)
        bonds_latex_df['r'] = bonds_latex_df.index.map(r_dict)
        return bonds_latex_df.reset_index(drop=True)
