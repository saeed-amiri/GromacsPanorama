"""
Reading and processing the force field parameters from the ITP file.
"""

import typing
import pandas as pd

from common import logger
from common import itp_to_df

if typing.TYPE_CHECKING:
    from omegaconf import DictConfig


class ProccessForceField:
    """Reads and processes the force field parameters."""
    __slots__ = ['cfg', 'atoms_df']

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
        atoms_df_list: list[pd.DataFrame] = []
        for itp in itps.values():
            atoms_df_i: pd.DataFrame = \
                self.atoms_to_latex_df(itp, charmm_atoms_types, log)
            atoms_df_list.append(atoms_df_i)
        self.atoms_df: pd.DataFrame = pd.concat(atoms_df_list)

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
