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
    __slots__ = ['cfg']

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
        self.make_df_to_latex(itps)

    def read_itp(self,
                 itp_files: pd.DataFrame,
                 ) -> dict[str, itp_to_df.Itp]:
        """Reads the ITP file."""
        itps: dict[str, itp_to_df.Itp] = {}
        for itp in itp_files.itertuples():
            itp_df: itp_to_df.Itp = itp_to_df.Itp(fname=itp.path)
            itps[str(itp)] = itp_df
        return itps

    def make_df_to_latex(self,
                         itps: dict[str, itp_to_df.Itp],
                         ) -> None:
        """Writes the data to a LaTeX file."""
        for itp in itps.values():
            self.atoms_to_latex_df(itp)

    def atoms_to_latex_df(self,
                          itp: itp_to_df.Itp,
                          ) -> pd.DataFrame:
        """Writes the atoms to a LaTeX file."""
        df_atoms: pd.DataFrame = \
            itp.atoms.drop_duplicates(subset=['atomtype', 'charge'])
        return df_atoms
