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
        itp_file: pd.DataFrame = pd.DataFrame.from_dict(
            self.cfg.files.ff_path.itps, orient='index', columns=['path'])
        itp_files: dict[str, itp_to_df.Itp] = self.read_itp(itp_file)
        self.make_df_to_latex(itp_files,)

    def read_itp(self,
                 itp_file: pd.DataFrame,
                 ) -> dict[str, itp_to_df.Itp]:
        """Reads the ITP file."""
        itp_files: dict[str, itp_to_df.Itp] = {}
        for itp in itp_file.itertuples():
            itp_df: itp_to_df.Itp = itp_to_df.Itp(fname=itp.path)
            itp_files[str(itp.path)] = itp_df
        return itp_files

    def make_df_to_latex(self,
                         itp_files: dict[str, itp_to_df.Itp],
                         ) -> None:
        """Writes the data to a LaTeX file."""
        for itp in itp_files.values():
            self.atoms_to_latex_df(itp)

    def atoms_to_latex_df(self,
                          itp: itp_to_df.Itp,
                          ) -> None:
        """Writes the atoms to a LaTeX file."""
        return itp.atoms.drop_duplicates(subset=['atomtype', 'charge'])
