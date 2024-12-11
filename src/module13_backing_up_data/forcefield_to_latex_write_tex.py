"""
Get the DF for the tex file and write it to a file.
"""

import typing
import pandas as pd

from common import logger

if typing.TYPE_CHECKING:
    from omegaconf import DictConfig
    from module13_backing_up_data.forcefield_to_latex_read_itp import \
        ProccessForceField


class WriteTex:
    """
    Writes the data to a LaTeX file.
    """

    __slots__ = ['cfg']

    def __init__(self,
                 latex_itp: "ProccessForceField",
                 cfg: "DictConfig",
                 log: logger.logging.Logger
                 ) -> None:
        self.cfg = cfg
        self.write_tex(latex_itp, log)

    def write_tex(self,
                  latex_itp: "ProccessForceField",
                  log: logger.logging.Logger
                  ) -> None:
        """
        Write the data to a LaTeX file.
        """
        self.atoms_to_latex(latex_itp.atoms_df, log)

    def atoms_to_latex(self,
                       atoms_df: pd.DataFrame,
                       log: logger.logging.Logger
                       ) -> None:
        """Writes the atoms to a LaTeX file."""
        residues: list[str] = list(atoms_df['resname'].unique())
        atoms_lines: list[str] = []
        for residue in residues:
            residue_name: str = \
                self.cfg.files.residue_names.get(residue.upper(), residue)
            atoms_lines.append(f'\\textbf{{{residue_name}}} & & & \\\\\n')
            df: pd.DataFrame = atoms_df[atoms_df['resname'] == residue]
            for _, row in df.iterrows():
                l_line: str = (
                    '\\hspace*{{4em}}'
                    f'{row["element"].upper()}, ({row["atomtype"].upper()}) & '
                    f'{row["charge"]:.3f} & '
                    f'{row["sigma"]:.3f} & '
                    f'{row["epsilon"]:.3f} \\\\\n'
                )
                atoms_lines.append(l_line)
            atoms_lines.append(' & & & \\\\\n')

        fname: str = self.cfg.files.latex_path['atoms']
        with open(fname, 'w', encoding='utf-8') as file:
            content: str = ''.join(
                self.cfg.files.latex_headers['atoms_header'] +
                atoms_lines +
                self.cfg.files.latex_headers['atoms_footer']
            )
            file.write(content)
        log.info(f'\tWrote the atoms to {fname}\n')
