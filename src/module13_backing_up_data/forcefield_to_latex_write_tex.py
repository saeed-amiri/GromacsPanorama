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
        self.bonds_to_latex(latex_itp.bonds_df, log)
        self.angles_to_latex(latex_itp.angles_df, log)

    def atoms_to_latex(self,
                       atoms_df: pd.DataFrame,
                       log: logger.logging.Logger
                       ) -> None:
        """Writes the atoms to a LaTeX file."""
        residues: list[str] = list(atoms_df['resname'].unique())
        residues = [residue.upper() for residue in residues]

        residue_table_groups: list[tuple[str, ...]] = \
            self.cfg.structures.structures['residue_table_groups']
        fname: str = self.cfg.files.latex_path['atoms']
        with open(fname, 'w', encoding='utf-8') as file:
            for group in residue_table_groups:
                atoms_lines: list[str] = self._make_atoms_lines(
                    group, residues, atoms_df)
                content: str = ''.join(
                    '\n'.join(self.cfg.files.latex_headers['atoms_header']) +
                    ''.join(atoms_lines) +
                    '\n'.join(self.cfg.files.latex_headers['atoms_footer']) +
                    '\n\n'
                )
                file.write(content)
                del content
        log.info(f'\tWrote the atoms to {fname}\n')

    def _make_atoms_lines(self,
                          group: tuple[str, ...],
                          residues: list[str],
                          atoms_df: pd.DataFrame,
                          ) -> list[str]:
        """Makes the lines for the atoms table."""
        structure_list: list[str] = \
            self.cfg.structures.structures['structure_list']
        atoms_lines: list[str] = []
        for residue in group:
            if residue not in residues:
                print(f'{residue} not in the residues')
                continue
            residue_name: str = \
                self.cfg.files.residue_names.get(residue, residue)
            atoms_lines.append(f'\\textbf{{{residue_name}}} & & & \\\\\n')
            df: pd.DataFrame = atoms_df[atoms_df['resname'] == residue.lower()]
            # sort the df by the atomtypes
            df = df.sort_values(by=['element', 'atomtype'])
            if residue in structure_list:
                structure: str = \
                    self.cfg.structures.structures[residue]
                atoms_lines.append('\\multicolumn{4}{c}{{}}\\\\\n')
                atoms_lines.append(
                    f"\\multicolumn{{4}}{{c}}{{{structure}}}\\\\\n")
            else:
                pass
            for _, row in df.iterrows():
                l_line: str = (
                    '\\hspace*{{1em}}'
                    f'{row["element"].upper()}, {row["atomtype"].upper()} & '
                    f'{row["charge"]:.3f} & '
                    f'{row["sigma"]:.3f} & '
                    f'{row["epsilon"]:.3f} \\\\\n'
                )
                atoms_lines.append(l_line)
            atoms_lines.append(' & & & \\\\\n')
        return atoms_lines

    def bonds_to_latex(self,
                       bonds_df: pd.DataFrame,
                       log: logger.logging.Logger
                       ) -> None:
        """Writes the bonds to a LaTeX file.
        columns: ai_atomtype aj_atomtype resname k r
        """
        # Get unique residues
        residues: list[str] = list(bonds_df['resname'].unique())
        bonds_lines: list[str] = []

        for residue in residues:
            residue_name: str = \
                self.cfg.files.residue_names.get(residue.upper(), residue)
            bonds_lines.append(f'\\textbf{{{residue_name}}} & & & \\\\\n')

            df: pd.DataFrame = bonds_df[bonds_df['resname'] == residue]
            # sort the df by the atomtypes
            df = df.sort_values(by=['ai_type', 'aj_type'])
            for _, row in df.iterrows():
                l_line: str = (
                    '\\hspace*{1em}'
                    f'{row["bondname"]} & '
                    f'{row["k"]:.3f} & {row["r"]:.3f} \\\\\n'
                )
                bonds_lines.append(l_line)
            bonds_lines.append(' & & & \\\\\n')

        fname: str = self.cfg.files.latex_path['bonds']
        with open(fname, 'w', encoding='utf-8') as file:
            content: str = ''.join(
                '\n'.join(self.cfg.files.latex_headers['bonds_header']) +
                ''.join(bonds_lines) +
                '\n'.join(self.cfg.files.latex_headers['bonds_footer'])
            )
            file.write(content)
        log.info(f'\tWrote the bonds to {fname}\n')

    def angles_to_latex(self,
                        angles_df: pd.DataFrame,
                        log: logger.logging.Logger
                        ) -> None:
        """Writes the angles to a LaTeX file.
        columns: ai_atomtype aj_atomtype ak_atomtype resname k theta
        th0 (Angle Equilibrium Value):
            The equilibrium bond angle (in degrees) between three atoms
            i, j, and k.
            This is the angle the system tries to maintain during
            simulations.

        cth (Force Constant):
            The force constant for the angle bending term (in
            kcal mol-1 rad-2).
            It determines the stiffness of the bond angle. A higher
            value means the bond angle resists deviation from its
            equilibrium value (th0) more strongly.

        S0 (Cubic Term):
            A cubic correction term that accounts for anharmonicity in
            the angle potential.
            It is optional and typically zero if anharmonicity is not
            considered.

        Kub (Quartic Term or Urey-Bradley Term):
            A quartic correction term (or Urey-Bradley constant in some
            contexts) used to add flexibility or fine-tune the angle
            potential energy function.
            For Urey-Bradley terms, it corresponds to the strength of
            a 1-3 interaction between atoms i and k (in kcal mol-1 AA-2).
        """
        # Get unique residues
        residues: list[str] = list(angles_df['resname'].unique())
        angles_lines: list[str] = []

        for residue in residues:
            residue_name: str = \
                self.cfg.files.residue_names.get(residue.upper(), residue)
            angles_lines.append(f'\\textbf{{{residue_name}}} & & & \\\\\n')

            df: pd.DataFrame = angles_df[angles_df['resname'] == residue]
            # sort by ai_type, aj_type, ak_type
            df = df.sort_values(by=['ai_type', 'aj_type', 'ak_type'])
            for _, row in df.iterrows():
                l_line: str = (
                    '\\hspace*{1em}'
                    f'{row["anglename"]} & '
                    f'{row["theta"]:.3f} & {row["cth"]:.3f} & '
                    f'{row["s_0"]:.3f} & {row["k"]:.3f} '
                    f'\\\\\n'
                )
                angles_lines.append(l_line)
            angles_lines.append(' & & & &\\\\\n')

        fname: str = self.cfg.files.latex_path['angles']

        with open(fname, 'w', encoding='utf-8') as file:
            content: str = ''.join(
                '\n'.join(self.cfg.files.latex_headers['angles_header']) +
                ''.join(angles_lines) +
                '\n'.join(self.cfg.files.latex_headers['angles_footer'])
            )
            file.write(content)
        log.info(f'\tWrote the angles to {fname}\n')
