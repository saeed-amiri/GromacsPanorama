"""reading itp files and return data several data frames
    Usually, the itp file contains information about atoms (not
    coordinates), bonds, angles, dihedrals, and improper dihedrals.

    informations in the itp file:
        [ moleculetype ] : defines the name of your molecule in this top
    and nrexcl = 3 stands for excluding non-bonded interactions between
    atoms that are no further than 3 bonds away.

        [ atoms ] : defines the molecule, where nr and type are fixed,
    the rest is user defined. So atom can be named as you like,
    cgnr made larger or smaller (if possible, the total charge of
    a charge group should be zero), and charges can be changed here
    too.

        [ bonds ] : no comment.

        [ pairs ] : LJ and Coulomb 1-4 interactions

        [ angles ] : no comment

        [ dihedrals ] : in this case there are 9 proper dihedrals
    (funct = 1), 3 improper (funct = 4) and no Ryckaert-Bellemans type
    dihedrals.
    """

import sys
import typing

import pandas as pd

from common.colors_text import TextColor as bcolors


# A helper function needed by most of the classes to clean the lines
def free_char_line(line: str  # line of the itp file
                   ) -> list[str]:  # Free from chars
    """cheack the lines and return the line free special chars"""
    char_list: list[str] = [';', '#', ':', '...']  # chars eliminate in lines
    l_line: list[str]  # Breaking the line cahrs
    l_line = line.strip().split(' ')
    l_line = [item for item in l_line if item]
    l_line = [item for item in l_line if item not in char_list]
    # make sure all the strings are in lower case
    l_line = [item.lower() for item in l_line]
    return l_line


class Itp:
    """Reads an ITP file and returns a DataFrame of the information
    within the file."""
    # pylint: disable=too-many-instance-attributes
    sections: dict[str, list[str]]

    def __init__(self,
                 fname: str,
                 section: typing.Union[str, None] = None
                 ) -> None:
        """Initializes the Itp class by reading the file."""
        print(f"{bcolors.OKBLUE}Reading '{fname}' ...{bcolors.ENDC}")
        self.sections = {
            'atoms': [],
            'bonds': [],
            'angles': [],
            'dihedrals': [],
            'impropers': [],
            'moleculetype': [],
            'atomtypes': [],
            'bondtypes': [],
            'angletypes': [],
            'dihedraltypes': [],
        }
        self.initialize_empty_dfs()
        self.read_file(fname)
        self.create_dataframes(section)

    def initialize_empty_dfs(self) -> None:
        """Initializes all section DataFrames as empty."""
        self.atoms = pd.DataFrame()
        self.bonds = pd.DataFrame()
        self.angles = pd.DataFrame()
        self.dihedrals = pd.DataFrame()
        self.impropers = pd.DataFrame()
        self.molecules = pd.DataFrame()
        self.atomtypes = pd.DataFrame()
        self.bondtypes = pd.DataFrame()
        self.angletypes = pd.DataFrame()
        self.dihedraltypes = pd.DataFrame()

    def read_file(self,
                  fname: str
                  ) -> None:
        """Reads the ITP file and organizes the data into sections."""
        current_section = None
        with open(fname, 'r', encoding='utf8') as file:
            for line in file:
                line = line.strip()
                if line.startswith('['):
                    section_name = line.split()[1]
                    current_section = section_name
                elif line and current_section in self.sections.keys():
                    self.sections[current_section].append(line)

    def create_dataframes(self,
                          section: typing.Union[str, None]
                          ) -> None:
        """
        Converts sections into DataFrames, optionally only processing
        a specified section.
        """
        if section is None or section == 'atoms':
            self.atoms = AtomsInfo(self.sections['atoms']).df
        if section is None or section == 'moleculetype':
            self.molecules = MoleculeInfo(self.sections['moleculetype']).df
        if section is None or section == 'bonds':
            self.bonds = BondsInfo(
                atoms=self.atoms, bonds=self.sections['bonds']).df
        if section is None or section == 'angles':
            self.angles = AnglesInfo(
                atoms=self.atoms, angles=self.sections['angles']).df
        if section is None or section == 'dihedrals':
            self.dihedrals = DihedralsInfo(
                atoms=self.atoms, dihedrals=self.sections['dihedrals']).df
        if section is None or section == 'atomtypes':
            self.atomtypes = AtomsTypes(self.sections['atomtypes']).df
        if section is None or section == 'bondtypes':
            self.bondtypes = BondsInfo(atoms=self.atoms,
                                       bonds=self.sections['bondtypes'],
                                       style='charmm'
                                       ).df
        if section is None or section == 'angletypes':
            self.angletypes = AnglesInfo(atoms=self.atoms,
                                         angles=self.sections['angletypes'],
                                         style='charmm'
                                         ).df
        if section is None or section == 'dihedraltypes':
            self.dihedraltypes = DihedralsInfo(
                atoms=self.atoms,
                dihedrals=self.sections['dihedraltypes'],
                style='charmm'
                ).df


class AtomsTypes:
    """Get the atomtypes info at the top of the charmm itp files"""
    # pylint: disable=too-few-public-methods

    def __init__(self,
                 atomtypes: list[str]
                 ) -> None:
        # pylint: disable=invalid-name
        self.df: pd.DataFrame = self.get_atoms_types(atomtypes)

    def get_atoms_types(self,
                        atomtypes: list[str]
                        ) -> pd.DataFrame:
        """parse the lines"""
        sparsed_lines: list[dict[str, typing.Any]] = []
        for line in atomtypes:
            if line.startswith(';'):
                pass
            else:
                sparsed_lines.append(self._process_line(line))
        return pd.DataFrame(sparsed_lines)

    def _process_line(self,
                      line: str
                      ) -> dict[str, typing.Any]:
        """sparse lines"""
        tmp: list[str] = line.split(' ')
        tmp = [item for item in tmp if item]
        return {
            'name': tmp[0],
            'atom_nr': int(tmp[1]),
            'mass': float(tmp[2]),
            'charge': float(tmp[3]),
            'ptype': tmp[4],
            'sigma': float(tmp[5]),
            'epsilon': float(tmp[6]),
        }


class MoleculeInfo:
    """get molecules wild information and return the line and its info"""
    def __init__(self,
                 molecules: list[str]  # line read by Itp class
                 ) -> None:
        self.get_molecule_info(molecules)

    def get_molecule_info(self,
                          molecules: list[str]  # line about molecules
                          ) -> None:
        """read and return data about molecule"""
        # pylint: disable=invalid-name
        l_line: list[str]  # Breaking the line chars
        columns: list[str] = ['name', 'nrexcl']
        name: list[str] = []  # Name of the molecules
        style: list[str] = []
        for line in molecules:
            tmp = line.strip().split('\t')
            tmp = [item for item in tmp if tmp]
            line = ' '.join(tmp)
            if line.startswith(';'):
                l_line = free_char_line(line)
                if l_line != columns:
                    sys.exit(f'{bcolors.FAIL}{self.__class__.__name__}:\n'
                             f'\tError in the [ atoms ] header of the '
                             f'itp file: {l_line = }, {columns = }\n'
                             f'{bcolors.ENDC}')
            else:
                l_line = line.split(' ')
                l_line = [item for item in l_line if item]
                name.append(l_line[0])
                style.append(l_line[1])
        self.df = self.mk_df(name, style, columns)

    @staticmethod
    def mk_df(name: list[str],  # Name of the molecules
              style: list[str],  # Style of the interactions(!)
              columns: list[str]  # Name of the columns
              ) -> pd.DataFrame:
        """make a dataframe to save all the data"""
        df_molecules = pd.DataFrame(columns=columns)
        df_molecules['Name'] = name
        df_molecules['nrexcl'] = style
        return df_molecules


class AtomsInfo:
    """get atoms wild information and retrun a DataFrame"""
    def __init__(self,
                 atoms: list[str]  # lines read by Itp class
                 ) -> None:
        # pylint: disable=invalid-name
        self.df = self.get_atoms_info(atoms)

    def get_atoms_info(self,
                       atoms: list[str]  # Lines of the atoms' section
                       ) -> pd.DataFrame:
        """get atoms info from the file"""
        l_line: list[str]  # Breaking the line chars
        # Check if header of the atoms section is same as the defeined one
        columns: list[str]   # columns for the atoms dict, name of each column
        columns = ['atomnr', 'atomtype', 'resnr', 'resname', 'atomname',
                   'chargegrp', 'charge', 'mass', 'element']
        atomnr: list[typing.Any] = []  # list to append info: atoms id
        atomtype: list[typing.Any] = []  # list to append info: forcefield type
        resnr: list[typing.Any] = []  # list to append info: res infos
        resname: list[typing.Any] = []  # list to append info: res number
        atomname: list[typing.Any] = []  # list to append info: atom name
        chargegrp: list[typing.Any] = []  # list to append info: charge group
        charge: list[typing.Any] = []  # list to append info: charge value
        mass: list[typing.Any] = []  # list to append info: mass value
        element: list[typing.Any] = []  # list to append info: real name
        for line in atoms:
            if line.startswith(';'):  # line start with ';' are commets&header
                l_line = free_char_line(line)
                if 'total' not in l_line:  # Not header!
                    if l_line != columns:
                        sys.exit(f'{bcolors.FAIL}{self.__class__.__name__}:\n'
                                 f'\tError in the [ atoms ] header of the '
                                 f'itp file:\n\t{l_line = }\n\t{columns = }'
                                 f'\n{bcolors.ENDC}')
            else:
                l_line = free_char_line(line)
                atomnr.append(int(l_line[0]))
                atomtype.append(l_line[1])
                resnr.append(int(l_line[2]))
                resname.append(l_line[3])
                atomname.append(l_line[4])
                chargegrp.append(l_line[5])
                charge.append(float(l_line[6]))
                mass.append(float(l_line[7]))
                element.append(l_line[8])
        df_atoms: pd.DataFrame  # DataFrame from the infos
        df_atoms = pd.DataFrame(columns=columns)
        df_atoms['atomnr'] = atomnr
        df_atoms['atomtype'] = atomtype
        df_atoms['resnr'] = resnr
        df_atoms['resname'] = resname
        df_atoms['atomname'] = atomname
        df_atoms['chargegrp'] = chargegrp
        df_atoms['charge'] = charge
        df_atoms['mass'] = mass
        df_atoms['element'] = self.drop_dot(element)
        return df_atoms

    @staticmethod
    def drop_dot(chars: list[str]  # to drop . from its items
                 ) -> list[str]:
        """drop dot (.) from the string"""
        return [item.replace('.', '') for item in chars]


class BondsInfo:
    """get the bonds list from Itp class and return a dataframe"""
    def __init__(self,
                 bonds: list[str],  # lines of bonds section read by Itp class
                 atoms: pd.DataFrame,  # atoms df from AtomsInfo to get names
                 style: str = 'normal'
                 ) -> None:
        """get the bonds infos"""
        if style == 'normal':
            self.mk_bonds_df(bonds, atoms)
        elif style == 'charmm':
            self.mk_charmm_bonds_df(bonds)

    def mk_bonds_df(self,
                    bonds: list[str],  # lines of bonds section
                    atoms: pd.DataFrame  # atoms df from AtomInfo
                    ) -> None:
        """call all the methods to make the bonds DataFrame"""
        # pylint: disable=invalid-name
        a_i: list[int]  # index of the 1st atoms in the bonds
        a_j: list[int]  # index of the 2nd atoms in the bonds
        funct: list[int]  # index of the type of the bonds
        names: list[str]  # name of the bonds
        a_i, a_j, funct, names = self.get_bonds(bonds)
        self.df = self.mk_df(a_i, a_j, funct, names, atoms)

    def get_bonds(self,
                  bonds: list[str]  # lines of bonds section read by Itp class
                  ) -> tuple[list[int], list[int], list[int], list[str]]:
        """return bonds dataframe to make bonds dataframe"""
        header_columns: list[str]  # Columns of the bonds wild
        alter_header_columns: list[str]  # Columns of the bonds wild, 2nd
        header_columns = ['ai', 'aj', 'typ', 'cmt', 'name']
        alter_header_columns = ['ai', 'aj', 'funct', 'r', 'k', 'name']
        a_i: list[int] = []  # index of the 1st atoms in the bonds
        a_j: list[int] = []  # index of the 2nd atoms in the bonds
        funct: list[int] = []  # Type of the function of the bond
        names: list[str] = []  # name of the bonds
        for line in bonds:
            if line.startswith(';'):  # line start with ';' are commets&header
                l_line = free_char_line(line)
                if ('total' not in l_line and
                   l_line != header_columns):
                    header_columns = alter_header_columns.copy()
                    if l_line != header_columns:
                        sys.exit(
                            f'{bcolors.FAIL}{self.__class__.__name__}:\n'
                            f'\tError in the [ bonds ] header of the '
                            f'itp file\n\t{l_line = }\n\t{header_columns = }'
                            f'{bcolors.ENDC}')
            else:
                tmp = line.split()
                tmp = [item for item in tmp if item]
                line = ' '.join(tmp)
                l_line = free_char_line(line)
                a_i.append(int(l_line[0]))
                a_j.append(int(l_line[1]))
                funct.append(int(l_line[2]))
                names.append(l_line[3])
        return a_i, a_j, funct, names

    def mk_df(self,
              a_i: list[int],  # index of the 1st atom in the bonds
              a_j: list[int],  # index of the 2nd atom in the bonds
              funct: list[int],  # Index of the type of the bond
              names: list[str],  # names of the bonds form bonds section
              atoms: pd.DataFrame  # atoms df from AtomsInfo to cehck the name
              ) -> pd.DataFrame:  # bonds DataFrame
        """make DataFrame and check if they are same as atoms name"""
        # pylint: disable=too-many-arguments
        df_bonds: pd.DataFrame  # to save the bonds_df
        df_bonds = pd.DataFrame(columns=['ai', 'aj', 'typ', 'cmt', 'name'])
        df_bonds['ai'] = a_i
        df_bonds['aj'] = a_j
        df_bonds['name'] = names
        df_bonds['cmt'] = [';' for _ in a_i]
        df_bonds['typ'] = funct
        df_bonds = self.check_names(df_bonds, atoms)
        df_bonds.index += 1
        return df_bonds

    @staticmethod
    def check_names(df_bonds: pd.DataFrame,  # Df to check its names column
                    atoms: pd.DataFrame  # Df source (mostly atoms df)
                    ) -> None:
        """checks the names column in the source file with comparing it
        with names from the atoms dataframe name column for each atom"""
        df_c: pd.DataFrame = df_bonds.copy()
        ai_name: list[str]  # name of the 1st atom from the list
        aj_name: list[str]  # name of the 2nd atom from the list
        ai_name = [atoms.loc[atoms['atomnr'] == item]['atomname'][item-1]
                   for item in df_bonds['ai']]
        aj_name = [atoms.loc[atoms['atomnr'] == item]['atomname'][item-1]
                   for item in df_bonds['aj']]
        names: list[str] = [f'{i}-{j}' for i, j in zip(ai_name, aj_name)]
        df_c['name'] = names
        return df_c

    def mk_charmm_bonds_df(self,
                           bonds: list[str],  # lines of bonds section
                           ) -> None:
        """call all the methods to make the bonds DataFrame"""
        # pylint: disable=invalid-name
        a_i: list[str] = []
        a_j: list[str] = []
        func: list[int] = []
        b_0: list[float] = []
        k_b: list[float] = []
        for line in bonds:
            if line.startswith(';'):
                pass
            else:
                tmp = line.split()
                tmp = [item for item in tmp if item]
                line = ' '.join(tmp)
                l_line = free_char_line(line)
                a_i.append(str(l_line[0]))
                a_j.append(str(l_line[1]))
                func.append(int(l_line[2]))
                b_0.append(float(l_line[3]))
                k_b.append(float(l_line[4]))
        self.df = self.mk_charmm_df(a_i, a_j, func, b_0, k_b)

    def mk_charmm_df(self,
                     a_i: list[int],  # index of the 1st atom in the bonds
                     a_j: list[int],  # index of the 2nd atom in the bonds
                     func: list[int],  # Index of the type of the bond
                     b_0: list[float],  # Equilibrium bond length
                     k_b: list[float],  # Force constant
                     ) -> pd.DataFrame:
        """make DataFrame and check if they are same as atoms name"""
        # pylint: disable=too-many-arguments
        df_bonds: pd.DataFrame
        df_bonds = pd.DataFrame(columns=['ai', 'aj', 'funct', 'r', 'k'])
        df_bonds['ai'] = a_i
        df_bonds['aj'] = a_j
        df_bonds['funct'] = func
        df_bonds['r'] = b_0
        df_bonds['k'] = k_b
        return df_bonds


class AnglesInfo:
    """get the angles list from Itp class and return a dataframe"""
    def __init__(self,
                 angles: list[str],  # lines of angles section by Itp class
                 atoms: pd.DataFrame,  # atoms df from AtomsInfo to get names
                 style: str = 'normal'
                 ) -> None:
        """get the angles infos"""
        if style == 'normal':
            self.mk_angles_df(angles, atoms)
        elif style == 'charmm':
            self.mk_charmm_angles_df(angles)

    def mk_angles_df(self,
                     angles: list[str],  # lines of angles section
                     atoms: pd.DataFrame  # atoms df from AtomInfo
                     ) -> None:
        """call all the methods to make the bonds DataFrame"""
        # pylint: disable=invalid-name
        a_i: list[int]  # index of the 1st atoms in the angles
        a_j: list[int]  # index of the 2nd atoms in the angles
        a_k: list[int]  # index of the 3rd atoms in the angles
        names: list[str]  # name of the angles
        a_i, a_j, a_k, funct, names = self.get_angles(angles)
        self.df = self.mk_df(a_i, a_j, a_k, funct, names, atoms)

    def get_angles(self,
                   angles: list[str],  # lines of angles section by Itp class
                   ) -> tuple[list[int], list[int], list[int],
                              list[int], list[str]]:
        """return bonds dataframe to make angles dataframe"""
        columns: list[str]  # Columns of the angles wild
        columns = ['ai', 'aj', 'ak', 'typ', 'cmt', 'name']
        alter_columns = ['ai', 'aj', 'ak', 'funct', 'theta', 'cth', 'name']
        a_i: list[int] = []  # index of the 1st atoms in the angles
        a_j: list[int] = []  # index of the 2nd atoms in the angles
        a_k: list[int] = []  # index of the 3rd atoms in the angles
        funct: list[int] = []  # index of the type (function) the angles
        names: list[str] = []  # name of the angles

        for line in angles:
            if line.startswith(';'):  # line start with ';' are commets&header
                l_line = free_char_line(line)
                if 'total' not in l_line and l_line != columns:
                    if l_line != alter_columns:
                        sys.exit(f'{bcolors.FAIL}{self.__class__.__name__}:\n'
                                 f'\tError in the [ angles ] header of the\n'
                                 f'itp file:\n\t{l_line = }\n\t{columns = }'
                                 f'{bcolors.ENDC}\n')
            else:
                tmp = line.strip().split('\t')
                tmp = [item for item in tmp if item]
                line = ' '.join(tmp)
                l_line = free_char_line(line)
                a_i.append(int(l_line[0]))
                a_j.append(int(l_line[1]))
                a_k.append(int(l_line[2]))
                funct.append(int(l_line[3]))
                if len(l_line) == 5:
                    names.append(l_line[4])
                else:
                    names.append('')
        return a_i, a_j, a_k, funct, names

    def mk_df(self,
              a_i: list[int],  # index of the 1st atom in the angles
              a_j: list[int],  # index of the 2nd atom in the angles
              a_k: list[int],  # index of the 3rd atom in the angles
              funct: list[int],  # Type (function) of the angles
              names: list[str],  # names of the bonds form angles section
              atoms: pd.DataFrame  # atoms df from AtomsInfo to cehck the name
              ) -> pd.DataFrame:  # angles DataFrame
        """make DataFrame and check if they are same as atoms name"""
        # pylint: disable=too-many-arguments
        df_angles: pd.DataFrame  # to save the angles_df
        df_angles = \
            pd.DataFrame(columns=['ai', 'aj', 'ak', 'typ', 'cmt', 'name'])
        df_angles['ai'] = a_i
        df_angles['aj'] = a_j
        df_angles['ak'] = a_k
        df_angles['name'] = names
        df_angles['cmt'] = [';' for _ in a_i]
        df_angles['typ'] = funct
        df_angles = self.check_names(df_angles, atoms)
        df_angles.index += 1
        return df_angles

    @staticmethod
    def check_names(df_angles: pd.DataFrame,  # Df to check its names column
                    atoms: pd.DataFrame  # Df source (mostly atoms df)
                    ) -> None:
        """ checks the names column in the source file with comparing it
        with names from the atoms dataframe name column for each atom"""
        df_c: pd.DataFrame = df_angles.copy()
        ai_name: list[str]  # name of the 1st atom from the list
        aj_name: list[str]  # name of the 2nd atom from the list
        ak_name: list[str]  # name of the 3rd atom from the list
        ai_name = [atoms.loc[atoms['atomnr'] == item]['atomname'][item-1]
                   for item in df_angles['ai']]
        aj_name = [atoms.loc[atoms['atomnr'] == item]['atomname'][item-1]
                   for item in df_angles['aj']]
        ak_name = [atoms.loc[atoms['atomnr'] == item]['atomname'][item-1]
                   for item in df_angles['ak']]
        names: list[str] = [f'{i}-{j}-{k}' for i, j, k
                            in zip(ai_name, aj_name, ak_name)]
        df_c['name'] = names
        return df_c

    def mk_charmm_angles_df(self,
                            angles: list[str]  # lines of angles section
                            ) -> None:
        """call all the methods to make the bonds DataFrame"""
        # pylint: disable=invalid-name
        a_i: list[str] = []
        a_j: list[str] = []
        a_k: list[str] = []
        func: list[int] = []
        theta: list[float] = []
        cth: list[float] = []
        s0: list[float] = []
        kub: list[float] = []
        for line in angles:
            if line.startswith(';'):
                pass
            else:
                tmp = line.split()
                tmp = [item for item in tmp if item]
                line = ' '.join(tmp)
                l_line = free_char_line(line)
                a_i.append(str(l_line[0]))
                a_j.append(str(l_line[1]))
                a_k.append(str(l_line[2]))
                func.append(int(l_line[3]))
                theta.append(float(l_line[4]))
                cth.append(float(l_line[5]))
                try:
                    s0.append(float(l_line[6]))
                except IndexError:
                    s0.append(0.0)
                try:
                    kub.append(float(l_line[7]))
                except IndexError:
                    kub.append(0.0)
        self.df = self.mk_charmm_df(a_i, a_j, a_k, func, theta, cth, s0, kub)

    def mk_charmm_df(self,
                     a_i: list[int],  # index of the 1st atom in the angles
                     a_j: list[int],  # index of the 2nd atom in the angles
                     a_k: list[int],  # index of the 3rd atom in the angles
                     func: list[int],  # Type (function) of the angles
                     theta: list[float],  # Equilibrium angle
                     cth: list[float],  # Force constant
                     s0: list[float],  # Force constant
                     kub: list[float]  # Force constant
                     ) -> pd.DataFrame:
        """make DataFrame and check if they are same as atoms name"""
        # pylint: disable=too-many-arguments
        df_angles: pd.DataFrame
        df_angles = pd.DataFrame(columns=['ai', 'aj', 'ak', 'funct', 'theta',
                                          'cth', 's0', 'kub'])
        df_angles['ai'] = a_i
        df_angles['aj'] = a_j
        df_angles['ak'] = a_k
        df_angles['funct'] = func
        df_angles['theta'] = theta
        df_angles['cth'] = cth
        df_angles['s0'] = s0
        df_angles['kub'] = kub
        return df_angles


class DihedralsInfo:
    """get the dihdrals list from Itp class and return a dataframe"""
    def __init__(self,
                 dihedrals: list[str],  # lines of dihedrals section by Itp
                 atoms: pd.DataFrame,  # atoms df from AtomsInfo to get names
                 style: str = 'normal'
                 ) -> None:
        """get the dihedrals infos"""
        if style == 'normal':
            self.mk_dihedrals_df(dihedrals, atoms)
        elif style == 'charmm':
            self.mk_charmm_dihedrals_df(dihedrals)

    def mk_dihedrals_df(self,
                        dihedrals: list[str],  # lines of dihedrals section
                        atoms: pd.DataFrame  # atoms df from AtomInfo
                        ) -> None:
        """call all the methods to make the bonds DataFrame"""
        # pylint: disable=invalid-name
        a_i: list[int]  # index of the 1st atoms in the dihedrals
        a_j: list[int]  # index of the 2nd atoms in the dihedrals
        a_k: list[int]  # index of the 3rd atoms in the dihedrals
        a_h: list[int]  # index of the 4th atoms in the dihedrals
        funct: list[int]  # index of the type of the dihedrals
        names: list[str]  # name of the dihedrals
        a_i, a_j, a_k, a_h, funct, names = self.get_dihedrals(dihedrals)
        self.df = self.mk_df(a_i, a_j, a_k, a_h, funct, names, atoms)

    def get_dihedrals(self,
                      dihedrals: list[str],  # lines of dihedrals section
                      ) -> pd.DataFrame:  # DataFrame of the dihedrals
        """return bonds dataframe to make dihedrals dataframe"""
        columns: list[str]  # Columns of the dihedrals wild
        columns = ['ai', 'aj', 'ak', 'ah', 'typ', 'cmt', 'name']
        alter_columns: list[str] = ['ai', 'aj', 'ak', 'ah', 'funct',
                                    'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'name']
        a_i: list[int] = []  # index of the 1st atoms in the dihedrals
        a_j: list[int] = []  # index of the 2nd atoms in the dihedrals
        a_k: list[int] = []  # index of the 3rd atoms in the dihedrals
        a_h: list[int] = []  # index of the 4th atoms in the dihedrals
        funct: list[int] = []  # index of the type of the dihedrals
        names: list[str] = []  # name of the dihedrals
        for line in dihedrals:
            if line.startswith(';'):  # line start with ';' are commets&header
                l_line = free_char_line(line)
                if 'total' not in l_line and l_line != columns:
                    if l_line != alter_columns:
                        sys.exit(f'{bcolors.FAIL}{self.__class__.__name__}:\n'
                                 f'\tError in the [ dihedrals ] header of the'
                                 f' itp file\n\t{l_line = }\n\t{columns = }\n'
                                 f'{bcolors.ENDC}')
            else:
                tmp = line.strip().split('\t')
                tmp = [item for item in tmp if item]
                line = ' '.join(tmp)
                l_line = free_char_line(line)
                a_i.append(int(l_line[0]))
                a_j.append(int(l_line[1]))
                a_k.append(int(l_line[2]))
                a_h.append(int(l_line[3]))
                funct.append(int(l_line[4]))
                if len(l_line) == 6:
                    names.append(l_line[5])
                else:
                    names.append('')
        return a_i, a_j, a_k, a_h, funct, names

    def mk_df(self,
              a_i: list[int],  # index of the 1st atom in the dihedrals
              a_j: list[int],  # index of the 2nd atom in the dihedrals
              a_k: list[int],  # index of the 3rd atom in the dihedrals
              a_h: list[int],  # index of the 4th atom in the dihedrals
              funct: list[int],  # types of the function
              names: list[str],  # names form dihedrals section
              atoms: pd.DataFrame  # atoms df from AtomsInfo to cehck the name
              ) -> pd.DataFrame:  # dihedrals DataFrame
        """make DataFrame and check if they are same as atoms name"""
        # pylint: disable=too-many-arguments
        df_dihedrals: pd.DataFrame  # to save the dihedrals_df
        df_dihedrals = pd.DataFrame(
            columns=['ai', 'aj', 'ak', 'ah', 'typ', 'cmt', 'name'])
        df_dihedrals['ai'] = a_i
        df_dihedrals['aj'] = a_j
        df_dihedrals['ak'] = a_k
        df_dihedrals['ah'] = a_h
        df_dihedrals['name'] = names
        df_dihedrals['cmt'] = [';' for _ in a_i]
        df_dihedrals['typ'] = funct
        df_dihedrals = self.check_names(df_dihedrals, atoms)
        df_dihedrals.index += 1
        return df_dihedrals

    @staticmethod
    def check_names(df_dihedrals: pd.DataFrame,  # Df to check its names column
                    atoms: pd.DataFrame  # Df source (mostly atoms df)
                    ) -> None:
        """ checks the names column in the source file with comparing it
        with names from the atoms dataframe name column for each atom"""
        df_c: pd.DataFrame = df_dihedrals.copy()
        ai_name: list[str]  # name of the 1st atom from the list
        aj_name: list[str]  # name of the 2nd atom from the list
        ak_name: list[str]  # name of the 3rd atom from the list
        ah_name: list[str]  # name of the 4th atom from the list
        ai_name = [atoms.loc[atoms['atomnr'] == item]['atomname'][item-1]
                   for item in df_dihedrals['ai']]
        aj_name = [atoms.loc[atoms['atomnr'] == item]['atomname'][item-1]
                   for item in df_dihedrals['aj']]
        ak_name = [atoms.loc[atoms['atomnr'] == item]['atomname'][item-1]
                   for item in df_dihedrals['ak']]
        ah_name = [atoms.loc[atoms['atomnr'] == item]['atomname'][item-1]
                   for item in df_dihedrals['ah']]
        names: list[str] = [f'{i}-{j}-{k}-{h}' for i, j, k, h
                            in zip(ai_name, aj_name, ak_name, ah_name)]
        df_c['name'] = names
        return df_c

    def mk_charmm_dihedrals_df(self,
                               dihedrals: list[str]  # lines of dihedrals
                               ) -> None:
        """call all the methods to make the bonds DataFrame"""
        # pylint: disable=invalid-name
        a_i: list[str] = []
        a_j: list[str] = []
        a_k: list[str] = []
        a_h: list[str] = []
        func: list[int] = []
        phi0: list[float] = []
        cp: list[float] = []
        mult: list[int] = []
        for line in dihedrals:
            if line.startswith(';'):
                pass
            else:
                tmp = line.split()
                tmp = [item for item in tmp if item]
                line = ' '.join(tmp)
                l_line = free_char_line(line)
                a_i.append(str(l_line[0]))
                a_j.append(str(l_line[1]))
                a_k.append(str(l_line[2]))
                a_h.append(str(l_line[3]))
                func.append(int(l_line[4]))
                phi0.append(float(l_line[5]))
                cp.append(float(l_line[6]))
                try:
                    mult.append(int(l_line[7]))
                except IndexError:
                    mult.append(0)
        self.df = self.mk_charmm_df(a_i, a_j, a_k, a_h, func, phi0, cp, mult)

    def mk_charmm_df(self,
                     a_i: list[int],  # index of the 1st atom in the dihedrals
                     a_j: list[int],  # index of the 2nd atom in the dihedrals
                     a_k: list[int],  # index of the 3rd atom in the dihedrals
                     a_h: list[int],  # index of the 4th atom in the dihedrals
                     func: list[int],  # Type (function) of the dihedrals
                     phi0: list[float],  # Equilibrium angle
                     cp: list[float],  # Force constant
                     mult: list[int]  # Multiplicity
                     ) -> pd.DataFrame:
        """make DataFrame and check if they are same as atoms name"""
        # pylint: disable=too-many-arguments
        df_dihedrals: pd.DataFrame
        df_dihedrals = pd.DataFrame(
            columns=['ai', 'aj', 'ak', 'ah', 'funct', 'phi0', 'cp', 'mult'])
        df_dihedrals['ai'] = a_i
        df_dihedrals['aj'] = a_j
        df_dihedrals['ak'] = a_k
        df_dihedrals['ah'] = a_h
        df_dihedrals['funct'] = func
        df_dihedrals['phi0'] = phi0
        df_dihedrals['cp'] = cp
        df_dihedrals['mult'] = mult
        return df_dihedrals
