"""read LAMMPS data file"""

import os
import re
import sys
import typing

import numpy as np
import pandas as pd

from common.colors_text import TextColor as bcolors


class Header:
    """
    read haeder of the data file
    check the number of the lines, atom, bond ... informations
    get the box , pairs, ... coefficents
    Use this class to read the header of the file (LAMMPS data file),
    and the file should have Masses with their name specified after (#)
    e.g.:
        Masses
        1 1.008000 # H
        2 16.000000 # OH
        3 16.000000 # OB
        4 28.059999 # Si
    it will return a few attributes for the class if they existed:
    Masses, Pair, and Angel and Dihedral coefficients. And also the name
    of the atoms types.
    The class BODY needs' names' to read the data file.
    """

    def __init__(self, infile) -> None:
        self.infile: str = infile
        print(f'\n{bcolors.OKCYAN}{self.__class__.__name__}: '
              f'({self.__module__})\n'
              f'\tReading: `{self.infile}`{bcolors.ENDC}')
        self.atomsLine: int
        self.diameter: float  # Diameter of the NP if its on the filename
        self.GROMACS_flag: bool  # If to convert to GROMACS pdb and itp
        self.file_exist(infile)
        self.atomsLine = self.check_file()
        self.set_attrs()
        self.set_attr_zero()
        self.read_header()

    def file_exist(self,
                   fname: str  # Name of the input file
                   ) -> None:
        """check if the file is exist"""
        if not os.path.exists(fname):
            sys.exit(f'\t{bcolors.FAIL}Error! `{fname}` does not exist!'
                     f'{bcolors.ENDC}\n')

    def check_file(self) -> int:
        """ Check header
        input:
            - INFILE (lammps data file)
        output:
            - number of header lines
        """
        # An integer to prevent over-reading in case of header bugs
        MAXHEADER: int = 1000
        # track the number of lines in the hedaer
        linecount: int = 0
        with open(self.infile, 'r') as f:
            while True:
                linecount += 1
                line = f.readline()
                if line.strip().startswith('Atoms'):
                    atomsLine = linecount
                    break
                if linecount > MAXHEADER:
                    sys.exit(f'{bcolors.FAIL}\tError! there is problem '
                             f'in the header of the `{self.infile}`, '
                             'maybe a long header or wrong file format!\n'
                             '\tThe input file should be in LAMMPS full '
                             f'atom format{bcolors.ENDC}')
                if not line:
                    sys.exit(f'{bcolors.FAIL}{self.__class__.__name__}'
                             f'wrong data file{bcolors.ENDC}\n')
        return atomsLine

    def read_header(self) -> None:
        """read header to get all the available info
        Read header now and get the data
        """
        # Setting dictionaries to save data of each block in the header

        self.set_diameter()
        # Setting flags to save data correctly
        Masses, PairCoeff, BondCoeff, AngleCoeff, DihedralCoeff, Atoms\
            = False, False, False, False, False, False
        # Track the number of lines
        linecount: int = 0
        with open(self.infile, 'r', encoding="utf8") as f:
            while True:
                linecount += 1
                if linecount > self.atomsLine:
                    break
                line: str = f.readline()
                if line.strip().endswith("atoms"):
                    self.NAtoms = int(line.strip().split(' ')[0])

                elif line.strip().endswith("atom types"):
                    self.NAtomTyp = int(line.strip().split(' ')[0])

                elif line.strip().endswith("bonds"):
                    self.NBonds = int(line.strip().split(' ')[0])

                elif line.strip().endswith("bond types"):
                    self.NBondTyp = int(line.strip().split(' ')[0])

                elif line.strip().endswith("angles"):
                    self.NAngles = int(line.strip().split(' ')[0])

                elif line.strip().endswith("angle types"):
                    self.NAngleTyp = int(line.strip().split(' ')[0])

                elif line.strip().endswith("dihedrals"):
                    self.NDihedrals = int(line.strip().split(' ')[0])

                elif line.strip().endswith("dihedral types"):
                    self.NDihedralTyp = int(line.strip().split(' ')[0])

                elif line.strip().endswith("xhi"):
                    self.Xlim = self.get_axis_lim(line.strip().split('xlo')[0])

                elif line.strip().endswith("yhi"):
                    self.Ylim = self.get_axis_lim(line.strip().split('ylo')[0])

                elif line.strip().endswith("zhi"):
                    self.Zlim = self.get_axis_lim(line.strip().split('zlo')[0])

                # setting up Flages for reading the cards of data in the file
                elif line.strip().startswith("Masses"):
                    Masses, PairCoeff, BondCoeff, AngleCoeff, DihedralCoeff,\
                        Atoms = True, False, False, False, False, False

                elif line.strip().startswith("Pair"):
                    Masses, PairCoeff, BondCoeff, AngleCoeff, DihedralCoeff,\
                        Atoms = False, True, False, False, False, False

                elif line.strip().startswith("Bond Coeffs"):
                    Masses, PairCoeff, BondCoeff, AngleCoeff, DihedralCoeff,\
                        Atoms = False, False, True, False, False, False

                elif line.strip().startswith("Angle Coeffs"):
                    Masses, PairCoeff, BondCoeff, AngleCoeff, DihedralCoeff,\
                        Atoms = False, False, False, True, False, False

                elif line.strip().startswith("Dihedral Coeffs"):
                    Masses, PairCoeff, BondCoeff, AngleCoeff, DihedralCoeff,\
                        Atoms = False, False, False, False, True, False

                elif line.strip().startswith("Atoms"):
                    Masses, PairCoeff, BondCoeff, AngleCoeff, DihedralCoeff,\
                        Atoms = False, False, False, False, False, True

                elif line.strip():
                    if Masses:
                        self.get_masses(line.strip(), 'Masses')
                    elif PairCoeff:
                        self.get_pair_coeff(line.strip(), 'Pair')
                    elif BondCoeff:
                        self.get_bond_coeff(line.strip(), 'Bond')
                    elif AngleCoeff:
                        self.get_angle_coeff(line.strip(), 'Angle')
                    elif DihedralCoeff:
                        self.get_dihedral_coeff(line.strip(), 'Dihedral')
                if Atoms:
                    break
                if not line:
                    break

    def set_attrs(self) -> None:
        """set the dictionaries for all the infos in the system"""
        self.Names: dict[int, str] = {}
        self.Masses: dict[int, float] = {}
        self.PairCoeff: dict[int, typing.Any] = {}
        self.BondCoeff: dict[int, typing.Any] = {}
        self.AngleCoeff: dict[int, typing.Any] = {}
        self.Bonds_Names: dict[int, str] = {}
        self.DihedralCoeff: dict[int, typing.Any] = {}

    def set_attr_zero(self) -> None:
        """set the intial values to zero"""
        self.NAtoms: int = 0
        self.NBonds: int = 0
        self.NAngles: int = 0
        self.NAtomTyp: int = 0
        self.NBondTyp: int = 0
        self.NAngleTyp: int = 0
        self.NDihedrals: int = 0
        self.NDihedralTyp: int = 0
        self.diameter: int = 0

    def get_axis_lim(self, lim: typing.Any) -> list:
        """get x limit of the data"""
        try:
            lim = lim.split(' ')
            lim = [float(item) for item in lim if item]
        except ValueError:
            # In case there is \t and space in lines
            lim = [item for item in lim if lim]
            lim = lim[0].split('\t')
            lim = [float(item) for item in lim if item]
        return lim

    def get_masses(self, line: str, check: str) -> None:
        """stting the nth row of the dictionary"""
        if check not in line:
            self.GROMACS_flag = False
            typ = int(line.split(' ')[0])
            mass = float(line.split(' ')[1])
            try:
                name = line.split('#')[1].strip()
                all_name = name.strip().split(' ')
                all_name = [item for item in all_name if item]
                atoms_info_len: int  # Numbers of chars in Masses section
                atoms_info_len = len(all_name)
                if atoms_info_len == 2:
                    atom_name = all_name[0]
                    bond_name = all_name[1]
                elif atoms_info_len == 3:  # Data problem
                    atom_name = all_name[0]
                    bond_name = all_name[1]
                    print(f'{bcolors.WARNING}read_data:\n'
                          f'\t Unclear style in the Masses section:'
                          f'\t{all_name} {bcolors.ENDC}')
                elif atoms_info_len == 5:  # Get data for GROMACS
                    atom_name = all_name
                    self.GROMACS_flag = True
                else:
                    atom_name = line.split('#')[1].strip()
                    bond_name = atom_name
            except IndexError:
                sys.exit(f'{bcolors.FAIL}\tAtoms name in the `Masses` '
                         f'section is not defined; Ex.:\n\t\t'
                         f'Masses\n\t\t1 1 # X{bcolors.ENDC}')
            self.Masses[typ] = mass
            self.Names[typ] = atom_name
            if not self.GROMACS_flag:
                self.Bonds_Names[typ] = bond_name
            else:
                self.Bonds_Names[typ] = 'Nan'

    def get_pair_coeff(self, line, check) -> None:
        """stting the nth row of the dictionary"""
        if check not in line:
            line = line.split(' ')
            typ = int(line[0])
            i_style = line[1]
            i_coeff = line[2:]
            self.PairCoeff[typ] = {"style": i_style,
                                   "coeff": i_coeff}

    def get_bond_coeff(self, line, check) -> None:
        """stting the nth row of the dictionary"""
        if check not in line:
            line = line.split(' ')
            typ = int(line[0])
            i_style = line[1]
            i_coeff = line[2:]
            self.BondCoeff[typ] = {"style": i_style,
                                   "coeff": i_coeff}

    def get_angle_coeff(self, line, check) -> None:
        """stting the nth row of the dictionary"""
        if check not in line:
            line = line.split(' ')
            typ = int(line[0])
            i_style = line[1]
            i_coeff = line[2:]
            self.AngleCoeff[typ] = {"style": i_style,
                                    "coeff": i_coeff}

    def get_dihedral_coeff(self, line, check) -> None:
        """stting the nth row of the dictionary"""
        if check not in line:
            line = line.split(' ')
            typ = int(line[0])
            i_style = line[1]
            i_coeff = line[2:]
            self.DihedralCoeff[typ] = {"style": i_style,
                                       "coeff": i_coeff}

    def set_diameter(self) -> None:
        """Set the radius if nanoparticle"""
        try:
            self.diameter = float(re.findall(r'\d+', self.infile)[0])
        except IndexError:
            pass


class Body(Header):
    """
    read the data for atoms, velocities, bonds, angles, dihedrals
    It needs the names of the atoms read by HEADER class
    """

    def __init__(self, infile) -> None:
        super().__init__(infile)
        self.infile: str = infile  # name for the IO file
        self.read_body()

    def read_body(self):
        """read body of the data file"""
        self.Atoms, self.Velocities, self.Bonds, self.Angles, self.Dihedrals\
            = {}, {}, {}, {}, {}
        Atoms, Velocities, Bonds, Angles, Dihedrals\
            = False, False, False, False, False
        self.q_flag: bool = False  # if there are charges in columns

        with open(self.infile, 'r', encoding="utf8") as f:
            while True:
                line = f.readline()
                if line.strip().startswith('Atoms'):
                    Atoms, Velocities, Bonds, Angles, Dihedrals\
                        = True, False, False, False, False
                    self.q_flag = self.get_atom_style(line)
                elif line.strip().startswith('Velocities'):
                    Atoms, Velocities, Bonds, Angles, Dihedrals\
                        = False, True, False, False, False
                elif line.strip().startswith('Bonds'):
                    Atoms, Velocities, Bonds, Angles, Dihedrals\
                        = False, False, True, False, False
                elif line.strip().startswith('Angles'):
                    Atoms, Velocities, Bonds, Angles, Dihedrals\
                        = False, False, False, True, False
                elif line.strip().startswith('Dihedrals'):
                    Atoms, Velocities, Bonds, Angles, Dihedrals\
                        = False, False, False, False, True
                elif line.strip():
                    if Atoms:
                        self.get_atoms(line.strip())
                    elif Velocities:
                        self.get_velocities(line.strip())
                    elif Bonds:
                        self.get_bonds(line.strip())
                    elif Angles:
                        self.get_angles(line.strip())
                    elif Dihedrals:
                        self.get_dihedrals(line.strip())
                if not line:
                    break
            Atoms_df = pd.DataFrame.from_dict(
                            self.Atoms, orient='columns').T
            self.Atoms_df = self.com_to_zero(Atoms_df)
            self.Bonds_df = pd.DataFrame.from_dict(self.Bonds).T
            self.Angles_df = pd.DataFrame.from_dict(self.Angles).T
            self.Dihedrals_df = pd.DataFrame.from_dict(self.Dihedrals).T
            self.Velocities_df = pd.DataFrame.from_dict(self.Velocities).T
            self.Masses_df = self.set_masses()
            self.Net_charge = float(f'{self.Atoms_df["charge"].sum(): .3f}')
            del self.Atoms, self.Bonds, self.Angles, self.Dihedrals

    def get_atoms(self, line) -> None:
        """stting the nth row of the dictionary"""
        if 'Atoms' not in line:
            line = line.split()
            line = [item for item in line if item]
            atom_id = int(line[0])
            i_mol = int(line[1])
            i_typ = int(line[2])
            if self.q_flag:
                i_charge = float(line[3])
                i_col = 3
            else:
                i_col = 2
            i_x = float(line[i_col + 1])
            i_y = float(line[i_col + 2])
            i_z = float(line[i_col + 3])
            if not self.GROMACS_flag:
                i_bond_name = self.Bonds_Names[i_typ]
                i_name = self.Names[i_typ]
            else:
                i_name = self.Names[i_typ][0]
                i_bond_name = self.Names[i_typ][2]
            try:
                i_nx = int(line[i_col + 4])
                i_ny = int(line[i_col + 5])
                i_nz = int(line[i_col + 6])
            except (ValueError, IndexError) as _:
                i_nx = 0
                i_ny = 0
                i_nz = 0
            if self.q_flag:
                self.Atoms[atom_id] = {
                                       "atom_id": atom_id,
                                       "mol": i_mol,
                                       "typ": i_typ,
                                       "charge": i_charge,
                                       "x": i_x,
                                       "y": i_y,
                                       "z": i_z,
                                       "nx": i_nx,
                                       "ny": i_ny,
                                       "nz": i_nz,
                                       "cmt": '#',
                                       "name": i_name,
                                       "b_name": i_bond_name
                                       }
            else:
                self.Atoms[atom_id] = {
                                       "atom_id": atom_id,
                                       "mol": i_mol,
                                       "typ": i_typ,
                                       "x": i_x,
                                       "y": i_y,
                                       "z": i_z,
                                       "nx": i_nx,
                                       "ny": i_ny,
                                       "nz": i_nz,
                                       "cmt": '#',
                                       "name": i_name,
                                       "b_name": i_bond_name
                                       }

    def com_to_zero(self,
                    Atoms: pd.DataFrame  # Atoms dataframe from read from data
                    ) -> pd.DataFrame:
        """set the center of mass to zero"""
        print(f'{bcolors.OKCYAN}\tMove the center of mass to zero'
              f'{bcolors.ENDC}')
        x_cm: float = np.average(Atoms['x'])
        y_cm: float = np.average(Atoms['y'])
        z_cm: float = np.average(Atoms['z'])
        df: pd.DataFrame = Atoms.copy()
        df['x'] -= x_cm
        df['y'] -= y_cm
        df['z'] -= z_cm
        return df

    def get_atom_style(self, line: str) -> bool:
        """return atom style for the atoms informations
            bond: there is no charges for the system
            full: with charge column
        """
        style: str  # style of the atoms section in LAMMPS data file
        style = line.split('#')[1].strip()
        return_check: bool  # To return the style mode
        if style:
            if style == 'bond':
                return_check = False
            elif style == 'full':
                return_check = True
        else:
            return_check = True
        return return_check

    def get_velocities(self, line) -> None:
        """stting the nth row of the dictionary"""
        if 'Velocities' not in line:
            line = line.split()
            line = [item for item in line if item]
            atom_id = int(line[0])
            i_vx = float(line[1])
            i_vy = float(line[2])
            i_vz = float(line[3])
            self.Velocities[atom_id] = {
                                        "vx": i_vx,
                                        "vy": i_vy,
                                        "vz": i_vz
                                        }

    def get_bonds(self, line) -> None:
        """stting the nth row of the dictionary"""
        cmt_flag: bool = False
        if 'Bonds' not in line:
            if '#' in line:
                cmt_flag = True
            line = line.split()
            line = [item for item in line]
            line = [item for item in line if item]
            line[:4] = [int(item) for item in line[:4]]
            bond_id = line[0]
            i_typ = int(line[1])
            i_ai = int(line[2])
            i_aj = int(line[3])
            if cmt_flag:
                i_cmt = line[4]
                i_name = line[5]
                self.Bonds[bond_id] = {
                                       "typ": i_typ,
                                       "ai": i_ai,
                                       "aj": i_aj,
                                       "cmt": i_cmt,
                                       "name": i_name
                                       }
            else:
                self.Bonds[bond_id] = {
                                       "typ": i_typ,
                                       "ai": i_ai,
                                       "aj": i_aj
                                       }

    def get_angles(self, line) -> None:
        """stting the nth row of the dictionary"""
        if "Angles" not in line:
            if '#' in line:
                line = line.split('#')[0]
            line = line.split()
            line = [int(item) for item in line if item]
            angle_id = line[0]
            i_typ = int(line[1])
            i_ai = int(line[2])
            i_aj = int(line[3])
            i_ak = int(line[4])
            self.Angles[angle_id] = {
                                     "typ": i_typ,
                                     "ai": i_ai,
                                     "aj": i_aj,
                                     "ak": i_ak
                                     }

    def get_dihedrals(self, line) -> None:
        """stting the nth row of the dictionary"""
        if "Dihedrals" not in line:
            if '#' in line:
                line = line.split('#')[0]
            line = line.split()
            line = [int(item) for item in line if item]
            dihedrals_id = line[0]
            i_typ = int(line[1])
            i_ai = int(line[2])
            i_aj = int(line[3])
            i_ak = int(line[4])
            i_ah = int(line[5])
            self.Dihedrals[dihedrals_id] = {
                                            "typ": i_typ,
                                            "ai": i_ai,
                                            "aj": i_aj,
                                            "ak": i_ak,
                                            "ah": i_ah
                                            }

    def set_masses(self) -> pd.DataFrame:
        names_list: list[str] = []  # list to store all the names
        b_names_list: list[str] = []  # All the name in bonds, angles, dihedral
        cmt_list: list[str] = []  # list to store '#'
        columns: list[str] = ['mass']  # column name of the DFs
        Masses_df = pd.DataFrame.from_dict(self.Masses,
                                           orient='index', columns=columns)
        Masses_df['typ'] = Masses_df.index
        for k, _ in self.Masses.items():
            names_list.append(self.Names[k])
            b_names_list.append(self.Bonds_Names[k])
            cmt_list.append('#')
        Masses_df['cmt'] = cmt_list
        if not self.GROMACS_flag:
            Masses_df['name'] = names_list
            Masses_df['b_name'] = b_names_list
        else:
            names: list[str] = []  # Atoms names
            residues: list[str] = []  # Residues names
            elements: list[str] = []  # Elements' symbols
            records: list[str] = []  # ATOMS or HATOM or TER
            ff_type: list[str] = []  # Force field type of the atom
            for key in self.Names.keys():
                names.append(self.Names[key][0])
                residues.append(self.Names[key][1])
                elements.append(self.Names[key][2])
                records.append(self.Names[key][3])
                ff_type.append(self.Names[key][4])
            Masses_df['names'] = names
            Masses_df['residues'] = residues
            Masses_df['elements'] = elements
            Masses_df['records'] = records
            Masses_df['ff_type'] = ff_type
        return Masses_df


class ReadData(Body):
    """reading the input file
    This class call all other classes and make one output obj
    """
    def __init__(self, infile) -> None:
        super().__init__(infile)
        self.__write_infos()

    def __write_infos(self) -> None:
        print(f'{bcolors.OKGREEN}\tRead Data Summary:\n'
              f'\t\t# Atoms: {self.NAtoms}, # Atom`s types: {self.NAtomTyp}\n'
              f'\t\t# Bonds: {self.NBonds}, # Bond`s types: {self.NBondTyp}\n'
              f'\t\t# Angles: {self.NAngles}, '
              f'# Angle`s types: {self.NAngleTyp}\n'
              f'\t\t# Dihedrals: {self.NDihedrals}, '
              f'# Dihedral`s types: {self.NDihedralTyp}\n'
              f'\t\tTotal charge: {self.Atoms_df["charge"].sum():.4f}\n'
              f'\t\tMin charge: {self.Atoms_df["charge"].min()}\n'
              f'\t\tMax charge: {self.Atoms_df["charge"].max()}'
              f'{bcolors.ENDC}'
              )


if __name__ == '__main__':
    ReadData(sys.argv[1])
