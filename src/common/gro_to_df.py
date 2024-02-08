"""reading GRO files:
Lines contain the following information (top to bottom):

        title string (free format string, optional time in ps after
        't=')
        number of atoms (free format integer)
        one line for each atom (fixed format, see below)
        box vectors (free format, space separated reals), values:
        v1(x) v2(y) v3(z) v1(y) v1(z) v2(x) v2(z) v3(x) v3(y),
        the last 6 values may be omitted (they will be set to zero).
        GROMACS only supports boxes with v1(y)=v1(z)=v2(z)=0.

This format is fixed, ie. all columns are in a fixed position.
Optionally (for now only yet with trjconv) you can write gro files
with any number of decimal places, the format will then be n+5
positions with n decimal places (n+1 for velocities) in stead of 8
with 3 (with 4 for velocities). Upon reading, the precision will be
inferred from the distance between the decimal points (which will be
n+5). Columns contain the following information (from left to right):

        residue number (5 positions, integer)
        residue name (5 characters)
        atom name (5 characters)
        atom number (5 positions, integer)
        position (in nm, x y z in 3 columns, each 8 positions with 3
        decimal places)
        velocity (in nm/ps (or km/s), x y z in 3 columns, each 8
        positions with 4 decimal places)

Note that separate molecules or ions (e.g. water or Cl-) are regarded
as residues. If you want to write such a file in your own program
without using the GROMACS libraries you can use the following formats:

C format
    "%5d%-5s%5s%5d%8.3f%8.3f%8.3f%8.4f%8.4f%8.4f"
Fortran format
    (i5,2a5,i5,3f8.3,3f8.4)
Pascal format
    This is left as an exercise for the user

Note that this is the format for writing, as in the above example
fields may be written without spaces, and therefore can not be read
with the same format statement in C.
GROMACS manual
based on the C or fortran format length of the data line is 68. but
python read it as 69!!!
Columns 1-5: Atom serial number.
Columns 6-10: Atom name or type.
Columns 11-15: Residue name.
Columns 16-20: Residue number.
Columns 21-28: Atom coordinates along the x-axis (in nanometers).
Columns 29-36: Atom coordinates along the y-axis (in nanometers).
Columns 37-44: Atom coordinates along the z-axis (in nanometers).
Columns 45-48: Atom velocity (optional; not always present in GRO files).
Note: The coordinates and velocities are typically given in nanometers
"""


import sys
import typing
import pandas as pd
from common import logger
from common.colors_text import TextColor as bcolors


class ReadGro:
    """reading GRO file based on the doc"""

    info_msg: str = 'Message:\n'  # Message to pass for logging and writing
    line_len_long: int = 69  # Length of the lines in the data file
    line_len_short: int = 45  # Length of the lines in the data file

    gro_data: pd.DataFrame  # All the informations in the file
    # The follwings will set in __process_header_tail method:
    title: str  # Name of the system
    number_atoms: int  # Total number of atoms in the system
    pbc_box: str  # Size of the box (its 3 floats but save as a string)

    def __init__(self,
                 fname: str,  # Name of the input file
                 log: logger.logging.Logger
                 ) -> None:
        self.gro_data = self.read_gro(fname)
        self.write_msg(log)
        self.info_msg = ''  # Empety the msg

    def read_gro(self,
                 fname: str  # gro file name
                 ) -> pd.DataFrame:
        """read gro file lien by line"""
        counter: int = 0  # To count number of lines
        processed_line: list[dict[str, typing.Any]] = []  # All proccesed lines
        with open(fname, 'r', encoding='utf8') as f_r:
            while True:
                line = f_r.readline()
                if len(line) not in [self.line_len_long, self.line_len_short]:
                    self.__process_header_tail(line.strip(), counter)
                else:
                    processed_line.append(self.__process_line(line.rstrip()))
                counter += 1
                if not line.strip():
                    break
        ReadGro.info_msg += (f'\tFile name is `{fname}`\n'
                             f'\tSystem title is `{self.title}`\n'
                             f'\tNumber of atoms is `{self.number_atoms}`\n'
                             f'\tBox boundary is `{self.pbc_box}`\n')
        return pd.DataFrame(processed_line)

    def __process_line(self,
                       line: str  # Data line
                       ) -> dict[str, typing.Any]:
        """process lines of information"""
        resnr = int(line[0:5])
        resname = line[5:10].strip()
        atomname = line[10:15].strip()
        atomnr = int(line[15:20])
        a_x = float(line[20:28])
        a_y = float(line[28:36])
        a_z = float(line[36:44])
        processed_line: dict[str, typing.Any] = {
                                                 'residue_number': resnr,
                                                 'residue_name': resname,
                                                 'atom_name': atomname,
                                                 'atom_id': atomnr,
                                                 'x': a_x,
                                                 'y': a_y,
                                                 'z': a_z}
        if len(line) == self.line_len_long:
            v_x = float(line[44:52])
            v_y = float(line[52:60])
            v_z = float(line[60:68])
            processed_line: dict[str, typing.Any] = {
                                                     'vx': v_x,
                                                     'vy': v_y,
                                                     'vz': v_z
                                                    }
        return processed_line

    def __process_header_tail(self,
                              line: str,  # Line in header or tail
                              counter: int  # Line number
                              ) -> None:
        """Get the header, number of atoms, and box size"""
        if counter == 0:
            self.title = line
        elif counter == 1:
            self.number_atoms = int(line)
        elif counter == self.number_atoms + 2:
            self.pbc_box = line

    def write_msg(self,
                    log: logger.logging.Logger
                    ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{ReadGro.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)

if __name__ == '__main__':
    ReadGro(sys.argv[1], log=logger.setup_logger('read_gro.log'))
