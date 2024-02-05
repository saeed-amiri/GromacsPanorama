"""Reading the PDB file and returning the coordinates.

    NONTE:
    It will return a dataframe without 'type' column since it is not
    in the PDB file. It will be added to the data frame by the read_itp
    scrips."""


import sys
import typing

import pandas as pd

from common import logger
from common.colors_text import TextColor as bcolors


class Pdb:
    """
    The PDB file consider is in standard PDB format.
    It reads the PDB file and return the information in DataFrames.
    convert LAMMPS data file to a standard PDB file format based on:
    [https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html]
    A PDB file has a format as:
    ATOM 	atomic coordinate record containing the X,Y,Z
        orthogonal Å coordinates for atoms in standard residues]
        (amino acids and nucleic acids).

    HATATM 	atomic coordinate record containing the X,Y,Z orthogonal Å
    coordinates for atoms in nonstandard residues. Nonstandard residues
    include inhibitors, cofactors, ions, and solvent. The only functional
    difference from ATOM records is that HETATM residues are by default
    not connected to other residues. Note that water residues should be
    in HETATM records.

    Protein Data Bank Format for ATOM:
      Coordinate Section
        Record Type	Columns	Data 	Justification	Data Type
        ATOM 	1-4	“ATOM”	                    	character
        7-11#	Atom serial number      	right	integer
        13-16	Atom name	                left*	character
        17	    Alternate location indicator		character
        18-20§	Residue name	            right	character
        22	    Chain identifier		            character
        23-26	Residue sequence number	    right	integer
        27	    Code for insertions of residues		character
        31-38	X orthogonal Å coordinate	right	real (8.3)
        39-46	Y orthogonal Å coordinate	right	real (8.3)
        47-54	Z orthogonal Å coordinate	right	real (8.3)
        55-60	Occupancy	                right	real (6.2)
        61-66	Temperature factor	        right	real (6.2)
        73-76	Segment identifier¶	        left	character
        77-78	Element symbol              right	character
        79-80	Charge		                        character

    Protein Data Bank Format for HATATM:
      Coordinate Section
        Record Type	Columns	Data 	Justification	Data Type
        1-6	“HETATM”		character
        7-80	same as ATOM records

    #Chimera allows (nonstandard) use of columns 6-11 for the integer
        atom serial number in ATOM records, and in TER records, only the
        “TER” is required.
    *Atom names start with element symbols right-justified in columns
        13-14 as permitted by the length of the name. For example, the
        symbol FE for iron appears in columns 13-14, whereas the symbol
        C for carbon appears in column 14 (see Misaligned Atom Names).
        If an atom name has four characters, however, it must start in
        column 13 even if the element symbol is a single character
        (for example, see Hydrogen Atoms).
    §Chimera allows (nonstandard) use of four-character residue names
        occupying an additional column to the right.
    ¶Segment identifier is obsolete, but still used by some programs.
        Chimera assigns it as the atom attribute pdbSegment to allow
        command-line specification.
    The format of ecah section is (fortran style):
    Format (A6,I5,1X,A4,A1,A3,1X,A1,I4,A1,3X,3F8.3,2F6.2,10X,A2,A2)
    """

    info_msg: str = 'Message:\n'  # Message to pass for logging and writing

    def __init__(self,
                 fname: str,  # PDB file name
                 log: logger.logging.Logger
                 ) -> None:
        self.info_msg += f"\tReading '{fname}' ...\n"
        self.pdb_df: pd.DataFrame = self.get_data(fname)
        self.__write_msg(log)
        self.info_msg = ''  # Empety the msg

    def get_data(self,
                 fname: str  # PDB file name
                 ) -> pd.DataFrame:
        """Read and get the atomic strcuture in lammps`"""
        # Read PDB file and return a list of list
        data = self.read_pdb(fname)
        # Convert the list to DataFrame
        return self.mk_df(data)

    def read_pdb(self,
                 fname: str  # PDB file name
                 ) -> list:
        """Reading line by line of the pdb file"""
        data_list: list[typing.Any] = []
        with open(fname, 'r', encoding='utf8') as f_i:
            while True:
                line = f_i.readline()
                if line.strip().startswith("ATOM"):
                    data_list.append(self.__process_atom(line))
                elif line.strip().startswith("HETATM"):
                    data_list.append(self.__process_hetatm(line))
                else:
                    pass
                if not line:
                    break
        return data_list

    @staticmethod
    def __process_atom(line: str) -> list:
        """Process the line on based on the PDB file, started with ATOM"""
        # first check the length of the line MUST be equal to 79
        if len(line) != 79:
            exit(f"ERROR! wrong line length: {len(line)} != 79")
        records: str = line[0:6].strip()
        atom_id: int = int(line[6:11].strip())
        atom_name: str = line[12:16].strip()
        residue_name: str = line[17:20].strip()
        residue_number: str = line[22:27].strip()
        x_i: float = float(line[30:39].strip())
        y_i: float = float(line[39:47].strip())
        z_i: float = float(line[47:55].strip())
        occupancy: float = float(line[55:61])
        temperature: float = float(line[61:67])
        atom_symbol: str = line[76:78].strip()
        return [records,
                atom_id,
                atom_name,
                residue_name,
                residue_number,
                x_i,
                y_i,
                z_i,
                occupancy,
                temperature,
                atom_symbol]

    @staticmethod
    def __process_hetatm(line: str) -> list:
        """Process the line on based on the PDB file, started with HATATM"""
        records: str = line[0:6].strip()
        atom_id: int = int(line[6:11].strip())
        atom_name: str = line[12:16].strip()
        residue_name: str = line[17:20].strip()
        residue_number: str = line[22:27].strip()
        x_i: float = float(line[30:39].strip())
        y_i: float = float(line[39:47].strip())
        z_i: float = float(line[47:55].strip())
        atom_symbol: str = line[76:78].strip()
        return [records,
                atom_id,
                atom_name,
                residue_name,
                residue_number,
                x_i,
                y_i,
                z_i,
                atom_symbol]

    @staticmethod
    def mk_df(data: list[list]) -> pd.DataFrame:
        """Making DataFrame from PDB file"""
        columns: list[str] = ['records',
                              'atom_id',
                              'atom_name',
                              'residue_name',
                              'residue_number',
                              'x',
                              'y',
                              'z',
                              'occupancy',
                              'temperature',
                              'atom_symbol']
        df_i = pd.DataFrame(data, columns=columns)
        return df_i

    @staticmethod
    def check_residue_number(df_i: pd.DataFrame) -> pd.DataFrame:
        """The residue number in the PDB file created by VMD usually
        does not have the correct order, and the numbering is almost
        random.
        Here we check that and if needed will be replaced with correct
        one.
        """
        # Make sure the residue type is integer
        df_i = df_i.astype({'residue_number': int})
        # Get the uniqe numbers of residues
        df_reduced: pd.DataFrame = df_i.groupby(df_i.residue_number).mean()
        # add a new index from 1
        df_reduced = df_reduced.reset_index()
        df_reduced.index += 1
        # make a dict from the new index and the residue
        residue_dict: dict[int, int] = dict(zip(df_reduced.residue_number,
                                                df_reduced.index))
        # Make a list of new residue number for the main DataFrame
        # I call it 'mol'
        mol: list[int] = []
        for _, row in df_i.iterrows():
            mol.append(residue_dict[row['residue_number']])
        # Add the new mol numbers to the DataFrame
        df_i['mol'] = mol
        del df_reduced
        del residue_dict
        return df_i

    def __write_msg(self,
                    log: logger.logging.Logger
                    ) -> None:
        """write and log messages"""
        print(f'{bcolors.OKCYAN}{self.__module__}:\n'
              f'\t{self.info_msg}{bcolors.ENDC}')
        log.info(self.info_msg)


if __name__ == '__main__':
    pdb = Pdb(sys.argv[1], log=logger.setup_logger('pdb2df.log'))
