"""
The PQR format is the primary input format for biomolecular structure
in APBS package. This format is a modification of the PDB format which
allows users to add charge and radius parameters to existing PDB data
while keeping it in a format amenable to visualization with standard
molecular graphics programs. The origins of the PQR format are somewhat
uncretain, but has been used by several computational biology software
programs, including MEAD and AutoDock. UHBD uses a very similar format
called QCD.

APBS reads very loosely-formatted PQR files: all fields are
-delimited, thereby allowing coordinates which are larger/smaller
than ± 999 Å.

APBS reads data on a per-line basis from PQR files using the following
format:
Field_name Atom_number Atom_name Residue_name Residue_number X Y Z \
    Charge Radius

where the whitespace is the most important feature of this format. The
fields are:

Field_name
    A string which specifies the type of PQR entry and should either be
    ATOM or HETATM in order to be parsed by APBS.

Atom_number
    An integer which provides the atom index.

Atom_name
    A string which provides the atom name.

Residue_name
    A string which provides the residue name.

Residue_number
    An integer which provides the residue index.

X Y Z
    3 floats which provide the atomic coordiantes.

Charge
    A float which provides the atomic charge (in electrons).

Radius
    A float which provides the atomic radius (in Å).

Clearly, this format can deviate wildly from PDB, particularly when
large coordinate values are used. However, in order to maintain
compatibility with most molecular graphics programs, the PDB2PQR
utilities provided with apbs (see the Parameterization section)
attemp to preserve the PDB format as much as possible.
From:
    https://ics.uci.edu/~dock/manuals/apbs/html/user-guide/a2566.html

"""

import sys
import typing

from dataclasses import dataclass

import pandas as pd

from common.colors_text import TextColor as bcolors


@dataclass
class PqrAtomData:
    """
    Dataclass to hold information about an atom in a PQR file.
    """
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=invalid-name
    field_name: str
    atom_id: int
    atom_name: str
    residue_name: str
    chain_id: str
    residue_number: int
    x: float
    y: float
    z: float
    charge: float
    radius: float

    # Define a class attribute for field slices
    field_slices = {
        'field_name': slice(0, 6),
        'atom_id': slice(6, 11),
        'atom_name': slice(12, 16),
        'residue_name': slice(17, 20),
        'chain_id': slice(21, 22),
        'residue_number': slice(22, 29),
        'x': slice(30, 38),
        'y': slice(38, 46),
        'z': slice(46, 54),
        'charge': slice(54, 62),
        'radius': slice(62, 69)
    }

    def __init__(self,
                 **kwargs
                 ) -> None:
        self.field_name = kwargs.get('field_name', '')
        self.atom_id = kwargs.get('atom_id', 0)
        self.atom_name = kwargs.get('atom_name', '')
        self.residue_name = kwargs.get('residue_name', '')
        self.chain_id = kwargs.get('chain_id', '')
        self.residue_number = kwargs.get('residue_number', 0)
        self.x = kwargs.get('x', 0.0)
        self.y = kwargs.get('y', 0.0)
        self.z = kwargs.get('z', 0.0)
        self.charge = kwargs.get('charge', 0.0)
        self.radius = kwargs.get('radius', 0.0)

    @staticmethod
    def from_pqr_line(line: str) -> 'PqrAtomData':
        """
        Factory method to create a PqrAtomData instance from a PQR file line.
        """
        # Use the field_slices to extract data from the line
        data = {field: line[slice_].strip() for field, slice_ in
                PqrAtomData.field_slices.items()}
        # Convert numeric fields to their appropriate types
        data_: dict[str, typing.Union[str, int, float]] = {}
        try:
            data_['field_name'] = data['field_name']
            data_['atom_id'] = int(data['atom_id'])
            data_['atom_name'] = data['atom_name']
            data_['residue_name'] = data['residue_name']
            data_['chain_id'] = data['chain_id']
            data_['residue_number'] = int(data['residue_number'])
            data_['x'] = float(data['x'])
            data_['y'] = float(data['y'])
            data_['z'] = float(data['z'])
            data_['charge'] = float(data['charge'])
            data_['radius'] = float(data['radius'])
        except ValueError as err:
            print(f"{bcolors.FAIL}Error parsing line:{bcolors.ENDC}")
            print(f"{bcolors.CAUTION}line: {line}{bcolors.ENDC}")
            print(f"{bcolors.WARNING}data: {data}{bcolors.ENDC}")
            sys.exit(f"{bcolors.FAIL}Error: {err}{bcolors.ENDC}")
        return PqrAtomData(**data_)


def read_pqr_to_dataframe(pqr_file_name: str) -> pd.DataFrame:
    """
    Reads a PQR file and returns a DataFrame with the atom data.
    """
    atom_data_list: list[dict[str, object]] = []

    with open(pqr_file_name, 'r', encoding='utf8') as file:
        for line in file:
            if line.startswith("ATOM"):
                # Assuming from_pqr_line is a method that parses a line
                # into a PqrAtomData instance
                atom_data_instance = PqrAtomData.from_pqr_line(line)
                # Convert the PqrAtomData instance to a dictionary (or
                # directly to a list of its values)
                atom_data_dict = {
                    'field_name': atom_data_instance.field_name,
                    'atom_id': atom_data_instance.atom_id,
                    'atom_name': atom_data_instance.atom_name,
                    'residue_name': atom_data_instance.residue_name,
                    'chain_id': atom_data_instance.chain_id,
                    'residue_number': atom_data_instance.residue_number,
                    'x': atom_data_instance.x,
                    'y': atom_data_instance.y,
                    'z': atom_data_instance.z,
                    'charge': atom_data_instance.charge,
                    'radius': atom_data_instance.radius
                }
                atom_data_list.append(atom_data_dict)

    return pd.DataFrame(atom_data_list)


if __name__ == "__main__":
    print(read_pqr_to_dataframe(sys.argv[1]))
