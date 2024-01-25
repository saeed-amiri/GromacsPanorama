"""
Analyzing the Order Parameter Data

This documentation outlines the process of analyzing order parameter
data using two Python modules: `order_parameter_pickle_parser.py` and
`common.com_file_parser.py.`

Data Reading:
    The `order_parameter_pickle_parser.py` reads the order parameter data,
    which is a dictionary of np.ndarray.
    The `common.com_file_parser.py` reads the center of mass data, also
    a dictionary of np.ndarray.

Data Structure and Keys:
    Both datasets have similar keys and structures, with a minor
    difference in the center of mass (COM) file.
    The keys of the dictionaries represent different residues:
        1: SOL (Water)
        2: CLA (Chloride ions)
        3: POT (Potassium ions)
        4: ODN (Octadecylamine: ODA)
        5: D10 (Decane)
    In the COM data, key 0 is reserved for the ODN amino group.

Array Layout:
    COM File:
        Layout: | Time | NP_x | NP_y | NP_z | Res1_x | Res1_y | Res1_z
        | ... | ResN_x | ResN_y | ResN_z | Odn1_x | Odn1_y | Odn1_z |
        ... | OdnN_z |
    Order Parameter File:
        Layout: | Time | NP_Sx | NP_Sy | NP_Sz | Res1_Sx | Res1_Sy |
        Res1_Sz | ... | ResN_Sx | ResN_Sy | ResN_Sz |
        'S' denotes the order parameter.

Special Considerations:
    The order parameter for ions (CLA and POT) is set to zero, as
    defining an order parameter for single-atom residues is not
    meaningful.
    The values for the nanoparticle (NP) are also set to zero.
    This approach is chosen for ease of data manipulation and alignment
    between the two datasets.

Purpose of COM Data:
    The COM data is essential for determining the center of mass of
    each residue.
    This information is valuable for further analysis, particularly in
    understanding the spatial distribution and orientation of
    molecules in the system.

Opt. ChatGPT
Author: Saeed
Date: 25 January 2024
"""
