"""
Brush Analysis Module

This module is designed for analyzing surfactant-rich systems,
specifically focusing on the behavior of octadecylamines (ODA) at the
interface of water and decane layers in the absence of nanoparticles.

Key Analysis Features:
    1. Surface Morphology: Examines the structural characteristics of
        the interface.
    2. ODA Distribution: Quantifies the number of ODA molecules at the
        interface and within the water phase.
    3. Order Parameters: Calculates the order parameters of ODA at the
        interface to assess alignment and organization.

Additional Analysis (if applicable):
    - Micelle Analysis: Investigates the formation and characteristics
        of ODA micelles in the water phase.

Input Data:
    - com_pickle: A serialized file representing the center of mass
     for all residues in the system. It is structured as an array with
     the following format:
     | time | NP_x | NP_y | NP_z | res1_x | res1_y | res1_z | ...
     | resN_x | resN_y | resN_z | odn1_x | odn1_y | odn1_z | ...
     | odnN_z |

    This data is loaded using the GetCom class from the
  common.com_file_parser module.
Jan 10 2023
Saeed
"""
