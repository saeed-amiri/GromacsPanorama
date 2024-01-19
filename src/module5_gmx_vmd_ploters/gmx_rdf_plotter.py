"""
Plot Radial Distribution Function (RDF) Calculated from GROMACS

This script plots the radial distribution function (RDF) and cumulative
distribution function (CDF) for Chloroacetate (CLA) at the surface of
a nanoparticle (NP). It utilizes data generated by GROMACS.

GROMACS offers two methods for calculating RDF:
    1. Based on the center of mass (COM) of the NP.
    2. Based on the outermost residues of the NP, specifically APTES
        (APTES being the functional groups on the NP).

The script generates the following plots:
    - RDF plots for both COM-based and outermost residue-based
        calculations.
    - CDF plots corresponding to both calculation methods.

Inputs:
    The script requires RDF and CDF data files for each calculation
        method. It will generate plots if these files are present.

Notes:
    - The script is specifically designed for RDF and CDF analysis in
        the context of nanoparticles and their surface functional
        groups.
    - Ensure that the input data files are in the correct format as
        expected by the script.
"""
