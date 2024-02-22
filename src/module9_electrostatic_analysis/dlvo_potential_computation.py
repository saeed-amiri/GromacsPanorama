"""
DLVO Electrostatic Potential Computation Module

This module calculates the electrostatic potential (phi) around a
nanoparticle (NP) at the water-decane interface using the DLVO model.
The computation requires an initial charge density (sigma) on the
surface of the NP, typically assumed to be uniformly distributed.

Prerequisites:
- The initial charge density is derived from spatially filtered and
    analyzed molecular dynamics simulation outputs, specifically using
    `spatial_filtering_and_analysing_trr.py`.
- It's important to note that the actual charge distribution might not
    be uniform due to the NP's partial immersion in the oil phase,
    affecting the electrostatic interactions.

Key Features:
- This module integrates with other computational modules to refine
    sigma by considering the NP's contact angle at the water-
    decane interface. This angle helps determine the extent of the
    NP's surface exposed to the water phase, where the charges are
    predominantly located.
- In the absence of explicit contact angle data (from `contact.xvg`),
    the module estimates sigma using an average contact angle
    value, ensuring robustness in calculations.
- The main output is the computed electrostatic potential phi,
    essential for understanding the colloidal stability and
    interactions governed by the DLVO theory.

Inputs:
- `charge_df.xvg`: File containing the charge density distribution
    data.
- `contact.xvg` (optional): File containing contact angle measurements.
    If unavailable, an average contact angle is used for calculations.

Physics Overview:
The module employs the linearized Poisson-Boltzmann equation to model
electrostatic interactions, considering the influence of the NP's
geometry and surface charge distribution on the potential phi.
This approach allows for the quantification of repulsive forces
between colloidal particles, crucial for predicting system stability.

Output:
The module calculates and returns the electrostatic potential phi
surrounding the NP, offering insights into the colloidal interactions
within the system.

Usage:
To use this module, ensure the prerequisite files are correctly
formatted and located in the specified directory. The module can be
executed as a standalone script or integrated into larger simulation
frameworks for comprehensive colloidal system analysis.
22 Feb 2024
Saeed
"""
