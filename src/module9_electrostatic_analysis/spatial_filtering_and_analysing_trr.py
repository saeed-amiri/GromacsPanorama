"""
Dynamic Trajectory Filtering and Analysis Script

This script performs dynamic selection of atoms within a specified
radius from the center of mass (COM) of a defined target (e.g., a
nanoparticle) in molecular dynamics (MD) simulations. For each frame
of the input trajectory, it identifies atoms (and their respective
esidues) within the defined radius, analyzes specified properties of
this dynamic selection, and writes a new trajectory file containing
only the selected atoms for further analysis.

The script is designed to facilitate the study of dynamic interactions
and the local environment around specified targets in MD simulations,
providing insights into spatial and temporal variations in properties
such as density, coordination number, or other custom analyses.

Usage:
  python spatial_filtering_and_analysing_trr.py  <input_trajectory>
The tpr file and gro file names will be created from the name of the
input trr file

Arguments that set by the script:
    topology_file : Topology file (e.g., .gro, .pdb) corresponding to
        the input trajectory.
    output_trajectory: Output trajectory file name for the filtered
        selection.
    radius: Radius (in Ångströms) for dynamic selection around the
        target's COM.
    statistics output: Output file name for numbers of residues and
        charges of each of them.
    charge density output: Output file name for the charge density and
        potential of the final system

Options:
    selection_string : MDAnalysis-compatible selection string for
        defining the target (default: 'COR_APT').

Features:
    - Dynamic selection based on spatial criteria, adjustable for each
        frame of the trajectory.
    - Analysis of selected properties for the dynamically selected
        atoms/residues.
    - Generation of a new, filtered trajectory file for targeted
        further analysis.

Example:
    python spatial_filtering_and_analysing_trr.py simulation.trr

Opt. by ChatGpt.
21 Feb 2023
Saeed
"""
