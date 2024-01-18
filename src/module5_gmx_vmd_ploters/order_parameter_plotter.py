"""
Plotting Order Parameters

This script plots the computed order parameters in a system with and
without nanoparticles (NP). It supports a variety of plots:

    1. Simple plot of the order parameter.
    2. Plot of the order parameter with error bars.
    3. Comparative plot of order parameters for systems both with and
        without NP, if data for both are available.
    4. Column plot showing the nominal number of Oda molecules with
        the actual number at the interface.

Data Format:
The data should be in a columnar format with the following columns:
    - 'name': Identifier of the data point (e.g., '0Oda', '5Oda').
    - 'nominal_oda': Nominal number of Oda molecules at the interface.
    - 'actual_oda': Average number of Oda molecules at the interface.
    - 'order_parameter': Computed order parameter.
    - 'std': Standard deviation of the order parameter.

Note: The inclusion of an error bar in the plot is optional.
Opt. by ChatGpt
Saeed
18 Jan 2023
"""
