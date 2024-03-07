import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

from src.module9_electrostatic_analysis.spatial_filtering_and_analysing_trr \
    import TrrFilterAnalysis


class TestTrrFilterAnalysis(unittest.TestCase):

    def test_compute_radius(self):
        # Create mock objects for the required arguments
        mock_trajectory = Mock()
        mock_log = Mock()

        # Create an instance of TrrFilterAnalysis
        my_class = TrrFilterAnalysis(mock_trajectory, mock_log)

        # Set up the configs.trajectory attribute on my_class
        my_class.configs = Mock()
        my_class.configs.trajectory = 'test_trajectory.trr'

        # Set up the input data
        ff_sigma = pd.DataFrame({'sigma': [1, 2, 3]})

        # Set the force_field.ff_sigma attribute of my_class
        my_class.force_field.ff_sigma = ff_sigma

        # Call the compute_radius method
        result = my_class.compute_radius()

        # Define the expected output
        expected_output = pd.DataFrame(
            {'sigma': [1, 2, 3], 'radius': [0.890898, 1.781796, 2.672694]})

        # Compare the result with the expected output
        pd.testing.assert_frame_equal(result, expected_output)
