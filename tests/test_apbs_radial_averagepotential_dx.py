"""
tests the RadialAveragePotential class in the:
    apbs_radial_averagepotential_dx module.
which averages the potential values in a 3D grid over radial distances
from a specified origin, read from a dx file.
"""

import unittest
from unittest.mock import MagicMock

import numpy as np

from src.module9_electrostatic_analysis.apbs_radial_averagepotential_dx \
    import RadialAveragePotential, InputConfig


class TestRadialAveragePotential(unittest.TestCase):
    """
    Test the RadialAveragePotential class.
    """

    def setUp(self):
        """
        Set up the test environment before each test.

        This method initializes test data and creates an instance of the 
        RadialAveragePotential class with a mocked logger.
        """
        # Set up known test data
        self.data = np.ones((5, 5, 5))  # uniform potential of 1
        self.grid_points = [5, 5, 5]
        self.grid_spacing = [1.0, 1.0, 1.0]
        self.origin = [0.0, 0.0, 0.0]

        # Mock logger
        self.mock_logger = MagicMock()

        # Create an instance of RadialAveragePotential with default config
        self.radial_average_potential = RadialAveragePotential(InputConfig())

    def test_uniform_potential(self):
        """
        Test the radial average calculation on a uniform potential.

        This test verifies that the radial average of a uniform potential
        is correctly computed, accounting for the potential unit
        conversion factor.
        """
        # pylint: disable=unused-variable
        # Test radial average on uniform potential
        data_arr = np.array(self.data).reshape(self.grid_points)
        radii, radial_average = self.radial_average_potential.radial_average(
            data_arr, self.grid_points, self.grid_spacing, self.origin)

        # Expect uniform potential to have the same average value scaled
        # by pot_unit_conversion
        expected_value = np.ones_like(radial_average) * \
            self.radial_average_potential.pot_unit_conversion
        np.testing.assert_almost_equal(radial_average, expected_value)

    def test_center_offset(self):
        """
        Test the radial average calculation with a center offset.

        This test verifies that the radial average calculation handles
        non-centered grids correctly, ensuring the average value
        remains consistent with the potential unit conversion factor.
        """
        # pylint: disable=unused-variable
        # Shift center to test handling of non-centered grids
        self.radial_average_potential.average_index_from = 2
        data_arr = np.array(self.data).reshape(self.grid_points)
        radii, radial_average = self.radial_average_potential.radial_average(
            data_arr, self.grid_points, self.grid_spacing, [2.0, 2.0, 2.0])

        # Expect uniform potential to have the same average value scaled
        # by pot_unit_conversion
        expected_value = np.ones_like(radial_average) * \
            self.radial_average_potential.pot_unit_conversion
        np.testing.assert_almost_equal(radial_average, expected_value)

    def test_calculate_center(self) -> None:
        """
        Test the calculation of the center of the grid.

        This test verifies that the center of the grid is correctly
        calculated by the radial average calculation.
        """
        # pylint: disable=unused-variable
        # Test calculation of center
        center = self.radial_average_potential._calculate_center(
            self.grid_points)
        expected_center = np.array([2.0, 2.0, 2.0])
        np.testing.assert_almost_equal(center, expected_center)


if __name__ == '__main__':
    unittest.main()
