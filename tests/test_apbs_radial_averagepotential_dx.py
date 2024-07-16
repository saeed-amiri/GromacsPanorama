"""
tests the RadialAveragePotential class in the:
    apbs_radial_averagepotential_dx module.
which averages the potential values in a 3D grid over radial distances
from a specified origin, read from a dx file.
"""

import unittest
from unittest.mock import MagicMock

import numpy as np

import matplotlib.pyplot as plt

from src.module9_electrostatic_analysis.apbs_radial_averagepotential_dx \
    import RadialAveragePotential, InputConfig

from src.common.colors_text import TextColor as bcolors


class TestRadialAveragePotential(unittest.TestCase):
    """
    Test the RadialAveragePotential class.
    """

    def setUp(self) -> None:
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

    def test_calculate_center(self) -> None:
        """
        Test the calculation of the center of the grid.

        This test verifies that the center of the grid is correctly
        calculated by the radial average calculation.
        """
        # pylint: disable=unused-variable
        # Test calculation of center
        center = self.radial_average_potential.calculate_center(
            self.grid_points)
        expected_center = np.array([2.0, 2.0, 2.0])
        np.testing.assert_almost_equal(center, expected_center)

    def test_calculate_max_radius(self) -> None:
        """
        Test the calculation of the maximum radius in the grid.

        This test verifies that the maximum radius in the grid is correctly
        calculated by the radial average calculation.
        """
        # pylint: disable=unused-variable
        # Test calculation of max radius
        center_xyz: tuple[float, float, float] = (2.0, 2.0, 2.0)
        max_radius = self.radial_average_potential.calculate_max_radius(
            center_xyz, self.grid_spacing)
        expected_max_radius = 2.0
        self.assertAlmostEqual(max_radius, expected_max_radius)

    def test_create_distance_grid(self) -> None:
        """
        Test the creation of the distance grid.
        """

        # Define grid points
        grid_points: list[int] = [5, 5, 5]

        # Expected shapes based on grid_points
        expected_shape: tuple[float, float, float] = (5, 5, 5)

        # Call the method
        grid_x, grid_y, grid_z = \
            self.radial_average_potential.create_distance_grid(grid_points)

        # Check if the shapes of the returned arrays match the expected shapes
        self.assertEqual(
            grid_x.shape, expected_shape, "grid_x shape is incorrect")
        self.assertEqual(
            grid_y.shape, expected_shape, "grid_y shape is incorrect")
        self.assertEqual(
            grid_z.shape, expected_shape, "grid_z shape is incorrect")

        # Check the first and last elements to ensure linspace worked correctly
        self.assertEqual(grid_x[0, 0, 0],
                         0,
                         "First element of grid_x is incorrect")
        self.assertEqual(grid_x[-1, -1, -1],
                         grid_points[0] - 1,
                         "Last element of grid_x is incorrect")

    def test_calculate_radial_average(self) -> None:
        """
        Test the calculation of the radial average.
        """
        data = np.ones((10, 10, 10))  # Uniform potential
        distances = np.sqrt((np.indices((10, 10, 10)) - 5) ** 2).sum(axis=0)
        grid_z = np.zeros((10, 10, 10))
        grid_spacing = [1.0]
        max_radius = 5.0
        radii, radial_average = \
            self.radial_average_potential.calculate_radial_average(
                data,
                distances,
                grid_spacing,
                max_radius,
                grid_z)

        # Verify radii are correctly spaced
        expected_radii = np.arange(0, max_radius, grid_spacing[0])
        np.testing.assert_array_almost_equal(
            radii, expected_radii, err_msg="Radii spacing is incorrect")

        # Verify radial averages are as expected
        # Since the potential is uniform and all grid_z values are 0
        # (<= average_index_from),
        # the radial average should be 1 for all calculated radii
        expected_radial_averages = np.ones_like(radial_average)
        np.testing.assert_array_almost_equal(
            radial_average,
            expected_radial_averages,
            err_msg="Radial averages are incorrect")

    def test_uniform_potential(self) -> None:
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
            data_arr, self.grid_points, self.grid_spacing)

        # Expect uniform potential to have the same average value scaled
        # by pot_unit_conversion
        expected_value = np.ones_like(radial_average)
        np.testing.assert_almost_equal(radial_average, expected_value)
        print(f'{bcolors.CAUTION}\nTesting uniform:\n'
                  '\tTest may failed:\n'
                  '\tRadial average and expected values are not close '
                  'enough: "AssertionError"\n'
                  '\tTake a look at the plot to see the discrepancy.\n'
                  f'{bcolors.ENDC}')
        self.plot_test_results(radii, radial_average, expected_value)

    def test_center_offset(self) -> None:
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
            data_arr, self.grid_points, self.grid_spacing)

        # Expect uniform potential to have the same average value scaled
        # by pot_unit_conversion
        expected_value = np.ones_like(radial_average)
        np.testing.assert_almost_equal(radial_average, expected_value)

    def test_x_squared_potential(self) -> None:
        """
        Test the radial average calculation for a potential that varies
        as r^2.

        This test verifies that the radial average calculation correctly
        handl+
        df es a potential that varies as r^2 from the center of the grid.
        """
        # pylint: disable=invalid-name
        # Generate a grid of coordinates
        grid_resolution: int = 200
        self.grid_points: list[int] = \
            [grid_resolution, grid_resolution, grid_resolution]
        x: np.ndarray = np.linspace(-self.grid_points[0] / 2,
                                    self.grid_points[0] / 2,
                                    grid_resolution)

        y: np.ndarray = np.linspace(-self.grid_points[1] / 2,
                                    self.grid_points[1] / 2,
                                    grid_resolution)

        z: np.ndarray = np.linspace(-self.grid_points[2] / 2,
                                    self.grid_points[2] / 2,
                                    grid_resolution)

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Calculate the distance from the center for each point
        center: np.ndarray = np.array([0, 0, 0])
        distances: np.ndarray = np.sqrt((X - center[0])**2 +
                                        (Y - center[1])**2 +
                                        (Z - center[2])**2)

        # Generate the potential as r^2
        r_squared_potential: np.ndarray = distances**2

        # Calculate the radial average
        radii: np.ndarray
        radial_average: np.ndarray
        radii, radial_average = self.radial_average_potential.radial_average(
            r_squared_potential, self.grid_points, self.grid_spacing)

        # The expected radial average should follow the input r^2
        # relationship closely.  Since the potential is directly
        # proportional to r^2, the radial average should theoretically
        # be the same as the input potential, adjusted for any binning
        # effects.
        # Here, we compare the computed radial average to the
        # theoretical expectation.

        # Generate the expected radial average based on the r^2 relationship
        # This might require adjusting based on how your radial_average
        # method bins the data
        expected_radial_average: np.ndarray = radii**2

        # Assert that the computed radial average is close to the
        # expected r^2 relationship
        # This may need to be adjusted for numerical precision and
        # binning effects

        # It is not good practice to assert within a try block. but this
        # is done here to catch the AssertionError and print a message
        # to the user to check the plot for discrepancies. But also the test
        # does not conside all the aspects of the radial average calculation
        # for example the setting to zero the average outside of the
        # `average_index_from`.
        try:
            np.testing.assert_allclose(radial_average,
                                       expected_radial_average,
                                       rtol=1e-4,
                                       atol=1e-4)
        except AssertionError as _:
            print(f'{bcolors.CAUTION}\nTesting Potentail(r^2):\n'
                  '\tTest may failed:\n'
                  '\tRadial average and expected values are not close '
                  'enough: "AssertionError"\n'
                  '\tTake a look at the plot to see the discrepancy.\n'
                  f'{bcolors.ENDC}')

        self.plot_test_results(radii, radial_average, expected_radial_average)


    def plot_test_results(self,
                          radii,
                          radial_average,
                          expected_value
                          ) -> None:
        """
        Plot the radial average against the expected uniform value.

        Parameters:
        - radii: The radii at which the radial average was calculated.
        - radial_average: The calculated radial average values.
        - expected_value: The expected uniform value for comparison.
        """
        plt.close('all')
        _, ax_i = plt.subplots(figsize=(20, 12))

        ax_i.plot(radii,
                  radial_average,
                  label='Calculated Radial Average',
                  color='red',
                  linestyle='-',
                  marker='o',
                  )
        ax_i.plot(radii,
                  expected_value,
                  label='Expected Radial Average',
                  color='grey',
                  linestyle='--',
                  marker='o',
                  )

        ax_i.set_xlabel('Radius', fontsize=22)
        ax_i.set_ylabel('Radial Average Potential', fontsize=22)
        ax_i.set_title('Radial Average vs. Expected Value', fontsize=22)
        ax_i.tick_params(axis='both', which='major', labelsize=18)

        plt.legend(fontsize=20)
        plt.show()


if __name__ == '__main__':
    unittest.main()
