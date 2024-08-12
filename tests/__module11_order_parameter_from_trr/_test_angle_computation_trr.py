import unittest
import numpy as np
from unittest.mock import Mock
from src.module11_order_parameter_from_trr.angle_computation_trr import \
    AngleProjectionComputation

class TestGetHeadTailVector(unittest.TestCase):
    def setUp(self):
        # Create mock versions of the required arguments
        self.log = Mock()
        self.universe = Mock()
        self.configs = Mock()
        # Set up the residues_tails attribute on the mock configs object
        self.configs.residues_tails = 'residues_tails'

    def test_get_head_tail_vector(self):
        # Create an instance of the class with the mock arguments
        instance = AngleProjectionComputation(self.log, self.universe, self.configs)
        print("instance", instance)
        # Define test inputs
        tail_positions = np.array([1, 2, 3])
        head_positions = np.array([4, 5, 6])

        # Call the method with the test inputs
        result = instance.get_head_tail_vector(tail_positions, head_positions)

        # Check that the result is correct
        expected_result = np.array([3, 3, 3])
        np.testing.assert_array_equal(result, expected_result)

    def test_get_head_tail_vector_mismatched_shapes(self):
        # Create an instance of the class with the mock arguments
        instance = AngleProjectionComputation(self.log, self.universe, self.configs)

        # Define test inputs with mismatched shapes
        tail_positions = np.array([1, 2, 3])
        head_positions = np.array([4, 5])

        # Check that the method raises a ValueError when called with inputs of mismatched shapes
        with self.assertRaises(ValueError):
            instance.get_head_tail_vector(tail_positions, head_positions)

if __name__ == '__main__':
    unittest.main()