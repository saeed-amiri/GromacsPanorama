import unittest
from unittest.mock import Mock, patch, PropertyMock
import numpy as np
from src.module11_order_parameter_from_trr.atom_selection_trr import \
    AtomSelection

class TestAtomSelection(unittest.TestCase):
    @patch('MDAnalysis.core.universe.Universe')
    @patch('src.module11_order_parameter_from_trr.config_classes_trr.AllConfig')
    def setUp(self, mock_universe, mock_all_config):
        # Create a mock Universe object
        self.universe = mock_universe

        # Mock the atoms property of the Universe object
        atoms = Mock()
        type(self.universe).atoms = PropertyMock(return_value=atoms)

        # Mock the positions property of the AtomGroup object
        type(atoms).positions = \
            PropertyMock(return_value=np.array([[1, 2, 3], [4, 5, 6]]))

        # Mock the residues property of the Universe object
        residues = Mock()
        type(self.universe).residues = PropertyMock(return_value=residues)

        # Mock the segments property of the Universe object
        segments = Mock()
        type(self.universe).segments = PropertyMock(return_value=segments)

        # Mock the filename property of the Universe object
        type(self.universe).filename = \
            PropertyMock(return_value='../data/center_npt.tpr')

        # Create a mock AllConfig object
        self.configs = mock_all_config

        # Mock the residues_tails property of the AllConfig object
        type(self.configs).residues_tails = PropertyMock(return_value='your_value')

        # Create a mock Logger object
        self.log = Mock()

    def test_init(self):
        atom_selection = AtomSelection(self.universe, self.configs, self.log)
        # Add your assertions here

if __name__ == '__main__':
    unittest.main()