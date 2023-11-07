"""
Module: get_data

A data processing module for splitting and organizing data based on
residue types.

This module defines a class 'GetCom' that reads input data from a
pickle file and splits it based on the residue types indicated by the
last row indices. Each split data is organized and stored separately
for further analysis.

Attributes:
    None

Classes:
    GetCom: A class for data processing and organization based on
    residue types.

Usage:
    from get_data import GetCom

    # Create an instance of GetCom
    data_processor = GetCom()

    # Access split data arrays organized by residue types
    split_data = data_processor.split_arr_dict

    # Get numerical information about the data
    numbers_info = data_processor.nr_dict

    # Get the dimensions of the box
    box_dimensions = data_processor.box_dims

    # Access specific methods of the GetCom class for data processing

"""

import pickle
import typing
import numpy as np
import common.static_info as stinfo


class GetCom:
    """
    A data processing class for splitting and organizing data based on
    residue types.

    This class reads input data from a pickle file and splits it based
    on the residue types indicated by the last row indices. Each split
    data is organized and stored separately for further analysis.

    Attributes:
        f_name (str): The filename of the input pickle file.
        split_arr_dict (dict[str, np.ndarray]): A dictionary containing
            split data arrays organized by residue types.
        nr_dict (dict[str, int]): A dictionary containing numerical
            information about the data, including the number of time
            frames and the number of residues for each type.
        box_dims (dict[str, float]): A dictionary containing the
            the dimensions of the box

    Methods:
        __init__(): Initialize the GetCom instance.
        initiate_data(): Read and split the data from the pickle file.
        get_numbers(data: dict[str, np.ndarray]) -> dict[str, int]:
            Get the number of time frames and the number of residues
            for each type.
        split_data(data: np.ndarray) -> dict[str, np.ndarray]:
            Split data based on residue types.
        find_key_by_value(dictionary: dict[typing.Any, typing.Any], \
            target_value: typing.Any) -> typing.Any:
            Find a key in a dictionary based on a target value.
        load_pickle() -> np.ndarray: Load data from the input pickle
        file.
    """

    def __init__(self,
                 fname: str = stinfo.files['com_pickle']
                 ) -> None:
        """
        Initialize the GetCom instance.

        The filename of the input pickle file is set based on the
        stinfo module.
        The data is read, split, and relevant numbers are calculated
        during initialization.
        """
        self.f_name: str = fname
        self.split_arr_dict: dict[str, np.ndarray] = self._initiate_data()
        self.nr_dict: dict[str, int] = self.get_numbers(self.split_arr_dict)
        self.box_dims: dict[str, float] = self.get_box_dimensions()

    def _initiate_data(self) -> dict[str, np.ndarray]:
        """
        Read and split the data from the pickle file.

        The data is loaded from the pickle file and split based on
        residue types.
        The split data is printed as a dictionary of residue names and
        corresponding arrays.
        """
        com_arr: np.ndarray = self.load_pickle()
        split_arr_dict: dict[str, np.ndarray] = self.split_data(com_arr[:, 4:])
        split_arr_dict['APT_COR'] = com_arr[:-2, 1:4]
        return split_arr_dict

    def get_box_dimensions(self) -> dict[str, float]:
        """
        Calculate the maximum and minimum values for x, y, and z axes
        of each residue.

        Returns:
            dict[str, np.float64]: A dictionary where keys are axis
            names ('xlo', 'xhi', etc.)
            and values are the corresponding minimum and maximum values.
        """
        box_dims: dict[str, float] = {}
        box_residues = ['SOL', 'D10']
        axis_names = ['x', 'y', 'z']

        # Initialize dictionaries for min and max values of each axis
        min_values = {axis: np.inf for axis in axis_names}
        max_values = {axis: -np.inf for axis in axis_names}

        for res in box_residues:
            arr = self.split_arr_dict[res]

            # Iterate through each axis (x, y, z)
            for axis_idx, axis in enumerate(axis_names):
                axis_values = arr[:-2, axis_idx::3]
                axis_min = np.min(axis_values)
                axis_max = np.max(axis_values)

                # Update min and max values for the axis
                min_values[axis] = min(min_values[axis], axis_min)
                max_values[axis] = max(max_values[axis], axis_max)
        box_dims = {
            f'{axis}_lo': min_values[axis] for axis in axis_names
        }

        box_dims.update({
            f'{axis}_hi': max_values[axis] for axis in axis_names
        })

        return box_dims

    @staticmethod
    def get_numbers(data: dict[str, np.ndarray]  # Splitted np.arrays
                    ) -> dict[str, int]:
        """
        Get the number of time frames and the number of residues for
        each type.

        Args:
            data (dict[str, np.ndarray]): The split data dictionary.

        Returns:
            dict[str, int]: A dictionary containing the number of time
            frames and the number of residues for each type.
        """
        nr_dict: dict[str, int] = {}
        nr_dict['nr_frames'] = np.shape(data['SOL'])[0] - 2
        for item, arr in data.items():
            nr_dict[item] = np.shape(arr)[1] // 3
        return nr_dict

    def split_data(self,
                   data: np.ndarray  # Loaded data without first 4 columns
                   ) -> dict[str, np.ndarray]:
        """
        Split data based on the type of the residues.

        Args:
            data (np.ndarray): The data to be split, excluding the
            first 4 columns.

        Returns:
            dict[str, np.ndarray]: A dictionary of residue names and
            their associated arrays.
        """
        # Get the last row of the array
        last_row: np.ndarray = data[-1]
        last_row_indices: np.ndarray = last_row.astype(int)
        unique_indices: np.ndarray = np.unique(last_row_indices)

        # Create an empty dictionary to store the split arrays
        result_dict: dict[int, list] = {index: [] for index in unique_indices}

        # Iterate through each column and split based on the indices
        for col_idx, column in enumerate(data.T):
            result_dict[last_row_indices[col_idx]].append(column)

        # Convert the dictionary values back to numpy arrays
        result: list[np.ndarray] = \
            [np.array(arr_list).T for arr_list in result_dict.values()]

        array_dict: dict[str, np.ndarray] = {}
        for i, arr in enumerate(result):
            residue_name = self.find_key_by_value(stinfo.reidues_id, i)
            array_dict[residue_name] = arr
        return array_dict

    @staticmethod
    def find_key_by_value(dictionary: dict[typing.Any, typing.Any],
                          target_value: typing.Any
                          ) -> typing.Any:
        """
        Find the key in the dictionary based on the target value.

        Args:
            dictionary (dict): The dictionary to search.
            target_value: The value to search for.

        Returns:
            str or None: The key associated with the target value, or
            None if not found.
        """
        return next((key for key, value in dictionary.items() if
                     value == target_value), None)

    def load_pickle(self) -> np.ndarray:
        """loading the input file"""
        with open(self.f_name, 'rb') as f_rb:
            com_arr = pickle.load(f_rb)
        return com_arr


if __name__ == '__main__':
    GetCom()
