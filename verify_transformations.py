import pickle
from typing import List, Union

import numpy as np


def verify_transformation(
    t_candidate: List[Union[int, None]],
    test_transformations: List[List[Union[int, None]]] = None,
    casename: str = None,
) -> bool:
    """
    Verify whether a single transformation candidate corresponds to a valid
    transformation from a given set of full transformations. If the transformation
    candidate is partial, meaning that it contains None-entries, then it is considered
    valid if it corresponds to a valid transformation in all entries which are not None.
    The function can be called by passing test_transformations or by using the casename
    parameter, in which case it will load the set of transformations from the disk.
    :param t_candidate: The transformation to verify.
    :param test_transformations: The given set of valid transformations.
    :param casename: The name of the testcase. The function will look for the pickled
    transformations in the file "data/test_transformations_{casename}.pickle".

    :return: Whether or not the given transformation candidate corresponds to any
    transformation in test
    """
    if casename is not None:
        with open(f"data/test_transformations_{casename}.pickle", "rb") as file:
            test_transformations = pickle.load(file)
    entries = len([elem for elem in t_candidate if elem is not None])
    for transformation in test_transformations:
        same_entries = 0
        for index, value in enumerate(t_candidate):
            if value is not None and transformation[index] == value:
                same_entries += 1
        if same_entries == entries:
            return True
    return False


def to_matrix(trafo: List[Union[int, None]]) -> np.ndarray:
    """
    Turn a given list format permutation into a matrix.
    :param trafo: The permutation in the list format.
    :return: The permutation in the matrix format (as a np array).
    """
    matrix = np.zeros((len(trafo), len(trafo)))
    for index, value in enumerate(trafo):
        if value is not None:
            matrix[index, value] = 1
        else:
            matrix[index, :] = 0  # Setting this to 0 is unnecessary, but I will leave
            # this line in case one wants to set it to some other value for
            # visualization.
    return matrix
