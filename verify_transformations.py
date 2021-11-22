import pickle
from typing import List, Union

import numpy as np


def verify_transformations(
    t_candidates: List[List[Union[int, None]]], casename: str
) -> int:
    """Return number of valid transformations within a list of transformations for a
    given case.
    :param t_candidates: The list of candidates.
    :param casename: The name of the testcase.
    :return: The number of valid transformations within the given transformation list.
    """
    with open(f"data/test_transformations_{casename}.pickle", "rb") as file:
        test_transformations = pickle.load(file)
    num_valid_transformations = 0
    for t_candidate in t_candidates:
        if verify_one_transformation(
            t_candidate, test_transformations=test_transformations
        ):
            num_valid_transformations += 1
    return num_valid_transformations


def verify_one_transformation(
    t_candidate: List[Union[int, None]],
    test_transformations: List[List[Union[int, None]]] = None,
    casename: str = None,
) -> bool:
    """
    Verify whether a single transformation candidate corresponds to a valid
    transformation (calculated from the full data set). If the transformation candidate
    is partial, meaning that it contains None-entries, then it is considered valid if it
    corresponds to a valid transformation in all entries which are not None.
    :param t_candidate: The transformation to verify.

    The function can be called by passing test_transformations or by using the casename
    parameter, in which case it will load the set of
    :param test_transformations: The given
    :param casename:
    :return:
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


def to_list(matrix):
    """
    Turn a given permutation matrix into the list format.
    :param matrix: The permutation in matrix format.
    :return: The permutation in list format.
    """
    perm = []
    for row in matrix:
        entry = np.argwhere(row == 1)[0][0]
        perm.append(entry)
    return perm


def matshow(v: np.ndarray):
    # Print ASCII-art of the matrix
    for row in v:
        line = "|"
        for col in row:
            line += " " if col == 0 else str(int(col))
        line += "|"
        print(line)
