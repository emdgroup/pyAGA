import pickle

import numpy as np


def verify_transformations(t_candidates, casename):
    with open(f"data/test_transformations_{casename}.pickle", "rb") as file:
        test_transformations = pickle.load(file)
    num_valid_transformations = 0
    for t_candidate in t_candidates:
        if verify_one_transformation(
                t_candidate,
                test_transformations=test_transformations
        ):
            num_valid_transformations += 1
    return num_valid_transformations


def verify_one_transformation(t_candidate, test_transformations=None, casename=None):
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


def to_matrix(trafo):
    """
    Turn a given list format permutation into a matrix.
    :param trafo:
    :return:
    """
    matrix = np.zeros((len(trafo), len(trafo)))
    for index, value in enumerate(trafo):
        if value is not None:
            matrix[index, value] = 1
        else:
            matrix[index, :] = 0 # Setting this to 0 is unnecessary, but I will leave
            # this line in case one wants to set it to some other value for
            # visualization.
    return matrix


def to_list(matrix):
    """
    Turn a given permutation matrix into the list format.
    :param matrix:
    :return:
    """
    perm = []
    for row in matrix:
        entry = np.argwhere(row == 1)[0][0]
        perm.append(entry)
    return perm


def matshow(v: np.ndarray):
    # Print ASCII-art of the matrix
    for row in v:
        line = '|'
        for col in row:
            line += ' ' if col == 0 else str(int(col))
        line += '|'
        print(line)

def delete_rows_and_cols(arr, ind):
    arr_1 = np.delete(arr, ind, axis=0)
    return np.delete(arr_1, ind, axis=1)
