import numpy as np
from typing import Tuple


def deviation_value(norm: 'Norm', P: np.ndarray, A: np.ndarray):
    from mipsym.mip import Norm
    if norm == Norm.L_INFINITY:
        return np.max(np.abs(P @ A - A @ P))
    elif norm == Norm.L_1:
        return np.sum(np.abs(P @ A - A @ P))
    elif norm == Norm.L_2:
        return np.sum((P @ A - A @ P)**2)


def hash_array(arr: np.ndarray) -> Tuple[int]:
    # Create a tuple with the indices of nonzero entries in arr
    # This is unique for each permuation matrix and can be used
    # e.g. as a key in a dict
    return tuple(np.nonzero(arr.flatten())[0])


def to_matrix(trafo) -> np.ndarray:
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


def to_ndarray(v, m, n, dtype=int) -> np.ndarray:
    # Convert pyomo variable to numpy array
    # TODO: when used for a permutation matrix, this should really be some sparse matrix format
    result = np.zeros((m, n), dtype=float)
    for i in range(m):
        for j in range(n):
            result[i, j] = v[i, j].value

    if dtype == int:
        result = np.around(result)

    return result.astype(dtype)


def to_list(matrix):
    perm = []
    for row in matrix:
        entry = np.argwhere(row == 1)[0][0]
        perm.append(entry)
    return perm


def matshow(v: np.ndarray):
    # Print ASCII-art of the matrix
    lines = ''
    for row in v:
        lines += '|'
        for col in row:
            lines += ' ' if col == 0 else str(int(col))
        lines += '|\n'

    return lines


def matshow_pyomo(v, m, n):
    # Print ASCII-art of the matrix
    lines = ''
    for row in m:
        lines += '|'
        for col in n:
            if not v[row, col].fixed:
                lines += '?'
            else:
                val = v[row, col].value
                lines += ' ' if val == 0 else str(val)
        lines += '|\n'
    return lines
