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
    for iRow in range(0, len(v), 2):
        rows = v[iRow:iRow+2]

        if len(rows) < 2:
            rows = np.vstack((rows, np.zeros(len(rows.T))))

        lines += '|'
        for col in rows.T:
            if col[0] == 0 and col[1] == 0:
                lines += ' '
            elif col[0] == 1 and col[1] == 0:
                lines += '▀'
            elif col[0] == 0 and col[1] == 1:
                lines += '▄'
            elif col[0] == 1 and col[1] == 1:
                lines += '█'
            else:
                assert False
        lines += '|\n'

    return lines
