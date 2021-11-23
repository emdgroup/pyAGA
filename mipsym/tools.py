import numpy as np


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
