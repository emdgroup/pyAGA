from typing import List
import logging
import time
from functools import lru_cache

import numpy as np
import pyomo.environ as po
from pyomo.core.util import quicksum

from mipsym.mip import Norm


logger = logging.getLogger("integer_programming")
logger.setLevel(
    level=logging.DEBUG
)  # set to logging.INFO for less, to logging.DEBUG for more verbosity


def create_reduced_mip_model(
    norm: Norm,
    A_l: np.ndarray,
    A_r: np.ndarray,
    col_index_map: List[int],
    row_index_map: List[int],
):
    """
    Create a reduced MIP model that only contains variables that still need to be fixed
    :param norm: The norm that shoiuld be used for the optimization
    :param A_l: Adjacency matrix with permutation multiplied from left
    :param A_r: Adjacency matrix with permutation multiplied from right
    :param col_index: List s.t. the i-th entry will be the index of the i-th row of A_row w.r.t. the original matrix
    :param row_index: List s.t. the i-th entry will be the index of the i-th column of A_col w.r.t. the original matrix
    :param permutation: The incomplete permutation
    :return: The model
    """
    assert isinstance(norm, Norm)
    assert A_l.ndim == 2

    A_row_l = A_l[row_index_map, :]
    A_row_r = A_r[row_index_map, :]
    A_col_l = A_l[:, col_index_map]
    A_col_r = A_r[:, col_index_map]

    n_unknowns = A_row_l.shape[0]
    n_nodes = A_row_l.shape[1]
    assert n_unknowns == A_col_l.shape[1]
    assert n_nodes == A_col_l.shape[0]
    assert len(col_index_map) == n_unknowns
    assert len(row_index_map) == n_unknowns

    start_time = time.time()

    model = po.ConcreteModel()
    model.U = po.Set(initialize=range(n_unknowns))
    model.N = po.Set(initialize=range(n_nodes))

    logger.debug("Creating Parameter for Concurrence Matrices")
    model.A_row_l = po.Param(
        model.U,
        model.N,
        initialize=lambda m, i, j: A_row_l[i, j],
        within=po.NonNegativeReals,
    )
    model.A_col_l = po.Param(
        model.N,
        model.U,
        initialize=lambda m, i, j: A_col_l[i, j],
        within=po.NonNegativeReals,
    )
    model.A_row_r = po.Param(
        model.U,
        model.N,
        initialize=lambda m, i, j: A_row_r[i, j],
        within=po.NonNegativeReals,
    )
    model.A_col_r = po.Param(
        model.N,
        model.U,
        initialize=lambda m, i, j: A_col_r[i, j],
        within=po.NonNegativeReals,
    )

    logger.debug("Creating Boolean Permutation Matrix")
    model.P = po.Var(model.U, model.U, within=po.Boolean)

    logger.debug("Creating Row Sum Constraint for the Permutation Matrix")
    model.rowSum = po.Constraint(
        model.U, rule=lambda m, i: 1 == sum(m.P[i, j] for j in m.U)
    )
    logger.debug("Creating Column Sum Constraint for the Permutation Matrix")
    model.colSum = po.Constraint(
        model.U, rule=lambda m, j: 1 == sum(m.P[i, j] for i in m.U)
    )

    @lru_cache(maxsize=None)
    def get(mat, row, col):
        return mat[row, col]

    @lru_cache(maxsize=None)
    def deviation_A_row(m, i, j):
        """
        Create entry (i,j) of the expression "P @ A_row-A_row @ P"
        As neither P nor A_row are square, this is a bit involved and requires some index juggling
        :param norm: The norm that shoiuld be used for the optimization
        :param m: The model
        :param i: Row index
        :param j: Column index
        """
        # A_row =
        #  | 11 12 13 14 15 |
        #  | 21 22 23 24 25 |
        #  | 31 32 33 34 35 |

        # P =
        # | 0 1 0 |
        # | 0 0 1 |
        # | 1 0 0 |

        # col_index_map = [2, 3, 5]

        # A_row @ P =
        #  | 11 15 12 14 13 |
        #  | 21 25 22 24 23 |
        #  | 31 35 32 34 33 |

        #  (P @ A_row_l - A_row_r @ P)_ij       i in m.U, j in m.N
        P_times_A_ij = quicksum(get(m.P, i, k) * get(m.A_row_l, k, j) for k in m.U)
        A_times_P_ij = (
            get(m.A_row_r, i, j)
            if j not in col_index_map
            else quicksum(
                get(m.A_row_r, i, col_index_map[k]) * get(m.P, k, col_index_map.index(j))
                for k in m.U
            )
        )

        return P_times_A_ij - A_times_P_ij

    @lru_cache(maxsize=None)
    def deviation_A_col(m, i, j):
        """
        Create entry (i,j) of the expression "P @ A_col-A_col @ P"
        As neither P nor A_col are square, this is a bit involved and requires some index juggling
        :param norm: The norm that should be used for the optimization
        :param m: The model
        :param i: Row index
        :param j: Column index
        """
        #  (P @ A_col_l - A_col_r @ P)_ij       i in m.N, j in m.U
        P_times_A_ij = (
            get(m.A_col_l, i, j)
            if i not in row_index_map
            else quicksum(
                get(m.P, row_index_map.index(i), k) * get(m.A_col_l, row_index_map[k], j)
                for k in m.U
            )
        )
        A_times_P_ij = quicksum(get(m.A_col_r, i, k) * get(m.P, k, j) for k in m.U)

        return P_times_A_ij - A_times_P_ij

    if norm == Norm.L_INFINITY:
        logger.debug("Creating Upper Limit Variable for the Maximum Error")
        model.T = po.Var(within=po.NonNegativeReals)

        logger.debug(
            "Creating Objective Function to Minimize L_INFINITY Norm of Deviation"
        )
        model.objective = po.Objective(expr=model.T, sense=po.minimize)

        logger.debug("Creating Constraints to Limit Positive Deviation")
        model.posDev_row = po.Constraint(
            model.U, model.N, rule=lambda m, i, j: deviation_A_row(m, i, j) <= m.T
        )
        model.posDev_col = po.Constraint(
            model.N, model.U, rule=lambda m, i, j: deviation_A_col(m, i, j) <= m.T
        )
        logger.debug("Creating Constraints to Limit Negative Deviation")
        model.negDev_row = po.Constraint(
            model.U, model.N, rule=lambda m, i, j: -m.T <= deviation_A_row(m, i, j)
        )
        model.negDev_col = po.Constraint(
            model.N, model.U, rule=lambda m, i, j: -m.T <= deviation_A_col(m, i, j)
        )
    elif norm == Norm.L_1:
        logger.debug("Creating Upper Limit Variable for the Maximum Error")
        model.T = po.Var(model.U, model.N, within=po.NonNegativeReals)

        logger.debug(
            "Creating Objective Function to Minimize L_INFINITY Norm of Deviation"
        )
        model.objective = po.Objective(
            expr=sum(model.T[i, j] for i in model.U for j in model.N), sense=po.minimize
        )

        logger.debug("Creating Constraints to Limit Positive Deviation")
        model.posDev_row = po.Constraint(
            model.U, model.N, rule=lambda m, i, j: deviation_A_row(m, i, j) <= m.T[i, j]
        )
        model.posDev_col = po.Constraint(
            model.N, model.U, rule=lambda m, i, j: deviation_A_col(m, i, j) <= m.T[j, i]
        )
        logger.debug("Creating Constraints to Limit Negative Deviation")
        model.negDev_row = po.Constraint(
            model.U,
            model.N,
            rule=lambda m, i, j: -m.T[i, j] <= deviation_A_row(m, i, j),
        )
        model.negDev_col = po.Constraint(
            model.N,
            model.U,
            rule=lambda m, i, j: -m.T[j, i] <= deviation_A_col(m, i, j),
        )
    else:
        raise ValueError(f"Unsupported Norm {norm}.")

    logger.debug(f"Finished creation of model in {time.time()-start_time:.2f} seconds.")

    return model
