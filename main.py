import pickle
from enum import Enum
from collections import deque
from itertools import count
import logging

from typing import Tuple

import numpy as np
import pyomo.environ as po
import time

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.DEBUG)  # set to logging.INFO vor less verbosity


class Norm(Enum):
    L0 = 0
    L1 = 1
    L2 = 2


def to_ndarray(v, m, n, dtype=int) -> np.ndarray:
    # Convert pyomo variable to numpy array
    # TODO: when used for a permutation matrix, this should really be some sparse matrix format
    result = np.zeros((m, n), dtype=dtype)
    for i in range(m):
        for j in range(n):
            result[i, j] = v[i, j].value
    return result


def matshow(v: np.ndarray):
    # Print ASCII-art of the matrix
    for row in v:
        line = '|'
        for col in row:
            line += ' ' if col == 0 else str(int(col))
        line += '|'
        logger.info(line)


def hash_array(arr: np.ndarray) -> Tuple[int]:
    # Create a tuple with the indices of nonzero entries in arr
    # This is unique for each permuation matrix and can be used
    # e.g. as a key in a dict
    return tuple(np.nonzero(arr.flatten())[0])


# TODO: Calculate level properly (lcm of cycle lengths or something similar?)
def create_permutation_combination_constraints(m, all_permutations, id):
    already_added = dict()
    todo_list = deque()
    todo_list.append((id, -1, 'id'))

    while len(todo_list) > 0:
        current_prod, previous_choice, name = todo_list.popleft()

        repr = hash_array(current_prod)
        assert len(repr) == current_prod.shape[0]
        assert current_prod.shape[0] == current_prod.shape[1]

        try:
            logger.info(f'Skipping [{name}] == [{already_added[repr]}]')
        except KeyError:
            logger.info(f'Adding Constraint for [{name}]')
            m.knownPermutations.add(expr=sum(m.P[tuple(ij)] for ij in np.argwhere(current_prod)) <= current_prod.shape[0] - 1)
            already_added[repr] = name

            for ip, permutation in enumerate(all_permutations):
                # prevent multiplying the same permutation that has just been used before
                if ip != previous_choice:
                    for p, power in enumerate(permutation):
                        todo_list.append((current_prod@power, ip, f'{name} P_{ip}^{p+1}'))


def find_permutations(A: np.ndarray, norm: Norm, objective_bound=100, glpk_time_limit=None):

    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]
    assert isinstance(norm, Norm)

    n_nodes = A.shape[0]

    start_time = time.time()

    model = po.ConcreteModel()
    model.N = po.Set(initialize=range(n_nodes))

    if norm == Norm.L2:
        logger.debug('Creating Solver using MindtPy')
        solver = po.SolverFactory('mindtpy')
    else:
        logger.debug('Creating Solver using glpk')
        solver = po.SolverFactory('glpk')
        # Set option for time limit and instruct to use a heuristic
        if glpk_time_limit is not None:
            solver.options['tmlim'] = glpk_time_limit
            solver.options['fpump'] = ''

    logger.debug('Creating Parameter for Concurrence Matrix')
    model.A = po.Param(model.N, model.N, initialize=A, within=po.Any)

    logger.debug('Creating Boolean Permutation Matrix')
    model.P = po.Var(model.N, model.N, within=po.Boolean)

    logger.debug('Creating Row Sum Constraint for the Permutation Matrix')
    model.rowSum = po.Constraint(model.N, rule=lambda m, i: 1 == sum(m.P[i, j] for j in m.N))
    logger.debug('Creating Column Sum Constraint for the Permutation Matrix')
    model.colSum = po.Constraint(model.N, rule=lambda m, j: 1 == sum(m.P[i, j] for i in m.N))

    logger.debug('Creating Constraint to Exclude Identity')
    model.identityConstraint = po.Constraint(expr=sum(model.P[i, i] for i in range(n_nodes)) <= n_nodes - 1)

    def deviation(m, i, j):
        return sum(m.P[i, k]*A[k, j] for k in m.N) - sum(A[i, k]*m.P[k, j] for k in m.N)

    if norm == Norm.L0:
        logger.debug('Creating Upper Limit Variable for the Maximum Error')
        model.T = po.Var(within=po.NonNegativeReals)

        logger.debug('Creating Objective Function to Minimize L0 Norm of Deviation')
        model.objective = po.Objective(expr=model.T, sense=po.minimize)

        logger.debug('Creating Constraint to Limit Positive Deviation')
        model.posDev = po.Constraint(model.N, model.N, rule=lambda m, i, j: deviation(m, i, j) <= m.T)
        logger.debug('Creating Constraint to Limit Negative Deviation')
        model.negDev = po.Constraint(model.N, model.N, rule=lambda m, i, j: -m.T <= deviation(m, i, j))
    elif norm == Norm.L1:
        logger.debug('Creating Upper Limit Variables for the Pointwise Error')
        model.T = po.Var(model.N, model.N, within=po.NonNegativeReals)

        logger.debug('Creating Objective Function to Minimize L1 Norm of Deviation')
        model.objective = po.Objective(expr=sum(model.T[i, j] for j in model.N for i in model.N), sense=po.minimize)

        logger.debug('Creating Constraint to Limit Positive Deviation')
        model.posDev = po.Constraint(model.N, model.N, rule=lambda m, i, j: deviation(m, i, j) <= m.T[i, j])
        logger.debug('Creating Constraint to Limit Negative Deviation')
        model.negDev = po.Constraint(model.N, model.N, rule=lambda m, i, j: -m.T[i, j] <= deviation(m, i, j))
    elif norm == Norm.L2:
        logger.debug('Creating Objective Function to Minimize L2 Norm of Deviation')
        model.objective = po.Objective(rule=lambda m: sum(deviation(m, i, j)**2 for j in m.N for i in m.N), sense=po.minimize)
    else:
        raise ValueError(f'Unsupported Norm {norm}.')

    logger.debug(f'Finished creation of model in {time.time()-start_time:.2f} seconds.')

    model.knownPermutations = po.ConstraintList()

    id = np.eye(n_nodes, dtype=int)

    # This will be a list of lists, with the inner lists
    # containing all powers of an identified permutation
    all_permutations = []

    # Last found objective function value
    # Used such that calculation stops once sufficiently bad solution was found
    last_objective = -1
    for i_result in count(0):
        iteration_start = time.time()

        if norm == Norm.L2:
            logger.debug('Solving using glpk and ipopt')
            results = solver.solve(model, mip_solver='glpk', nlp_solver='ipopt')
        else:
            logger.debug('Solving using glpk')
            results = solver.solve(model, tee=logger.getEffectiveLevel() == logging.DEBUG)

        logger.info('Solver Result:\n' + str(results))
        last_objective = model.objective.expr()
        if last_objective >= objective_bound:
            logger.warning('Current objective value exceeds the given bound; calculation is stopped')
            break

        permutation = to_ndarray(model.P, n_nodes, n_nodes)
        logger.info(f'P_{i_result} =')
        matshow(permutation)

        logger.debug(f'Computing Cycle of Solution P_{i_result}')
        all_powers = []
        power = permutation.copy()
        while True:
            all_powers.append(power)
            power = power@permutation
            if np.allclose(power, id):
                break

        logger.info(f'Cycle Length is {len(all_powers) + 1}')
        all_permutations.append(all_powers)

        logger.debug('Creating Constraints to Exclude Known Solutions, their Powers and their Combinatorial Products')
        model.del_component(model.knownPermutations)
        model.del_component(model.knownPermutations_index)
        model.knownPermutations = po.ConstraintList()
        create_permutation_combination_constraints(m=model, all_permutations=all_permutations, id=id)
        logger.info(f'Created {len(model.knownPermutations)} constraints.')

        logger.debug(f'Iteration {i_result+1} finished in {time.time()-iteration_start:.2f} seconds.')


if __name__ == '__main__':
    filename = 'data/one_letter_words_10x5_concurrence_matrix_100.pickle'
    logger.info(f'Loading file {filename}')
    with open(filename, 'rb') as f:
        A = pickle.load(f)

    find_permutations(A=A, norm=Norm.L1, objective_bound=0.01, glpk_time_limit=None)
