import pickle
from enum import Enum
from itertools import count

from typing import Tuple

import numpy as np
import pyomo.environ as po

# This is not the best comment in the world.
# This is just a tribute. 
# Couldn't remember the greatest comment in the world, yeah - no!

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
            line += ' ' if col == 0 else '#'
        line += '|'
        print(line)


def hash_array(arr: np.ndarray) -> Tuple[int]:
    # Create a tuple with the indices of nonzero entries in arr
    # This is unique for each permuation matrix and can be used
    # e.g. as a key in a dict
    return tuple(np.nonzero(arr.flatten())[0])


def create_permutation_combination_constraints(m, all_permutations, level, already_added, current_prod, previous_choice=-1, name='id'):
    # TODO: should focus on adding short products first (might need conversion of the recursion to iteration),
    # so that skipping of already added transformations happens more early. This should allow to omit the bound
    # on level completely
    repr = hash_array(current_prod)
    assert len(repr) == current_prod.shape[0]
    assert current_prod.shape[0] == current_prod.shape[1]

    try:
        print(f'Skipping [{name}] == [{already_added[repr]}]')
    except KeyError:
        print(f'Adding Constraint for [{name}]')
        m.knownPermutations.add(expr=sum(m.P[tuple(ij)] for ij in np.argwhere(current_prod)) <= current_prod.shape[0] - 1)
        already_added[repr] = name

        if level >= 0:
            for ip, permutation in enumerate(all_permutations):
                if ip == previous_choice:
                    continue
                for p, power in enumerate(permutation):
                    create_permutation_combination_constraints(
                        m=m,
                        all_permutations=all_permutations,
                        level=level-1,
                        already_added=already_added,
                        current_prod=current_prod@power,
                        previous_choice=ip,
                        name=f'{name} P_{ip}^{p+1}',
                    )


def find_permutations(A: np.ndarray, norm: Norm):

    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]
    assert isinstance(norm, Norm)

    n_nodes = A.shape[0]

    model = po.ConcreteModel()
    model.N = po.Set(initialize=range(n_nodes))

    if norm == Norm.L2:
        print('Creating Solver using MindtPy')
        solver = po.SolverFactory('mindtpy')
    else:
        print('Creating Solver using glpk')
        solver = po.SolverFactory('glpk')

    print('Creating Parameter for Concurrence Matrix')
    model.A = po.Param(model.N, model.N, initialize=A, within=po.Any)

    print('Creating Boolean Permutation Matrix')
    model.P = po.Var(model.N, model.N, within=po.Boolean)

    print('Creating Row Sum Constraint for the Permutation Matrix')
    model.rowSum = po.Constraint(model.N, rule=lambda m, i: 1 == sum(m.P[i, j] for j in m.N))
    print('Creating Column Sum Constraint for the Permutation Matrix')
    model.colSum = po.Constraint(model.N, rule=lambda m, j: 1 == sum(m.P[i, j] for i in m.N))

    def deviation(m, i, j):
        return sum(m.P[i, k]*A[k, j] for k in m.N) - sum(A[i, k]*m.P[k, j] for k in m.N)

    if norm == Norm.L0:
        print('Creating Upper Limit Variable for the Maximum Error')
        model.T = po.Var(within=po.NonNegativeReals)

        print('Creating Objective Function to Minimize L0 Norm of Deviation')
        model.objective = po.Objective(expr=model.T, sense=po.minimize)

        print('Creating Constraint to Limit Positive Deviation')
        model.posDev = po.Constraint(model.N, model.N, rule=lambda m, i, j: deviation(m, i, j) <= m.T)
        print('Creating Constraint to Limit Negative Deviation')
        model.negDev = po.Constraint(model.N, model.N, rule=lambda m, i, j: -m.T <= deviation(m, i, j))
    elif norm == Norm.L1:
        print('Creating Upper Limit Variables for the Pointwise Error')
        model.T = po.Var(model.N, model.N, within=po.NonNegativeReals)

        print('Creating Objective Function to Minimize L1 Norm of Deviation')
        model.objective = po.Objective(expr=sum(model.T[i, j] for j in model.N for i in model.N), sense=po.minimize)

        print('Creating Constraint to Limit Positive Deviation')
        model.posDev = po.Constraint(model.N, model.N, rule=lambda m, i, j: deviation(m, i, j) <= m.T[i, j])
        print('Creating Constraint to Limit Negative Deviation')
        model.negDev = po.Constraint(model.N, model.N, rule=lambda m, i, j: -m.T[i, j] <= deviation(m, i, j))
    elif norm == Norm.L2:
        print('Creating Objective Function to Minimize L2 Norm of Deviation')
        model.objective = po.Objective(rule=lambda m: sum(deviation(m, i, j)**2 for j in m.N for i in m.N), sense=po.minimize)
    else:
        raise ValueError(f'Unsupported Norm {norm}.')

    model.knownPermutations = po.ConstraintList()

    id = np.eye(n_nodes, dtype=int)

    # This will be a list of lists, with the inner lists
    # containing all powers of an identified permutation
    all_permutations = []

    for i_result in count(0):
        if norm == Norm.L2:
            print('Solving using glpk and ipopt')
            results = solver.solve(model, mip_solver='glpk', nlp_solver='ipopt')
        else:
            print('Solving using glpk')
            results = solver.solve(model, tee=True)

        print(results)

        permutation = to_ndarray(model.P, n_nodes, n_nodes)
        print(f'P_{i_result} =')
        matshow(permutation)

        print(f'Computing Cycle of Solution P_{i_result}')
        all_powers = []
        power = permutation.copy()
        while True:
            all_powers.append(power)
            power = power@permutation
            if np.allclose(power, id):
                break

        print(f'Cycle Length is {len(all_powers) + 1}')
        all_permutations.append(all_powers)

        print('Creating Constraints to Exclude Known Solutions, their Powers and their Combinatorial Products')
        model.del_component(model.knownPermutations)
        model.del_component(model.knownPermutations_index)
        model.knownPermutations = po.ConstraintList()
        create_permutation_combination_constraints(
            m=model,
            all_permutations=all_permutations,
            level=5,
            already_added=dict(),
            current_prod=id,
        )
        print(f'Created {len(model.knownPermutations)} constraints.')


if __name__ == '__main__':
    with open('data/one_letter_words_5x5_concurrence_matrix_100.pickle', 'rb') as f:
        A = pickle.load(f)

    find_permutations(A=A, norm=Norm.L1)
