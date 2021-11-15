import pickle
from itertools import count

import numpy as np
import pyomo.environ as po


def to_ndarray(v, m, n, dtype=int):
    result = np.zeros((m, n), dtype=dtype)
    for i in range(m):
        for j in range(n):
            result[i, j] = v[i, j].value
    return result


def matshow(v):
    for row in v:
        line = '|'
        for col in row:
            line += ' ' if col == 0 else '#'
        line += '|'
        print(line)


def hash_array(arr):
    return tuple(np.nonzero(arr.flatten())[0])


def create_permutation_combination_constraints(m, id, n_nodes, all_permutations, level, already_added, fullprod=id, last_choice=-1, fullstr='id'):
    repr = hash_array(fullprod)
    assert len(repr) == n_nodes

    try:
        print(f'Skipping [{fullstr}] == [{already_added[repr]}]')
    except KeyError:
        print(f'Adding Constraint for [{fullstr}]')
        m.knownPermutations.add(expr=sum(m.P[tuple(ij)] for ij in np.argwhere(fullprod)) <= n_nodes - 1)
        already_added[repr] = fullstr

        if level >= 0:
            for ip, permutation in enumerate(all_permutations):
                if ip == last_choice:
                    continue
                for p, power in enumerate(permutation):
                    create_permutation_combination_constraints(
                        m=m,
                        id=id,
                        n_nodes=n_nodes,
                        all_permutations=all_permutations,
                        level=level-1,
                        already_added=already_added,
                        fullprod=fullprod@power,
                        last_choice=ip,
                        fullstr=f'{fullstr} P_{ip}^{p+1}',
                    )


def main():
    with open('one_letter_words_5x5_concurrence_matrix_100.pickle', 'rb') as f:
        A = pickle.load(f)

    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]

    n_nodes = A.shape[0]

    model = po.ConcreteModel()
    model.N = po.Set(initialize=range(n_nodes))

    print('Creating Solver using glpk')
    solver = po.SolverFactory('glpk')

    print('Creating Parameter for Concurrence Matrix')
    model.A = po.Param(model.N, model.N, initialize=A)  # , within=po.Reals)

    print('Creating Boolean Permutation Matrix')
    model.P = po.Var(model.N, model.N, within=po.Boolean)

    print('Creating Upper Limit Variables for the Pointwise Error')
    model.T = po.Var(model.N, model.N, within=po.NonNegativeReals)

    print('Creating Row Sum Constraint for the Permutation Matrix')
    model.rowSum = po.Constraint(model.N, rule=lambda m, i: 1 == sum(m.P[i, j] for j in m.N))
    print('Creating Column Sum Constraint for the Permutation Matrix')
    model.colSum = po.Constraint(model.N, rule=lambda m, j: 1 == sum(m.P[i, j] for i in m.N))

    print('Creating Objective Function')
    model.objective = po.Objective(expr=sum(model.T[i, j] for j in model.N for i in model.N), sense=po.minimize)

    def deviation(m, i, j):
        return sum(m.P[i, k]*A[k, j] for k in m.N) - sum(A[i, k]*m.P[k, j] for k in m.N)

    print('Creating Constraint to Limit Positive Deviation')
    model.posDev = po.Constraint(model.N, model.N, rule=lambda m, i, j: deviation(m, i, j) <= m.T[i, j])
    print('Creating Constraint to Limit Negative Deviation')
    model.negDev = po.Constraint(model.N, model.N, rule=lambda m, i, j: -m.T[i, j] <= deviation(m, i, j))

    # Will be excluded below anyway
    # print('Creating Constraint to Exclude Identity')
    # model.identityConstraint = po.Constraint(expr=sum(model.P[i, i] for i in range(n_nodes)) <= n_nodes - 1)

    model.knownPermutations = po.ConstraintList()

    id = np.eye(n_nodes, dtype=int)

    all_permutations = []

    for i_result in count(0):
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

        all_permutations.append(all_powers)

        print('Creating Constraints to Exclude Known Solutions, their Powers and their Combinatorial Products')
        model.del_component(model.knownPermutations)
        model.del_component(model.knownPermutations_index)
        model.knownPermutations = po.ConstraintList()
        create_permutation_combination_constraints(
            m=model,
            id=id,
            n_nodes=n_nodes,
            all_permutations=all_permutations,
            level=5,
            already_added=dict(),
            fullprod=id,
        )
        print(f'Created {len(model.knownPermutations)} constraints.')

    #print('Creating Solver using MindtPy')
    #solver = po.SolverFactory('mindtpy')
    #print('Solving using glpk and ipopt')
    #solver.solve(model, mip_solver='glpk', nlp_solver='ipopt')


if __name__ == '__main__':
    main()
