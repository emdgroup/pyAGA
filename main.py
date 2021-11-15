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
            line += ' ' if col==0 else '#'
        line += '|'
        print(line)


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
    model.A = po.Param(model.N, model.N, initialize=A) #, within=po.Reals)

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

    model.knownPermutations = po.ConstraintList()

    id = np.eye(n_nodes)
    known_result = id

    found_transformations = []

    for i_result in count(0):
        print(f'Computing Cycle of Solution P_{i_result}')
        all_powers = []
        power = known_result.copy()
        while True:
            all_powers.append(power)
            power = power@known_result
            if np.allclose(power, id):
                break

        found_transformations.append(all_powers)

        print('Creating Constraints to Exclude Known Solution and its Powers')
        for p, power in enumerate(all_powers):
            print(f'P_{i_result}^{p}')
            model.knownPermutations.add(expr=sum(model.P[tuple(ij)] for ij in np.argwhere(power)) <= n_nodes - 1)

        print('Solving using glpk')
        results = solver.solve(model, tee=True)
        print(results)

        known_result = to_ndarray(model.P, n_nodes, n_nodes)
        print(f'P_{i_result + 1} =')
        matshow(known_result)

    #print('Creating Solver using MindtPy')
    #solver = po.SolverFactory('mindtpy')
    #print('Solving using glpk and ipopt')
    #solver.solve(model, mip_solver='glpk', nlp_solver='ipopt')


if __name__ == '__main__':
    main()
