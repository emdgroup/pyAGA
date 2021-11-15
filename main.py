import pickle

import numpy as np
import pyomo.environ as po


def to_ndarray(v, m, n, dtype=int):
    result = np.zeros((m, n), dtype=dtype)
    for i in range(m):
        for j in range(n):
            result[i, j] = v[i, j].value
    return result


def main():
    with open('one_letter_words_5x5_concurrence_matrix_100.pickle', 'rb') as f:
        A = pickle.load(f)

    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]

    n_nodes = A.shape[0]

    model = po.ConcreteModel()
    model.N = po.Set(initialize=range(n_nodes))

    print('Creating Parameter for Concurrence Matrix')
    model.A = po.Param(model.N, model.N, initialize=A)

    print('Creating Boolean Permutation Matrix')
    model.P = po.Var(model.N, model.N, within=po.Boolean)

    print('Creating Upper Limit Variables for the Pointwise Error')
    model.T = po.Var(model.N, model.N, within=po.NonNegativeReals)

    print('Creating Row Sum Constraint for the Permutation Matrix')
    model.rowSum = po.Constraint(model.N, rule=lambda m, i: 1 == sum(m.P[i, j] for j in m.N))
    print('Creating Column Sum Constraint for the Permutation Matrix')
    model.colSum = po.Constraint(model.N, rule=lambda m, j: 1 == sum(m.P[i, j] for i in m.N))

    def deviation(m, i, j):
        return sum(m.P[i, k]*A[k, j] for k in m.N) - sum(A[i, k]*m.P[k, j] for k in m.N)

    print('Creating Constraint to Limit Positive Deviation')
    model.posDev = po.Constraint(model.N, model.N, rule=lambda m, i, j: deviation(m, i, j) <= m.T[i, j])
    print('Creating Constraint to Limit Negative Deviation')
    model.negDev = po.Constraint(model.N, model.N, rule=lambda m, i, j: -m.T[i, j] <= deviation(m, i, j))

    print('Creating Objective Function')
    model.objective = po.Objective(expr=sum(model.T[i, j] for j in model.N for i in model.N), sense=po.minimize)

    print('Creating Solver using glpk')
    solver = po.SolverFactory('glpk')
    print('Solving using glpk')
    results = solver.solve(model)

    print(results)

    #print('Creating Solver using MindtPy')
    #solver = po.SolverFactory('mindtpy')
    #print('Solving using glpk and ipopt')
    #solver.solve(model, mip_solver='glpk', nlp_solver='ipopt')

    print(to_ndarray(model.P, n_nodes, n_nodes))


if __name__ == '__main__':
    main()
