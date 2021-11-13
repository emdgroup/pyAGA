import pickle

import pyomo.environ as po


def main():
    with open('one_letter_words_5x5_concurrence_matrix_100.pickle', 'rb') as f:
        A = pickle.load(f)

    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]

    n_nodes = A.shape[0]

    model = po.ConcreteModel()
    model.N = po.Set(initialize=range(n_nodes))

    print('Creating Boolean Permutation Matrix')
    model.P = po.Var(model.N, model.N, within=po.Boolean)

    print('Creating Row Sum Constraint for the Permutation Matrix')
    model.rowSum = po.Constraint(model.N, rule=lambda m, i: 1 >= sum(m.P[i, j] for j in m.N))
    print('Creating Column Sum Constraint for the Permutation Matrix')
    model.colSum = po.Constraint(model.N, rule=lambda m, j: 1 == sum(m.P[i, j] for i in m.N))

    print('Creating Objective Function')
    model.objective = po.Objective(rule=lambda m: sum(((m.P@A-A@m.P)**2)[ij] for ij in m.N*m.N), sense=po.minimize)

    print('Creating Solver using MindtPy')
    solver = po.SolverFactory('mindtpy')
    print('Solving using glpk and ipopt')
    solver.solve(model, mip_solver='glpk', nlp_solver='ipopt')

    model.P.display()


if __name__ == '__main__':
    main()
