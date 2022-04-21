from enum import Enum
import logging

import pyomo.environ as po
from pyomo.opt import ProblemFormat

from mipsym import scip  # noqa: F401


logger = logging.getLogger("integer_programming")
logger.setLevel(
    level=logging.DEBUG
)  # set to logging.INFO for less, to logging.DEBUG for more verbosity


class Norm(Enum):
    L_INFINITY = 0
    L_1 = 1
    L_2 = 2


class Solver(Enum):
    GLPK = 0
    IPOPT = 1
    SCIP = 2


def create_mip_solver(solver: Solver, norm: Norm):
    if solver == Solver.GLPK:
        solver_factory_params = dict(_name="glpk")
        solver_executable = []
        solver_options = dict(fpump="")
        solve_params = dict()
    elif solver == Solver.IPOPT:
        solver_factory_params = dict(_name="mindtpy")
        solver_executable = []
        solver_options = dict()
        solve_params = dict(mip_solver="glpk", nlp_solver="ipopt")
    elif solver == Solver.SCIP:
        solver_factory_params = dict(_name="scip")
        solver_executable = [
            "C:/Program Files/SCIPOptSuite 7.0.3/bin/scip",
            "/Users/m290886/Downloads/SCIPOptSuite-7.0.3-Darwin/bin/scip",
        ]
        solver_options = dict()
        solve_params = dict()
    else:
        raise ValueError(f"Unsupported solver {solver}")

    logger.debug(f"Creating Solver using params {solver_factory_params}")
    ip_solver = po.SolverFactory(**solver_factory_params)

    for executable in solver_executable:
        try:
            ip_solver.set_executable(executable, validate=True)
        except ValueError:
            continue

        logger.debug(f"Using Solver Executable {executable}")
        break

    ip_solver.options = solver_options

    if norm == Norm.L_2:
        assert solver in (Solver.IPOPT, Solver.SCIP)
        ip_solver.set_problem_format(ProblemFormat.mps)

    return ip_solver, solve_params
