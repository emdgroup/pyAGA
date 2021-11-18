import logging
import re
import subprocess
from enum import Enum

from pyomo.common.tempfiles import TempfileManager

from pyomo.common import Executable
from pyomo.common.collections import Bunch
from pyomo.opt import SolverFactory, OptSolver, ProblemFormat, ResultsFormat, SolverResults, TerminationCondition, SolutionStatus
from pyomo.opt.base.solvers import _extract_version
from pyomo.opt.solver import SystemCallSolver

logger = logging.getLogger('pyomo.solvers')

_highs_version = None


def configure_highs():
    global _highs_version
    if _highs_version is not None:
        return
    _highs_version = _extract_version('')
    if not Executable("highs"):
        return
    result = subprocess.run([Executable('highs').path(), '-h'],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=1, universal_newlines=True)
    if not result.returncode:
        _highs_version = _extract_version(result.stdout)


@SolverFactory.register('highs', doc='HiGHS - high performance software for linear optimization')
class HiGHS(OptSolver):
    """HiGHS - high performance software for linear optimization"""

    def __new__(cls, *args, **kwds):
        configure_highs()
        try:
            mode = kwds['solver_io']
            if mode is None:
                mode = 'lp'
            del kwds['solver_io']
        except KeyError:
            mode = 'lp'

        opt = SolverFactory('_highs_shell', **kwds)

        if mode == 'lp':
            pass
        elif mode == 'mps':
            opt.set_problem_format(ProblemFormat.mps)

        return opt


@SolverFactory.register('_highs_shell', doc='Shell interface to HiGHS - high performance software for linear optimization')
class HiGHS_Shell(SystemCallSolver):
    """Shell interface to HiGHS - high performance software for linear optimization"""

    def __init__(self, **kwargs):
        configure_highs()
        #
        # Call base constructor
        #
        kwargs['type'] = 'highs'
        SystemCallSolver.__init__(self, **kwargs)

        self._rawfile = None

        #
        # Valid problem formats, and valid results for each format
        #
        self._valid_problem_formats = [ProblemFormat.cpxlp,
                                       ProblemFormat.mps]
        self._valid_result_formats = {
          ProblemFormat.cpxlp: ResultsFormat.soln,
          ProblemFormat.mps:   ResultsFormat.soln,
        }
        self.set_problem_format(ProblemFormat.cpxlp)

        # Note: Undefined capabilities default to 'None'
        self._capabilities = Bunch()
        self._capabilities.linear = True
        self._capabilities.integer = True

    def _default_results_format(self, prob_format):
        return ResultsFormat.soln

    def _default_executable(self):
        executable = Executable('highs')
        if not executable:
            msg = ("Could not locate the 'highs' executable, which is "
                   "required for solver '%s'")
            logger.warning(msg % self.name)
            self.enable = False
            return None
        return executable.path()

    def _get_version(self):
        """
        Returns a tuple describing the solver executable version.
        """
        if _highs_version is None:
            return _extract_version('')
        return _highs_version

    def create_command_line(self, executable, problem_files):
        #
        # Define log file
        #
        if self._log_file is None:
            self._log_file = TempfileManager.create_tempfile(suffix='.highs.log')

        #
        # Define solution file
        #
        self._soln_file = TempfileManager.create_tempfile(suffix='.highs.sol')

        #
        # Define options file
        #
        self._options_file = TempfileManager.create_tempfile(suffix='.highs.set')

        # TODO: see https://github.com/ERGO-Code/HiGHS/blob/master/src/lp_data/HighsOptions.h
        options_file_content = dict(
            log_dev_level=3,
            log_to_console='true',
            output_flag='true',
            # Presolve option: "off", "choose" or "on"
            # [type: string, advanced: false, default: "choose"]
            presolve='choose',
            # Solver option: "simplex", "choose" or "ipm"
            # [type: string, advanced: false, default: "choose"]
            solver='choose',
            # Parallel option: "off", "choose" or "on"
            # [type: string, advanced: false, default: "choose"]
            parallel='choose',
            # Time limit
            # [type: double, advanced: false, range: [0, inf], default: inf]
            time_limit=self._timelimit if self._timelimit is not None and self._timelimit > 0.0 else 'inf',
            # Limit on cost coefficient: values larger than this will be treated as infinite
            # [type: double, advanced: false, range: [1e+15, 1e+25], default: 1e+20]
            infinite_cost=1e+20,
            # Limit on |constraint bound|: values larger than this will be treated as infinite
            # [type: double, advanced: false, range: [1e+15, 1e+25], default: 1e+20]
            infinite_bound=1e+20,
            # Lower limit on |matrix entries|: values smaller than this will be treated as zero
            # [type: double, advanced: false, range: [1e-12, inf], default: 1e-09]
            small_matrix_value=1e-09,
            # Upper limit on |matrix entries|: values larger than this will be treated as infinite
            # [type: double, advanced: false, range: [1, 1e+20], default: 1e+15]
            large_matrix_value=1e+15,
            # Primal feasibility tolerance
            # [type: double, advanced: false, range: [1e-10, inf], default: 1e-07]
            primal_feasibility_tolerance=1e-07,
            # Dual feasibility tolerance
            # [type: double, advanced: false, range: [1e-10, inf], default: 1e-07]
            dual_feasibility_tolerance=1e-07,
            # Debugging level in HiGHS
            # [type: int, advanced: false, range: {0, 3}, default: 0]
            highs_debug_level=3,
            # Strategy for simplex solver
            # [type: int, advanced: false, range: {0, 4}, default: 1]
            simplex_strategy=1,
            # Strategy for scaling before simplex solver: off / on (0/1)
            # [type: int, advanced: false, range: {0, 5}, default: 2]
            simplex_scale_strategy=2,
            # Strategy for simplex crash: off / LTSSF / Bixby (0/1/2)
            # [type: int, advanced: false, range: {0, 9}, default: 0]
            simplex_crash_strategy=0,
            # Strategy for simplex dual edge weights: Dantzig / Devex / Steepest Edge (0/1/2)
            # [type: int, advanced: false, range: {0, 4}, default: 2]
            simplex_dual_edge_weight_strategy=2,
            # Strategy for simplex primal edge weights: Dantzig / Devex (0/1)
            # [type: int, advanced: false, range: {0, 1}, default: 0]
            simplex_primal_edge_weight_strategy=0,
            # Iteration limit for simplex solver
            # [type: int, advanced: false, range: {0, 2147483647}, default: 2147483647]
            simplex_iteration_limit=2147483647,
            # Limit on the number of simplex UPDATE operations
            # [type: int, advanced: false, range: {0, 2147483647}, default: 5000]
            simplex_update_limit=5000,
            # Iteration limit for IPM solver
            # [type: int, advanced: false, range: {0, 2147483647}, default: 2147483647]
            ipm_iteration_limit=2147483647,
            # Minimum number of threads in parallel execution
            # [type: int, advanced: false, range: {1, 8}, default: 1]
            highs_min_threads=1,
            # Maximum number of threads in parallel execution
            # [type: int, advanced: false, range: {1, 8}, default: 8]
            highs_max_threads=8,
            # Solution file
            # [type: string, advanced: false, default: ""]
            solution_file=self._soln_file,
            # Write the primal and dual solution to a file
            # [type: bool, advanced: false, range: {false, true}, default: false]
            write_solution_to_file='true',
            # Write the primal and dual solution in a pretty (human-readable) format
            # [type: bool, advanced: false, range: {false, true}, default: false]
            write_solution_pretty='false',
            # MIP solver max number of nodes
            # [type: int, advanced: false, range: {0, 2147483647}, default: 2147483647]
            mip_max_nodes=2147483647,
            # MIP solver reporting level
            # [type: int, advanced: false, range: {0, 2}, default: 1]
            mip_report_level=2,
            # Use the free format MPS file reader
            # [type: bool, advanced: true, range: {false, true}, default: true]
            mps_parser_type_free='true',
            # For multiple N-rows in MPS files: delete rows / delete entries / keep rows (-1/0/1)
            # [type: int, advanced: true, range: {-1, 1}, default: -1]
            keep_n_rows=-1,
            # Largest power-of-two factor permitted when scaling the constraint matrix for the simplex solver
            # [type: int, advanced: true, range: {0, 20}, default: 10]
            allowed_simplex_matrix_scale_factor=10,
            # Largest power-of-two factor permitted when scaling the costs for the simplex solver
            # [type: int, advanced: true, range: {0, 20}, default: 0]
            allowed_simplex_cost_scale_factor=0,
            # Strategy for dualising before simplex
            # [type: int, advanced: true, range: {-1, 1}, default: -1]
            simplex_dualise_strategy=-1,
            # Strategy for permuting before simplex
            # [type: int, advanced: true, range: {-1, 1}, default: -1]
            simplex_permute_strategy=-1,
            # Strategy for PRICE in simplex
            # [type: int, advanced: true, range: {0, 4}, default: 3]
            simplex_price_strategy=3,
            # Perform initial basis condition check in simplex
            # [type: bool, advanced: true, range: {false, true}, default: true]
            simplex_initial_condition_check='true',
            # Tolerance on initial basis condition in simplex
            # [type: double, advanced: true, range: [1, inf], default: 1e+14]
            simplex_initial_condition_tolerance=1e+14,
            # Dual simplex cost perturbation multiplier: 0 => no perturbation
            # [type: double, advanced: true, range: [0, inf], default: 1]
            dual_simplex_cost_perturbation_multiplier=1,
            # Use original HFactor logic for sparse vs hyper-sparse TRANs
            # [type: bool, advanced: true, range: {false, true}, default: true]
            use_original_HFactor_logic='true',
            # Check whether LP is candidate for LiDSE
            # [type: bool, advanced: true, range: {false, true}, default: true]
            less_infeasible_DSE_check='true',
            # Use LiDSE if LP has right properties
            # [type: bool, advanced: true, range: {false, true}, default: true]
            less_infeasible_DSE_choose_row='true',
        )

        # TODO: should transfer stuff from self.options into options_file_content
        # to make it adjustable

        with open(self._options_file, 'w') as f:
            for key, value in options_file_content.items():
                f.write(f'{key} = {value}\n')

        #
        # Define command line
        #
        cmd = [executable]
        if self._timer:
            cmd.insert(0, self._timer)
        for key in self.options:
            opt = self.options[key]
            if opt is None or (isinstance(opt, str) and opt.strip() == ''):
                # Handle the case for options that must be
                # specified without a value
                cmd.append("--%s" % key)
            else:
                cmd.extend(["--%s" % key, str(opt)])

        cmd.extend(['--options_file', self._options_file])
        cmd.extend(['--model_file', problem_files[0]])

        return Bunch(cmd=cmd, log_file=self._log_file, env=None)

    def process_logfile(self):
        """
        Process logfile
        """
        results = SolverResults()

        # For the lazy programmer, handle long variable names
        prob   = results.problem
        solv   = results.solver
        solv.termination_condition = TerminationCondition.unknown
        stats  = results.solver.statistics
        bbound = stats.branch_and_bound

        prob.upper_bound = float('inf')
        prob.lower_bound = float('-inf')
        bbound.number_of_created_subproblems = 0
        bbound.number_of_bounded_subproblems = 0

        #TODO: properly support output file
        with open(self._log_file, 'r') as output:
            for line in output:
                toks = line.split()
                if 'tree is empty' in line:
                    bbound.number_of_created_subproblems = toks[-1][:-1]
                    bbound.number_of_bounded_subproblems = toks[-1][:-1]
                elif len(toks) == 2 and toks[0] == "sys":
                    solv.system_time = toks[1]
                elif len(toks) == 2 and toks[0] == "user":
                    solv.user_time = toks[1]
                elif len(toks) > 2 and (toks[0], toks[2]) == ("TIME", "EXCEEDED;"):
                    solv.termination_condition = TerminationCondition.maxTimeLimit
                elif len(toks) > 5 and (toks[:6] == ['PROBLEM', 'HAS', 'NO', 'DUAL', 'FEASIBLE', 'SOLUTION']):
                    solv.termination_condition = TerminationCondition.unbounded
                elif len(toks) > 5 and (toks[:6] == ['PROBLEM', 'HAS', 'NO', 'PRIMAL', 'FEASIBLE', 'SOLUTION']):
                    solv.termination_condition = TerminationCondition.infeasible
                elif len(toks) > 4 and (toks[:5] == ['PROBLEM', 'HAS', 'NO', 'FEASIBLE', 'SOLUTION']):
                    solv.termination_condition = TerminationCondition.infeasible
                elif len(toks) > 6 and (toks[:7] == ['LP', 'RELAXATION', 'HAS', 'NO', 'DUAL', 'FEASIBLE', 'SOLUTION']):
                    solv.termination_condition = TerminationCondition.unbounded

        return results

    def process_soln_file(self, results):
        class ReaderState(Enum):
            UNKNOWN = 0
            MODEL_STATUS = 1
            PRIMAL_SOLUTION_VALUES = 2
            DUAL_SOLUTION_VALUES = 3
            BASIS = 4
            COLUMNS = 5
            ROWS = 6

        line_types = (
            (re.compile(r'Model status'), ReaderState.MODEL_STATUS),
            (re.compile(r'# Primal solution values'), ReaderState.PRIMAL_SOLUTION_VALUES),
            (re.compile(r'# Dual solution values'), ReaderState.DUAL_SOLUTION_VALUES),
            (re.compile(r'# Basis'), ReaderState.BASIS),
            (re.compile(r'# Columns (\d+)'), ReaderState.COLUMNS),
            (re.compile(r'# Rows (\d+)'), ReaderState.ROWS),
        )

        results.problem.name = 'unknown'
        soln = results.solution.add()

        current_state = ReaderState.UNKNOWN
        with open(self._soln_file, 'r') as f:
            for line in f:
                line = line.strip()

                if 0 == len(line):
                    continue
                if 'None' == line:
                    continue

                new_state = False
                for r, s in line_types:
                    m = r.match(line)
                    if m is not None:
                        current_state = s
                        new_state = True

                if new_state:
                    continue

                if current_state == ReaderState.MODEL_STATUS:
                    if line == 'Optimal':
                        results.solver.termination_condition = TerminationCondition.optimal
                        continue
                elif current_state == ReaderState.PRIMAL_SOLUTION_VALUES:
                    if line.startswith('Objective '):
                        obj_val = float(line[10:])
                        results.problem.lower_bound = obj_val
                        results.problem.upper_bound = obj_val
                        # TODO: for proper output, we should try to find out, what the objective name is to run
                        # soln.objective[obj_name] = {'Value': obj_val}
                        continue
                    if line == 'Feasible':
                        soln.status = SolutionStatus.feasible
                        continue
                elif current_state == ReaderState.DUAL_SOLUTION_VALUES:
                    pass
                elif current_state == ReaderState.BASIS:
                    if 'HiGHS v1' == line:
                        continue
                elif current_state == ReaderState.COLUMNS:
                    if 'x' == line[0]:
                        name, val = line.split(' ')
                        soln.variable[name] = {"Value": float(val)}
                        continue
                    elif line.startswith('ONE_VAR_CONSTANT '):
                        continue
                elif current_state == ReaderState.ROWS:
                    if 'R' == line[0]:
                        name, val = line.split(' ')
                        soln.constraint[name] = {"Dual": val}
                        continue
                else:
                    pass

                logger.error(f'Unexpected line "{line}" while parsing {self._soln_file} in state {current_state}')
