import logging
import re
import subprocess

from pyomo.common.tempfiles import TempfileManager

from pyomo.common import Executable
from pyomo.common.collections import Bunch
from pyomo.opt import SolverFactory, OptSolver, ProblemFormat, ResultsFormat, SolverResults, TerminationCondition, SolutionStatus
from pyomo.opt.base.solvers import _extract_version
from pyomo.opt.solver import SystemCallSolver

logger = logging.getLogger('pyomo.solvers')

_scip_version = None


def configure_scip():
    global _scip_version
    if _scip_version is not None:
        return
    _scip_version = _extract_version("")
    if not Executable("scip"):
        return
    result = subprocess.run([Executable('scip').path(), '-v'],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=1, universal_newlines=True)
    if not result.returncode:
        _scip_version = _extract_version(result.stdout)


@SolverFactory.register('scip', doc='SCIP - Solving Constraint Integer Programs')
class SCIP(OptSolver):
    """SCIP - Solving Constraint Integer Programs"""

    def __new__(cls, *args, **kwds):
        configure_scip()
        try:
            mode = kwds['solver_io']
            if mode is None:
                mode = 'lp'
            del kwds['solver_io']
        except KeyError:
            mode = 'lp'

        opt = SolverFactory('_scip_shell', **kwds)

        if mode == 'lp':
            pass
        elif mode == 'mps':
            opt.set_problem_format(ProblemFormat.mps)

        return opt


@SolverFactory.register('_scip_shell', doc='Shell interface to SCIP - Solving Constraint Integer Programs')
class SCIP_Shell(SystemCallSolver):
    """Shell interface to SCIP - Solving Constraint Integer Programs"""

    def __init__(self, **kwargs):
        configure_scip()
        #
        # Call base constructor
        #
        kwargs['type'] = 'scip'
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
        executable = Executable('scip')
        if not executable:
            msg = ("Could not locate the 'scip' executable, which is "
                   "required for solver '%s'")
            logger.warning(msg % self.name)
            self.enable = False
            return None
        return executable.path()

    def _get_version(self):
        """
        Returns a tuple describing the solver executable version.
        """
        if _scip_version is None:
            return _extract_version('')
        return _scip_version

    def create_command_line(self, executable, problem_files):
        #
        # Define log file
        #
        if self._log_file is None:
            self._log_file = TempfileManager.create_tempfile(suffix='.scip.log')

        #
        # Define solution file
        #
        self._soln_file = TempfileManager.create_tempfile(suffix='.scip.sol')

        #
        # Define options file
        #
        self._options_file = TempfileManager.create_tempfile(suffix='.scip.set')

        # run ./scip -c "set save scip.set" to get the currently active settings; the following is a copy of a local run of this on scip 7.0.3
        options_file_content = {
            # branching score function ('s'um, 'p'roduct, 'q'uotient)
            # [type: char, advanced: TRUE, range: {spq}, default: p]
            'branching/scorefunc': 'p',
            # branching score factor to weigh downward and upward gain prediction in sum score function
            # [type: real, advanced: TRUE, range: [0,1], default: 0.167]
            'branching/scorefac': '0.167',
            # should branching on binary variables be preferred?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'branching/preferbinary': 'FALSE',
            # minimal relative distance of branching point to bounds when branching on a continuous variable
            # [type: real, advanced: FALSE, range: [0,0.5], default: 0.2]
            'branching/clamp': '0.2',
            # fraction by which to move branching point of a continuous variable towards the middle of the domain; a value of 1.0 leads to branching always in the middle of the domain
            # [type: real, advanced: FALSE, range: [0,1], default: 0.75]
            'branching/midpull': '0.75',
            # multiply midpull by relative domain width if the latter is below this value
            # [type: real, advanced: FALSE, range: [0,1], default: 0.5]
            'branching/midpullreldomtrig': '0.5',
            # strategy for normalization of LP gain when updating pseudocosts of continuous variables (divide by movement of 'l'p value, reduction in 'd'omain width, or reduction in domain width of 's'ibling)
            # [type: char, advanced: FALSE, range: {dls}, default: s]
            'branching/lpgainnormalize': 's',
            # should updating pseudo costs for continuous variables be delayed to the time after separation?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'branching/delaypscostupdate': 'TRUE',
            # should pseudo costs be updated also in diving and probing mode?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'branching/divingpscost': 'TRUE',
            # should all strong branching children be regarded even if one is detected to be infeasible? (only with propagation)
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'branching/forceallchildren': 'FALSE',
            # child node to be regarded first during strong branching (only with propagation): 'u'p child, 'd'own child, 'h'istory-based, or 'a'utomatic
            # [type: char, advanced: TRUE, range: {aduh}, default: a]
            'branching/firstsbchild': 'a',
            # should LP solutions during strong branching with propagation be checked for feasibility?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'branching/checksol': 'TRUE',
            # should LP solutions during strong branching with propagation be rounded? (only when checksbsol=TRUE)
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'branching/roundsbsol': 'TRUE',
            # score adjustment near zero by adding epsilon (TRUE) or using maximum (FALSE)
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'branching/sumadjustscore': 'FALSE',
            # should automatic tree compression after the presolving be enabled?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'compression/enable': 'FALSE',
            # should conflict analysis be enabled?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'conflict/enable': 'TRUE',
            # should conflicts based on an old cutoff bound be removed from the conflict pool after improving the primal bound?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'conflict/cleanboundexceedings': 'TRUE',
            # use local rows to construct infeasibility proofs
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'conflict/uselocalrows': 'TRUE',
            # should propagation conflict analysis be used?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'conflict/useprop': 'TRUE',
            # should infeasible LP conflict analysis be used? ('o'ff, 'c'onflict graph, 'd'ual ray, 'b'oth conflict graph and dual ray)
            # [type: char, advanced: FALSE, range: {ocdb}, default: b]
            'conflict/useinflp': 'b',
            # should bound exceeding LP conflict analysis be used? ('o'ff, 'c'onflict graph, 'd'ual ray, 'b'oth conflict graph and dual ray)
            # [type: char, advanced: FALSE, range: {ocdb}, default: b]
            'conflict/useboundlp': 'b',
            # should infeasible/bound exceeding strong branching conflict analysis be used?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'conflict/usesb': 'TRUE',
            # should pseudo solution conflict analysis be used?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'conflict/usepseudo': 'TRUE',
            # maximal fraction of variables involved in a conflict constraint
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0.15]
            'conflict/maxvarsfac': '0.15',
            # minimal absolute maximum of variables involved in a conflict constraint
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 0]
            'conflict/minmaxvars': '0',
            # maximal number of LP resolving loops during conflict analysis (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 2]
            'conflict/maxlploops': '2',
            # maximal number of LP iterations in each LP resolving loop (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 10]
            'conflict/lpiterations': '10',
            # number of depth levels up to which first UIP's are used in conflict analysis (-1: use All-FirstUIP rule)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'conflict/fuiplevels': '-1',
            # maximal number of intermediate conflict constraints generated in conflict graph (-1: use every intermediate constraint)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'conflict/interconss': '-1',
            # number of depth levels up to which UIP reconvergence constraints are generated (-1: generate reconvergence constraints in all depth levels)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'conflict/reconvlevels': '-1',
            # maximal number of conflict constraints accepted at an infeasible node (-1: use all generated conflict constraints)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 10]
            'conflict/maxconss': '10',
            # maximal size of conflict store (-1: auto, 0: disable storage)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 10000]
            'conflict/maxstoresize': '10000',
            # should binary conflicts be preferred?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'conflict/preferbinary': 'FALSE',
            # prefer infeasibility proof to boundexceeding proof
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'conflict/prefinfproof': 'TRUE',
            # should conflict constraints be generated that are only valid locally?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'conflict/allowlocal': 'TRUE',
            # should conflict constraints be attached only to the local subtree where they can be useful?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'conflict/settlelocal': 'FALSE',
            # should earlier nodes be repropagated in order to replace branching decisions by deductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'conflict/repropagate': 'TRUE',
            # should constraints be kept for repropagation even if they are too long?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'conflict/keepreprop': 'TRUE',
            # should the conflict constraints be separated?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'conflict/separate': 'TRUE',
            # should the conflict constraints be subject to aging?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'conflict/dynamic': 'TRUE',
            # should the conflict's relaxations be subject to LP aging and cleanup?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'conflict/removable': 'TRUE',
            # score factor for depth level in bound relaxation heuristic
            # [type: real, advanced: TRUE, range: [-1.79769313486232e+308,1.79769313486232e+308], default: 1]
            'conflict/graph/depthscorefac': '1',
            # score factor for impact on acticity in bound relaxation heuristic
            # [type: real, advanced: TRUE, range: [-1.79769313486232e+308,1.79769313486232e+308], default: 1]
            'conflict/proofscorefac': '1',
            # score factor for up locks in bound relaxation heuristic
            # [type: real, advanced: TRUE, range: [-1.79769313486232e+308,1.79769313486232e+308], default: 0]
            'conflict/uplockscorefac': '0',
            # score factor for down locks in bound relaxation heuristic
            # [type: real, advanced: TRUE, range: [-1.79769313486232e+308,1.79769313486232e+308], default: 0]
            'conflict/downlockscorefac': '0',
            # factor to decrease importance of variables' earlier conflict scores
            # [type: real, advanced: TRUE, range: [1e-06,1], default: 0.98]
            'conflict/scorefac': '0.98',
            # number of successful conflict analysis calls that trigger a restart (0: disable conflict restarts)
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 0]
            'conflict/restartnum': '0',
            # factor to increase restartnum with after each restart
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 1.5]
            'conflict/restartfac': '1.5',
            # should relaxed bounds be ignored?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'conflict/ignorerelaxedbd': 'FALSE',
            # maximal number of variables to try to detect global bound implications and shorten the whole conflict set (0: disabled)
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 250]
            'conflict/maxvarsdetectimpliedbounds': '250',
            # try to shorten the whole conflict set or terminate early (depending on the 'maxvarsdetectimpliedbounds' parameter)
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'conflict/fullshortenconflict': 'TRUE',
            # the weight the VSIDS score is weight by updating the VSIDS for a variable if it is part of a conflict
            # [type: real, advanced: FALSE, range: [0,1], default: 0]
            'conflict/conflictweight': '0',
            # the weight the VSIDS score is weight by updating the VSIDS for a variable if it is part of a conflict graph
            # [type: real, advanced: FALSE, range: [0,1], default: 1]
            'conflict/conflictgraphweight': '1',
            # minimal improvement of primal bound to remove conflicts based on a previous incumbent
            # [type: real, advanced: TRUE, range: [0,1], default: 0.05]
            'conflict/minimprove': '0.05',
            # weight of the size of a conflict used in score calculation
            # [type: real, advanced: TRUE, range: [0,1], default: 0.001]
            'conflict/weightsize': '0.001',
            # weight of the repropagation depth of a conflict used in score calculation
            # [type: real, advanced: TRUE, range: [0,1], default: 0.1]
            'conflict/weightrepropdepth': '0.1',
            # weight of the valid depth of a conflict used in score calculation
            # [type: real, advanced: TRUE, range: [0,1], default: 1]
            'conflict/weightvaliddepth': '1',
            # apply cut generating functions to construct alternative proofs
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'conflict/sepaaltproofs': 'FALSE',
            # maximum age an unnecessary constraint can reach before it is deleted (0: dynamic, -1: keep all constraints)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 0]
            'constraints/agelimit': '0',
            # age of a constraint after which it is marked obsolete (0: dynamic, -1 do not mark constraints obsolete)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'constraints/obsoleteage': '-1',
            # should enforcement of pseudo solution be disabled?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/disableenfops': 'FALSE',
            # verbosity level of output
            # [type: int, advanced: FALSE, range: [0,5], default: 4]
            'display/verblevel': '4',
            # maximal number of characters in a node information line
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 143]
            'display/width': '143',
            # frequency for displaying node information lines
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 100]
            'display/freq': '100',
            # frequency for displaying header lines (every n'th node information line)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 15]
            'display/headerfreq': '15',
            # should the LP solver display status messages?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'display/lpinfo': 'FALSE',
            # display all violations for a given start solution / the best solution after the solving process?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'display/allviols': 'FALSE',
            # should the relevant statistics be displayed at the end of solving?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'display/relevantstats': 'TRUE',
            # should setting of common subscip parameters include the activation of the UCT node selector?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/useuctsubscip': 'FALSE',
            # should statistics be collected for variable domain value pairs?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'history/valuebased': 'FALSE',
            # should variable histories be merged from sub-SCIPs whenever possible?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'history/allowmerge': 'FALSE',
            # should variable histories be transferred to initialize SCIP copies?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'history/allowtransfer': 'FALSE',
            # maximal time in seconds to run
            # [type: real, advanced: FALSE, range: [0,1e+20], default: 1e+20]
            'limits/time': self._timelimit if self._timelimit is not None and self._timelimit > 0.0 else '1e+20',
            # maximal number of nodes to process (-1: no limit)
            # [type: longint, advanced: FALSE, range: [-1,9223372036854775807], default: -1]
            'limits/nodes': '-1',
            # maximal number of total nodes (incl. restarts) to process (-1: no limit)
            # [type: longint, advanced: FALSE, range: [-1,9223372036854775807], default: -1]
            'limits/totalnodes': '-1',
            # solving stops, if the given number of nodes was processed since the last improvement of the primal solution value (-1: no limit)
            # [type: longint, advanced: FALSE, range: [-1,9223372036854775807], default: -1]
            'limits/stallnodes': '-1',
            # maximal memory usage in MB; reported memory usage is lower than real memory usage!
            # [type: real, advanced: FALSE, range: [0,8796093022207], default: 8796093022207]
            'limits/memory': '8796093022207',
            # solving stops, if the relative gap': '|primal - dual|/MIN(|dual|,|primal|) is below the given value, the gap is 'Infinity', if primal and dual bound have opposite signs',
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 0]
            'limits/gap': '0',
            # solving stops, if the absolute gap': '|primalbound - dualbound| is below the given value',
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 0]
            'limits/absgap': '0',
            # solving stops, if the given number of solutions were found (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'limits/solutions': '-1',
            # solving stops, if the given number of solution improvements were found (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'limits/bestsol': '-1',
            # maximal number of solutions to store in the solution storage
            # [type: int, advanced: FALSE, range: [1,2147483647], default: 100]
            'limits/maxsol': '100',
            # maximal number of solutions candidates to store in the solution storage of the original problem
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 10]
            'limits/maxorigsol': '10',
            # solving stops, if the given number of restarts was triggered (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'limits/restarts': '-1',
            # if solve exceeds this number of nodes for the first time, an automatic restart is triggered (-1: no automatic restart)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'limits/autorestartnodes': '-1',
            # frequency for solving LP at the nodes (-1: never; 0: only root LP)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'lp/solvefreq': '1',
            # iteration limit for each single LP solve (-1: no limit)
            # [type: longint, advanced: TRUE, range: [-1,9223372036854775807], default: -1]
            'lp/iterlim': '-1',
            # iteration limit for initial root LP solve (-1: no limit)
            # [type: longint, advanced: TRUE, range: [-1,9223372036854775807], default: -1]
            'lp/rootiterlim': '-1',
            # maximal depth for solving LP at the nodes (-1: no depth limit)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'lp/solvedepth': '-1',
            # LP algorithm for solving initial LP relaxations (automatic 's'implex, 'p'rimal simplex, 'd'ual simplex, 'b'arrier, barrier with 'c'rossover)
            # [type: char, advanced: FALSE, range: {spdbc}, default: s]
            'lp/initalgorithm': 's',
            # LP algorithm for resolving LP relaxations if a starting basis exists (automatic 's'implex, 'p'rimal simplex, 'd'ual simplex, 'b'arrier, barrier with 'c'rossover)
            # [type: char, advanced: FALSE, range: {spdbc}, default: s]
            'lp/resolvealgorithm': 's',
            # LP pricing strategy ('l'pi default, 'a'uto, 'f'ull pricing, 'p'artial, 's'teepest edge pricing, 'q'uickstart steepest edge pricing, 'd'evex pricing)
            # [type: char, advanced: FALSE, range: {lafpsqd}, default: l]
            'lp/pricing': 'l',
            # should lp state be cleared at the end of probing mode when lp was initially unsolved, e.g., when called right after presolving?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'lp/clearinitialprobinglp': 'TRUE',
            # should the LP be resolved to restore the state at start of diving (if FALSE we buffer the solution values)?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'lp/resolverestore': 'FALSE',
            # should the buffers for storing LP solution values during diving be freed at end of diving?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'lp/freesolvalbuffers': 'FALSE',
            # maximum age a dynamic column can reach before it is deleted from the LP (-1: don't delete columns due to aging)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 10]
            'lp/colagelimit': '10',
            # maximum age a dynamic row can reach before it is deleted from the LP (-1: don't delete rows due to aging)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 10]
            'lp/rowagelimit': '10',
            # should new non-basic columns be removed after LP solving?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'lp/cleanupcols': 'FALSE',
            # should new non-basic columns be removed after root LP solving?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'lp/cleanupcolsroot': 'FALSE',
            # should new basic rows be removed after LP solving?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'lp/cleanuprows': 'TRUE',
            # should new basic rows be removed after root LP solving?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'lp/cleanuprowsroot': 'TRUE',
            # should LP solver's return status be checked for stability?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'lp/checkstability': 'TRUE',
            # maximum condition number of LP basis counted as stable (-1.0: no limit)
            # [type: real, advanced: TRUE, range: [-1,1.79769313486232e+308], default: -1]
            'lp/conditionlimit': '-1',
            # minimal Markowitz threshold to control sparsity/stability in LU factorization
            # [type: real, advanced: TRUE, range: [0.0001,0.9999], default: 0.01]
            'lp/minmarkowitz': '0.01',
            # should LP solutions be checked for primal feasibility, resolving LP when numerical troubles occur?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'lp/checkprimfeas': 'TRUE',
            # should LP solutions be checked for dual feasibility, resolving LP when numerical troubles occur?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'lp/checkdualfeas': 'TRUE',
            # should infeasibility proofs from the LP be checked?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'lp/checkfarkas': 'TRUE',
            # which FASTMIP setting of LP solver should be used? 0: off, 1: low
            # [type: int, advanced: TRUE, range: [0,1], default: 1]
            'lp/fastmip': '1',
            # LP scaling (0: none, 1: normal, 2: aggressive)
            # [type: int, advanced: TRUE, range: [0,2], default: 1]
            'lp/scaling': '1',
            # should presolving of LP solver be used?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'lp/presolving': 'TRUE',
            # should the lexicographic dual algorithm be used?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'lp/lexdualalgo': 'FALSE',
            # should the lexicographic dual algorithm be applied only at the root node
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'lp/lexdualrootonly': 'TRUE',
            # maximum number of rounds in the lexicographic dual algorithm (-1: unbounded)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 2]
            'lp/lexdualmaxrounds': '2',
            # choose fractional basic variables in lexicographic dual algorithm?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'lp/lexdualbasic': 'FALSE',
            # turn on the lex dual algorithm only when stalling?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'lp/lexdualstalling': 'TRUE',
            # disable the cutoff bound in the LP solver? (0: enabled, 1: disabled, 2: auto)
            # [type: int, advanced: TRUE, range: [0,2], default: 2]
            'lp/disablecutoff': '2',
            # simplex algorithm shall use row representation of the basis if number of rows divided by number of columns exceeds this value (-1.0 to disable row representation)
            # [type: real, advanced: TRUE, range: [-1,1.79769313486232e+308], default: 1.2]
            'lp/rowrepswitch': '1.2',
            # number of threads used for solving the LP (0: automatic)
            # [type: int, advanced: TRUE, range: [0,64], default: 0]
            'lp/threads': '0',
            # factor of average LP iterations that is used as LP iteration limit for LP resolve (-1: unlimited)
            # [type: real, advanced: TRUE, range: [-1,1.79769313486232e+308], default: -1]
            'lp/resolveiterfac': '-1',
            # minimum number of iterations that are allowed for LP resolve
            # [type: int, advanced: TRUE, range: [1,2147483647], default: 1000]
            'lp/resolveitermin': '1000',
            # LP solution polishing method (0: disabled, 1: only root, 2: always, 3: auto)
            # [type: int, advanced: TRUE, range: [0,3], default: 3]
            'lp/solutionpolishing': '3',
            # LP refactorization interval (0: auto)
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 0]
            'lp/refactorinterval': '0',
            # should the Farkas duals always be collected when an LP is found to be infeasible?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'lp/alwaysgetduals': 'FALSE',
            # solver to use for solving NLPs; leave empty to select NLPI with highest priority
            # [type: string, advanced: FALSE, default: ""]
            'nlp/solver': '""',
            # should the NLP relaxation be always disabled (also for NLPs/MINLPs)?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'nlp/disable': 'FALSE',
            # fraction of maximal memory usage resulting in switch to memory saving mode
            # [type: real, advanced: FALSE, range: [0,1], default: 0.8]
            'memory/savefac': '0.8',
            # memory growing factor for dynamically allocated arrays
            # [type: real, advanced: TRUE, range: [1,10], default: 1.2]
            'memory/arraygrowfac': '1.2',
            # initial size of dynamically allocated arrays
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 4]
            'memory/arraygrowinit': '4',
            # memory growing factor for tree array
            # [type: real, advanced: TRUE, range: [1,10], default: 2]
            'memory/treegrowfac': '2',
            # initial size of tree array
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 65536]
            'memory/treegrowinit': '65536',
            # memory growing factor for path array
            # [type: real, advanced: TRUE, range: [1,10], default: 2]
            'memory/pathgrowfac': '2',
            # initial size of path array
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 256]
            'memory/pathgrowinit': '256',
            # should the CTRL-C interrupt be caught by SCIP?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'misc/catchctrlc': 'TRUE',
            # should a hashtable be used to map from variable names to variables?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'misc/usevartable': 'TRUE',
            # should a hashtable be used to map from constraint names to constraints?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'misc/useconstable': 'TRUE',
            # should smaller hashtables be used? yields better performance for small problems with about 100 variables
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'misc/usesmalltables': 'FALSE',
            # should the statistics be reset if the transformed problem is freed (in case of a Benders' decomposition this parameter should be set to FALSE)
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'misc/resetstat': 'TRUE',
            # should only solutions be checked which improve the primal bound
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'misc/improvingsols': 'FALSE',
            # should the reason be printed if a given start solution is infeasible
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'misc/printreason': 'TRUE',
            # should the usage of external memory be estimated?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'misc/estimexternmem': 'TRUE',
            # should SCIP try to transfer original solutions to the transformed space (after presolving)?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'misc/transorigsols': 'TRUE',
            # should SCIP try to transfer transformed solutions to the original space (after solving)?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'misc/transsolsorig': 'TRUE',
            # should SCIP calculate the primal dual integral value?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'misc/calcintegral': 'TRUE',
            # should SCIP try to remove infinite fixings from solutions copied to the solution store?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'misc/finitesolutionstore': 'FALSE',
            # should the best solution be transformed to the orignal space and be output in command line run?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'misc/outputorigsol': 'TRUE',
            # should strong dual reductions be allowed in propagation and presolving?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'misc/allowstrongdualreds': 'TRUE',
            # should weak dual reductions be allowed in propagation and presolving?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'misc/allowweakdualreds': 'TRUE',
            # should the objective function be scaled so that it is always integer?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'misc/scaleobj': 'TRUE',
            # objective value for reference purposes
            # [type: real, advanced: FALSE, range: [-1.79769313486232e+308,1.79769313486232e+308], default: 1e+99]
            'misc/referencevalue': '1e+99',
            # bitset describing used symmetry handling technique (0: off; 1: polyhedral (orbitopes and/or symresacks); 2: orbital fixing; 3: orbitopes and orbital fixing), see type_symmetry.h.
            # [type: int, advanced: FALSE, range: [0,3], default: 3]
            'misc/usesymmetry': '3',
            # global shift of all random seeds in the plugins and the LP random seed
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 0]
            'randomization/randomseedshift': '0',
            # seed value for permuting the problem after reading/transformation (0: no permutation)
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 0]
            'randomization/permutationseed': '0',
            # should order of constraints be permuted (depends on permutationseed)?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'randomization/permuteconss': 'TRUE',
            # should order of variables be permuted (depends on permutationseed)?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'randomization/permutevars': 'FALSE',
            # random seed for LP solver, e.g. for perturbations in the simplex (0: LP default)
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 0]
            'randomization/lpseed': '0',
            # child selection rule ('d'own, 'u'p, 'p'seudo costs, 'i'nference, 'l'p value, 'r'oot LP value difference, 'h'ybrid inference/root LP value difference)
            # [type: char, advanced: FALSE, range: {dupilrh}, default: h]
            'nodeselection/childsel': 'h',
            # values larger than this are considered infinity
            # [type: real, advanced: FALSE, range: [10000000000,1e+98], default: 1e+20]
            'numerics/infinity': '1e+20',
            # absolute values smaller than this are considered zero
            # [type: real, advanced: FALSE, range: [1e-20,0.001], default: 1e-09]
            'numerics/epsilon': '1e-09',
            # absolute values of sums smaller than this are considered zero
            # [type: real, advanced: FALSE, range: [1e-17,0.001], default: 1e-06]
            'numerics/sumepsilon': '1e-06',
            # feasibility tolerance for constraints
            # [type: real, advanced: FALSE, range: [1e-17,0.001], default: 1e-06]
            'numerics/feastol': '1e-06',
            # feasibility tolerance factor; for checking the feasibility of the best solution
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 1]
            'numerics/checkfeastolfac': '1',
            # factor w.r.t. primal feasibility tolerance that determines default (and maximal) primal feasibility tolerance of LP solver
            # [type: real, advanced: FALSE, range: [1e-06,1], default: 1]
            'numerics/lpfeastolfactor': '1',
            # feasibility tolerance for reduced costs in LP solution
            # [type: real, advanced: FALSE, range: [1e-17,0.001], default: 1e-07]
            'numerics/dualfeastol': '1e-07',
            # LP convergence tolerance used in barrier algorithm
            # [type: real, advanced: TRUE, range: [1e-17,0.001], default: 1e-10]
            'numerics/barrierconvtol': '1e-10',
            # minimal relative improve for strengthening bounds
            # [type: real, advanced: TRUE, range: [1e-17,1e+98], default: 0.05]
            'numerics/boundstreps': '0.05',
            # minimal variable distance value to use for branching pseudo cost updates
            # [type: real, advanced: TRUE, range: [1e-17,1], default: 0.1]
            'numerics/pseudocosteps': '0.1',
            # minimal objective distance value to use for branching pseudo cost updates
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0.0001]
            'numerics/pseudocostdelta': '0.0001',
            # minimal decrease factor that causes the recomputation of a value (e.g., pseudo objective) instead of an update
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 10000000]
            'numerics/recomputefac': '10000000',
            # values larger than this are considered huge and should be handled separately (e.g., in activity computation)
            # [type: real, advanced: TRUE, range: [0,1e+98], default: 1e+15]
            'numerics/hugeval': '1e+15',
            # maximal number of presolving rounds (-1: unlimited, 0: off)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'presolving/maxrounds': '-1',
            # abort presolve, if at most this fraction of the problem was changed in last presolve round
            # [type: real, advanced: TRUE, range: [0,1], default: 0.0008]
            'presolving/abortfac': '0.0008',
            # maximal number of restarts (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'presolving/maxrestarts': '-1',
            # fraction of integer variables that were fixed in the root node triggering a restart with preprocessing after root node evaluation
            # [type: real, advanced: TRUE, range: [0,1], default: 0.025]
            'presolving/restartfac': '0.025',
            # limit on number of entries in clique table relative to number of problem nonzeros
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 2]
            'presolving/clqtablefac': '2',
            # fraction of integer variables that were fixed in the root node triggering an immediate restart with preprocessing
            # [type: real, advanced: TRUE, range: [0,1], default: 0.1]
            'presolving/immrestartfac': '0.1',
            # fraction of integer variables that were globally fixed during the solving process triggering a restart with preprocessing
            # [type: real, advanced: TRUE, range: [0,1], default: 1]
            'presolving/subrestartfac': '1',
            # minimal fraction of integer variables removed after restart to allow for an additional restart
            # [type: real, advanced: TRUE, range: [0,1], default: 0.1]
            'presolving/restartminred': '0.1',
            # should multi-aggregation of variables be forbidden?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'presolving/donotmultaggr': 'FALSE',
            # should aggregation of variables be forbidden?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'presolving/donotaggr': 'FALSE',
            # maximal number of variables priced in per pricing round
            # [type: int, advanced: FALSE, range: [1,2147483647], default: 100]
            'pricing/maxvars': '100',
            # maximal number of priced variables at the root node
            # [type: int, advanced: FALSE, range: [1,2147483647], default: 2000]
            'pricing/maxvarsroot': '2000',
            # pricing is aborted, if fac * pricing/maxvars pricing candidates were found
            # [type: real, advanced: FALSE, range: [1,1.79769313486232e+308], default: 2]
            'pricing/abortfac': '2',
            # should variables created at the current node be deleted when the node is solved in case they are not present in the LP anymore?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'pricing/delvars': 'FALSE',
            # should variables created at the root node be deleted when the root is solved in case they are not present in the LP anymore?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'pricing/delvarsroot': 'FALSE',
            # should the variables be labelled for the application of Benders' decomposition?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'decomposition/benderslabels': 'FALSE',
            # if a decomposition exists, should Benders' decomposition be applied?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'decomposition/applybenders': 'FALSE',
            # maximum number of edges in block graph computation, or -1 for no limit
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 10000]
            'decomposition/maxgraphedge': '10000',
            # the tolerance used for checking optimality in Benders' decomposition. tol where optimality is given by LB + tol > UB.
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 1e-06]
            'benders/solutiontol': '1e-06',
            # should Benders' cuts be generated from the solution to the LP relaxation?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'benders/cutlpsol': 'TRUE',
            # should Benders' decomposition be copied for use in sub-SCIPs?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'benders/copybenders': 'TRUE',
            # maximal number of propagation rounds per node (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 100]
            'propagating/maxrounds': '100',
            # maximal number of propagation rounds in the root node (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 1000]
            'propagating/maxroundsroot': '1000',
            # should propagation be aborted immediately? setting this to FALSE could help conflict analysis to produce more conflict constraints
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'propagating/abortoncutoff': 'TRUE',
            # should reoptimization used?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'reoptimization/enable': 'FALSE',
            # maximal number of saved nodes
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 2147483647]
            'reoptimization/maxsavednodes': '2147483647',
            # maximal number of bound changes between two stored nodes on one path
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 2147483647]
            'reoptimization/maxdiffofnodes': '2147483647',
            # save global constraints to separate infeasible subtrees.
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'reoptimization/globalcons/sepainfsubtrees': 'TRUE',
            # separate the optimal solution, i.e., for constrained shortest path
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'reoptimization/sepabestsol': 'FALSE',
            # use variable history of the previous solve if the objctive function has changed only slightly
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'reoptimization/storevarhistory': 'FALSE',
            # re-use pseudo costs if the objective function changed only slightly
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'reoptimization/usepscost': 'FALSE',
            # at which reopttype should the LP be solved? (1: transit, 3: strong branched, 4: w/ added logicor, 5: only leafs).
            # [type: int, advanced: TRUE, range: [1,5], default: 1]
            'reoptimization/solvelp': '1',
            # maximal number of bound changes at node to skip solving the LP
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 1]
            'reoptimization/solvelpdiff': '1',
            # number of best solutions which should be saved for the following runs. (-1: save all)
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 2147483647]
            'reoptimization/savesols': '2147483647',
            # similarity of two sequential objective function to disable solving the root LP.
            # [type: real, advanced: TRUE, range: [-1,1], default: 0.8]
            'reoptimization/objsimrootLP': '0.8',
            # similarity of two objective functions to re-use stored solutions
            # [type: real, advanced: TRUE, range: [-1,1], default: -1]
            'reoptimization/objsimsol': '-1',
            # minimum similarity for using reoptimization of the search tree.
            # [type: real, advanced: TRUE, range: [-1,1], default: -1]
            'reoptimization/delay': '-1',
            # time limit over all reoptimization rounds?.
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'reoptimization/commontimelimit': 'FALSE',
            # replace branched inner nodes by their child nodes, if the number of bound changes is not to large
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'reoptimization/shrinkinner': 'TRUE',
            # try to fix variables at the root node before reoptimizing by probing like strong branching
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'reoptimization/strongbranchinginit': 'TRUE',
            # delete stored nodes which were not reoptimized
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'reoptimization/reducetofrontier': 'TRUE',
            # force a restart if the last n optimal solutions were found by heuristic reoptsols
            # [type: int, advanced: TRUE, range: [1,2147483647], default: 3]
            'reoptimization/forceheurrestart': '3',
            # save constraint propagations
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'reoptimization/saveconsprop': 'FALSE',
            # use constraints to reconstruct the subtree pruned be dual reduction when reactivating the node
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'reoptimization/usesplitcons': 'TRUE',
            # use 'd'efault, 'r'andom or a variable ordering based on 'i'nference score for interdiction branching used during reoptimization
            # [type: char, advanced: TRUE, range: {dir}, default: d]
            'reoptimization/varorderinterdiction': 'd',
            # reoptimize cuts found at the root node
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'reoptimization/usecuts': 'FALSE',
            # maximal age of a cut to be use for reoptimization
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 0]
            'reoptimization/maxcutage': '0',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying separation (0.0: only on current best node, 1.0: on all nodes)
            # [type: real, advanced: FALSE, range: [0,1], default: 1]
            'separating/maxbounddist': '1',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying local separation (0.0: only on current best node, 1.0: on all nodes)
            # [type: real, advanced: FALSE, range: [0,1], default: 0]
            'separating/maxlocalbounddist': '0',
            # maximal ratio between coefficients in strongcg, cmir, and flowcover cuts
            # [type: real, advanced: FALSE, range: [1,1e+98], default: 10000]
            'separating/maxcoefratio': '10000',
            # minimal efficacy for a cut to enter the LP
            # [type: real, advanced: FALSE, range: [0,1e+98], default: 0.0001]
            'separating/minefficacy': '0.0001',
            # minimal efficacy for a cut to enter the LP in the root node
            # [type: real, advanced: FALSE, range: [0,1e+98], default: 0.0001]
            'separating/minefficacyroot': '0.0001',
            # minimal orthogonality for a cut to enter the LP
            # [type: real, advanced: FALSE, range: [0,1], default: 0.9]
            'separating/minortho': '0.9',
            # minimal orthogonality for a cut to enter the LP in the root node
            # [type: real, advanced: FALSE, range: [0,1], default: 0.9]
            'separating/minorthoroot': '0.9',
            # factor to scale objective parallelism of cut in separation score calculation
            # [type: real, advanced: TRUE, range: [0,1e+98], default: 0.1]
            'separating/objparalfac': '0.1',
            # factor to scale directed cutoff distance of cut in score calculation
            # [type: real, advanced: TRUE, range: [0,1e+98], default: 0.5]
            'separating/dircutoffdistfac': '0.5',
            # factor to scale efficacy of cut in score calculation
            # [type: real, advanced: TRUE, range: [0,1e+98], default: 0.6]
            'separating/efficacyfac': '0.6',
            # factor to scale integral support of cut in separation score calculation
            # [type: real, advanced: TRUE, range: [0,1e+98], default: 0.1]
            'separating/intsupportfac': '0.1',
            # minimum cut activity quotient to convert cuts into constraints during a restart (0.0: all cuts are converted)
            # [type: real, advanced: FALSE, range: [0,1], default: 0.8]
            'separating/minactivityquot': '0.8',
            # function used for calc. scalar prod. in orthogonality test ('e'uclidean, 'd'iscrete)
            # [type: char, advanced: TRUE, range: {ed}, default: e]
            'separating/orthofunc': 'e',
            # row norm to use for efficacy calculation ('e'uclidean, 'm'aximum, 's'um, 'd'iscrete)
            # [type: char, advanced: TRUE, range: {emsd}, default: e]
            'separating/efficacynorm': 'e',
            # cut selection during restart ('a'ge, activity 'q'uotient)
            # [type: char, advanced: TRUE, range: {aq}, default: a]
            'separating/cutselrestart': 'a',
            # cut selection for sub SCIPs  ('a'ge, activity 'q'uotient)
            # [type: char, advanced: TRUE, range: {aq}, default: a]
            'separating/cutselsubscip': 'a',
            # maximal number of runs for which separation is enabled (-1: unlimited)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'separating/maxruns': '-1',
            # maximal number of separation rounds per node (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'separating/maxrounds': '-1',
            # maximal number of separation rounds in the root node (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'separating/maxroundsroot': '-1',
            # maximal number of separation rounds in the root node of a subsequent run (-1: unlimited)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'separating/maxroundsrootsubrun': '-1',
            # maximal additional number of separation rounds in subsequent price-and-cut loops (-1: no additional restriction)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 1]
            'separating/maxaddrounds': '1',
            # maximal number of consecutive separation rounds without objective or integrality improvement in local nodes (-1: no additional restriction)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 1]
            'separating/maxstallrounds': '1',
            # maximal number of consecutive separation rounds without objective or integrality improvement in the root node (-1: no additional restriction)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 10]
            'separating/maxstallroundsroot': '10',
            # maximal number of cuts separated per separation round (0: disable local separation)
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 100]
            'separating/maxcuts': '100',
            # maximal number of separated cuts at the root node (0: disable root node separation)
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 2000]
            'separating/maxcutsroot': '2000',
            # maximum age a cut can reach before it is deleted from the global cut pool, or -1 to keep all cuts
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 80]
            'separating/cutagelimit': '80',
            # separation frequency for the global cut pool (-1: disable global cut pool, 0: only separate pool at the root)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 10]
            'separating/poolfreq': '10',
            # parallel optimisation mode, 0: opportunistic or 1: deterministic.
            # [type: int, advanced: FALSE, range: [0,1], default: 1]
            'parallel/mode': '1',
            # the minimum number of threads used during parallel solve
            # [type: int, advanced: FALSE, range: [0,64], default: 1]
            'parallel/minnthreads': '1',
            # the maximum number of threads used during parallel solve
            # [type: int, advanced: FALSE, range: [0,64], default: 8]
            'parallel/maxnthreads': '8',
            # set different random seeds in each concurrent solver?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'concurrent/changeseeds': 'TRUE',
            # use different child selection rules in each concurrent solver?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'concurrent/changechildsel': 'TRUE',
            # should the concurrent solvers communicate global variable bound changes?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'concurrent/commvarbnds': 'TRUE',
            # should the problem be presolved before it is copied to the concurrent solvers?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'concurrent/presolvebefore': 'TRUE',
            # maximum number of solutions that will be shared in a one synchronization
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 5131912]
            'concurrent/initseed': '5131912',
            # initial frequency of synchronization with other threads
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 10]
            'concurrent/sync/freqinit': '10',
            # maximal frequency of synchronization with other threads
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 10]
            'concurrent/sync/freqmax': '10',
            # factor by which the frequency of synchronization is changed
            # [type: real, advanced: FALSE, range: [1,1.79769313486232e+308], default: 1.5]
            'concurrent/sync/freqfactor': '1.5',
            # when adapting the synchronization frequency this value is the targeted relative difference by which the absolute gap decreases per synchronization
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 0.001]
            'concurrent/sync/targetprogress': '0.001',
            # maximum number of solutions that will be shared in a single synchronization
            # [type: int, advanced: FALSE, range: [0,1000], default: 3]
            'concurrent/sync/maxnsols': '3',
            # maximum number of synchronizations before reading is enforced regardless of delay
            # [type: int, advanced: TRUE, range: [0,100], default: 7]
            'concurrent/sync/maxnsyncdelay': '7',
            # minimum delay before synchronization data is read
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 10]
            'concurrent/sync/minsyncdelay': '10',
            # how many of the N best solutions should be considered for synchronization?
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 10]
            'concurrent/sync/nbestsols': '10',
            # path prefix for parameter setting files of concurrent solvers
            # [type: string, advanced: FALSE, default: ""]
            'concurrent/paramsetprefix': '""',
            # default clock type (1: CPU user seconds, 2: wall clock time)
            # [type: int, advanced: FALSE, range: [1,2], default: 2]
            'timing/clocktype': '2',
            # is timing enabled?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'timing/enabled': 'TRUE',
            # belongs reading time to solving time?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'timing/reading': 'FALSE',
            # should clock checks of solving time be performed less frequently (note: time limit could be exceeded slightly)
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'timing/rareclockcheck': 'FALSE',
            # should timing for statistic output be performed?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'timing/statistictiming': 'TRUE',
            # name of the VBC tool output file, or - if no VBC tool output should be created
            # [type: string, advanced: FALSE, default: "-"]
            'visual/vbcfilename': '"-"',
            # name of the BAK tool output file, or - if no BAK tool output should be created
            # [type: string, advanced: FALSE, default: "-"]
            'visual/bakfilename': '"-"',
            # should the real solving time be used instead of a time step counter in visualization?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'visual/realtime': 'TRUE',
            # should the node where solutions are found be visualized?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'visual/dispsols': 'FALSE',
            # should lower bound information be visualized?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'visual/displb': 'FALSE',
            # should be output the external value of the objective?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'visual/objextern': 'TRUE',
            # should model constraints be marked as initial?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'reading/initialconss': 'TRUE',
            # should model constraints be subject to aging?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'reading/dynamicconss': 'TRUE',
            # should columns be added and removed dynamically to the LP?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'reading/dynamiccols': 'FALSE',
            # should rows be added and removed dynamically to the LP?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'reading/dynamicrows': 'FALSE',
            # should all constraints be written (including the redundant constraints)?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'write/allconss': 'FALSE',
            # should variables set to zero be printed?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'write/printzeros': 'TRUE',  # !! do not change this. Otherwise, we cannot parse the solution properly for pyomo
            # when writing a generic problem the index for the first variable should start with?
            # [type: int, advanced: FALSE, range: [0,1073741823], default: 0]
            'write/genericnamesoffset': '0',
            # frequency for separating cuts (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'constraints/nonlinear/sepafreq': '1',
            # frequency for propagating domains (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'constraints/nonlinear/propfreq': '1',
            # timing when constraint propagation should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS)
            # [type: int, advanced: TRUE, range: [1,15], default: 1]
            'constraints/nonlinear/proptiming': '1',
            # frequency for using all instead of only the useful constraints in separation, propagation and enforcement (-1: never, 0: only in first evaluation)
            # [type: int, advanced: TRUE, range: [-1,65534], default: 100]
            'constraints/nonlinear/eagerfreq': '100',
            # maximal number of presolving rounds the constraint handler participates in (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'constraints/nonlinear/maxprerounds': '-1',
            # should separation method be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/nonlinear/delaysepa': 'FALSE',
            # should propagation method be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/nonlinear/delayprop': 'FALSE',
            # timing mask of the constraint handler's presolving method (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 28]
            'constraints/nonlinear/presoltiming': '28',
            # maximal coef range of a cut (maximal coefficient divided by minimal coefficient) in order to be added to LP relaxation
            # [type: real, advanced: FALSE, range: [0,1e+20], default: 10000000]
            'constraints/nonlinear/cutmaxrange': '10000000',
            # whether to try to make solutions in check function feasible by shifting a linear variable (esp. useful if constraint was actually objective function)
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/nonlinear/linfeasshift': 'TRUE',
            # whether to assume that nonlinear functions in inequalities (<=) are convex (disables reformulation)
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/nonlinear/assumeconvex': 'FALSE',
            # limit on number of propagation rounds for a single constraint within one round of SCIP propagation
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 1]
            'constraints/nonlinear/maxproprounds': '1',
            # whether to reformulate expression graph
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/nonlinear/reformulate': 'TRUE',
            # maximal exponent where still expanding non-monomial polynomials in expression simplification
            # [type: int, advanced: TRUE, range: [1,2147483647], default: 2]
            'constraints/nonlinear/maxexpansionexponent': '2',
            # minimal required fraction of continuous variables in problem to use solution of NLP relaxation in root for separation
            # [type: real, advanced: FALSE, range: [0,2], default: 1]
            'constraints/nonlinear/sepanlpmincont': '1',
            # are cuts added during enforcement removable from the LP in the same node?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/nonlinear/enfocutsremovable': 'FALSE',
            # frequency for separating cuts (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'constraints/quadratic/sepafreq': '1',
            # frequency for propagating domains (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'constraints/quadratic/propfreq': '1',
            # timing when constraint propagation should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS)
            # [type: int, advanced: TRUE, range: [1,15], default: 1]
            'constraints/quadratic/proptiming': '1',
            # frequency for using all instead of only the useful constraints in separation, propagation and enforcement (-1: never, 0: only in first evaluation)
            # [type: int, advanced: TRUE, range: [-1,65534], default: 100]
            'constraints/quadratic/eagerfreq': '100',
            # maximal number of presolving rounds the constraint handler participates in (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'constraints/quadratic/maxprerounds': '-1',
            # should separation method be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/quadratic/delaysepa': 'FALSE',
            # should propagation method be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/quadratic/delayprop': 'FALSE',
            # timing mask of the constraint handler's presolving method (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 28]
            'constraints/quadratic/presoltiming': '28',
            # max. length of linear term which when multiplied with a binary variables is replaced by an auxiliary variable and a linear reformulation (0 to turn off)
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 2147483647]
            'constraints/quadratic/replacebinaryprod': '2147483647',
            # empathy level for using the AND constraint handler: 0 always avoid using AND; 1 use AND sometimes; 2 use AND as often as possible
            # [type: int, advanced: FALSE, range: [0,2], default: 2]
            'constraints/quadratic/empathy4and': '2',
            # whether to make non-varbound linear constraints added due to replacing products with binary variables initial
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/quadratic/binreforminitial': 'FALSE',
            # whether to consider only binary variables when replacing products with binary variables
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/quadratic/binreformbinaryonly': 'TRUE',
            # limit (as factor on 1/feastol) on coefficients and coef. range in linear constraints created when replacing products with binary variables
            # [type: real, advanced: TRUE, range: [0,1e+20], default: 0.0001]
            'constraints/quadratic/binreformmaxcoef': '0.0001',
            # maximal coef range of a cut (maximal coefficient divided by minimal coefficient) in order to be added to LP relaxation
            # [type: real, advanced: TRUE, range: [0,1e+20], default: 10000000]
            'constraints/quadratic/cutmaxrange': '10000000',
            # minimal curvature of constraints to be considered when returning bilinear terms to other plugins
            # [type: real, advanced: TRUE, range: [-1e+20,1e+20], default: 0.8]
            'constraints/quadratic/mincurvcollectbilinterms': '0.8',
            # whether linearizations of convex quadratic constraints should be added to cutpool in a solution found by some heuristic
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/quadratic/linearizeheursol': 'TRUE',
            # whether multivariate quadratic functions should be checked for convexity/concavity
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/quadratic/checkcurvature': 'TRUE',
            # whether constraint functions should be checked to be factorable
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/quadratic/checkfactorable': 'TRUE',
            # whether quadratic variables contained in a single constraint should be forced to be at their lower or upper bounds ('d'isable, change 't'ype, add 'b'ound disjunction)
            # [type: char, advanced: TRUE, range: {bdt}, default: t]
            'constraints/quadratic/checkquadvarlocks': 't',
            # whether to try to make solutions in check function feasible by shifting a linear variable (esp. useful if constraint was actually objective function)
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/quadratic/linfeasshift': 'TRUE',
            # maximum number of created constraints when disaggregating a quadratic constraint (<= 1: off)
            # [type: int, advanced: FALSE, range: [1,2147483647], default: 1]
            'constraints/quadratic/maxdisaggrsize': '1',
            # strategy how to merge independent blocks to reach maxdisaggrsize limit (keep 'b'iggest blocks and merge others; keep 's'mallest blocks and merge other; merge small blocks into bigger blocks to reach 'm'ean sizes)
            # [type: char, advanced: TRUE, range: {bms}, default: m]
            'constraints/quadratic/disaggrmergemethod': 'm',
            # limit on number of propagation rounds for a single constraint within one round of SCIP propagation during solve
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 1]
            'constraints/quadratic/maxproprounds': '1',
            # limit on number of propagation rounds for a single constraint within one round of SCIP presolve
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 10]
            'constraints/quadratic/maxproproundspresolve': '10',
            # maximum number of enforcement rounds before declaring the LP relaxation infeasible (-1: no limit); WARNING: changing this parameter might lead to incorrect results!
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'constraints/quadratic/enfolplimit': '-1',
            # minimal required fraction of continuous variables in problem to use solution of NLP relaxation in root for separation
            # [type: real, advanced: FALSE, range: [0,2], default: 1]
            'constraints/quadratic/sepanlpmincont': '1',
            # are cuts added during enforcement removable from the LP in the same node?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/quadratic/enfocutsremovable': 'FALSE',
            # should convex quadratics generated strong cuts via gauge function?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/quadratic/gaugecuts': 'FALSE',
            # how the interior point for gauge cuts should be computed: 'a'ny point per constraint, 'm'ost interior per constraint
            # [type: char, advanced: TRUE, range: {am}, default: a]
            'constraints/quadratic/interiorcomputation': 'a',
            # should convex quadratics generated strong cuts via projections?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/quadratic/projectedcuts': 'FALSE',
            # which score to give branching candidates: convexification 'g'ap, constraint 'v'iolation, 'c'entrality of variable value in domain
            # [type: char, advanced: TRUE, range: {cgv}, default: g]
            'constraints/quadratic/branchscoring': 'g',
            # should linear inequalities be consindered when computing the branching scores for bilinear terms?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/quadratic/usebilinineqbranch': 'FALSE',
            # minimal required score in order to use linear inequalities for tighter bilinear relaxations
            # [type: real, advanced: FALSE, range: [0,1], default: 0.01]
            'constraints/quadratic/minscorebilinterms': '0.01',
            # maximum number of separation rounds to use linear inequalities for the bilinear term relaxation in a local node
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 3]
            'constraints/quadratic/bilinineqmaxseparounds': '3',
            # enable nonlinear upgrading for constraint handler <quadratic>
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/nonlinear/upgrade/quadratic': 'TRUE',
            # priority of conflict handler <linear>
            # [type: int, advanced: TRUE, range: [-2147483648,2147483647], default: -1000000]
            'conflict/linear/priority': '-1000000',
            # frequency for separating cuts (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 0]
            'constraints/linear/sepafreq': '0',
            # frequency for propagating domains (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'constraints/linear/propfreq': '1',
            # timing when constraint propagation should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS)
            # [type: int, advanced: TRUE, range: [1,15], default: 1]
            'constraints/linear/proptiming': '1',
            # frequency for using all instead of only the useful constraints in separation, propagation and enforcement (-1: never, 0: only in first evaluation)
            # [type: int, advanced: TRUE, range: [-1,65534], default: 100]
            'constraints/linear/eagerfreq': '100',
            # maximal number of presolving rounds the constraint handler participates in (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'constraints/linear/maxprerounds': '-1',
            # should separation method be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/linear/delaysepa': 'FALSE',
            # should propagation method be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/linear/delayprop': 'FALSE',
            # timing mask of the constraint handler's presolving method (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 20]
            'constraints/linear/presoltiming': '20',
            # enable quadratic upgrading for constraint handler <linear>
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/quadratic/upgrade/linear': 'TRUE',
            # enable nonlinear upgrading for constraint handler <linear>
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/nonlinear/upgrade/linear': 'TRUE',
            # multiplier on propagation frequency, how often the bounds are tightened (-1: never, 0: only at root)
            # [type: int, advanced: TRUE, range: [-1,65534], default: 1]
            'constraints/linear/tightenboundsfreq': '1',
            # maximal number of separation rounds per node (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 5]
            'constraints/linear/maxrounds': '5',
            # maximal number of separation rounds per node in the root node (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'constraints/linear/maxroundsroot': '-1',
            # maximal number of cuts separated per separation round
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 50]
            'constraints/linear/maxsepacuts': '50',
            # maximal number of cuts separated per separation round in the root node
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 200]
            'constraints/linear/maxsepacutsroot': '200',
            # should pairwise constraint comparison be performed in presolving?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/linear/presolpairwise': 'TRUE',
            # should hash table be used for detecting redundant constraints in advance
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/linear/presolusehashing': 'TRUE',
            # number for minimal pairwise presolve comparisons
            # [type: int, advanced: TRUE, range: [1,2147483647], default: 200000]
            'constraints/linear/nmincomparisons': '200000',
            # minimal gain per minimal pairwise presolve comparisons to repeat pairwise comparison round
            # [type: real, advanced: TRUE, range: [0,1], default: 1e-06]
            'constraints/linear/mingainpernmincomparisons': '1e-06',
            # maximal allowed relative gain in maximum norm for constraint aggregation (0.0: disable constraint aggregation)
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0]
            'constraints/linear/maxaggrnormscale': '0',
            # maximum activity delta to run easy propagation on linear constraint (faster, but numerically less stable)
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 1000000]
            'constraints/linear/maxeasyactivitydelta': '1000000',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for separating knapsack cardinality cuts
            # [type: real, advanced: TRUE, range: [0,1], default: 0]
            'constraints/linear/maxcardbounddist': '0',
            # should all constraints be subject to cardinality cut generation instead of only the ones with non-zero dual value?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/linear/separateall': 'FALSE',
            # should presolving search for aggregations in equations
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/linear/aggregatevariables': 'TRUE',
            # should presolving try to simplify inequalities
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/linear/simplifyinequalities': 'TRUE',
            # should dual presolving steps be performed?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/linear/dualpresolving': 'TRUE',
            # should stuffing of singleton continuous variables be performed?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/linear/singletonstuffing': 'TRUE',
            # should single variable stuffing be performed, which tries to fulfill constraints using the cheapest variable?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/linear/singlevarstuffing': 'FALSE',
            # apply binaries sorting in decr. order of coeff abs value?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/linear/sortvars': 'TRUE',
            # should the violation for a constraint with side 0.0 be checked relative to 1.0 (FALSE) or to the maximum absolute value in the activity (TRUE)?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/linear/checkrelmaxabs': 'FALSE',
            # should presolving try to detect constraints parallel to the objective function defining an upper bound and prevent these constraints from entering the LP?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/linear/detectcutoffbound': 'TRUE',
            # should presolving try to detect constraints parallel to the objective function defining a lower bound and prevent these constraints from entering the LP?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/linear/detectlowerbound': 'TRUE',
            # should presolving try to detect subsets of constraints parallel to the objective function?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/linear/detectpartialobjective': 'TRUE',
            # should presolving and propagation try to improve bounds, detect infeasibility, and extract sub-constraints from ranged rows and equations?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/linear/rangedrowpropagation': 'TRUE',
            # should presolving and propagation extract sub-constraints from ranged rows and equations?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/linear/rangedrowartcons': 'TRUE',
            # maximum depth to apply ranged row propagation
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 2147483647]
            'constraints/linear/rangedrowmaxdepth': '2147483647',
            # frequency for applying ranged row propagation
            # [type: int, advanced: TRUE, range: [1,65534], default: 1]
            'constraints/linear/rangedrowfreq': '1',
            # should multi-aggregations only be performed if the constraint can be removed afterwards?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/linear/multaggrremove': 'FALSE',
            # maximum coefficient dynamism (ie. maxabsval / minabsval) for primal multiaggregation
            # [type: real, advanced: TRUE, range: [1,1.79769313486232e+308], default: 1000]
            'constraints/linear/maxmultaggrquot': '1000',
            # maximum coefficient dynamism (ie. maxabsval / minabsval) for dual multiaggregation
            # [type: real, advanced: TRUE, range: [1,1.79769313486232e+308], default: 1e+20]
            'constraints/linear/maxdualmultaggrquot': '1e+20',
            # should Cliques be extracted?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/linear/extractcliques': 'TRUE',
            # frequency for separating cuts (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'constraints/abspower/sepafreq': '1',
            # frequency for propagating domains (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'constraints/abspower/propfreq': '1',
            # timing when constraint propagation should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS)
            # [type: int, advanced: TRUE, range: [1,15], default: 15]
            'constraints/abspower/proptiming': '15',
            # frequency for using all instead of only the useful constraints in separation, propagation and enforcement (-1: never, 0: only in first evaluation)
            # [type: int, advanced: TRUE, range: [-1,65534], default: 100]
            'constraints/abspower/eagerfreq': '100',
            # maximal number of presolving rounds the constraint handler participates in (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'constraints/abspower/maxprerounds': '-1',
            # should separation method be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/abspower/delaysepa': 'FALSE',
            # should propagation method be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/abspower/delayprop': 'FALSE',
            # timing mask of the constraint handler's presolving method (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 12]
            'constraints/abspower/presoltiming': '12',
            # enable quadratic upgrading for constraint handler <abspower>
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/quadratic/upgrade/abspower': 'TRUE',
            # enable nonlinear upgrading for constraint handler <abspower>
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/nonlinear/upgrade/abspower': 'TRUE',
            # maximal coef range of a cut (maximal coefficient divided by minimal coefficient) in order to be added to LP relaxation
            # [type: real, advanced: FALSE, range: [0,1e+20], default: 10000000]
            'constraints/abspower/cutmaxrange': '10000000',
            # whether to project the reference point when linearizing an absolute power constraint in a convex region
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/abspower/projectrefpoint': 'TRUE',
            # how much to prefer branching on 0.0 when sign of variable is not fixed yet: 0 no preference, 1 prefer if LP solution will be cutoff in both child nodes, 2 prefer always, 3 ensure always
            # [type: int, advanced: FALSE, range: [0,3], default: 1]
            'constraints/abspower/preferzerobranch': '1',
            # whether to compute branching point such that the convexification error is minimized (after branching on 0.0)
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/abspower/branchminconverror': 'FALSE',
            # should variable bound constraints be added for derived variable bounds?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/abspower/addvarboundcons': 'TRUE',
            # whether to try to make solutions in check function feasible by shifting the linear variable z
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/abspower/linfeasshift': 'TRUE',
            # should dual presolve be applied?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/abspower/dualpresolve': 'TRUE',
            # whether to separate linearization cuts only in the variable bounds (does not affect enforcement)
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/abspower/sepainboundsonly': 'FALSE',
            # minimal required fraction of continuous variables in problem to use solution of NLP relaxation in root for separation
            # [type: real, advanced: FALSE, range: [0,2], default: 1]
            'constraints/abspower/sepanlpmincont': '1',
            # are cuts added during enforcement removable from the LP in the same node?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/abspower/enfocutsremovable': 'FALSE',
            # frequency for separating cuts (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'constraints/and/sepafreq': '1',
            # frequency for propagating domains (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'constraints/and/propfreq': '1',
            # timing when constraint propagation should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS)
            # [type: int, advanced: TRUE, range: [1,15], default: 1]
            'constraints/and/proptiming': '1',
            # frequency for using all instead of only the useful constraints in separation, propagation and enforcement (-1: never, 0: only in first evaluation)
            # [type: int, advanced: TRUE, range: [-1,65534], default: 100]
            'constraints/and/eagerfreq': '100',
            # maximal number of presolving rounds the constraint handler participates in (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'constraints/and/maxprerounds': '-1',
            # should separation method be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/and/delaysepa': 'FALSE',
            # should propagation method be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/and/delayprop': 'FALSE',
            # timing mask of the constraint handler's presolving method (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 20]
            'constraints/and/presoltiming': '20',
            # should pairwise constraint comparison be performed in presolving?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/and/presolpairwise': 'TRUE',
            # should hash table be used for detecting redundant constraints in advance
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/and/presolusehashing': 'TRUE',
            # should the AND-constraint get linearized and removed (in presolving)?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/and/linearize': 'FALSE',
            # should cuts be separated during LP enforcing?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/and/enforcecuts': 'TRUE',
            # should an aggregated linearization be used?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/and/aggrlinearization': 'FALSE',
            # should all binary resultant variables be upgraded to implicit binary variables?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/and/upgraderesultant': 'TRUE',
            # should dual presolving be performed?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/and/dualpresolving': 'TRUE',
            # enable nonlinear upgrading for constraint handler <and>
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/nonlinear/upgrade/and': 'TRUE',
            # frequency for separating cuts (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'constraints/benders/sepafreq': '-1',
            # frequency for propagating domains (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'constraints/benders/propfreq': '-1',
            # timing when constraint propagation should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS)
            # [type: int, advanced: TRUE, range: [1,15], default: 1]
            'constraints/benders/proptiming': '1',
            # frequency for using all instead of only the useful constraints in separation, propagation and enforcement (-1: never, 0: only in first evaluation)
            # [type: int, advanced: TRUE, range: [-1,65534], default: 100]
            'constraints/benders/eagerfreq': '100',
            # maximal number of presolving rounds the constraint handler participates in (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 0]
            'constraints/benders/maxprerounds': '0',
            # should separation method be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/benders/delaysepa': 'FALSE',
            # should propagation method be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/benders/delayprop': 'FALSE',
            # timing mask of the constraint handler's presolving method (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 4]
            'constraints/benders/presoltiming': '4',
            # is the Benders' decomposition constraint handler active?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/benders/active': 'FALSE',
            # frequency for separating cuts (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'constraints/benderslp/sepafreq': '-1',
            # frequency for propagating domains (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'constraints/benderslp/propfreq': '-1',
            # timing when constraint propagation should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS)
            # [type: int, advanced: TRUE, range: [1,15], default: 1]
            'constraints/benderslp/proptiming': '1',
            # frequency for using all instead of only the useful constraints in separation, propagation and enforcement (-1: never, 0: only in first evaluation)
            # [type: int, advanced: TRUE, range: [-1,65534], default: 100]
            'constraints/benderslp/eagerfreq': '100',
            # maximal number of presolving rounds the constraint handler participates in (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 0]
            'constraints/benderslp/maxprerounds': '0',
            # should separation method be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/benderslp/delaysepa': 'FALSE',
            # should propagation method be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/benderslp/delayprop': 'FALSE',
            # timing mask of the constraint handler's presolving method (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 28]
            'constraints/benderslp/presoltiming': '28',
            # depth at which Benders' decomposition cuts are generated from the LP solution (-1: always, 0: only at root)
            # [type: int, advanced: TRUE, range: [-1,65534], default: 0]
            'constraints/benderslp/maxdepth': '0',
            # the depth frequency for generating LP cuts after the max depth is reached (0: never, 1: all nodes, ...)
            # [type: int, advanced: TRUE, range: [0,65534], default: 0]
            'constraints/benderslp/depthfreq': '0',
            # the number of nodes processed without a dual bound improvement before enforcing the LP relaxation, 0: no stall count applied
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 100]
            'constraints/benderslp/stalllimit': '100',
            # after the root node, only iterlimit fractional LP solutions are used at each node to generate Benders' decomposition cuts.
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 100]
            'constraints/benderslp/iterlimit': '100',
            # is the Benders' decomposition LP cut constraint handler active?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/benderslp/active': 'FALSE',
            # frequency for separating cuts (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'constraints/bivariate/sepafreq': '1',
            # frequency for propagating domains (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'constraints/bivariate/propfreq': '1',
            # timing when constraint propagation should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS)
            # [type: int, advanced: TRUE, range: [1,15], default: 1]
            'constraints/bivariate/proptiming': '1',
            # frequency for using all instead of only the useful constraints in separation, propagation and enforcement (-1: never, 0: only in first evaluation)
            # [type: int, advanced: TRUE, range: [-1,65534], default: 100]
            'constraints/bivariate/eagerfreq': '100',
            # maximal number of presolving rounds the constraint handler participates in (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'constraints/bivariate/maxprerounds': '-1',
            # should separation method be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/bivariate/delaysepa': 'FALSE',
            # should propagation method be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/bivariate/delayprop': 'FALSE',
            # timing mask of the constraint handler's presolving method (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 4]
            'constraints/bivariate/presoltiming': '4',
            # enable quadratic upgrading for constraint handler <bivariate>
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/quadratic/upgrade/bivariate': 'FALSE',
            # enable nonlinear upgrading for constraint handler <bivariate>
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/nonlinear/upgrade/bivariate': 'FALSE',
            # maximal coef range of a cut (maximal coefficient divided by minimal coefficient) in order to be added to LP relaxation
            # [type: real, advanced: TRUE, range: [0,1e+20], default: 10000000]
            'constraints/bivariate/cutmaxrange': '10000000',
            # whether to try to make solutions in check function feasible by shifting a linear variable (esp. useful if constraint was actually objective function)
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/bivariate/linfeasshift': 'TRUE',
            # limit on number of propagation rounds for a single constraint within one round of SCIP propagation
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 1]
            'constraints/bivariate/maxproprounds': '1',
            # number of reference points in each direction where to compute linear support for envelope in LP initialization
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 3]
            'constraints/bivariate/ninitlprefpoints': '3',
            # are cuts added during enforcement removable from the LP in the same node?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/bivariate/enfocutsremovable': 'FALSE',
            # maximal percantage of continuous variables within a conflict
            # [type: real, advanced: FALSE, range: [0,1], default: 0.4]
            'conflict/bounddisjunction/continuousfrac': '0.4',
            # priority of conflict handler <bounddisjunction>
            # [type: int, advanced: TRUE, range: [-2147483648,2147483647], default: -3000000]
            'conflict/bounddisjunction/priority': '-3000000',
            # frequency for separating cuts (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'constraints/bounddisjunction/sepafreq': '-1',
            # frequency for propagating domains (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'constraints/bounddisjunction/propfreq': '1',
            # timing when constraint propagation should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS)
            # [type: int, advanced: TRUE, range: [1,15], default: 1]
            'constraints/bounddisjunction/proptiming': '1',
            # frequency for using all instead of only the useful constraints in separation, propagation and enforcement (-1: never, 0: only in first evaluation)
            # [type: int, advanced: TRUE, range: [-1,65534], default: 100]
            'constraints/bounddisjunction/eagerfreq': '100',
            # maximal number of presolving rounds the constraint handler participates in (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'constraints/bounddisjunction/maxprerounds': '-1',
            # should separation method be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/bounddisjunction/delaysepa': 'FALSE',
            # should propagation method be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/bounddisjunction/delayprop': 'FALSE',
            # timing mask of the constraint handler's presolving method (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 4]
            'constraints/bounddisjunction/presoltiming': '4',
            # enable quadratic upgrading for constraint handler <bounddisjunction>
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/quadratic/upgrade/bounddisjunction': 'TRUE',
            # frequency for separating cuts (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 10]
            'constraints/cardinality/sepafreq': '10',
            # frequency for propagating domains (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'constraints/cardinality/propfreq': '1',
            # timing when constraint propagation should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS)
            # [type: int, advanced: TRUE, range: [1,15], default: 1]
            'constraints/cardinality/proptiming': '1',
            # frequency for using all instead of only the useful constraints in separation, propagation and enforcement (-1: never, 0: only in first evaluation)
            # [type: int, advanced: TRUE, range: [-1,65534], default: 100]
            'constraints/cardinality/eagerfreq': '100',
            # maximal number of presolving rounds the constraint handler participates in (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'constraints/cardinality/maxprerounds': '-1',
            # should separation method be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/cardinality/delaysepa': 'FALSE',
            # should propagation method be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/cardinality/delayprop': 'FALSE',
            # timing mask of the constraint handler's presolving method (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 4]
            'constraints/cardinality/presoltiming': '4',
            # whether to use balanced instead of unbalanced branching
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/cardinality/branchbalanced': 'FALSE',
            # maximum depth for using balanced branching (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 20]
            'constraints/cardinality/balanceddepth': '20',
            # determines that balanced branching is only used if the branching cut off value w.r.t. the current LP solution is greater than a given value
            # [type: real, advanced: TRUE, range: [0.01,1.79769313486232e+308], default: 2]
            'constraints/cardinality/balancedcutoff': '2',
            # frequency for separating cuts (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'constraints/conjunction/sepafreq': '-1',
            # frequency for propagating domains (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'constraints/conjunction/propfreq': '-1',
            # timing when constraint propagation should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS)
            # [type: int, advanced: TRUE, range: [1,15], default: 1]
            'constraints/conjunction/proptiming': '1',
            # frequency for using all instead of only the useful constraints in separation, propagation and enforcement (-1: never, 0: only in first evaluation)
            # [type: int, advanced: TRUE, range: [-1,65534], default: 100]
            'constraints/conjunction/eagerfreq': '100',
            # maximal number of presolving rounds the constraint handler participates in (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'constraints/conjunction/maxprerounds': '-1',
            # should separation method be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/conjunction/delaysepa': 'FALSE',
            # should propagation method be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/conjunction/delayprop': 'FALSE',
            # timing mask of the constraint handler's presolving method (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 4]
            'constraints/conjunction/presoltiming': '4',
            # frequency for separating cuts (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'constraints/countsols/sepafreq': '-1',
            # frequency for propagating domains (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'constraints/countsols/propfreq': '-1',
            # timing when constraint propagation should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS)
            # [type: int, advanced: TRUE, range: [1,15], default: 1]
            'constraints/countsols/proptiming': '1',
            # frequency for using all instead of only the useful constraints in separation, propagation and enforcement (-1: never, 0: only in first evaluation)
            # [type: int, advanced: TRUE, range: [-1,65534], default: 100]
            'constraints/countsols/eagerfreq': '100',
            # maximal number of presolving rounds the constraint handler participates in (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 0]
            'constraints/countsols/maxprerounds': '0',
            # should separation method be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/countsols/delaysepa': 'FALSE',
            # should propagation method be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/countsols/delayprop': 'FALSE',
            # timing mask of the constraint handler's presolving method (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 28]
            'constraints/countsols/presoltiming': '28',
            # is the constraint handler active?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/countsols/active': 'FALSE',
            # should the sparse solution test be turned on?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/countsols/sparsetest': 'TRUE',
            # is it allowed to discard solutions?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/countsols/discardsols': 'TRUE',
            # should the solutions be collected?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/countsols/collect': 'FALSE',
            # counting stops, if the given number of solutions were found (-1: no limit)
            # [type: longint, advanced: FALSE, range: [-1,9223372036854775807], default: -1]
            'constraints/countsols/sollimit': '-1',
            # display activation status of display column <sols> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 0]
            'display/sols/active': '0',
            # display activation status of display column <feasST> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 0]
            'display/feasST/active': '0',
            # frequency for separating cuts (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'constraints/cumulative/sepafreq': '1',
            # frequency for propagating domains (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'constraints/cumulative/propfreq': '1',
            # timing when constraint propagation should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS)
            # [type: int, advanced: TRUE, range: [1,15], default: 1]
            'constraints/cumulative/proptiming': '1',
            # frequency for using all instead of only the useful constraints in separation, propagation and enforcement (-1: never, 0: only in first evaluation)
            # [type: int, advanced: TRUE, range: [-1,65534], default: 100]
            'constraints/cumulative/eagerfreq': '100',
            # maximal number of presolving rounds the constraint handler participates in (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'constraints/cumulative/maxprerounds': '-1',
            # should separation method be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/cumulative/delaysepa': 'FALSE',
            # should propagation method be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/cumulative/delayprop': 'FALSE',
            # timing mask of the constraint handler's presolving method (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 28]
            'constraints/cumulative/presoltiming': '28',
            # should time-table (core-times) propagator be used to infer bounds?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/cumulative/ttinfer': 'TRUE',
            # should edge-finding be used to detect an overload?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/cumulative/efcheck': 'FALSE',
            # should edge-finding be used to infer bounds?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/cumulative/efinfer': 'FALSE',
            # should edge-finding be executed?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/cumulative/useadjustedjobs': 'FALSE',
            # should time-table edge-finding be used to detect an overload?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/cumulative/ttefcheck': 'TRUE',
            # should time-table edge-finding be used to infer bounds?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/cumulative/ttefinfer': 'TRUE',
            # should the binary representation be used?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/cumulative/usebinvars': 'FALSE',
            # should cuts be added only locally?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/cumulative/localcuts': 'FALSE',
            # should covering cuts be added every node?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/cumulative/usecovercuts': 'TRUE',
            # should the cumulative constraint create cuts as knapsack constraints?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/cumulative/cutsasconss': 'TRUE',
            # shall old sepa algo be applied?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/cumulative/sepaold': 'TRUE',
            # should branching candidates be added to storage?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/cumulative/fillbranchcands': 'FALSE',
            # should dual presolving be applied?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/cumulative/dualpresolve': 'TRUE',
            # should coefficient tightening be applied?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/cumulative/coeftightening': 'FALSE',
            # should demands and capacity be normalized?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/cumulative/normalize': 'TRUE',
            # should pairwise constraint comparison be performed in presolving?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/cumulative/presolpairwise': 'TRUE',
            # extract disjunctive constraints?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/cumulative/disjunctive': 'TRUE',
            # number of branch-and-bound nodes to solve an independent cumulative constraint (-1: no limit)?
            # [type: longint, advanced: FALSE, range: [-1,9223372036854775807], default: 10000]
            'constraints/cumulative/maxnodes': '10000',
            # search for conflict set via maximal cliques to detect disjunctive constraints
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/cumulative/detectdisjunctive': 'TRUE',
            # search for conflict set via maximal cliques to detect variable bound constraints
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/cumulative/detectvarbounds': 'TRUE',
            # should bound widening be used during the conflict analysis?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/cumulative/usebdwidening': 'TRUE',
            # frequency for separating cuts (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'constraints/disjunction/sepafreq': '-1',
            # frequency for propagating domains (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'constraints/disjunction/propfreq': '-1',
            # timing when constraint propagation should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS)
            # [type: int, advanced: TRUE, range: [1,15], default: 1]
            'constraints/disjunction/proptiming': '1',
            # frequency for using all instead of only the useful constraints in separation, propagation and enforcement (-1: never, 0: only in first evaluation)
            # [type: int, advanced: TRUE, range: [-1,65534], default: 100]
            'constraints/disjunction/eagerfreq': '100',
            # maximal number of presolving rounds the constraint handler participates in (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'constraints/disjunction/maxprerounds': '-1',
            # should separation method be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/disjunction/delaysepa': 'FALSE',
            # should propagation method be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/disjunction/delayprop': 'FALSE',
            # timing mask of the constraint handler's presolving method (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 4]
            'constraints/disjunction/presoltiming': '4',
            # alawys perform branching if one of the constraints is violated, otherwise only if all integers are fixed
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/disjunction/alwaysbranch': 'TRUE',
            # frequency for separating cuts (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 10]
            'constraints/indicator/sepafreq': '10',
            # frequency for propagating domains (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'constraints/indicator/propfreq': '1',
            # timing when constraint propagation should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS)
            # [type: int, advanced: TRUE, range: [1,15], default: 1]
            'constraints/indicator/proptiming': '1',
            # frequency for using all instead of only the useful constraints in separation, propagation and enforcement (-1: never, 0: only in first evaluation)
            # [type: int, advanced: TRUE, range: [-1,65534], default: 100]
            'constraints/indicator/eagerfreq': '100',
            # maximal number of presolving rounds the constraint handler participates in (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'constraints/indicator/maxprerounds': '-1',
            # should separation method be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/indicator/delaysepa': 'FALSE',
            # should propagation method be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/indicator/delayprop': 'FALSE',
            # timing mask of the constraint handler's presolving method (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 4]
            'constraints/indicator/presoltiming': '4',
            # enable linear upgrading for constraint handler <indicator>
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/linear/upgrade/indicator': 'TRUE',
            # priority of conflict handler <indicatorconflict>
            # [type: int, advanced: TRUE, range: [-2147483648,2147483647], default: 200000]
            'conflict/indicatorconflict/priority': '200000',
            # Branch on indicator constraints in enforcing?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/indicator/branchindicators': 'FALSE',
            # Generate logicor constraints instead of cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/indicator/genlogicor': 'FALSE',
            # Add coupling constraints or rows if big-M is small enough?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/indicator/addcoupling': 'TRUE',
            # maximum coefficient for binary variable in coupling constraint
            # [type: real, advanced: TRUE, range: [0,1000000000], default: 10000]
            'constraints/indicator/maxcouplingvalue': '10000',
            # Add initial variable upper bound constraints, if 'addcoupling' is true?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/indicator/addcouplingcons': 'FALSE',
            # Should the coupling inequalities be separated dynamically?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/indicator/sepacouplingcuts': 'TRUE',
            # Allow to use local bounds in order to separate coupling inequalities?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/indicator/sepacouplinglocal': 'FALSE',
            # maximum coefficient for binary variable in separated coupling constraint
            # [type: real, advanced: TRUE, range: [0,1000000000], default: 10000]
            'constraints/indicator/sepacouplingvalue': '10000',
            # Separate cuts based on perspective formulation?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/indicator/sepaperspective': 'FALSE',
            # Allow to use local bounds in order to separate perspective cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/indicator/sepapersplocal': 'TRUE',
            # maximal number of separated non violated IISs, before separation is stopped
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 3]
            'constraints/indicator/maxsepanonviolated': '3',
            # Update bounds of original variables for separation?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/indicator/updatebounds': 'FALSE',
            # maximum estimated condition of the solution basis matrix of the alternative LP to be trustworthy (0.0 to disable check)
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0]
            'constraints/indicator/maxconditionaltlp': '0',
            # maximal number of cuts separated per separation round
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 100]
            'constraints/indicator/maxsepacuts': '100',
            # maximal number of cuts separated per separation round in the root node
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 2000]
            'constraints/indicator/maxsepacutsroot': '2000',
            # Remove indicator constraint if corresponding variable bound constraint has been added?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/indicator/removeindicators': 'FALSE',
            # Do not generate indicator constraint, but a bilinear constraint instead?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/indicator/generatebilinear': 'FALSE',
            # Scale slack variable coefficient at construction time?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/indicator/scaleslackvar': 'FALSE',
            # Try to make solutions feasible by setting indicator variables?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/indicator/trysolutions': 'TRUE',
            # In enforcing try to generate cuts (only if sepaalternativelp is true)?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/indicator/enforcecuts': 'FALSE',
            # Should dual reduction steps be performed?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/indicator/dualreductions': 'TRUE',
            # Add opposite inequality in nodes in which the binary variable has been fixed to 0?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/indicator/addopposite': 'FALSE',
            # Try to upgrade bounddisjunction conflicts by replacing slack variables?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/indicator/conflictsupgrade': 'FALSE',
            # fraction of binary variables that need to be fixed before restart occurs (in forcerestart)
            # [type: real, advanced: TRUE, range: [0,1], default: 0.9]
            'constraints/indicator/restartfrac': '0.9',
            # Collect other constraints to alternative LP?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/indicator/useotherconss': 'FALSE',
            # Use objective cut with current best solution to alternative LP?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/indicator/useobjectivecut': 'FALSE',
            # Try to construct a feasible solution from a cover?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/indicator/trysolfromcover': 'FALSE',
            # Try to upgrade linear constraints to indicator constraints?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/indicator/upgradelinear': 'FALSE',
            # Separate using the alternative LP?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/indicator/sepaalternativelp': 'FALSE',
            # Force restart if absolute gap is 1 or enough binary variables have been fixed?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/indicator/forcerestart': 'FALSE',
            # Decompose problem (do not generate linear constraint if all variables are continuous)?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/indicator/nolinconscont': 'FALSE',
            # frequency for separating cuts (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'constraints/integral/sepafreq': '-1',
            # frequency for propagating domains (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'constraints/integral/propfreq': '-1',
            # timing when constraint propagation should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS)
            # [type: int, advanced: TRUE, range: [1,15], default: 1]
            'constraints/integral/proptiming': '1',
            # frequency for using all instead of only the useful constraints in separation, propagation and enforcement (-1: never, 0: only in first evaluation)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'constraints/integral/eagerfreq': '-1',
            # maximal number of presolving rounds the constraint handler participates in (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 0]
            'constraints/integral/maxprerounds': '0',
            # should separation method be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/integral/delaysepa': 'FALSE',
            # should propagation method be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/integral/delayprop': 'FALSE',
            # timing mask of the constraint handler's presolving method (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 28]
            'constraints/integral/presoltiming': '28',
            # frequency for separating cuts (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 0]
            'constraints/knapsack/sepafreq': '0',
            # frequency for propagating domains (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'constraints/knapsack/propfreq': '1',
            # timing when constraint propagation should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS)
            # [type: int, advanced: TRUE, range: [1,15], default: 1]
            'constraints/knapsack/proptiming': '1',
            # frequency for using all instead of only the useful constraints in separation, propagation and enforcement (-1: never, 0: only in first evaluation)
            # [type: int, advanced: TRUE, range: [-1,65534], default: 100]
            'constraints/knapsack/eagerfreq': '100',
            # maximal number of presolving rounds the constraint handler participates in (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'constraints/knapsack/maxprerounds': '-1',
            # should separation method be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/knapsack/delaysepa': 'FALSE',
            # should propagation method be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/knapsack/delayprop': 'FALSE',
            # timing mask of the constraint handler's presolving method (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 28]
            'constraints/knapsack/presoltiming': '28',
            # enable linear upgrading for constraint handler <knapsack>
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/linear/upgrade/knapsack': 'TRUE',
            # multiplier on separation frequency, how often knapsack cuts are separated (-1: never, 0: only at root)
            # [type: int, advanced: TRUE, range: [-1,65534], default: 1]
            'constraints/knapsack/sepacardfreq': '1',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for separating knapsack cuts
            # [type: real, advanced: TRUE, range: [0,1], default: 0]
            'constraints/knapsack/maxcardbounddist': '0',
            # lower clique size limit for greedy clique extraction algorithm (relative to largest clique)
            # [type: real, advanced: TRUE, range: [0,1], default: 0.5]
            'constraints/knapsack/cliqueextractfactor': '0.5',
            # maximal number of separation rounds per node (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 5]
            'constraints/knapsack/maxrounds': '5',
            # maximal number of separation rounds per node in the root node (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'constraints/knapsack/maxroundsroot': '-1',
            # maximal number of cuts separated per separation round
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 50]
            'constraints/knapsack/maxsepacuts': '50',
            # maximal number of cuts separated per separation round in the root node
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 200]
            'constraints/knapsack/maxsepacutsroot': '200',
            # should disaggregation of knapsack constraints be allowed in preprocessing?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/knapsack/disaggregation': 'TRUE',
            # should presolving try to simplify knapsacks
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/knapsack/simplifyinequalities': 'TRUE',
            # should negated clique information be used in solving process
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/knapsack/negatedclique': 'TRUE',
            # should pairwise constraint comparison be performed in presolving?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/knapsack/presolpairwise': 'TRUE',
            # should hash table be used for detecting redundant constraints in advance
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/knapsack/presolusehashing': 'TRUE',
            # should dual presolving steps be performed?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/knapsack/dualpresolving': 'TRUE',
            # should GUB information be used for separation?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/knapsack/usegubs': 'FALSE',
            # should presolving try to detect constraints parallel to the objective function defining an upper bound and prevent these constraints from entering the LP?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/knapsack/detectcutoffbound': 'TRUE',
            # should presolving try to detect constraints parallel to the objective function defining a lower bound and prevent these constraints from entering the LP?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/knapsack/detectlowerbound': 'TRUE',
            # should clique partition information be updated when old partition seems outdated?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/knapsack/updatecliquepartitions': 'FALSE',
            # factor on the growth of global cliques to decide when to update a previous (negated) clique partition (used only if updatecliquepartitions is set to TRUE)
            # [type: real, advanced: TRUE, range: [1,10], default: 1.5]
            'constraints/knapsack/clqpartupdatefac': '1.5',
            # frequency for separating cuts (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'constraints/linking/sepafreq': '1',
            # frequency for propagating domains (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'constraints/linking/propfreq': '1',
            # timing when constraint propagation should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS)
            # [type: int, advanced: TRUE, range: [1,15], default: 1]
            'constraints/linking/proptiming': '1',
            # frequency for using all instead of only the useful constraints in separation, propagation and enforcement (-1: never, 0: only in first evaluation)
            # [type: int, advanced: TRUE, range: [-1,65534], default: 100]
            'constraints/linking/eagerfreq': '100',
            # maximal number of presolving rounds the constraint handler participates in (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'constraints/linking/maxprerounds': '-1',
            # should separation method be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/linking/delaysepa': 'FALSE',
            # should propagation method be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/linking/delayprop': 'FALSE',
            # timing mask of the constraint handler's presolving method (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 8]
            'constraints/linking/presoltiming': '8',
            # this constraint will not propagate or separate, linear and setppc are used?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/linking/linearize': 'FALSE',
            # priority of conflict handler <logicor>
            # [type: int, advanced: TRUE, range: [-2147483648,2147483647], default: 800000]
            'conflict/logicor/priority': '800000',
            # frequency for separating cuts (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 0]
            'constraints/logicor/sepafreq': '0',
            # frequency for propagating domains (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'constraints/logicor/propfreq': '1',
            # timing when constraint propagation should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS)
            # [type: int, advanced: TRUE, range: [1,15], default: 1]
            'constraints/logicor/proptiming': '1',
            # frequency for using all instead of only the useful constraints in separation, propagation and enforcement (-1: never, 0: only in first evaluation)
            # [type: int, advanced: TRUE, range: [-1,65534], default: 100]
            'constraints/logicor/eagerfreq': '100',
            # maximal number of presolving rounds the constraint handler participates in (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'constraints/logicor/maxprerounds': '-1',
            # should separation method be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/logicor/delaysepa': 'FALSE',
            # should propagation method be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/logicor/delayprop': 'FALSE',
            # timing mask of the constraint handler's presolving method (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 28]
            'constraints/logicor/presoltiming': '28',
            # enable linear upgrading for constraint handler <logicor>
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/linear/upgrade/logicor': 'TRUE',
            # should pairwise constraint comparison be performed in presolving?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/logicor/presolpairwise': 'TRUE',
            # should hash table be used for detecting redundant constraints in advance
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/logicor/presolusehashing': 'TRUE',
            # should dual presolving steps be performed?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/logicor/dualpresolving': 'TRUE',
            # should negated clique information be used in presolving
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/logicor/negatedclique': 'TRUE',
            # should implications/cliques be used in presolving
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/logicor/implications': 'TRUE',
            # should pairwise constraint comparison try to strengthen constraints by removing superflous non-zeros?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/logicor/strengthen': 'TRUE',
            # frequency for separating cuts (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 0]
            'constraints/or/sepafreq': '0',
            # frequency for propagating domains (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'constraints/or/propfreq': '1',
            # timing when constraint propagation should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS)
            # [type: int, advanced: TRUE, range: [1,15], default: 1]
            'constraints/or/proptiming': '1',
            # frequency for using all instead of only the useful constraints in separation, propagation and enforcement (-1: never, 0: only in first evaluation)
            # [type: int, advanced: TRUE, range: [-1,65534], default: 100]
            'constraints/or/eagerfreq': '100',
            # maximal number of presolving rounds the constraint handler participates in (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'constraints/or/maxprerounds': '-1',
            # should separation method be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/or/delaysepa': 'FALSE',
            # should propagation method be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/or/delayprop': 'FALSE',
            # timing mask of the constraint handler's presolving method (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 8]
            'constraints/or/presoltiming': '8',
            # frequency for separating cuts (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 5]
            'constraints/orbisack/sepafreq': '5',
            # frequency for propagating domains (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 5]
            'constraints/orbisack/propfreq': '5',
            # timing when constraint propagation should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS)
            # [type: int, advanced: TRUE, range: [1,15], default: 1]
            'constraints/orbisack/proptiming': '1',
            # frequency for using all instead of only the useful constraints in separation, propagation and enforcement (-1: never, 0: only in first evaluation)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'constraints/orbisack/eagerfreq': '-1',
            # maximal number of presolving rounds the constraint handler participates in (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'constraints/orbisack/maxprerounds': '-1',
            # should separation method be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/orbisack/delaysepa': 'FALSE',
            # should propagation method be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/orbisack/delayprop': 'FALSE',
            # timing mask of the constraint handler's presolving method (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 16]
            'constraints/orbisack/presoltiming': '16',
            # Separate cover inequalities for orbisacks?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/orbisack/coverseparation': 'TRUE',
            # Separate orbisack inequalities?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/orbisack/orbiSeparation': 'FALSE',
            # Maximum size of coefficients for orbisack inequalities
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 1000000]
            'constraints/orbisack/coeffbound': '1000000',
            # Upgrade orbisack constraints to packing/partioning orbisacks?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/orbisack/checkpporbisack': 'TRUE',
            # Whether orbisack constraints should be forced to be copied to sub SCIPs.
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/orbisack/forceconscopy': 'FALSE',
            # frequency for separating cuts (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'constraints/orbitope/sepafreq': '-1',
            # frequency for propagating domains (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'constraints/orbitope/propfreq': '1',
            # timing when constraint propagation should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS)
            # [type: int, advanced: TRUE, range: [1,15], default: 1]
            'constraints/orbitope/proptiming': '1',
            # frequency for using all instead of only the useful constraints in separation, propagation and enforcement (-1: never, 0: only in first evaluation)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'constraints/orbitope/eagerfreq': '-1',
            # maximal number of presolving rounds the constraint handler participates in (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'constraints/orbitope/maxprerounds': '-1',
            # should separation method be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/orbitope/delaysepa': 'FALSE',
            # should propagation method be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/orbitope/delayprop': 'FALSE',
            # timing mask of the constraint handler's presolving method (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 8]
            'constraints/orbitope/presoltiming': '8',
            # Strengthen orbitope constraints to packing/partioning orbitopes?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/orbitope/checkpporbitope': 'TRUE',
            # Whether we separate inequalities for full orbitopes?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/orbitope/sepafullorbitope': 'FALSE',
            # Whether we use a dynamic version of the propagation routine.
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/orbitope/usedynamicprop': 'TRUE',
            # Whether orbitope constraints should be forced to be copied to sub SCIPs.
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/orbitope/forceconscopy': 'FALSE',
            # frequency for separating cuts (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'constraints/pseudoboolean/sepafreq': '-1',
            # frequency for propagating domains (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'constraints/pseudoboolean/propfreq': '-1',
            # timing when constraint propagation should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS)
            # [type: int, advanced: TRUE, range: [1,15], default: 1]
            'constraints/pseudoboolean/proptiming': '1',
            # frequency for using all instead of only the useful constraints in separation, propagation and enforcement (-1: never, 0: only in first evaluation)
            # [type: int, advanced: TRUE, range: [-1,65534], default: 100]
            'constraints/pseudoboolean/eagerfreq': '100',
            # maximal number of presolving rounds the constraint handler participates in (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'constraints/pseudoboolean/maxprerounds': '-1',
            # should separation method be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/pseudoboolean/delaysepa': 'FALSE',
            # should propagation method be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/pseudoboolean/delayprop': 'FALSE',
            # timing mask of the constraint handler's presolving method (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 8]
            'constraints/pseudoboolean/presoltiming': '8',
            # decompose all normal pseudo boolean constraint into a "linear" constraint and "and" constraints
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/pseudoboolean/decomposenormal': 'FALSE',
            # decompose all indicator pseudo boolean constraint into a "linear" constraint and "and" constraints
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/pseudoboolean/decomposeindicator': 'TRUE',
            # should the nonlinear constraints be separated during LP processing?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/pseudoboolean/nlcseparate': 'TRUE',
            # should the nonlinear constraints be propagated during node processing?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/pseudoboolean/nlcpropagate': 'TRUE',
            # should the nonlinear constraints be removable?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/pseudoboolean/nlcremovable': 'TRUE',
            # priority of conflict handler <setppc>
            # [type: int, advanced: TRUE, range: [-2147483648,2147483647], default: 700000]
            'conflict/setppc/priority': '700000',
            # frequency for separating cuts (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 0]
            'constraints/setppc/sepafreq': '0',
            # frequency for propagating domains (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'constraints/setppc/propfreq': '1',
            # timing when constraint propagation should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS)
            # [type: int, advanced: TRUE, range: [1,15], default: 1]
            'constraints/setppc/proptiming': '1',
            # frequency for using all instead of only the useful constraints in separation, propagation and enforcement (-1: never, 0: only in first evaluation)
            # [type: int, advanced: TRUE, range: [-1,65534], default: 100]
            'constraints/setppc/eagerfreq': '100',
            # maximal number of presolving rounds the constraint handler participates in (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'constraints/setppc/maxprerounds': '-1',
            # should separation method be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/setppc/delaysepa': 'FALSE',
            # should propagation method be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/setppc/delayprop': 'FALSE',
            # timing mask of the constraint handler's presolving method (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 28]
            'constraints/setppc/presoltiming': '28',
            # enable linear upgrading for constraint handler <setppc>
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/linear/upgrade/setppc': 'TRUE',
            # enable quadratic upgrading for constraint handler <setppc>
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/quadratic/upgrade/setppc': 'TRUE',
            # number of children created in pseudo branching (0: disable pseudo branching)
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 2]
            'constraints/setppc/npseudobranches': '2',
            # should pairwise constraint comparison be performed in presolving?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/setppc/presolpairwise': 'TRUE',
            # should hash table be used for detecting redundant constraints in advance
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/setppc/presolusehashing': 'TRUE',
            # should dual presolving steps be performed?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/setppc/dualpresolving': 'TRUE',
            #  should we try to lift variables into other clique constraints, fix variables, aggregate them, and also shrink the amount of variables in clique constraints
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/setppc/cliquelifting': 'FALSE',
            # should we try to generate extra cliques out of all binary variables to maybe fasten redundant constraint detection
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/setppc/addvariablesascliques': 'FALSE',
            # should we try to shrink the number of variables in a clique constraints, by replacing more than one variable by only one
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/setppc/cliqueshrinking': 'TRUE',
            # frequency for separating cuts (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'constraints/soc/sepafreq': '1',
            # frequency for propagating domains (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'constraints/soc/propfreq': '1',
            # timing when constraint propagation should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS)
            # [type: int, advanced: TRUE, range: [1,15], default: 1]
            'constraints/soc/proptiming': '1',
            # frequency for using all instead of only the useful constraints in separation, propagation and enforcement (-1: never, 0: only in first evaluation)
            # [type: int, advanced: TRUE, range: [-1,65534], default: 100]
            'constraints/soc/eagerfreq': '100',
            # maximal number of presolving rounds the constraint handler participates in (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'constraints/soc/maxprerounds': '-1',
            # should separation method be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/soc/delaysepa': 'FALSE',
            # should propagation method be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/soc/delayprop': 'FALSE',
            # timing mask of the constraint handler's presolving method (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 28]
            'constraints/soc/presoltiming': '28',
            # enable quadratic upgrading for constraint handler <soc>
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/quadratic/upgrade/soc': 'TRUE',
            # whether the reference point of a cut should be projected onto the feasible set of the SOC constraint
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/soc/projectpoint': 'FALSE',
            # number of auxiliary variables to use when creating a linear outer approx. of a SOC3 constraint; 0 to turn off
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 0]
            'constraints/soc/nauxvars': '0',
            # whether the Glineur Outer Approximation should be used instead of Ben-Tal Nemirovski
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/soc/glineur': 'TRUE',
            # whether to sparsify cuts
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/soc/sparsify': 'FALSE',
            # maximal loss in cut efficacy by sparsification
            # [type: real, advanced: TRUE, range: [0,1], default: 0.2]
            'constraints/soc/sparsifymaxloss': '0.2',
            # growth rate of maximal allowed nonzeros in cuts in sparsification
            # [type: real, advanced: TRUE, range: [1.000001,1e+20], default: 1.3]
            'constraints/soc/sparsifynzgrowth': '1.3',
            # whether to try to make solutions feasible in check by shifting the variable on the right hand side
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/soc/linfeasshift': 'TRUE',
            # which formulation to use when adding a SOC constraint to the NLP (a: automatic, q: nonconvex quadratic form, s: convex sqrt form, e: convex exponential-sqrt form, d: convex division form)
            # [type: char, advanced: FALSE, range: {aqsed}, default: a]
            'constraints/soc/nlpform': 'a',
            # minimal required fraction of continuous variables in problem to use solution of NLP relaxation in root for separation
            # [type: real, advanced: FALSE, range: [0,2], default: 1]
            'constraints/soc/sepanlpmincont': '1',
            # are cuts added during enforcement removable from the LP in the same node?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/soc/enfocutsremovable': 'FALSE',
            # try to upgrade more general quadratics to soc?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/soc/generalsocupgrade': 'TRUE',
            # try to completely disaggregate soc?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/soc/disaggregate': 'TRUE',
            # frequency for separating cuts (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 10]
            'constraints/SOS1/sepafreq': '10',
            # frequency for propagating domains (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'constraints/SOS1/propfreq': '1',
            # timing when constraint propagation should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS)
            # [type: int, advanced: TRUE, range: [1,15], default: 1]
            'constraints/SOS1/proptiming': '1',
            # frequency for using all instead of only the useful constraints in separation, propagation and enforcement (-1: never, 0: only in first evaluation)
            # [type: int, advanced: TRUE, range: [-1,65534], default: 100]
            'constraints/SOS1/eagerfreq': '100',
            # maximal number of presolving rounds the constraint handler participates in (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'constraints/SOS1/maxprerounds': '-1',
            # should separation method be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/SOS1/delaysepa': 'FALSE',
            # should propagation method be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/SOS1/delayprop': 'FALSE',
            # timing mask of the constraint handler's presolving method (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 8]
            'constraints/SOS1/presoltiming': '8',
            # do not create an adjacency matrix if number of SOS1 variables is larger than predefined value (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 10000]
            'constraints/SOS1/maxsosadjacency': '10000',
            # maximal number of extensions that will be computed for each SOS1 constraint  (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 1]
            'constraints/SOS1/maxextensions': '1',
            # maximal number of bound tightening rounds per presolving round (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 5]
            'constraints/SOS1/maxtightenbds': '5',
            # if TRUE then perform implication graph analysis (might add additional SOS1 constraints)
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/SOS1/perfimplanalysis': 'FALSE',
            # number of recursive calls of implication graph analysis (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'constraints/SOS1/depthimplanalysis': '-1',
            # whether to use conflict graph propagation
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/SOS1/conflictprop': 'TRUE',
            # whether to use implication graph propagation
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/SOS1/implprop': 'TRUE',
            # whether to use SOS1 constraint propagation
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/SOS1/sosconsprop': 'FALSE',
            # which branching rule should be applied ? ('n': neighborhood, 'b': bipartite, 's': SOS1/clique) (note: in some cases an automatic switching to SOS1 branching is possible)
            # [type: char, advanced: TRUE, range: {nbs}, default: n]
            'constraints/SOS1/branchingrule': 'n',
            # if TRUE then automatically switch to SOS1 branching if the SOS1 constraints do not overlap
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/SOS1/autosos1branch': 'TRUE',
            # if neighborhood branching is used, then fix the branching variable (if positive in sign) to the value of the feasibility tolerance
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/SOS1/fixnonzero': 'FALSE',
            # if TRUE then add complementarity constraints to the branching nodes (can be used in combination with neighborhood or bipartite branching)
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/SOS1/addcomps': 'FALSE',
            # maximal number of complementarity constraints added per branching node (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'constraints/SOS1/maxaddcomps': '-1',
            # minimal feasibility value for complementarity constraints in order to be added to the branching node
            # [type: real, advanced: TRUE, range: [-1.79769313486232e+308,1.79769313486232e+308], default: -0.6]
            'constraints/SOS1/addcompsfeas': '-0.6',
            # minimal feasibility value for bound inequalities in order to be added to the branching node
            # [type: real, advanced: TRUE, range: [-1.79769313486232e+308,1.79769313486232e+308], default: 1]
            'constraints/SOS1/addbdsfeas': '1',
            # should added complementarity constraints be extended to SOS1 constraints to get tighter bound inequalities
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/SOS1/addextendedbds': 'TRUE',
            # Use SOS1 branching in enforcing (otherwise leave decision to branching rules)? This value can only be set to false if all SOS1 variables are binary
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/SOS1/branchsos': 'TRUE',
            # Branch on SOS constraint with most number of nonzeros?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/SOS1/branchnonzeros': 'FALSE',

            '# Branch on SOS cons. with highest nonzero-variable weight for branching (needs branchnonzeros': 'false)?',
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/SOS1/branchweight': 'FALSE',
            # only add complementarity constraints to branching nodes for predefined depth (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 30]
            'constraints/SOS1/addcompsdepth': '30',
            # maximal number of strong branching rounds to perform for each node (-1: auto); only available for neighborhood and bipartite branching
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 0]
            'constraints/SOS1/nstrongrounds': '0',
            # maximal number LP iterations to perform for each strong branching round (-2: auto, -1: no limit)
            # [type: int, advanced: TRUE, range: [-2,2147483647], default: 10000]
            'constraints/SOS1/nstrongiter': '10000',
            # if TRUE separate bound inequalities from initial SOS1 constraints
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/SOS1/boundcutsfromsos1': 'FALSE',
            # if TRUE separate bound inequalities from the conflict graph
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/SOS1/boundcutsfromgraph': 'TRUE',
            # if TRUE then automatically switch to separating initial SOS1 constraints if the SOS1 constraints do not overlap
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/SOS1/autocutsfromsos1': 'TRUE',
            # frequency for separating bound cuts; zero means to separate only in the root node
            # [type: int, advanced: TRUE, range: [-1,65534], default: 10]
            'constraints/SOS1/boundcutsfreq': '10',
            # node depth of separating bound cuts (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 40]
            'constraints/SOS1/boundcutsdepth': '40',
            # maximal number of bound cuts separated per branching node
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 50]
            'constraints/SOS1/maxboundcuts': '50',
            # maximal number of bound cuts separated per iteration in the root node
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 150]
            'constraints/SOS1/maxboundcutsroot': '150',
            # if TRUE then bound cuts are strengthened in case bound variables are available
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/SOS1/strthenboundcuts': 'TRUE',
            # frequency for separating implied bound cuts; zero means to separate only in the root node
            # [type: int, advanced: TRUE, range: [-1,65534], default: 0]
            'constraints/SOS1/implcutsfreq': '0',
            # node depth of separating implied bound cuts (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 40]
            'constraints/SOS1/implcutsdepth': '40',
            # maximal number of implied bound cuts separated per branching node
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 50]
            'constraints/SOS1/maximplcuts': '50',
            # maximal number of implied bound cuts separated per iteration in the root node
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 150]
            'constraints/SOS1/maximplcutsroot': '150',
            # frequency for separating cuts (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 0]
            'constraints/SOS2/sepafreq': '0',
            # frequency for propagating domains (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'constraints/SOS2/propfreq': '1',
            # timing when constraint propagation should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS)
            # [type: int, advanced: TRUE, range: [1,15], default: 1]
            'constraints/SOS2/proptiming': '1',
            # frequency for using all instead of only the useful constraints in separation, propagation and enforcement (-1: never, 0: only in first evaluation)
            # [type: int, advanced: TRUE, range: [-1,65534], default: 100]
            'constraints/SOS2/eagerfreq': '100',
            # maximal number of presolving rounds the constraint handler participates in (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'constraints/SOS2/maxprerounds': '-1',
            # should separation method be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/SOS2/delaysepa': 'FALSE',
            # should propagation method be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/SOS2/delayprop': 'FALSE',
            # timing mask of the constraint handler's presolving method (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 4]
            'constraints/SOS2/presoltiming': '4',
            # frequency for separating cuts (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'constraints/superindicator/sepafreq': '-1',
            # frequency for propagating domains (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'constraints/superindicator/propfreq': '1',
            # timing when constraint propagation should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS)
            # [type: int, advanced: TRUE, range: [1,15], default: 1]
            'constraints/superindicator/proptiming': '1',
            # frequency for using all instead of only the useful constraints in separation, propagation and enforcement (-1: never, 0: only in first evaluation)
            # [type: int, advanced: TRUE, range: [-1,65534], default: 100]
            'constraints/superindicator/eagerfreq': '100',
            # maximal number of presolving rounds the constraint handler participates in (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'constraints/superindicator/maxprerounds': '-1',
            # should separation method be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/superindicator/delaysepa': 'FALSE',
            # should propagation method be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/superindicator/delayprop': 'FALSE',
            # timing mask of the constraint handler's presolving method (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 8]
            'constraints/superindicator/presoltiming': '8',
            # should type of slack constraint be checked when creating superindicator constraint?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/superindicator/checkslacktype': 'TRUE',
            # maximum big-M coefficient of binary variable in upgrade to a linear constraint (relative to smallest coefficient)
            # [type: real, advanced: TRUE, range: [0,1e+15], default: 10000]
            'constraints/superindicator/maxupgdcoeflinear': '10000',
            # priority for upgrading to an indicator constraint (-1: never)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 1]
            'constraints/superindicator/upgdprioindicator': '1',
            # priority for upgrading to an indicator constraint (-1: never)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 2]
            'constraints/superindicator/upgdpriolinear': '2',
            # frequency for separating cuts (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 5]
            'constraints/symresack/sepafreq': '5',
            # frequency for propagating domains (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 5]
            'constraints/symresack/propfreq': '5',
            # timing when constraint propagation should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS)
            # [type: int, advanced: TRUE, range: [1,15], default: 1]
            'constraints/symresack/proptiming': '1',
            # frequency for using all instead of only the useful constraints in separation, propagation and enforcement (-1: never, 0: only in first evaluation)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'constraints/symresack/eagerfreq': '-1',
            # maximal number of presolving rounds the constraint handler participates in (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'constraints/symresack/maxprerounds': '-1',
            # should separation method be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/symresack/delaysepa': 'FALSE',
            # should propagation method be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/symresack/delayprop': 'FALSE',
            # timing mask of the constraint handler's presolving method (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 16]
            'constraints/symresack/presoltiming': '16',
            # Upgrade symresack constraints to packing/partioning symresacks?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/symresack/ppsymresack': 'TRUE',
            # Check whether permutation is monotone when upgrading to packing/partioning symresacks?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/symresack/checkmonotonicity': 'TRUE',
            # Whether symresack constraints should be forced to be copied to sub SCIPs.
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/symresack/forceconscopy': 'FALSE',
            # frequency for separating cuts (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 0]
            'constraints/varbound/sepafreq': '0',
            # frequency for propagating domains (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'constraints/varbound/propfreq': '1',
            # timing when constraint propagation should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS)
            # [type: int, advanced: TRUE, range: [1,15], default: 1]
            'constraints/varbound/proptiming': '1',
            # frequency for using all instead of only the useful constraints in separation, propagation and enforcement (-1: never, 0: only in first evaluation)
            # [type: int, advanced: TRUE, range: [-1,65534], default: 100]
            'constraints/varbound/eagerfreq': '100',
            # maximal number of presolving rounds the constraint handler participates in (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'constraints/varbound/maxprerounds': '-1',
            # should separation method be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/varbound/delaysepa': 'FALSE',
            # should propagation method be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/varbound/delayprop': 'FALSE',
            # timing mask of the constraint handler's presolving method (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 12]
            'constraints/varbound/presoltiming': '12',
            # enable linear upgrading for constraint handler <varbound>
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/linear/upgrade/varbound': 'TRUE',
            # should pairwise constraint comparison be performed in presolving?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/varbound/presolpairwise': 'TRUE',
            # maximum coefficient in varbound constraint to be added as a row into LP
            # [type: real, advanced: TRUE, range: [0,1e+20], default: 1000000000]
            'constraints/varbound/maxlpcoef': '1000000000',
            # should bound widening be used in conflict analysis?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/varbound/usebdwidening': 'TRUE',
            # frequency for separating cuts (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 0]
            'constraints/xor/sepafreq': '0',
            # frequency for propagating domains (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'constraints/xor/propfreq': '1',
            # timing when constraint propagation should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS)
            # [type: int, advanced: TRUE, range: [1,15], default: 1]
            'constraints/xor/proptiming': '1',
            # frequency for using all instead of only the useful constraints in separation, propagation and enforcement (-1: never, 0: only in first evaluation)
            # [type: int, advanced: TRUE, range: [-1,65534], default: 100]
            'constraints/xor/eagerfreq': '100',
            # maximal number of presolving rounds the constraint handler participates in (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'constraints/xor/maxprerounds': '-1',
            # should separation method be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/xor/delaysepa': 'FALSE',
            # should propagation method be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/xor/delayprop': 'FALSE',
            # timing mask of the constraint handler's presolving method (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 28]
            'constraints/xor/presoltiming': '28',
            # enable linear upgrading for constraint handler <xor>
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/linear/upgrade/xor': 'TRUE',
            # should pairwise constraint comparison be performed in presolving?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/xor/presolpairwise': 'TRUE',
            # should hash table be used for detecting redundant constraints in advance?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/xor/presolusehashing': 'TRUE',
            # should the extended formulation be added in presolving?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/xor/addextendedform': 'FALSE',
            # should the extended flow formulation be added (nonsymmetric formulation otherwise)?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/xor/addflowextended': 'FALSE',
            # should parity inequalities be separated?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/xor/separateparity': 'FALSE',
            # frequency for applying the Gauss propagator
            # [type: int, advanced: TRUE, range: [-1,65534], default: 5]
            'constraints/xor/gausspropfreq': '5',
            # frequency for separating cuts (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'constraints/components/sepafreq': '-1',
            # frequency for propagating domains (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'constraints/components/propfreq': '1',
            # timing when constraint propagation should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS)
            # [type: int, advanced: TRUE, range: [1,15], default: 1]
            'constraints/components/proptiming': '1',
            # frequency for using all instead of only the useful constraints in separation, propagation and enforcement (-1: never, 0: only in first evaluation)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'constraints/components/eagerfreq': '-1',
            # maximal number of presolving rounds the constraint handler participates in (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'constraints/components/maxprerounds': '-1',
            # should separation method be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'constraints/components/delaysepa': 'FALSE',
            # should propagation method be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'constraints/components/delayprop': 'TRUE',
            # timing mask of the constraint handler's presolving method (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 32]
            'constraints/components/presoltiming': '32',
            # maximum depth of a node to run components detection (-1: disable component detection during solving)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'constraints/components/maxdepth': '-1',
            # maximum number of integer (or binary) variables to solve a subproblem during presolving (-1: unlimited)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 500]
            'constraints/components/maxintvars': '500',
            # minimum absolute size (in terms of variables) to solve a component individually during branch-and-bound
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 50]
            'constraints/components/minsize': '50',
            # minimum relative size (in terms of variables) to solve a component individually during branch-and-bound
            # [type: real, advanced: TRUE, range: [0,1], default: 0.1]
            'constraints/components/minrelsize': '0.1',
            # maximum number of nodes to be solved in subproblems during presolving
            # [type: longint, advanced: FALSE, range: [-1,9223372036854775807], default: 10000]
            'constraints/components/nodelimit': '10000',
            # the weight of an integer variable compared to binary variables
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 1]
            'constraints/components/intfactor': '1',
            # factor to increase the feasibility tolerance of the main SCIP in all sub-SCIPs, default value 1.0
            # [type: real, advanced: TRUE, range: [0,1000000], default: 1]
            'constraints/components/feastolfactor': '1',
            # should possible "and" constraint be linearized when writing the mps file?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'reading/mpsreader/linearize-and-constraints': 'TRUE',
            # should an aggregated linearization for and constraints be used?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'reading/mpsreader/aggrlinearization-ands': 'TRUE',
            # should possible "and" constraint be linearized when writing the lp file?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'reading/lpreader/linearize-and-constraints': 'TRUE',
            # should an aggregated linearization for and constraints be used?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'reading/lpreader/aggrlinearization-ands': 'TRUE',
            # have integer variables no upper bound by default (depending on GAMS version)?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'reading/gmsreader/freeints': 'FALSE',
            # shall characters '#', '*', '+', '/', and '-' in variable and constraint names be replaced by '_'?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'reading/gmsreader/replaceforbiddenchars': 'FALSE',
            # default M value for big-M reformulation of indicator constraints in case no bound on slack variable is given
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 1000000]
            'reading/gmsreader/bigmdefault': '1000000',
            # which reformulation to use for indicator constraints: 'b'ig-M, 's'os1
            # [type: char, advanced: FALSE, range: {bs}, default: s]
            'reading/gmsreader/indicatorreform': 's',
            # is it allowed to use the gams function signpower(x,a)?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'reading/gmsreader/signpower': 'FALSE',
            # should the current directory be changed to that of the ZIMPL file before parsing?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'reading/zplreader/changedir': 'TRUE',
            # should ZIMPL starting solutions be forwarded to SCIP?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'reading/zplreader/usestartsol': 'TRUE',
            # additional parameter string passed to the ZIMPL parser (or - for no additional parameters)
            # [type: string, advanced: FALSE, default: "-"]
            'reading/zplreader/parameters': '"-"',
            # should model constraints be subject to aging?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'reading/opbreader/dynamicconss': 'FALSE',
            # use '*' between coefficients and variables by writing to problem?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'reading/opbreader/multisymbol': 'FALSE',
            # should an artificial objective, depending on the number of clauses a variable appears in, be used?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'reading/cnfreader/useobj': 'FALSE',
            # should fixed and aggregated variables be printed (if not, re-parsing might fail)
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'reading/cipreader/writefixedvars': 'TRUE',
            # should Benders' decomposition be used?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'reading/storeader/usebenders': 'FALSE',
            # only use improving bounds
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'reading/bndreader/improveonly': 'FALSE',
            # should the coloring values be relativ or absolute
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'reading/ppmreader/rgbrelativ': 'TRUE',
            # should the output format be binary(P6) (otherwise plain(P3) format)
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'reading/ppmreader/rgbascii': 'TRUE',
            # splitting coefficients in this number of intervals
            # [type: int, advanced: FALSE, range: [3,16], default: 3]
            'reading/ppmreader/coefficientlimit': '3',
            # maximal color value
            # [type: int, advanced: FALSE, range: [0,255], default: 160]
            'reading/ppmreader/rgblimit': '160',
            # should the output format be binary(P4) (otherwise plain(P1) format)
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'reading/pbmreader/binary': 'TRUE',
            # maximum number of rows in the scaled picture (-1 for no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 1000]
            'reading/pbmreader/maxrows': '1000',
            # maximum number of columns in the scaled picture (-1 for no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 1000]
            'reading/pbmreader/maxcols': '1000',
            # priority of presolver <boundshift>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 7900000]
            'presolving/boundshift/priority': '7900000',
            # maximal number of presolving rounds the presolver participates in (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 0]
            'presolving/boundshift/maxrounds': '0',
            # timing mask of presolver <boundshift> (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 4]
            'presolving/boundshift/timing': '4',
            # absolute value of maximum shift
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 9223372036854775807]
            'presolving/boundshift/maxshift': '9223372036854775807',
            # is flipping allowed (multiplying with -1)?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'presolving/boundshift/flipping': 'TRUE',
            # shift only integer ranges?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'presolving/boundshift/integer': 'TRUE',
            # priority of presolver <convertinttobin>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 6000000]
            'presolving/convertinttobin/priority': '6000000',
            # maximal number of presolving rounds the presolver participates in (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 0]
            'presolving/convertinttobin/maxrounds': '0',
            # timing mask of presolver <convertinttobin> (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 4]
            'presolving/convertinttobin/timing': '4',
            # absolute value of maximum domain size for converting an integer variable to binaries variables
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 9223372036854775807]
            'presolving/convertinttobin/maxdomainsize': '9223372036854775807',
            # should only integer variables with a domain size of 2^p - 1 be converted(, there we don't need an knapsack-constraint for restricting the sum of the binaries)
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'presolving/convertinttobin/onlypoweroftwo': 'FALSE',
            # should only integer variables with uplocks equals downlocks be converted
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'presolving/convertinttobin/samelocksinbothdirections': 'FALSE',
            # priority of presolver <domcol>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1000]
            'presolving/domcol/priority': '-1000',
            # maximal number of presolving rounds the presolver participates in (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'presolving/domcol/maxrounds': '-1',
            # timing mask of presolver <domcol> (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 16]
            'presolving/domcol/timing': '16',
            # minimal number of pair comparisons
            # [type: int, advanced: FALSE, range: [100,1048576], default: 1024]
            'presolving/domcol/numminpairs': '1024',
            # maximal number of pair comparisons
            # [type: int, advanced: FALSE, range: [1024,1000000000], default: 1048576]
            'presolving/domcol/nummaxpairs': '1048576',
            # should predictive bound strengthening be applied?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'presolving/domcol/predbndstr': 'FALSE',
            # should reductions for continuous variables be performed?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'presolving/domcol/continuousred': 'TRUE',
            # priority of presolver <dualagg>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -12000]
            'presolving/dualagg/priority': '-12000',
            # maximal number of presolving rounds the presolver participates in (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 0]
            'presolving/dualagg/maxrounds': '0',
            # timing mask of presolver <dualagg> (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 16]
            'presolving/dualagg/timing': '16',
            # priority of presolver <dualcomp>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -50]
            'presolving/dualcomp/priority': '-50',
            # maximal number of presolving rounds the presolver participates in (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'presolving/dualcomp/maxrounds': '-1',
            # timing mask of presolver <dualcomp> (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 16]
            'presolving/dualcomp/timing': '16',
            # should only discrete variables be compensated?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'presolving/dualcomp/componlydisvars': 'FALSE',
            # priority of presolver <dualinfer>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -3000]
            'presolving/dualinfer/priority': '-3000',
            # maximal number of presolving rounds the presolver participates in (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 0]
            'presolving/dualinfer/maxrounds': '0',
            # timing mask of presolver <dualinfer> (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 16]
            'presolving/dualinfer/timing': '16',
            # use convex combination of columns for determining dual bounds
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'presolving/dualinfer/twocolcombine': 'TRUE',
            # maximal number of dual bound strengthening loops
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 12]
            'presolving/dualinfer/maxdualbndloops': '12',
            # maximal number of considered non-zeros within one column (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 100]
            'presolving/dualinfer/maxconsiderednonzeros': '100',
            # maximal number of consecutive useless hashtable retrieves
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 1000]
            'presolving/dualinfer/maxretrievefails': '1000',
            # maximal number of consecutive useless column combines
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 1000]
            'presolving/dualinfer/maxcombinefails': '1000',
            # Maximum number of hashlist entries as multiple of number of columns in the problem (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 10]
            'presolving/dualinfer/maxhashfac': '10',
            # Maximum number of processed column pairs as multiple of the number of columns in the problem (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 1]
            'presolving/dualinfer/maxpairfac': '1',
            # Maximum number of row's non-zeros for changing inequality to equality
            # [type: int, advanced: FALSE, range: [2,2147483647], default: 3]
            'presolving/dualinfer/maxrowsupport': '3',
            # priority of presolver <gateextraction>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 1000000]
            'presolving/gateextraction/priority': '1000000',
            # maximal number of presolving rounds the presolver participates in (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'presolving/gateextraction/maxrounds': '-1',
            # timing mask of presolver <gateextraction> (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 16]
            'presolving/gateextraction/timing': '16',
            # should we only try to extract set-partitioning constraints and no and-constraints
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'presolving/gateextraction/onlysetpart': 'FALSE',
            # should we try to extract set-partitioning constraint out of one logicor and one corresponding set-packing constraint
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'presolving/gateextraction/searchequations': 'TRUE',
            # order logicor contraints to extract big-gates before smaller ones (-1), do not order them (0) or order them to extract smaller gates at first (1)
            # [type: int, advanced: TRUE, range: [-1,1], default: 1]
            'presolving/gateextraction/sorting': '1',
            # priority of presolver <implics>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -10000]
            'presolving/implics/priority': '-10000',
            # maximal number of presolving rounds the presolver participates in (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'presolving/implics/maxrounds': '-1',
            # timing mask of presolver <implics> (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 8]
            'presolving/implics/timing': '8',
            # priority of presolver <inttobinary>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 7000000]
            'presolving/inttobinary/priority': '7000000',
            # maximal number of presolving rounds the presolver participates in (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'presolving/inttobinary/maxrounds': '-1',
            # timing mask of presolver <inttobinary> (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 4]
            'presolving/inttobinary/timing': '4',
            # priority of presolver <milp>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 9999999]
            'presolving/milp/priority': '9999999',
            # maximal number of presolving rounds the presolver participates in (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'presolving/milp/maxrounds': '-1',
            # timing mask of presolver <milp> (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 8]
            'presolving/milp/timing': '8',
            # maximum number of threads presolving may use (0: automatic)
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 1]
            'presolving/milp/threads': '1',
            # maximal possible fillin for substitutions to be considered
            # [type: int, advanced: FALSE, range: [-2147483648,2147483647], default: 3]
            'presolving/milp/maxfillinpersubstitution': '3',
            # maximal amount of nonzeros allowed to be shifted to make space for substitutions
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 10]
            'presolving/milp/maxshiftperrow': '10',
            # the random seed used for randomization of tie breaking
            # [type: int, advanced: FALSE, range: [-2147483648,2147483647], default: 0]
            'presolving/milp/randomseed': '0',
            # modify SCIP constraints when the number of nonzeros or rows is at most this factor times the number of nonzeros or rows before presolving
            # [type: real, advanced: FALSE, range: [0,1], default: 0.8]
            'presolving/milp/modifyconsfac': '0.8',
            # the markowitz tolerance used for substitutions
            # [type: real, advanced: FALSE, range: [0,1], default: 0.01]
            'presolving/milp/markowitztolerance': '0.01',
            # absolute bound value that is considered too huge for activitity based calculations
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 100000000]
            'presolving/milp/hugebound': '100000000',
            # should the parallel rows presolver be enabled within the presolve library?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'presolving/milp/enableparallelrows': 'TRUE',
            # should the dominated column presolver be enabled within the presolve library?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'presolving/milp/enabledomcol': 'TRUE',
            # should the dualinfer presolver be enabled within the presolve library?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'presolving/milp/enabledualinfer': 'TRUE',
            # should the multi-aggregation presolver be enabled within the presolve library?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'presolving/milp/enablemultiaggr': 'TRUE',
            # should the probing presolver be enabled within the presolve library?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'presolving/milp/enableprobing': 'TRUE',
            # should the sparsify presolver be enabled within the presolve library?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'presolving/milp/enablesparsify': 'FALSE',
            # priority of presolver <qpkktref>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1]
            'presolving/qpkktref/priority': '-1',
            # maximal number of presolving rounds the presolver participates in (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'presolving/qpkktref/maxrounds': '-1',
            # timing mask of presolver <qpkktref> (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 8]
            'presolving/qpkktref/timing': '8',
            # if TRUE then allow binary variables for KKT update
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'presolving/qpkktref/addkktbinary': 'FALSE',
            # if TRUE then only apply the update to QPs with bounded variables; if the variables are not bounded then a finite optimal solution might not exist and the KKT conditions would then be invalid
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'presolving/qpkktref/updatequadbounded': 'TRUE',
            # if TRUE then apply quadratic constraint update even if the quadratic constraint matrix is known to be indefinite
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'presolving/qpkktref/updatequadindef': 'FALSE',
            # priority of presolver <redvub>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -9000000]
            'presolving/redvub/priority': '-9000000',
            # maximal number of presolving rounds the presolver participates in (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 0]
            'presolving/redvub/maxrounds': '0',
            # timing mask of presolver <redvub> (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 16]
            'presolving/redvub/timing': '16',
            # priority of presolver <trivial>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 9000000]
            'presolving/trivial/priority': '9000000',
            # maximal number of presolving rounds the presolver participates in (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'presolving/trivial/maxrounds': '-1',
            # timing mask of presolver <trivial> (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 4]
            'presolving/trivial/timing': '4',
            # priority of presolver <tworowbnd>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -2000]
            'presolving/tworowbnd/priority': '-2000',
            # maximal number of presolving rounds the presolver participates in (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 0]
            'presolving/tworowbnd/maxrounds': '0',
            # timing mask of presolver <tworowbnd> (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 16]
            'presolving/tworowbnd/timing': '16',
            # should tworowbnd presolver be copied to sub-SCIPs?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'presolving/tworowbnd/enablecopy': 'TRUE',
            # maximal number of considered non-zeros within one row (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 100]
            'presolving/tworowbnd/maxconsiderednonzeros': '100',
            # maximal number of consecutive useless hashtable retrieves
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 1000]
            'presolving/tworowbnd/maxretrievefails': '1000',
            # maximal number of consecutive useless row combines
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 1000]
            'presolving/tworowbnd/maxcombinefails': '1000',
            # Maximum number of hashlist entries as multiple of number of rows in the problem (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 10]
            'presolving/tworowbnd/maxhashfac': '10',
            # Maximum number of processed row pairs as multiple of the number of rows in the problem (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 1]
            'presolving/tworowbnd/maxpairfac': '1',
            # priority of presolver <sparsify>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -24000]
            'presolving/sparsify/priority': '-24000',
            # maximal number of presolving rounds the presolver participates in (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'presolving/sparsify/maxrounds': '-1',
            # timing mask of presolver <sparsify> (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 16]
            'presolving/sparsify/timing': '16',
            # should sparsify presolver be copied to sub-SCIPs?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'presolving/sparsify/enablecopy': 'TRUE',
            # should we cancel nonzeros in constraints of the linear constraint handler?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'presolving/sparsify/cancellinear': 'TRUE',
            # should we forbid cancellations that destroy integer coefficients?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'presolving/sparsify/preserveintcoefs': 'TRUE',
            # maximal fillin for continuous variables (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 0]
            'presolving/sparsify/maxcontfillin': '0',
            # maximal fillin for binary variables (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 0]
            'presolving/sparsify/maxbinfillin': '0',
            # maximal fillin for integer variables including binaries (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 0]
            'presolving/sparsify/maxintfillin': '0',
            # maximal support of one equality to be used for cancelling (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'presolving/sparsify/maxnonzeros': '-1',
            # maximal number of considered non-zeros within one row (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 70]
            'presolving/sparsify/maxconsiderednonzeros': '70',
            # order in which to process inequalities ('n'o sorting, 'i'ncreasing nonzeros, 'd'ecreasing nonzeros)
            # [type: char, advanced: TRUE, range: {nid}, default: d]
            'presolving/sparsify/rowsort': 'd',
            # limit on the number of useless vs. useful hashtable retrieves as a multiple of the number of constraints
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 100]
            'presolving/sparsify/maxretrievefac': '100',
            # number of calls to wait until next execution as a multiple of the number of useless calls
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 2]
            'presolving/sparsify/waitingfac': '2',
            # priority of presolver <dualsparsify>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -240000]
            'presolving/dualsparsify/priority': '-240000',
            # maximal number of presolving rounds the presolver participates in (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'presolving/dualsparsify/maxrounds': '-1',
            # timing mask of presolver <dualsparsify> (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 16]
            'presolving/dualsparsify/timing': '16',
            # should dualsparsify presolver be copied to sub-SCIPs?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'presolving/dualsparsify/enablecopy': 'TRUE',
            # should we forbid cancellations that destroy integer coefficients?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'presolving/dualsparsify/preserveintcoefs': 'FALSE',
            # should we preserve good locked properties of variables (at most one lock in one direction)?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'presolving/dualsparsify/preservegoodlocks': 'FALSE',
            # maximal fillin for continuous variables (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 1]
            'presolving/dualsparsify/maxcontfillin': '1',
            # maximal fillin for binary variables (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 1]
            'presolving/dualsparsify/maxbinfillin': '1',
            # maximal fillin for integer variables including binaries (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 1]
            'presolving/dualsparsify/maxintfillin': '1',
            # maximal number of considered nonzeros within one column (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 70]
            'presolving/dualsparsify/maxconsiderednonzeros': '70',
            # minimal eliminated nonzeros within one column if we need to add a constraint to the problem
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 100]
            'presolving/dualsparsify/mineliminatednonzeros': '100',
            # limit on the number of useless vs. useful hashtable retrieves as a multiple of the number of constraints
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 100]
            'presolving/dualsparsify/maxretrievefac': '100',
            # number of calls to wait until next execution as a multiple of the number of useless calls
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 2]
            'presolving/dualsparsify/waitingfac': '2',
            # priority of presolver <stuffing>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -100]
            'presolving/stuffing/priority': '-100',
            # maximal number of presolving rounds the presolver participates in (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 0]
            'presolving/stuffing/maxrounds': '0',
            # timing mask of presolver <stuffing> (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [4,60], default: 16]
            'presolving/stuffing/timing': '16',
            # priority of node selection rule <bfs> in standard mode
            # [type: int, advanced: FALSE, range: [-536870912,1073741823], default: 100000]
            'nodeselection/bfs/stdpriority': '100000',
            # priority of node selection rule <bfs> in memory saving mode
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 0]
            'nodeselection/bfs/memsavepriority': '0',
            # minimal plunging depth, before new best node may be selected (-1 for dynamic setting)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'nodeselection/bfs/minplungedepth': '-1',
            # maximal plunging depth, before new best node is forced to be selected (-1 for dynamic setting)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'nodeselection/bfs/maxplungedepth': '-1',
            # maximal quotient (curlowerbound - lowerbound)/(cutoffbound - lowerbound) where plunging is performed
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0.25]
            'nodeselection/bfs/maxplungequot': '0.25',
            # priority of node selection rule <breadthfirst> in standard mode
            # [type: int, advanced: FALSE, range: [-536870912,1073741823], default: -10000]
            'nodeselection/breadthfirst/stdpriority': '-10000',
            # priority of node selection rule <breadthfirst> in memory saving mode
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1000000]
            'nodeselection/breadthfirst/memsavepriority': '-1000000',
            # priority of node selection rule <dfs> in standard mode
            # [type: int, advanced: FALSE, range: [-536870912,1073741823], default: 0]
            'nodeselection/dfs/stdpriority': '0',
            # priority of node selection rule <dfs> in memory saving mode
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 100000]
            'nodeselection/dfs/memsavepriority': '100000',
            # priority of node selection rule <estimate> in standard mode
            # [type: int, advanced: FALSE, range: [-536870912,1073741823], default: 200000]
            'nodeselection/estimate/stdpriority': '200000',
            # priority of node selection rule <estimate> in memory saving mode
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 100]
            'nodeselection/estimate/memsavepriority': '100',
            # minimal plunging depth, before new best node may be selected (-1 for dynamic setting)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'nodeselection/estimate/minplungedepth': '-1',
            # maximal plunging depth, before new best node is forced to be selected (-1 for dynamic setting)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'nodeselection/estimate/maxplungedepth': '-1',
            # maximal quotient (estimate - lowerbound)/(cutoffbound - lowerbound) where plunging is performed
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0.25]
            'nodeselection/estimate/maxplungequot': '0.25',
            # frequency at which the best node instead of the best estimate is selected (0: never)
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 10]
            'nodeselection/estimate/bestnodefreq': '10',
            # depth until breadth-first search is applied
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'nodeselection/estimate/breadthfirstdepth': '-1',
            # number of nodes before doing plunging the first time
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 0]
            'nodeselection/estimate/plungeoffset': '0',
            # priority of node selection rule <hybridestim> in standard mode
            # [type: int, advanced: FALSE, range: [-536870912,1073741823], default: 50000]
            'nodeselection/hybridestim/stdpriority': '50000',
            # priority of node selection rule <hybridestim> in memory saving mode
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 50]
            'nodeselection/hybridestim/memsavepriority': '50',
            # minimal plunging depth, before new best node may be selected (-1 for dynamic setting)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'nodeselection/hybridestim/minplungedepth': '-1',
            # maximal plunging depth, before new best node is forced to be selected (-1 for dynamic setting)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'nodeselection/hybridestim/maxplungedepth': '-1',
            # maximal quotient (estimate - lowerbound)/(cutoffbound - lowerbound) where plunging is performed
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0.25]
            'nodeselection/hybridestim/maxplungequot': '0.25',
            # frequency at which the best node instead of the hybrid best estimate / best bound is selected (0: never)
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 1000]
            'nodeselection/hybridestim/bestnodefreq': '1000',
            # weight of estimate value in node selection score (0: pure best bound search, 1: pure best estimate search)
            # [type: real, advanced: TRUE, range: [0,1], default: 0.1]
            'nodeselection/hybridestim/estimweight': '0.1',
            # priority of node selection rule <restartdfs> in standard mode
            # [type: int, advanced: FALSE, range: [-536870912,1073741823], default: 10000]
            'nodeselection/restartdfs/stdpriority': '10000',
            # priority of node selection rule <restartdfs> in memory saving mode
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 50000]
            'nodeselection/restartdfs/memsavepriority': '50000',
            # frequency for selecting the best node instead of the deepest one
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 100]
            'nodeselection/restartdfs/selectbestfreq': '100',
            # count only leaf nodes (otherwise all nodes)?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'nodeselection/restartdfs/countonlyleaves': 'TRUE',
            # priority of node selection rule <uct> in standard mode
            # [type: int, advanced: FALSE, range: [-536870912,1073741823], default: 10]
            'nodeselection/uct/stdpriority': '10',
            # priority of node selection rule <uct> in memory saving mode
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 0]
            'nodeselection/uct/memsavepriority': '0',
            # maximum number of nodes before switching to default rule
            # [type: int, advanced: TRUE, range: [0,1000000], default: 31]
            'nodeselection/uct/nodelimit': '31',
            # weight for visit quotient of node selection rule
            # [type: real, advanced: TRUE, range: [0,1], default: 0.1]
            'nodeselection/uct/weight': '0.1',
            # should the estimate (TRUE) or lower bound of a node be used for UCT score?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'nodeselection/uct/useestimate': 'FALSE',
            # priority of branching rule <allfullstrong>
            # [type: int, advanced: FALSE, range: [-536870912,536870911], default: -1000]
            'branching/allfullstrong/priority': '-1000',
            # maximal depth level, up to which branching rule <allfullstrong> should be used (-1 for no limit)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'branching/allfullstrong/maxdepth': '-1',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying branching rule (0.0: only on current best node, 1.0: on all nodes)
            # [type: real, advanced: FALSE, range: [0,1], default: 1]
            'branching/allfullstrong/maxbounddist': '1',
            # priority of branching rule <cloud>
            # [type: int, advanced: FALSE, range: [-536870912,536870911], default: 0]
            'branching/cloud/priority': '0',
            # maximal depth level, up to which branching rule <cloud> should be used (-1 for no limit)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'branching/cloud/maxdepth': '-1',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying branching rule (0.0: only on current best node, 1.0: on all nodes)
            # [type: real, advanced: FALSE, range: [0,1], default: 1]
            'branching/cloud/maxbounddist': '1',
            # should a cloud of points be used?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'branching/cloud/usecloud': 'TRUE',
            # should only F2 be used?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'branching/cloud/onlyF2': 'FALSE',
            # should the union of candidates be used?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'branching/cloud/useunion': 'FALSE',
            # maximum number of points for the cloud (-1 means no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'branching/cloud/maxpoints': '-1',
            # minimum success rate for the cloud
            # [type: real, advanced: FALSE, range: [0,1], default: 0]
            'branching/cloud/minsuccessrate': '0',
            # minimum success rate for the union
            # [type: real, advanced: FALSE, range: [0,1], default: 0]
            'branching/cloud/minsuccessunion': '0',
            # maximum depth for the union
            # [type: int, advanced: FALSE, range: [0,65000], default: 65000]
            'branching/cloud/maxdepthunion': '65000',
            # priority of branching rule <distribution>
            # [type: int, advanced: FALSE, range: [-536870912,536870911], default: 0]
            'branching/distribution/priority': '0',
            # maximal depth level, up to which branching rule <distribution> should be used (-1 for no limit)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'branching/distribution/maxdepth': '-1',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying branching rule (0.0: only on current best node, 1.0: on all nodes)
            # [type: real, advanced: FALSE, range: [0,1], default: 1]
            'branching/distribution/maxbounddist': '1',
            # the score;largest 'd'ifference, 'l'owest cumulative probability,'h'ighest c.p., 'v'otes lowest c.p., votes highest c.p.('w')
            # [type: char, advanced: TRUE, range: {dhlvw}, default: v]
            'branching/distribution/scoreparam': 'v',
            # should only rows which are active at the current node be considered?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'branching/distribution/onlyactiverows': 'FALSE',
            # should the branching score weigh up- and down-scores of a variable
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'branching/distribution/weightedscore': 'FALSE',
            # priority of branching rule <fullstrong>
            # [type: int, advanced: FALSE, range: [-536870912,536870911], default: 0]
            'branching/fullstrong/priority': '0',
            # maximal depth level, up to which branching rule <fullstrong> should be used (-1 for no limit)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'branching/fullstrong/maxdepth': '-1',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying branching rule (0.0: only on current best node, 1.0: on all nodes)
            # [type: real, advanced: FALSE, range: [0,1], default: 1]
            'branching/fullstrong/maxbounddist': '1',
            # number of intermediate LPs solved to trigger reevaluation of strong branching value for a variable that was already evaluated at the current node
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 10]
            'branching/fullstrong/reevalage': '10',
            # maximum number of propagation rounds to be performed during strong branching before solving the LP (-1: no limit, -2: parameter settings)
            # [type: int, advanced: TRUE, range: [-3,2147483647], default: -2]
            'branching/fullstrong/maxproprounds': '-2',
            # should valid bounds be identified in a probing-like fashion during strong branching (only with propagation)?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'branching/fullstrong/probingbounds': 'TRUE',
            # should strong branching be applied even if there is just a single candidate?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'branching/fullstrong/forcestrongbranch': 'FALSE',
            # priority of branching rule <inference>
            # [type: int, advanced: FALSE, range: [-536870912,536870911], default: 1000]
            'branching/inference/priority': '1000',
            # maximal depth level, up to which branching rule <inference> should be used (-1 for no limit)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'branching/inference/maxdepth': '-1',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying branching rule (0.0: only on current best node, 1.0: on all nodes)
            # [type: real, advanced: FALSE, range: [0,1], default: 1]
            'branching/inference/maxbounddist': '1',
            # weight in score calculations for conflict score
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 1000]
            'branching/inference/conflictweight': '1000',
            # weight in score calculations for inference score
            # [type: real, advanced: TRUE, range: [-1.79769313486232e+308,1.79769313486232e+308], default: 1]
            'branching/inference/inferenceweight': '1',
            # weight in score calculations for cutoff score
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 1]
            'branching/inference/cutoffweight': '1',
            # should branching on LP solution be restricted to the fractional variables?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'branching/inference/fractionals': 'TRUE',
            # should a weighted sum of inference, conflict and cutoff weights be used?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'branching/inference/useweightedsum': 'TRUE',
            # weight in score calculations for conflict score
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0.001]
            'branching/inference/reliablescore': '0.001',
            # priority of branching rule <leastinf>
            # [type: int, advanced: FALSE, range: [-536870912,536870911], default: 50]
            'branching/leastinf/priority': '50',
            # maximal depth level, up to which branching rule <leastinf> should be used (-1 for no limit)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'branching/leastinf/maxdepth': '-1',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying branching rule (0.0: only on current best node, 1.0: on all nodes)
            # [type: real, advanced: FALSE, range: [0,1], default: 1]
            'branching/leastinf/maxbounddist': '1',
            # priority of branching rule <lookahead>
            # [type: int, advanced: FALSE, range: [-536870912,536870911], default: 0]
            'branching/lookahead/priority': '0',
            # maximal depth level, up to which branching rule <lookahead> should be used (-1 for no limit)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'branching/lookahead/maxdepth': '-1',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying branching rule (0.0: only on current best node, 1.0: on all nodes)
            # [type: real, advanced: FALSE, range: [0,1], default: 1]
            'branching/lookahead/maxbounddist': '1',
            # should binary constraints be collected and applied?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'branching/lookahead/useimpliedbincons': 'FALSE',
            # should binary constraints be added as rows to the base LP? (0: no, 1: separate, 2: as initial rows)
            # [type: int, advanced: TRUE, range: [0,2], default: 0]
            'branching/lookahead/addbinconsrow': '0',
            # how many constraints that are violated by the base lp solution should be gathered until the rule is stopped and they are added? [0 for unrestricted]
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 1]
            'branching/lookahead/maxnviolatedcons': '1',
            # how many binary constraints that are violated by the base lp solution should be gathered until the rule is stopped and they are added? [0 for unrestricted]
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 0]
            'branching/lookahead/maxnviolatedbincons': '0',
            # how many domain reductions that are violated by the base lp solution should be gathered until the rule is stopped and they are added? [0 for unrestricted]
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 1]
            'branching/lookahead/maxnviolateddomreds': '1',
            # max number of LPs solved after which a previous prob branching results are recalculated
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 10]
            'branching/lookahead/reevalage': '10',
            # max number of LPs solved after which a previous FSB scoring results are recalculated
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 10]
            'branching/lookahead/reevalagefsb': '10',
            # the max depth of LAB.
            # [type: int, advanced: TRUE, range: [1,2147483647], default: 2]
            'branching/lookahead/recursiondepth': '2',
            # should domain reductions be collected and applied?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'branching/lookahead/usedomainreduction': 'TRUE',
            # should domain reductions of feasible siblings should be merged?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'branching/lookahead/mergedomainreductions': 'FALSE',
            # should domain reductions only be applied if there are simple bound changes?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'branching/lookahead/prefersimplebounds': 'FALSE',
            # should only domain reductions that violate the LP solution be applied?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'branching/lookahead/onlyvioldomreds': 'FALSE',
            # should binary constraints, that are not violated by the base LP, be collected and added?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'branching/lookahead/addnonviocons': 'FALSE',
            # toggles the abbreviated LAB.
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'branching/lookahead/abbreviated': 'TRUE',
            # if abbreviated: The max number of candidates to consider at the node.
            # [type: int, advanced: TRUE, range: [1,2147483647], default: 4]
            'branching/lookahead/maxncands': '4',
            # if abbreviated: The max number of candidates to consider per deeper node.
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 2]
            'branching/lookahead/maxndeepercands': '2',
            # if abbreviated: Should the information gathered to obtain the best candidates be reused?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'branching/lookahead/reusebasis': 'TRUE',
            # if only non violating constraints are added, should the branching decision be stored till the next call?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'branching/lookahead/storeunviolatedsol': 'TRUE',
            # if abbreviated: Use pseudo costs to estimate the score of a candidate.
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'branching/lookahead/abbrevpseudo': 'FALSE',
            # should the average score be used for uninitialized scores in level 2?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'branching/lookahead/level2avgscore': 'FALSE',
            # should uninitialized scores in level 2 be set to 0?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'branching/lookahead/level2zeroscore': 'FALSE',
            # add binary constraints with two variables found at the root node also as a clique
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'branching/lookahead/addclique': 'FALSE',
            # should domain propagation be executed before each temporary node is solved?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'branching/lookahead/propagate': 'TRUE',
            # should branching data generated at depth level 2 be stored for re-using it?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'branching/lookahead/uselevel2data': 'TRUE',
            # should bounds known for child nodes be applied?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'branching/lookahead/applychildbounds': 'FALSE',
            # should the maximum number of domain reductions maxnviolateddomreds be enforced?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'branching/lookahead/enforcemaxdomreds': 'FALSE',
            # should branching results (and scores) be updated w.r.t. proven dual bounds?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'branching/lookahead/updatebranchingresults': 'FALSE',
            # maximum number of propagation rounds to perform at each temporary node (-1: unlimited, 0: SCIP default)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 0]
            'branching/lookahead/maxproprounds': '0',
            # scoring function to be used at the base level
            # [type: char, advanced: TRUE, range: {dfswplcra}, default: a]
            'branching/lookahead/scoringfunction': 'a',
            # scoring function to be used at deeper levels
            # [type: char, advanced: TRUE, range: {dfswlcrx}, default: x]
            'branching/lookahead/deeperscoringfunction': 'x',
            # scoring function to be used during FSB scoring
            # [type: char, advanced: TRUE, range: {dfswlcr}, default: d]
            'branching/lookahead/scoringscoringfunction': 'd',
            # if scoringfunction is 's', this value is used to weight the min of the gains of two child problems in the convex combination
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0.8]
            'branching/lookahead/minweight': '0.8',
            # if the FSB score is of a candidate is worse than the best by this factor, skip this candidate (-1: disable)
            # [type: real, advanced: TRUE, range: [-1,1.79769313486232e+308], default: -1]
            'branching/lookahead/worsefactor': '-1',
            # should lookahead branching only be applied if the max gain in level 1 is not uniquely that of the best candidate?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'branching/lookahead/filterbymaxgain': 'FALSE',
            # priority of branching rule <mostinf>
            # [type: int, advanced: FALSE, range: [-536870912,536870911], default: 100]
            'branching/mostinf/priority': '100',
            # maximal depth level, up to which branching rule <mostinf> should be used (-1 for no limit)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'branching/mostinf/maxdepth': '-1',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying branching rule (0.0: only on current best node, 1.0: on all nodes)
            # [type: real, advanced: FALSE, range: [0,1], default: 1]
            'branching/mostinf/maxbounddist': '1',
            # priority of branching rule <multaggr>
            # [type: int, advanced: FALSE, range: [-536870912,536870911], default: 0]
            'branching/multaggr/priority': '0',
            # maximal depth level, up to which branching rule <multaggr> should be used (-1 for no limit)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'branching/multaggr/maxdepth': '-1',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying branching rule (0.0: only on current best node, 1.0: on all nodes)
            # [type: real, advanced: FALSE, range: [0,1], default: 1]
            'branching/multaggr/maxbounddist': '1',
            # number of intermediate LPs solved to trigger reevaluation of strong branching value for a variable that was already evaluated at the current node
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 0]
            'branching/multaggr/reevalage': '0',
            # maximum number of propagation rounds to be performed during multaggr branching before solving the LP (-1: no limit, -2: parameter settings)
            # [type: int, advanced: TRUE, range: [-2,2147483647], default: 0]
            'branching/multaggr/maxproprounds': '0',
            # should valid bounds be identified in a probing-like fashion during multaggr branching (only with propagation)?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'branching/multaggr/probingbounds': 'TRUE',
            # priority of branching rule <nodereopt>
            # [type: int, advanced: FALSE, range: [-536870912,536870911], default: -9000000]
            'branching/nodereopt/priority': '-9000000',
            # maximal depth level, up to which branching rule <nodereopt> should be used (-1 for no limit)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'branching/nodereopt/maxdepth': '-1',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying branching rule (0.0: only on current best node, 1.0: on all nodes)
            # [type: real, advanced: FALSE, range: [0,1], default: 1]
            'branching/nodereopt/maxbounddist': '1',
            # priority of branching rule <pscost>
            # [type: int, advanced: FALSE, range: [-536870912,536870911], default: 2000]
            'branching/pscost/priority': '2000',
            # maximal depth level, up to which branching rule <pscost> should be used (-1 for no limit)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'branching/pscost/maxdepth': '-1',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying branching rule (0.0: only on current best node, 1.0: on all nodes)
            # [type: real, advanced: FALSE, range: [0,1], default: 1]
            'branching/pscost/maxbounddist': '1',
            # strategy for utilizing pseudo-costs of external branching candidates (multiply as in pseudo costs 'u'pdate rule, or by 'd'omain reduction, or by domain reduction of 's'ibling, or by 'v'ariable score)
            # [type: char, advanced: FALSE, range: {dsuv}, default: u]
            'branching/pscost/strategy': 'u',
            # weight for minimum of scores of a branching candidate when building weighted sum of min/max/sum of scores
            # [type: real, advanced: TRUE, range: [-1e+20,1e+20], default: 0.8]
            'branching/pscost/minscoreweight': '0.8',
            # weight for maximum of scores of a branching candidate when building weighted sum of min/max/sum of scores
            # [type: real, advanced: TRUE, range: [-1e+20,1e+20], default: 1.3]
            'branching/pscost/maxscoreweight': '1.3',
            # weight for sum of scores of a branching candidate when building weighted sum of min/max/sum of scores
            # [type: real, advanced: TRUE, range: [-1e+20,1e+20], default: 0.1]
            'branching/pscost/sumscoreweight': '0.1',
            # number of children to create in n-ary branching
            # [type: int, advanced: FALSE, range: [2,2147483647], default: 2]
            'branching/pscost/nchildren': '2',
            # maximal depth where to do n-ary branching, -1 to turn off
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'branching/pscost/narymaxdepth': '-1',
            # minimal domain width in children when doing n-ary branching, relative to global bounds
            # [type: real, advanced: FALSE, range: [0,1], default: 0.001]
            'branching/pscost/naryminwidth': '0.001',
            # factor of domain width in n-ary branching when creating nodes with increasing distance from branching value
            # [type: real, advanced: FALSE, range: [1,1.79769313486232e+308], default: 2]
            'branching/pscost/narywidthfactor': '2',
            # priority of branching rule <random>
            # [type: int, advanced: FALSE, range: [-536870912,536870911], default: -100000]
            'branching/random/priority': '-100000',
            # maximal depth level, up to which branching rule <random> should be used (-1 for no limit)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'branching/random/maxdepth': '-1',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying branching rule (0.0: only on current best node, 1.0: on all nodes)
            # [type: real, advanced: FALSE, range: [0,1], default: 1]
            'branching/random/maxbounddist': '1',
            # initial random seed value
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 41]
            'branching/random/seed': '41',
            # priority of branching rule <relpscost>
            # [type: int, advanced: FALSE, range: [-536870912,536870911], default: 10000]
            'branching/relpscost/priority': '10000',
            # maximal depth level, up to which branching rule <relpscost> should be used (-1 for no limit)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'branching/relpscost/maxdepth': '-1',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying branching rule (0.0: only on current best node, 1.0: on all nodes)
            # [type: real, advanced: FALSE, range: [0,1], default: 1]
            'branching/relpscost/maxbounddist': '1',
            # weight in score calculations for conflict score
            # [type: real, advanced: TRUE, range: [-1.79769313486232e+308,1.79769313486232e+308], default: 0.01]
            'branching/relpscost/conflictweight': '0.01',
            # weight in score calculations for conflict length score
            # [type: real, advanced: TRUE, range: [-1.79769313486232e+308,1.79769313486232e+308], default: 0]
            'branching/relpscost/conflictlengthweight': '0',
            # weight in score calculations for inference score
            # [type: real, advanced: TRUE, range: [-1.79769313486232e+308,1.79769313486232e+308], default: 0.0001]
            'branching/relpscost/inferenceweight': '0.0001',
            # weight in score calculations for cutoff score
            # [type: real, advanced: TRUE, range: [-1.79769313486232e+308,1.79769313486232e+308], default: 0.0001]
            'branching/relpscost/cutoffweight': '0.0001',
            # weight in score calculations for pseudo cost score
            # [type: real, advanced: TRUE, range: [-1.79769313486232e+308,1.79769313486232e+308], default: 1]
            'branching/relpscost/pscostweight': '1',
            # weight in score calculations for nlcount score
            # [type: real, advanced: TRUE, range: [-1.79769313486232e+308,1.79769313486232e+308], default: 0.1]
            'branching/relpscost/nlscoreweight': '0.1',
            # minimal value for minimum pseudo cost size to regard pseudo cost value as reliable
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 1]
            'branching/relpscost/minreliable': '1',
            # maximal value for minimum pseudo cost size to regard pseudo cost value as reliable
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 5]
            'branching/relpscost/maxreliable': '5',
            # maximal fraction of strong branching LP iterations compared to node relaxation LP iterations
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 0.5]
            'branching/relpscost/sbiterquot': '0.5',
            # additional number of allowed strong branching LP iterations
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 100000]
            'branching/relpscost/sbiterofs': '100000',
            # maximal number of further variables evaluated without better score
            # [type: int, advanced: TRUE, range: [1,2147483647], default: 9]
            'branching/relpscost/maxlookahead': '9',
            # maximal number of candidates initialized with strong branching per node
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 100]
            'branching/relpscost/initcand': '100',
            # iteration limit for strong branching initializations of pseudo cost entries (0: auto)
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 0]
            'branching/relpscost/inititer': '0',
            # maximal number of bound tightenings before the node is reevaluated (-1: unlimited)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 5]
            'branching/relpscost/maxbdchgs': '5',
            # maximum number of propagation rounds to be performed during strong branching before solving the LP (-1: no limit, -2: parameter settings)
            # [type: int, advanced: TRUE, range: [-2,2147483647], default: -2]
            'branching/relpscost/maxproprounds': '-2',
            # should valid bounds be identified in a probing-like fashion during strong branching (only with propagation)?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'branching/relpscost/probingbounds': 'TRUE',
            # should reliability be based on relative errors?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'branching/relpscost/userelerrorreliability': 'FALSE',
            # low relative error tolerance for reliability
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0.05]
            'branching/relpscost/lowerrortol': '0.05',
            # high relative error tolerance for reliability
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 1]
            'branching/relpscost/higherrortol': '1',
            # should strong branching result be considered for pseudo costs if the other direction was infeasible?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'branching/relpscost/storesemiinitcosts': 'FALSE',
            # should the scoring function use only local cutoff and inference information obtained for strong branching candidates?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'branching/relpscost/usesblocalinfo': 'FALSE',
            # should the strong branching decision be based on a hypothesis test?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'branching/relpscost/usehyptestforreliability': 'FALSE',
            # should the confidence level be adjusted dynamically?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'branching/relpscost/usedynamicconfidence': 'FALSE',
            # should branching rule skip candidates that have a low probability to be better than the best strong-branching or pseudo-candidate?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'branching/relpscost/skipbadinitcands': 'TRUE',
            # the confidence level for statistical methods, between 0 (Min) and 4 (Max).
            # [type: int, advanced: TRUE, range: [0,4], default: 2]
            'branching/relpscost/confidencelevel': '2',
            # should candidates be initialized in randomized order?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'branching/relpscost/randinitorder': 'FALSE',
            # should smaller weights be used for pseudo cost updates after hitting the LP iteration limit?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'branching/relpscost/usesmallweightsitlim': 'FALSE',
            # should the weights of the branching rule be adjusted dynamically during solving based on objective and infeasible leaf counters?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'branching/relpscost/dynamicweights': 'TRUE',
            # should degeneracy be taken into account to update weights and skip strong branching? (0: off, 1: after root, 2: always)
            # [type: int, advanced: TRUE, range: [0,2], default: 1]
            'branching/relpscost/degeneracyaware': '1',
            # start seed for random number generation
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 5]
            'branching/relpscost/startrandseed': '5',
            # Use symmetry to filter branching candidates?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'branching/relpscost/filtercandssym': 'FALSE',
            # Transfer pscost information to symmetric variables?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'branching/relpscost/transsympscost': 'FALSE',
            # should candidate branching variables be scored using the Treemodel branching rules?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'branching/treemodel/enable': 'FALSE',
            # scoring function to use at nodes predicted to be high in the tree ('d'efault, 's'vts, 'r'atio, 't'ree sample)
            # [type: char, advanced: FALSE, range: {dsrt}, default: r]
            'branching/treemodel/highrule': 'r',
            # scoring function to use at nodes predicted to be low in the tree ('d'efault, 's'vts, 'r'atio, 't'ree sample)
            # [type: char, advanced: FALSE, range: {dsrt}, default: r]
            'branching/treemodel/lowrule': 'r',
            # estimated tree height at which we switch from using the low rule to the high rule
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 10]
            'branching/treemodel/height': '10',
            # should dominated candidates be filtered before using the high scoring function? ('a'uto, 't'rue, 'f'alse)
            # [type: char, advanced: TRUE, range: {atf}, default: a]
            'branching/treemodel/filterhigh': 'a',
            # should dominated candidates be filtered before using the low scoring function? ('a'uto, 't'rue, 'f'alse)
            # [type: char, advanced: TRUE, range: {atf}, default: a]
            'branching/treemodel/filterlow': 'a',
            # maximum number of fixed-point iterations when computing the ratio
            # [type: int, advanced: TRUE, range: [1,2147483647], default: 24]
            'branching/treemodel/maxfpiter': '24',
            # maximum height to compute the SVTS score exactly before approximating
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 100]
            'branching/treemodel/maxsvtsheight': '100',
            # which method should be used as a fallback if the tree size estimates are infinite? ('d'efault, 'r'atio)
            # [type: char, advanced: TRUE, range: {dr}, default: r]
            'branching/treemodel/fallbackinf': 'r',
            # which method should be used as a fallback if there is no primal bound available? ('d'efault, 'r'atio)
            # [type: char, advanced: TRUE, range: {dr}, default: r]
            'branching/treemodel/fallbacknoprim': 'r',
            # threshold at which pseudocosts are considered small, making hybrid scores more likely to be the deciding factor in branching
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0.1]
            'branching/treemodel/smallpscost': '0.1',
            # priority of branching rule <vanillafullstrong>
            # [type: int, advanced: FALSE, range: [-536870912,536870911], default: -2000]
            'branching/vanillafullstrong/priority': '-2000',
            # maximal depth level, up to which branching rule <vanillafullstrong> should be used (-1 for no limit)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'branching/vanillafullstrong/maxdepth': '-1',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying branching rule (0.0: only on current best node, 1.0: on all nodes)
            # [type: real, advanced: FALSE, range: [0,1], default: 1]
            'branching/vanillafullstrong/maxbounddist': '1',
            # should integral variables in the current LP solution be considered as branching candidates?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'branching/vanillafullstrong/integralcands': 'FALSE',
            # should strong branching side-effects be prevented (e.g., domain changes, stat updates etc.)?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'branching/vanillafullstrong/idempotent': 'FALSE',
            # should strong branching scores be computed for all candidates, or can we early stop when a variable has infinite score?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'branching/vanillafullstrong/scoreall': 'FALSE',
            # should strong branching scores be collected?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'branching/vanillafullstrong/collectscores': 'FALSE',
            # should candidates only be scored, but no branching be performed?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'branching/vanillafullstrong/donotbranch': 'FALSE',
            # restart policy: (a)lways, (c)ompletion, (e)stimation, (n)ever
            # [type: char, advanced: FALSE, range: {acen}, default: e]
            'estimation/restarts/restartpolicy': 'e',
            # tree size estimation method: (c)ompletion, (e)nsemble, time series forecasts on either (g)ap, (l)eaf frequency, (o)open nodes, tree (w)eight, (s)sg, or (t)ree profile or w(b)e
            # [type: char, advanced: FALSE, range: {bceglostw}, default: w]
            'estimation/method': 'w',
            # restart limit
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 1]
            'estimation/restarts/restartlimit': '1',
            # minimum number of nodes before restart
            # [type: longint, advanced: FALSE, range: [-1,9223372036854775807], default: 1000]
            'estimation/restarts/minnodes': '1000',
            # should only leaves count for the minnodes parameter?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'estimation/restarts/countonlyleaves': 'FALSE',
            # factor by which the estimated number of nodes should exceed the current number of nodes
            # [type: real, advanced: FALSE, range: [1,1.79769313486232e+308], default: 50]
            'estimation/restarts/restartfactor': '50',
            # whether to apply a restart when nonlinear constraints are present
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'estimation/restarts/restartnonlinear': 'FALSE',
            # whether to apply a restart when active pricers are used
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'estimation/restarts/restartactpricers': 'FALSE',
            # coefficient of tree weight in monotone approximation of search completion
            # [type: real, advanced: FALSE, range: [0,1], default: 0.3667]
            'estimation/coefmonoweight': '0.3667',
            # coefficient of 1 - SSG in monotone approximation of search completion
            # [type: real, advanced: FALSE, range: [0,1], default: 0.6333]
            'estimation/coefmonossg': '0.6333',
            # limit on the number of successive samples to really trigger a restart
            # [type: int, advanced: FALSE, range: [1,2147483647], default: 50]
            'estimation/restarts/hitcounterlim': '50',
            # report frequency on estimation: -1: never, 0:always, k >= 1: k times evenly during search
            # [type: int, advanced: TRUE, range: [-1,1073741823], default: -1]
            'estimation/reportfreq': '-1',
            # user regression forest in RFCSV format
            # [type: string, advanced: FALSE, default: "-"]
            'estimation/regforestfilename': '"-"',
            # approximation of search tree completion: (a)uto, (g)ap, tree (w)eight, (m)onotone regression, (r)egression forest, (s)sg
            # [type: char, advanced: FALSE, range: {agmrsw}, default: a]
            'estimation/completiontype': 'a',
            # should the event handler collect data?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'estimation/treeprofile/enabled': 'FALSE',
            # minimum average number of nodes at each depth before producing estimations
            # [type: real, advanced: FALSE, range: [1,1.79769313486232e+308], default: 20]
            'estimation/treeprofile/minnodesperdepth': '20',
            # use leaf nodes as basic observations for time series, or all nodes?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'estimation/useleafts': 'TRUE',
            # the maximum number of individual SSG subtrees; -1: no limit
            # [type: int, advanced: FALSE, range: [-1,1073741823], default: -1]
            'estimation/ssg/nmaxsubtrees': '-1',
            # minimum number of nodes to process between two consecutive SSG splits
            # [type: longint, advanced: FALSE, range: [0,9223372036854775807], default: 0]
            'estimation/ssg/nminnodeslastsplit': '0',
            # is statistics table <estim> active
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'table/estim/active': 'TRUE',
            # display activation status of display column <completed> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/completed/active': '1',
            # display activation status of display column <nrank1nodes> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 0]
            'display/nrank1nodes/active': '0',
            # display activation status of display column <nnodesbelowinc> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 0]
            'display/nnodesbelowinc/active': '0',
            # should the event handler adapt the solver behavior?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'solvingphases/enabled': 'FALSE',
            # should the event handler test all phase transitions?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'solvingphases/testmode': 'FALSE',
            # settings file for feasibility phase -- precedence over emphasis settings
            # [type: string, advanced: FALSE, default: "-"]
            'solvingphases/feassetname': '"-"',
            # settings file for improvement phase -- precedence over emphasis settings
            # [type: string, advanced: FALSE, default: "-"]
            'solvingphases/improvesetname': '"-"',
            # settings file for proof phase -- precedence over emphasis settings
            # [type: string, advanced: FALSE, default: "-"]
            'solvingphases/proofsetname': '"-"',
            # node offset for rank-1 and estimate transitions
            # [type: longint, advanced: FALSE, range: [1,9223372036854775807], default: 50]
            'solvingphases/nodeoffset': '50',
            # should the event handler fall back from optimal phase?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'solvingphases/fallback': 'FALSE',
            # transition method: Possible options are 'e'stimate,'l'ogarithmic regression,'o'ptimal-value based,'r'ank-1
            # [type: char, advanced: FALSE, range: {elor}, default: r]
            'solvingphases/transitionmethod': 'r',
            # should the event handler interrupt the solving process after optimal solution was found?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'solvingphases/interruptoptimal': 'FALSE',
            # should a restart be applied between the feasibility and improvement phase?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'solvingphases/userestart1to2': 'FALSE',
            # should a restart be applied between the improvement and the proof phase?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'solvingphases/userestart2to3': 'FALSE',
            # optimal solution value for problem
            # [type: real, advanced: FALSE, range: [-1.79769313486232e+308,1.79769313486232e+308], default: 1e+99]
            'solvingphases/optimalvalue': '0',
            # x-type for logarithmic regression - (t)ime, (n)odes, (l)p iterations
            # [type: char, advanced: FALSE, range: {lnt}, default: n]
            'solvingphases/xtype': 'n',
            # should emphasis settings for the solving phases be used, or settings files?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'solvingphases/useemphsettings': 'TRUE',
            # priority of compression <largestrepr>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 2000]
            'compression/largestrepr/priority': '2000',
            # minimal number of leave nodes for calling tree compression <largestrepr>
            # [type: int, advanced: FALSE, range: [1,2147483647], default: 20]
            'compression/largestrepr/minnleaves': '20',
            # number of runs in the constrained part.
            # [type: int, advanced: FALSE, range: [1,2147483647], default: 5]
            'compression/largestrepr/iterations': '5',
            # minimal number of common variables.
            # [type: int, advanced: FALSE, range: [1,2147483647], default: 3]
            'compression/largestrepr/mincommonvars': '3',
            # priority of compression <weakcompr>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 1000]
            'compression/weakcompr/priority': '1000',
            # minimal number of leave nodes for calling tree compression <weakcompr>
            # [type: int, advanced: FALSE, range: [1,2147483647], default: 50]
            'compression/weakcompr/minnleaves': '50',
            # convert constraints into nodes
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'compression/weakcompr/convertconss': 'FALSE',
            # priority of heuristic <actconsdiving>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1003700]
            'heuristics/actconsdiving/priority': '-1003700',
            # frequency for calling primal heuristic <actconsdiving> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'heuristics/actconsdiving/freq': '-1',
            # frequency offset for calling primal heuristic <actconsdiving>
            # [type: int, advanced: FALSE, range: [0,65534], default: 5]
            'heuristics/actconsdiving/freqofs': '5',
            # maximal depth level to call primal heuristic <actconsdiving> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/actconsdiving/maxdepth': '-1',
            # minimal relative depth to start diving
            # [type: real, advanced: TRUE, range: [0,1], default: 0]
            'heuristics/actconsdiving/minreldepth': '0',
            # maximal relative depth to start diving
            # [type: real, advanced: TRUE, range: [0,1], default: 1]
            'heuristics/actconsdiving/maxreldepth': '1',
            # maximal fraction of diving LP iterations compared to node LP iterations
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 0.05]
            'heuristics/actconsdiving/maxlpiterquot': '0.05',
            # additional number of allowed LP iterations
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 1000]
            'heuristics/actconsdiving/maxlpiterofs': '1000',
            # maximal quotient (curlowerbound - lowerbound)/(cutoffbound - lowerbound) where diving is performed (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1], default: 0.8]
            'heuristics/actconsdiving/maxdiveubquot': '0.8',
            # maximal quotient (curlowerbound - lowerbound)/(avglowerbound - lowerbound) where diving is performed (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0]
            'heuristics/actconsdiving/maxdiveavgquot': '0',
            # maximal UBQUOT when no solution was found yet (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1], default: 1]
            'heuristics/actconsdiving/maxdiveubquotnosol': '1',
            # maximal AVGQUOT when no solution was found yet (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 1]
            'heuristics/actconsdiving/maxdiveavgquotnosol': '1',
            # use one level of backtracking if infeasibility is encountered?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/actconsdiving/backtrack': 'TRUE',
            # percentage of immediate domain changes during probing to trigger LP resolve
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 0.15]
            'heuristics/actconsdiving/lpresolvedomchgquot': '0.15',
            # LP solve frequency for diving heuristics (0: only after enough domain changes have been found)
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 0]
            'heuristics/actconsdiving/lpsolvefreq': '0',
            # should only LP branching candidates be considered instead of the slower but more general constraint handler diving variable selection?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/actconsdiving/onlylpbranchcands': 'TRUE',
            # priority of heuristic <adaptivediving>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -70000]
            'heuristics/adaptivediving/priority': '-70000',
            # frequency for calling primal heuristic <adaptivediving> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 5]
            'heuristics/adaptivediving/freq': '5',
            # frequency offset for calling primal heuristic <adaptivediving>
            # [type: int, advanced: FALSE, range: [0,65534], default: 3]
            'heuristics/adaptivediving/freqofs': '3',
            # maximal depth level to call primal heuristic <adaptivediving> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/adaptivediving/maxdepth': '-1',
            # parameter that increases probability of exploration among divesets (only active if seltype is 'e')
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 1]
            'heuristics/adaptivediving/epsilon': '1',
            # score parameter for selection: minimize either average 'n'odes, LP 'i'terations,backtrack/'c'onflict ratio, 'd'epth, 1 / 's'olutions, or 1 / solutions'u'ccess
            # [type: char, advanced: FALSE, range: {cdinsu}, default: c]
            'heuristics/adaptivediving/scoretype': 'c',
            # selection strategy: (e)psilon-greedy, (w)eighted distribution, (n)ext diving
            # [type: char, advanced: FALSE, range: {enw}, default: w]
            'heuristics/adaptivediving/seltype': 'w',
            # should the heuristic use its own statistics, or shared statistics?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/adaptivediving/useadaptivecontext': 'FALSE',
            # coefficient c to decrease initial confidence (calls + 1.0) / (calls + c) in scores
            # [type: real, advanced: FALSE, range: [1,2147483647], default: 10]
            'heuristics/adaptivediving/selconfidencecoeff': '10',
            # maximal fraction of diving LP iterations compared to node LP iterations
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 0.1]
            'heuristics/adaptivediving/maxlpiterquot': '0.1',
            # additional number of allowed LP iterations
            # [type: longint, advanced: FALSE, range: [0,2147483647], default: 1500]
            'heuristics/adaptivediving/maxlpiterofs': '1500',
            # weight of incumbent solutions compared to other solutions in computation of LP iteration limit
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 10]
            'heuristics/adaptivediving/bestsolweight': '10',
            # priority of heuristic <bound>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1107000]
            'heuristics/bound/priority': '-1107000',
            # frequency for calling primal heuristic <bound> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'heuristics/bound/freq': '-1',
            # frequency offset for calling primal heuristic <bound>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/bound/freqofs': '0',
            # maximal depth level to call primal heuristic <bound> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/bound/maxdepth': '-1',
            # Should heuristic only be executed if no primal solution was found, yet?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/bound/onlywithoutsol': 'TRUE',
            # maximum number of propagation rounds during probing (-1 infinity, -2 parameter settings)
            # [type: int, advanced: TRUE, range: [-1,536870911], default: 0]
            'heuristics/bound/maxproprounds': '0',
            # to which bound should integer variables be fixed? ('l'ower, 'u'pper, or 'b'oth)
            # [type: char, advanced: FALSE, range: {lub}, default: l]
            'heuristics/bound/bound': 'l',
            # priority of heuristic <clique>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 5000]
            'heuristics/clique/priority': '5000',
            # frequency for calling primal heuristic <clique> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 0]
            'heuristics/clique/freq': '0',
            # frequency offset for calling primal heuristic <clique>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/clique/freqofs': '0',
            # maximal depth level to call primal heuristic <clique> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/clique/maxdepth': '-1',
            # minimum percentage of integer variables that have to be fixable
            # [type: real, advanced: FALSE, range: [0,1], default: 0.65]
            'heuristics/clique/minintfixingrate': '0.65',
            # minimum percentage of fixed variables in the sub-MIP
            # [type: real, advanced: FALSE, range: [0,1], default: 0.65]
            'heuristics/clique/minmipfixingrate': '0.65',
            # maximum number of nodes to regard in the subproblem
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 5000]
            'heuristics/clique/maxnodes': '5000',
            # number of nodes added to the contingent of the total nodes
            # [type: longint, advanced: FALSE, range: [0,9223372036854775807], default: 500]
            'heuristics/clique/nodesofs': '500',
            # minimum number of nodes required to start the subproblem
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 500]
            'heuristics/clique/minnodes': '500',
            # contingent of sub problem nodes in relation to the number of nodes of the original problem
            # [type: real, advanced: FALSE, range: [0,1], default: 0.1]
            'heuristics/clique/nodesquot': '0.1',
            # factor by which clique heuristic should at least improve the incumbent
            # [type: real, advanced: TRUE, range: [0,1], default: 0.01]
            'heuristics/clique/minimprove': '0.01',
            # maximum number of propagation rounds during probing (-1 infinity)
            # [type: int, advanced: TRUE, range: [-1,536870911], default: 2]
            'heuristics/clique/maxproprounds': '2',
            # should all active cuts from cutpool be copied to constraints in subproblem?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/clique/copycuts': 'TRUE',
            # should more variables be fixed based on variable locks if the fixing rate was not reached?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/clique/uselockfixings': 'FALSE',
            # maximum number of backtracks during the fixing process
            # [type: int, advanced: TRUE, range: [-1,536870911], default: 10]
            'heuristics/clique/maxbacktracks': '10',
            # priority of heuristic <coefdiving>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1001000]
            'heuristics/coefdiving/priority': '-1001000',
            # frequency for calling primal heuristic <coefdiving> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'heuristics/coefdiving/freq': '-1',
            # frequency offset for calling primal heuristic <coefdiving>
            # [type: int, advanced: FALSE, range: [0,65534], default: 1]
            'heuristics/coefdiving/freqofs': '1',
            # maximal depth level to call primal heuristic <coefdiving> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/coefdiving/maxdepth': '-1',
            # minimal relative depth to start diving
            # [type: real, advanced: TRUE, range: [0,1], default: 0]
            'heuristics/coefdiving/minreldepth': '0',
            # maximal relative depth to start diving
            # [type: real, advanced: TRUE, range: [0,1], default: 1]
            'heuristics/coefdiving/maxreldepth': '1',
            # maximal fraction of diving LP iterations compared to node LP iterations
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 0.05]
            'heuristics/coefdiving/maxlpiterquot': '0.05',
            # additional number of allowed LP iterations
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 1000]
            'heuristics/coefdiving/maxlpiterofs': '1000',
            # maximal quotient (curlowerbound - lowerbound)/(cutoffbound - lowerbound) where diving is performed (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1], default: 0.8]
            'heuristics/coefdiving/maxdiveubquot': '0.8',
            # maximal quotient (curlowerbound - lowerbound)/(avglowerbound - lowerbound) where diving is performed (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0]
            'heuristics/coefdiving/maxdiveavgquot': '0',
            # maximal UBQUOT when no solution was found yet (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1], default: 0.1]
            'heuristics/coefdiving/maxdiveubquotnosol': '0.1',
            # maximal AVGQUOT when no solution was found yet (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0]
            'heuristics/coefdiving/maxdiveavgquotnosol': '0',
            # use one level of backtracking if infeasibility is encountered?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/coefdiving/backtrack': 'TRUE',
            # percentage of immediate domain changes during probing to trigger LP resolve
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 0.15]
            'heuristics/coefdiving/lpresolvedomchgquot': '0.15',
            # LP solve frequency for diving heuristics (0: only after enough domain changes have been found)
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 0]
            'heuristics/coefdiving/lpsolvefreq': '0',
            # should only LP branching candidates be considered instead of the slower but more general constraint handler diving variable selection?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/coefdiving/onlylpbranchcands': 'FALSE',
            # priority of heuristic <completesol>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 0]
            'heuristics/completesol/priority': '0',
            # frequency for calling primal heuristic <completesol> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 0]
            'heuristics/completesol/freq': '0',
            # frequency offset for calling primal heuristic <completesol>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/completesol/freqofs': '0',
            # maximal depth level to call primal heuristic <completesol> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: 0]
            'heuristics/completesol/maxdepth': '0',
            # maximum number of nodes to regard in the subproblem
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 5000]
            'heuristics/completesol/maxnodes': '5000',
            # minimum number of nodes required to start the subproblem
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 50]
            'heuristics/completesol/minnodes': '50',
            # maximal rate of unknown solution values
            # [type: real, advanced: FALSE, range: [0,1], default: 0.85]
            'heuristics/completesol/maxunknownrate': '0.85',
            # should all subproblem solutions be added to the original SCIP?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/completesol/addallsols': 'FALSE',
            # number of nodes added to the contingent of the total nodes
            # [type: longint, advanced: FALSE, range: [0,9223372036854775807], default: 500]
            'heuristics/completesol/nodesofs': '500',
            # contingent of sub problem nodes in relation to the number of nodes of the original problem
            # [type: real, advanced: FALSE, range: [0,1], default: 0.1]
            'heuristics/completesol/nodesquot': '0.1',
            # factor by which the limit on the number of LP depends on the node limit
            # [type: real, advanced: TRUE, range: [1,1.79769313486232e+308], default: 2]
            'heuristics/completesol/lplimfac': '2',
            # weight of the original objective function (1: only original objective)
            # [type: real, advanced: TRUE, range: [0.001,1], default: 1]
            'heuristics/completesol/objweight': '1',
            # bound widening factor applied to continuous variables (0: fix variables to given solution values, 1: relax to global bounds)
            # [type: real, advanced: TRUE, range: [0,1], default: 0.1]
            'heuristics/completesol/boundwidening': '0.1',
            # factor by which the incumbent should be improved at least
            # [type: real, advanced: TRUE, range: [0,1], default: 0.01]
            'heuristics/completesol/minimprove': '0.01',
            # should number of continuous variables be ignored?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/completesol/ignorecont': 'FALSE',
            # heuristic stops, if the given number of improving solutions were found (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 5]
            'heuristics/completesol/solutions': '5',
            # maximal number of iterations in propagation (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 10]
            'heuristics/completesol/maxproprounds': '10',
            # should the heuristic run before presolving?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/completesol/beforepresol': 'TRUE',
            # maximal number of LP iterations (-1: no limit)
            # [type: longint, advanced: FALSE, range: [-1,9223372036854775807], default: -1]
            'heuristics/completesol/maxlpiter': '-1',
            # maximal number of continuous variables after presolving
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'heuristics/completesol/maxcontvars': '-1',
            # priority of heuristic <conflictdiving>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1000100]
            'heuristics/conflictdiving/priority': '-1000100',
            # frequency for calling primal heuristic <conflictdiving> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 10]
            'heuristics/conflictdiving/freq': '10',
            # frequency offset for calling primal heuristic <conflictdiving>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/conflictdiving/freqofs': '0',
            # maximal depth level to call primal heuristic <conflictdiving> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/conflictdiving/maxdepth': '-1',
            # minimal relative depth to start diving
            # [type: real, advanced: TRUE, range: [0,1], default: 0]
            'heuristics/conflictdiving/minreldepth': '0',
            # maximal relative depth to start diving
            # [type: real, advanced: TRUE, range: [0,1], default: 1]
            'heuristics/conflictdiving/maxreldepth': '1',
            # maximal fraction of diving LP iterations compared to node LP iterations
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 0.15]
            'heuristics/conflictdiving/maxlpiterquot': '0.15',
            # additional number of allowed LP iterations
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 1000]
            'heuristics/conflictdiving/maxlpiterofs': '1000',
            # maximal quotient (curlowerbound - lowerbound)/(cutoffbound - lowerbound) where diving is performed (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1], default: 0.8]
            'heuristics/conflictdiving/maxdiveubquot': '0.8',
            # maximal quotient (curlowerbound - lowerbound)/(avglowerbound - lowerbound) where diving is performed (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0]
            'heuristics/conflictdiving/maxdiveavgquot': '0',
            # maximal UBQUOT when no solution was found yet (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1], default: 0.1]
            'heuristics/conflictdiving/maxdiveubquotnosol': '0.1',
            # maximal AVGQUOT when no solution was found yet (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0]
            'heuristics/conflictdiving/maxdiveavgquotnosol': '0',
            # use one level of backtracking if infeasibility is encountered?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/conflictdiving/backtrack': 'TRUE',
            # percentage of immediate domain changes during probing to trigger LP resolve
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 0.15]
            'heuristics/conflictdiving/lpresolvedomchgquot': '0.15',
            # LP solve frequency for diving heuristics (0: only after enough domain changes have been found)
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 0]
            'heuristics/conflictdiving/lpsolvefreq': '0',
            # should only LP branching candidates be considered instead of the slower but more general constraint handler diving variable selection?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/conflictdiving/onlylpbranchcands': 'FALSE',
            # try to maximize the violation
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/conflictdiving/maxviol': 'TRUE',
            # perform rounding like coefficient diving
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/conflictdiving/likecoef': 'FALSE',
            # minimal number of conflict locks per variable
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 5]
            'heuristics/conflictdiving/minconflictlocks': '5',
            # weight used in a convex combination of conflict and variable locks
            # [type: real, advanced: TRUE, range: [0,1], default: 0.75]
            'heuristics/conflictdiving/lockweight': '0.75',
            # priority of heuristic <crossover>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1104000]
            'heuristics/crossover/priority': '-1104000',
            # frequency for calling primal heuristic <crossover> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 30]
            'heuristics/crossover/freq': '30',
            # frequency offset for calling primal heuristic <crossover>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/crossover/freqofs': '0',
            # maximal depth level to call primal heuristic <crossover> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/crossover/maxdepth': '-1',
            # number of nodes added to the contingent of the total nodes
            # [type: longint, advanced: FALSE, range: [0,9223372036854775807], default: 500]
            'heuristics/crossover/nodesofs': '500',
            # maximum number of nodes to regard in the subproblem
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 5000]
            'heuristics/crossover/maxnodes': '5000',
            # minimum number of nodes required to start the subproblem
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 50]
            'heuristics/crossover/minnodes': '50',
            # number of solutions to be taken into account
            # [type: int, advanced: FALSE, range: [2,2147483647], default: 3]
            'heuristics/crossover/nusedsols': '3',
            # number of nodes without incumbent change that heuristic should wait
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 200]
            'heuristics/crossover/nwaitingnodes': '200',
            # contingent of sub problem nodes in relation to the number of nodes of the original problem
            # [type: real, advanced: FALSE, range: [0,1], default: 0.1]
            'heuristics/crossover/nodesquot': '0.1',
            # minimum percentage of integer variables that have to be fixed
            # [type: real, advanced: FALSE, range: [0,1], default: 0.666]
            'heuristics/crossover/minfixingrate': '0.666',
            # factor by which Crossover should at least improve the incumbent
            # [type: real, advanced: TRUE, range: [0,1], default: 0.01]
            'heuristics/crossover/minimprove': '0.01',
            # factor by which the limit on the number of LP depends on the node limit
            # [type: real, advanced: TRUE, range: [1,1.79769313486232e+308], default: 2]
            'heuristics/crossover/lplimfac': '2',
            # should the choice which sols to take be randomized?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/crossover/randomization': 'TRUE',
            # should the nwaitingnodes parameter be ignored at the root node?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/crossover/dontwaitatroot': 'FALSE',
            # should subproblem be created out of the rows in the LP rows?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/crossover/uselprows': 'FALSE',
            # if uselprows == FALSE, should all active cuts from cutpool be copied to constraints in subproblem?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/crossover/copycuts': 'TRUE',
            # should the subproblem be permuted to increase diversification?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/crossover/permute': 'FALSE',
            # limit on number of improving incumbent solutions in sub-CIP
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'heuristics/crossover/bestsollimit': '-1',
            # should uct node selection be used at the beginning of the search?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/crossover/useuct': 'FALSE',
            # priority of heuristic <dins>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1105000]
            'heuristics/dins/priority': '-1105000',
            # frequency for calling primal heuristic <dins> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'heuristics/dins/freq': '-1',
            # frequency offset for calling primal heuristic <dins>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/dins/freqofs': '0',
            # maximal depth level to call primal heuristic <dins> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/dins/maxdepth': '-1',
            # number of nodes added to the contingent of the total nodes
            # [type: longint, advanced: FALSE, range: [0,9223372036854775807], default: 5000]
            'heuristics/dins/nodesofs': '5000',
            # contingent of sub problem nodes in relation to the number of nodes of the original problem
            # [type: real, advanced: FALSE, range: [0,1], default: 0.05]
            'heuristics/dins/nodesquot': '0.05',
            # minimum number of nodes required to start the subproblem
            # [type: longint, advanced: FALSE, range: [0,9223372036854775807], default: 50]
            'heuristics/dins/minnodes': '50',
            # number of pool-solutions to be checked for flag array update (for hard fixing of binary variables)
            # [type: int, advanced: FALSE, range: [1,2147483647], default: 5]
            'heuristics/dins/solnum': '5',
            # radius (using Manhattan metric) of the incumbent's neighborhood to be searched
            # [type: int, advanced: FALSE, range: [1,2147483647], default: 18]
            'heuristics/dins/neighborhoodsize': '18',
            # maximum number of nodes to regard in the subproblem
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 5000]
            'heuristics/dins/maxnodes': '5000',
            # factor by which dins should at least improve the incumbent
            # [type: real, advanced: TRUE, range: [0,1], default: 0.01]
            'heuristics/dins/minimprove': '0.01',
            # number of nodes without incumbent change that heuristic should wait
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 200]
            'heuristics/dins/nwaitingnodes': '200',
            # factor by which the limit on the number of LP depends on the node limit
            # [type: real, advanced: TRUE, range: [1,1.79769313486232e+308], default: 1.5]
            'heuristics/dins/lplimfac': '1.5',
            # minimum percentage of integer variables that have to be fixable
            # [type: real, advanced: FALSE, range: [0,1], default: 0.3]
            'heuristics/dins/minfixingrate': '0.3',
            # should subproblem be created out of the rows in the LP rows?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/dins/uselprows': 'FALSE',
            # if uselprows == FALSE, should all active cuts from cutpool be copied to constraints in subproblem?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/dins/copycuts': 'TRUE',
            # should uct node selection be used at the beginning of the search?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/dins/useuct': 'FALSE',
            # limit on number of improving incumbent solutions in sub-CIP
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 3]
            'heuristics/dins/bestsollimit': '3',
            # priority of heuristic <distributiondiving>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1003300]
            'heuristics/distributiondiving/priority': '-1003300',
            # frequency for calling primal heuristic <distributiondiving> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 10]
            'heuristics/distributiondiving/freq': '10',
            # frequency offset for calling primal heuristic <distributiondiving>
            # [type: int, advanced: FALSE, range: [0,65534], default: 3]
            'heuristics/distributiondiving/freqofs': '3',
            # maximal depth level to call primal heuristic <distributiondiving> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/distributiondiving/maxdepth': '-1',
            # minimal relative depth to start diving
            # [type: real, advanced: TRUE, range: [0,1], default: 0]
            'heuristics/distributiondiving/minreldepth': '0',
            # maximal relative depth to start diving
            # [type: real, advanced: TRUE, range: [0,1], default: 1]
            'heuristics/distributiondiving/maxreldepth': '1',
            # maximal fraction of diving LP iterations compared to node LP iterations
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 0.05]
            'heuristics/distributiondiving/maxlpiterquot': '0.05',
            # additional number of allowed LP iterations
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 1000]
            'heuristics/distributiondiving/maxlpiterofs': '1000',
            # maximal quotient (curlowerbound - lowerbound)/(cutoffbound - lowerbound) where diving is performed (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1], default: 0.8]
            'heuristics/distributiondiving/maxdiveubquot': '0.8',
            # maximal quotient (curlowerbound - lowerbound)/(avglowerbound - lowerbound) where diving is performed (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0]
            'heuristics/distributiondiving/maxdiveavgquot': '0',
            # maximal UBQUOT when no solution was found yet (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1], default: 0.1]
            'heuristics/distributiondiving/maxdiveubquotnosol': '0.1',
            # maximal AVGQUOT when no solution was found yet (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0]
            'heuristics/distributiondiving/maxdiveavgquotnosol': '0',
            # use one level of backtracking if infeasibility is encountered?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/distributiondiving/backtrack': 'TRUE',
            # percentage of immediate domain changes during probing to trigger LP resolve
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 0.15]
            'heuristics/distributiondiving/lpresolvedomchgquot': '0.15',
            # LP solve frequency for diving heuristics (0: only after enough domain changes have been found)
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 0]
            'heuristics/distributiondiving/lpsolvefreq': '0',
            # should only LP branching candidates be considered instead of the slower but more general constraint handler diving variable selection?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/distributiondiving/onlylpbranchcands': 'TRUE',
            # the score;largest 'd'ifference, 'l'owest cumulative probability,'h'ighest c.p., 'v'otes lowest c.p., votes highest c.p.('w'), 'r'evolving
            # [type: char, advanced: TRUE, range: {lvdhwr}, default: r]
            'heuristics/distributiondiving/scoreparam': 'r',
            # priority of heuristic <dualval>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 0]
            'heuristics/dualval/priority': '0',
            # frequency for calling primal heuristic <dualval> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'heuristics/dualval/freq': '-1',
            # frequency offset for calling primal heuristic <dualval>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/dualval/freqofs': '0',
            # maximal depth level to call primal heuristic <dualval> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/dualval/maxdepth': '-1',
            # exit if objective doesn't improve
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/dualval/forceimprovements': 'FALSE',
            # add constraint to ensure that discrete vars are improving
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/dualval/onlycheaper': 'TRUE',
            # disable the heuristic if it was not called at a leaf of the B&B tree
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/dualval/onlyleaves': 'FALSE',
            # relax the indicator variables by introducing continuous copies
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/dualval/relaxindicators': 'FALSE',
            # relax the continous variables
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/dualval/relaxcontvars': 'FALSE',
            # verblevel of the heuristic, default is 0 to display nothing
            # [type: int, advanced: FALSE, range: [0,4], default: 0]
            'heuristics/dualval/heurverblevel': '0',
            # verblevel of the nlp solver, can be 0 or 1
            # [type: int, advanced: FALSE, range: [0,1], default: 0]
            'heuristics/dualval/nlpverblevel': '0',
            # number of ranks that should be displayed when the heuristic is called
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 10]
            'heuristics/dualval/rankvalue': '10',
            # maximal number of recursive calls of the heuristic (if dynamicdepth is off)
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 25]
            'heuristics/dualval/maxcalls': '25',
            # says if and how the recursion depth is computed at runtime
            # [type: int, advanced: FALSE, range: [0,1], default: 0]
            'heuristics/dualval/dynamicdepth': '0',
            # maximal number of variables that may have maximal rank, quit if there are more, turn off by setting -1
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 50]
            'heuristics/dualval/maxequalranks': '50',
            # minimal gap for which we still run the heuristic, if gap is less we return without doing anything
            # [type: real, advanced: FALSE, range: [0,100], default: 5]
            'heuristics/dualval/mingap': '5',
            # value added to objective of slack variables, must not be zero
            # [type: real, advanced: FALSE, range: [0.1,1e+20], default: 1]
            'heuristics/dualval/lambdaslack': '1',
            # scaling factor for the objective function
            # [type: real, advanced: FALSE, range: [0,1], default: 0]
            'heuristics/dualval/lambdaobj': '0',
            # priority of heuristic <farkasdiving>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -900000]
            'heuristics/farkasdiving/priority': '-900000',
            # frequency for calling primal heuristic <farkasdiving> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 10]
            'heuristics/farkasdiving/freq': '10',
            # frequency offset for calling primal heuristic <farkasdiving>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/farkasdiving/freqofs': '0',
            # maximal depth level to call primal heuristic <farkasdiving> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/farkasdiving/maxdepth': '-1',
            # minimal relative depth to start diving
            # [type: real, advanced: TRUE, range: [0,1], default: 0]
            'heuristics/farkasdiving/minreldepth': '0',
            # maximal relative depth to start diving
            # [type: real, advanced: TRUE, range: [0,1], default: 1]
            'heuristics/farkasdiving/maxreldepth': '1',
            # maximal fraction of diving LP iterations compared to node LP iterations
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 0.05]
            'heuristics/farkasdiving/maxlpiterquot': '0.05',
            # additional number of allowed LP iterations
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 1000]
            'heuristics/farkasdiving/maxlpiterofs': '1000',
            # maximal quotient (curlowerbound - lowerbound)/(cutoffbound - lowerbound) where diving is performed (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1], default: 0.8]
            'heuristics/farkasdiving/maxdiveubquot': '0.8',
            # maximal quotient (curlowerbound - lowerbound)/(avglowerbound - lowerbound) where diving is performed (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0]
            'heuristics/farkasdiving/maxdiveavgquot': '0',
            # maximal UBQUOT when no solution was found yet (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1], default: 0.1]
            'heuristics/farkasdiving/maxdiveubquotnosol': '0.1',
            # maximal AVGQUOT when no solution was found yet (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0]
            'heuristics/farkasdiving/maxdiveavgquotnosol': '0',
            # use one level of backtracking if infeasibility is encountered?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/farkasdiving/backtrack': 'TRUE',
            # percentage of immediate domain changes during probing to trigger LP resolve
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 0.15]
            'heuristics/farkasdiving/lpresolvedomchgquot': '0.15',
            # LP solve frequency for diving heuristics (0: only after enough domain changes have been found)
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 1]
            'heuristics/farkasdiving/lpsolvefreq': '1',
            # should only LP branching candidates be considered instead of the slower but more general constraint handler diving variable selection?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/farkasdiving/onlylpbranchcands': 'FALSE',
            # should diving candidates be checked before running?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/farkasdiving/checkcands': 'FALSE',
            # should the score be scaled?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/farkasdiving/scalescore': 'TRUE',
            # should the heuristic only run within the tree if at least one solution was found at the root node?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/farkasdiving/rootsuccess': 'TRUE',
            # maximal occurance factor of an objective coefficient
            # [type: real, advanced: TRUE, range: [0,1], default: 1]
            'heuristics/farkasdiving/maxobjocc': '1',
            # minimal objective dynamism (log) to run
            # [type: real, advanced: TRUE, range: [0,1e+20], default: 0.0001]
            'heuristics/farkasdiving/objdynamism': '0.0001',
            # scale score by [f]ractionality or [i]mpact on farkasproof
            # [type: char, advanced: TRUE, range: {fi}, default: i]
            'heuristics/farkasdiving/scaletype': 'i',
            # priority of heuristic <feaspump>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1000000]
            'heuristics/feaspump/priority': '-1000000',
            # frequency for calling primal heuristic <feaspump> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 20]
            'heuristics/feaspump/freq': '20',
            # frequency offset for calling primal heuristic <feaspump>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/feaspump/freqofs': '0',
            # maximal depth level to call primal heuristic <feaspump> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/feaspump/maxdepth': '-1',
            # maximal fraction of diving LP iterations compared to node LP iterations
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 0.01]
            'heuristics/feaspump/maxlpiterquot': '0.01',
            # factor by which the regard of the objective is decreased in each round, 1.0 for dynamic
            # [type: real, advanced: FALSE, range: [0,1], default: 0.1]
            'heuristics/feaspump/objfactor': '0.1',
            # initial weight of the objective function in the convex combination
            # [type: real, advanced: FALSE, range: [0,1], default: 1]
            'heuristics/feaspump/alpha': '1',
            # threshold difference for the convex parameter to perform perturbation
            # [type: real, advanced: FALSE, range: [0,1], default: 1]
            'heuristics/feaspump/alphadiff': '1',
            # additional number of allowed LP iterations
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 1000]
            'heuristics/feaspump/maxlpiterofs': '1000',
            # total number of feasible solutions found up to which heuristic is called (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 10]
            'heuristics/feaspump/maxsols': '10',
            # maximal number of pumping loops (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 10000]
            'heuristics/feaspump/maxloops': '10000',
            # maximal number of pumping rounds without fractionality improvement (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 10]
            'heuristics/feaspump/maxstallloops': '10',
            # minimum number of random variables to flip, if a 1-cycle is encountered
            # [type: int, advanced: TRUE, range: [1,2147483647], default: 10]
            'heuristics/feaspump/minflips': '10',
            # maximum length of cycles to be checked explicitly in each round
            # [type: int, advanced: TRUE, range: [1,100], default: 3]
            'heuristics/feaspump/cyclelength': '3',
            # number of iterations until a random perturbation is forced
            # [type: int, advanced: TRUE, range: [1,2147483647], default: 100]
            'heuristics/feaspump/perturbfreq': '100',
            # radius (using Manhattan metric) of the neighborhood to be searched in stage 3
            # [type: int, advanced: FALSE, range: [1,2147483647], default: 18]
            'heuristics/feaspump/neighborhoodsize': '18',
            # should the feasibility pump be called at root node before cut separation?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/feaspump/beforecuts': 'TRUE',
            # should an iterative round-and-propagate scheme be used to find the integral points?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/feaspump/usefp20': 'FALSE',
            # should a random perturbation be performed if a feasible solution was found?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/feaspump/pertsolfound': 'TRUE',
            # should we solve a local branching sub-MIP if no solution could be found?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/feaspump/stage3': 'FALSE',
            # should all active cuts from cutpool be copied to constraints in subproblem?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/feaspump/copycuts': 'TRUE',
            # priority of heuristic <fixandinfer>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -500000]
            'heuristics/fixandinfer/priority': '-500000',
            # frequency for calling primal heuristic <fixandinfer> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'heuristics/fixandinfer/freq': '-1',
            # frequency offset for calling primal heuristic <fixandinfer>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/fixandinfer/freqofs': '0',
            # maximal depth level to call primal heuristic <fixandinfer> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/fixandinfer/maxdepth': '-1',
            # maximal number of propagation rounds in probing subproblems (-1: no limit, 0: auto)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 0]
            'heuristics/fixandinfer/proprounds': '0',
            # minimal number of fixings to apply before dive may be aborted
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 100]
            'heuristics/fixandinfer/minfixings': '100',
            # priority of heuristic <fracdiving>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1003000]
            'heuristics/fracdiving/priority': '-1003000',
            # frequency for calling primal heuristic <fracdiving> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 10]
            'heuristics/fracdiving/freq': '10',
            # frequency offset for calling primal heuristic <fracdiving>
            # [type: int, advanced: FALSE, range: [0,65534], default: 3]
            'heuristics/fracdiving/freqofs': '3',
            # maximal depth level to call primal heuristic <fracdiving> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/fracdiving/maxdepth': '-1',
            # minimal relative depth to start diving
            # [type: real, advanced: TRUE, range: [0,1], default: 0]
            'heuristics/fracdiving/minreldepth': '0',
            # maximal relative depth to start diving
            # [type: real, advanced: TRUE, range: [0,1], default: 1]
            'heuristics/fracdiving/maxreldepth': '1',
            # maximal fraction of diving LP iterations compared to node LP iterations
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 0.05]
            'heuristics/fracdiving/maxlpiterquot': '0.05',
            # additional number of allowed LP iterations
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 1000]
            'heuristics/fracdiving/maxlpiterofs': '1000',
            # maximal quotient (curlowerbound - lowerbound)/(cutoffbound - lowerbound) where diving is performed (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1], default: 0.8]
            'heuristics/fracdiving/maxdiveubquot': '0.8',
            # maximal quotient (curlowerbound - lowerbound)/(avglowerbound - lowerbound) where diving is performed (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0]
            'heuristics/fracdiving/maxdiveavgquot': '0',
            # maximal UBQUOT when no solution was found yet (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1], default: 0.1]
            'heuristics/fracdiving/maxdiveubquotnosol': '0.1',
            # maximal AVGQUOT when no solution was found yet (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0]
            'heuristics/fracdiving/maxdiveavgquotnosol': '0',
            # use one level of backtracking if infeasibility is encountered?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/fracdiving/backtrack': 'TRUE',
            # percentage of immediate domain changes during probing to trigger LP resolve
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 0.15]
            'heuristics/fracdiving/lpresolvedomchgquot': '0.15',
            # LP solve frequency for diving heuristics (0: only after enough domain changes have been found)
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 0]
            'heuristics/fracdiving/lpsolvefreq': '0',
            # should only LP branching candidates be considered instead of the slower but more general constraint handler diving variable selection?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/fracdiving/onlylpbranchcands': 'FALSE',
            # priority of heuristic <gins>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1103000]
            'heuristics/gins/priority': '-1103000',
            # frequency for calling primal heuristic <gins> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 20]
            'heuristics/gins/freq': '20',
            # frequency offset for calling primal heuristic <gins>
            # [type: int, advanced: FALSE, range: [0,65534], default: 8]
            'heuristics/gins/freqofs': '8',
            # maximal depth level to call primal heuristic <gins> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/gins/maxdepth': '-1',
            # number of nodes added to the contingent of the total nodes
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 500]
            'heuristics/gins/nodesofs': '500',
            # maximum number of nodes to regard in the subproblem
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 5000]
            'heuristics/gins/maxnodes': '5000',
            # minimum number of nodes required to start the subproblem
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 50]
            'heuristics/gins/minnodes': '50',
            # number of nodes without incumbent change that heuristic should wait
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 100]
            'heuristics/gins/nwaitingnodes': '100',
            # contingent of sub problem nodes in relation to the number of nodes of the original problem
            # [type: real, advanced: FALSE, range: [0,1], default: 0.15]
            'heuristics/gins/nodesquot': '0.15',
            # percentage of integer variables that have to be fixed
            # [type: real, advanced: FALSE, range: [1e-06,0.999999], default: 0.66]
            'heuristics/gins/minfixingrate': '0.66',
            # factor by which gins should at least improve the incumbent
            # [type: real, advanced: TRUE, range: [0,1], default: 0.01]
            'heuristics/gins/minimprove': '0.01',
            # should subproblem be created out of the rows in the LP rows?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/gins/uselprows': 'FALSE',
            # if uselprows == FALSE, should all active cuts from cutpool be copied to constraints in subproblem?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/gins/copycuts': 'TRUE',
            # should continuous variables outside the neighborhoods be fixed?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/gins/fixcontvars': 'FALSE',
            # limit on number of improving incumbent solutions in sub-CIP
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 3]
            'heuristics/gins/bestsollimit': '3',
            # maximum distance to selected variable to enter the subproblem, or -1 to select the distance that best approximates the minimum fixing rate from below
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 3]
            'heuristics/gins/maxdistance': '3',
            # the reference point to compute the neighborhood potential: (r)oot, (l)ocal lp, or (p)seudo solution
            # [type: char, advanced: TRUE, range: {lpr}, default: r]
            'heuristics/gins/potential': 'r',
            # should the heuristic solve a sequence of sub-MIP's around the first selected variable
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/gins/userollinghorizon': 'TRUE',
            # should dense constraints (at least as dense as 1 - minfixingrate) be ignored by connectivity graph?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/gins/relaxdenseconss': 'FALSE',
            # limiting percentage for variables already used in sub-SCIPs to terminate rolling horizon approach
            # [type: real, advanced: TRUE, range: [0,1], default: 0.4]
            'heuristics/gins/rollhorizonlimfac': '0.4',
            # overlap of blocks between runs - 0.0: no overlap, 1.0: shift by only 1 block
            # [type: real, advanced: TRUE, range: [0,1], default: 0]
            'heuristics/gins/overlap': '0',
            # should user decompositions be considered, if available?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/gins/usedecomp': 'TRUE',
            # should user decompositions be considered for initial selection in rolling horizon, if available?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/gins/usedecomprollhorizon': 'FALSE',
            # should random initial variable selection be used if decomposition was not successful?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/gins/useselfallback': 'TRUE',
            # should blocks be treated consecutively (sorted by ascending label?)
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/gins/consecutiveblocks': 'TRUE',
            # priority of heuristic <guideddiving>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1007000]
            'heuristics/guideddiving/priority': '-1007000',
            # frequency for calling primal heuristic <guideddiving> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 10]
            'heuristics/guideddiving/freq': '10',
            # frequency offset for calling primal heuristic <guideddiving>
            # [type: int, advanced: FALSE, range: [0,65534], default: 7]
            'heuristics/guideddiving/freqofs': '7',
            # maximal depth level to call primal heuristic <guideddiving> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/guideddiving/maxdepth': '-1',
            # minimal relative depth to start diving
            # [type: real, advanced: TRUE, range: [0,1], default: 0]
            'heuristics/guideddiving/minreldepth': '0',
            # maximal relative depth to start diving
            # [type: real, advanced: TRUE, range: [0,1], default: 1]
            'heuristics/guideddiving/maxreldepth': '1',
            # maximal fraction of diving LP iterations compared to node LP iterations
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 0.05]
            'heuristics/guideddiving/maxlpiterquot': '0.05',
            # additional number of allowed LP iterations
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 1000]
            'heuristics/guideddiving/maxlpiterofs': '1000',
            # maximal quotient (curlowerbound - lowerbound)/(cutoffbound - lowerbound) where diving is performed (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1], default: 0.8]
            'heuristics/guideddiving/maxdiveubquot': '0.8',
            # maximal quotient (curlowerbound - lowerbound)/(avglowerbound - lowerbound) where diving is performed (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0]
            'heuristics/guideddiving/maxdiveavgquot': '0',
            # maximal UBQUOT when no solution was found yet (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1], default: 1]
            'heuristics/guideddiving/maxdiveubquotnosol': '1',
            # maximal AVGQUOT when no solution was found yet (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 1]
            'heuristics/guideddiving/maxdiveavgquotnosol': '1',
            # use one level of backtracking if infeasibility is encountered?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/guideddiving/backtrack': 'TRUE',
            # percentage of immediate domain changes during probing to trigger LP resolve
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 0.15]
            'heuristics/guideddiving/lpresolvedomchgquot': '0.15',
            # LP solve frequency for diving heuristics (0: only after enough domain changes have been found)
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 0]
            'heuristics/guideddiving/lpsolvefreq': '0',
            # should only LP branching candidates be considered instead of the slower but more general constraint handler diving variable selection?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/guideddiving/onlylpbranchcands': 'FALSE',
            # priority of heuristic <zeroobj>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 100]
            'heuristics/zeroobj/priority': '100',
            # frequency for calling primal heuristic <zeroobj> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'heuristics/zeroobj/freq': '-1',
            # frequency offset for calling primal heuristic <zeroobj>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/zeroobj/freqofs': '0',
            # maximal depth level to call primal heuristic <zeroobj> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: 0]
            'heuristics/zeroobj/maxdepth': '0',
            # maximum number of nodes to regard in the subproblem
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 1000]
            'heuristics/zeroobj/maxnodes': '1000',
            # number of nodes added to the contingent of the total nodes
            # [type: longint, advanced: FALSE, range: [0,9223372036854775807], default: 100]
            'heuristics/zeroobj/nodesofs': '100',
            # minimum number of nodes required to start the subproblem
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 100]
            'heuristics/zeroobj/minnodes': '100',
            # maximum number of LP iterations to be performed in the subproblem
            # [type: longint, advanced: TRUE, range: [-1,9223372036854775807], default: 5000]
            'heuristics/zeroobj/maxlpiters': '5000',
            # contingent of sub problem nodes in relation to the number of nodes of the original problem
            # [type: real, advanced: FALSE, range: [0,1], default: 0.1]
            'heuristics/zeroobj/nodesquot': '0.1',
            # factor by which zeroobj should at least improve the incumbent
            # [type: real, advanced: TRUE, range: [0,1], default: 0.01]
            'heuristics/zeroobj/minimprove': '0.01',
            # should all subproblem solutions be added to the original SCIP?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/zeroobj/addallsols': 'FALSE',
            # should heuristic only be executed if no primal solution was found, yet?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/zeroobj/onlywithoutsol': 'TRUE',
            # should uct node selection be used at the beginning of the search?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/zeroobj/useuct': 'FALSE',
            # priority of heuristic <indicator>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -20200]
            'heuristics/indicator/priority': '-20200',
            # frequency for calling primal heuristic <indicator> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'heuristics/indicator/freq': '1',
            # frequency offset for calling primal heuristic <indicator>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/indicator/freqofs': '0',
            # maximal depth level to call primal heuristic <indicator> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/indicator/maxdepth': '-1',
            # whether the one-opt heuristic should be started
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/indicator/oneopt': 'FALSE',
            # Try to improve other solutions by one-opt?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/indicator/improvesols': 'FALSE',
            # priority of heuristic <intdiving>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1003500]
            'heuristics/intdiving/priority': '-1003500',
            # frequency for calling primal heuristic <intdiving> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'heuristics/intdiving/freq': '-1',
            # frequency offset for calling primal heuristic <intdiving>
            # [type: int, advanced: FALSE, range: [0,65534], default: 9]
            'heuristics/intdiving/freqofs': '9',
            # maximal depth level to call primal heuristic <intdiving> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/intdiving/maxdepth': '-1',
            # minimal relative depth to start diving
            # [type: real, advanced: TRUE, range: [0,1], default: 0]
            'heuristics/intdiving/minreldepth': '0',
            # maximal relative depth to start diving
            # [type: real, advanced: TRUE, range: [0,1], default: 1]
            'heuristics/intdiving/maxreldepth': '1',
            # maximal fraction of diving LP iterations compared to node LP iterations
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 0.05]
            'heuristics/intdiving/maxlpiterquot': '0.05',
            # additional number of allowed LP iterations
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 1000]
            'heuristics/intdiving/maxlpiterofs': '1000',
            # maximal quotient (curlowerbound - lowerbound)/(cutoffbound - lowerbound) where diving is performed (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1], default: 0.8]
            'heuristics/intdiving/maxdiveubquot': '0.8',
            # maximal quotient (curlowerbound - lowerbound)/(avglowerbound - lowerbound) where diving is performed (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0]
            'heuristics/intdiving/maxdiveavgquot': '0',
            # maximal UBQUOT when no solution was found yet (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1], default: 0.1]
            'heuristics/intdiving/maxdiveubquotnosol': '0.1',
            # maximal AVGQUOT when no solution was found yet (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0]
            'heuristics/intdiving/maxdiveavgquotnosol': '0',
            # use one level of backtracking if infeasibility is encountered?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/intdiving/backtrack': 'TRUE',
            # priority of heuristic <intshifting>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -10000]
            'heuristics/intshifting/priority': '-10000',
            # frequency for calling primal heuristic <intshifting> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 10]
            'heuristics/intshifting/freq': '10',
            # frequency offset for calling primal heuristic <intshifting>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/intshifting/freqofs': '0',
            # maximal depth level to call primal heuristic <intshifting> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/intshifting/maxdepth': '-1',
            # priority of heuristic <linesearchdiving>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1006000]
            'heuristics/linesearchdiving/priority': '-1006000',
            # frequency for calling primal heuristic <linesearchdiving> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 10]
            'heuristics/linesearchdiving/freq': '10',
            # frequency offset for calling primal heuristic <linesearchdiving>
            # [type: int, advanced: FALSE, range: [0,65534], default: 6]
            'heuristics/linesearchdiving/freqofs': '6',
            # maximal depth level to call primal heuristic <linesearchdiving> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/linesearchdiving/maxdepth': '-1',
            # minimal relative depth to start diving
            # [type: real, advanced: TRUE, range: [0,1], default: 0]
            'heuristics/linesearchdiving/minreldepth': '0',
            # maximal relative depth to start diving
            # [type: real, advanced: TRUE, range: [0,1], default: 1]
            'heuristics/linesearchdiving/maxreldepth': '1',
            # maximal fraction of diving LP iterations compared to node LP iterations
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 0.05]
            'heuristics/linesearchdiving/maxlpiterquot': '0.05',
            # additional number of allowed LP iterations
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 1000]
            'heuristics/linesearchdiving/maxlpiterofs': '1000',
            # maximal quotient (curlowerbound - lowerbound)/(cutoffbound - lowerbound) where diving is performed (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1], default: 0.8]
            'heuristics/linesearchdiving/maxdiveubquot': '0.8',
            # maximal quotient (curlowerbound - lowerbound)/(avglowerbound - lowerbound) where diving is performed (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0]
            'heuristics/linesearchdiving/maxdiveavgquot': '0',
            # maximal UBQUOT when no solution was found yet (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1], default: 0.1]
            'heuristics/linesearchdiving/maxdiveubquotnosol': '0.1',
            # maximal AVGQUOT when no solution was found yet (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0]
            'heuristics/linesearchdiving/maxdiveavgquotnosol': '0',
            # use one level of backtracking if infeasibility is encountered?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/linesearchdiving/backtrack': 'TRUE',
            # percentage of immediate domain changes during probing to trigger LP resolve
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 0.15]
            'heuristics/linesearchdiving/lpresolvedomchgquot': '0.15',
            # LP solve frequency for diving heuristics (0: only after enough domain changes have been found)
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 0]
            'heuristics/linesearchdiving/lpsolvefreq': '0',
            # should only LP branching candidates be considered instead of the slower but more general constraint handler diving variable selection?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/linesearchdiving/onlylpbranchcands': 'FALSE',
            # priority of heuristic <localbranching>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1102000]
            'heuristics/localbranching/priority': '-1102000',
            # frequency for calling primal heuristic <localbranching> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'heuristics/localbranching/freq': '-1',
            # frequency offset for calling primal heuristic <localbranching>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/localbranching/freqofs': '0',
            # maximal depth level to call primal heuristic <localbranching> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/localbranching/maxdepth': '-1',
            # number of nodes added to the contingent of the total nodes
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 1000]
            'heuristics/localbranching/nodesofs': '1000',
            # radius (using Manhattan metric) of the incumbent's neighborhood to be searched
            # [type: int, advanced: FALSE, range: [1,2147483647], default: 18]
            'heuristics/localbranching/neighborhoodsize': '18',
            # contingent of sub problem nodes in relation to the number of nodes of the original problem
            # [type: real, advanced: FALSE, range: [0,1], default: 0.05]
            'heuristics/localbranching/nodesquot': '0.05',
            # factor by which the limit on the number of LP depends on the node limit
            # [type: real, advanced: TRUE, range: [1,1.79769313486232e+308], default: 1.5]
            'heuristics/localbranching/lplimfac': '1.5',
            # minimum number of nodes required to start the subproblem
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 1000]
            'heuristics/localbranching/minnodes': '1000',
            # maximum number of nodes to regard in the subproblem
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 10000]
            'heuristics/localbranching/maxnodes': '10000',
            # number of nodes without incumbent change that heuristic should wait
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 200]
            'heuristics/localbranching/nwaitingnodes': '200',
            # factor by which localbranching should at least improve the incumbent
            # [type: real, advanced: TRUE, range: [0,1], default: 0.01]
            'heuristics/localbranching/minimprove': '0.01',
            # should subproblem be created out of the rows in the LP rows?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/localbranching/uselprows': 'FALSE',
            # if uselprows == FALSE, should all active cuts from cutpool be copied to constraints in subproblem?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/localbranching/copycuts': 'TRUE',
            # limit on number of improving incumbent solutions in sub-CIP
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 3]
            'heuristics/localbranching/bestsollimit': '3',
            # priority of heuristic <locks>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 3000]
            'heuristics/locks/priority': '3000',
            # frequency for calling primal heuristic <locks> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 0]
            'heuristics/locks/freq': '0',
            # frequency offset for calling primal heuristic <locks>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/locks/freqofs': '0',
            # maximal depth level to call primal heuristic <locks> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/locks/maxdepth': '-1',
            # maximum number of propagation rounds to be performed in each propagation call (-1: no limit, -2: parameter settings)
            # [type: int, advanced: TRUE, range: [-2,2147483647], default: 2]
            'heuristics/locks/maxproprounds': '2',
            # minimum percentage of integer variables that have to be fixable
            # [type: real, advanced: FALSE, range: [0,1], default: 0.65]
            'heuristics/locks/minfixingrate': '0.65',
            # probability for rounding a variable up in case of ties
            # [type: real, advanced: FALSE, range: [0,1], default: 0.67]
            'heuristics/locks/roundupprobability': '0.67',
            # should a final sub-MIP be solved to costruct a feasible solution if the LP was not roundable?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/locks/usefinalsubmip': 'TRUE',
            # maximum number of nodes to regard in the subproblem
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 5000]
            'heuristics/locks/maxnodes': '5000',
            # number of nodes added to the contingent of the total nodes
            # [type: longint, advanced: FALSE, range: [0,9223372036854775807], default: 500]
            'heuristics/locks/nodesofs': '500',
            # minimum number of nodes required to start the subproblem
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 500]
            'heuristics/locks/minnodes': '500',
            # contingent of sub problem nodes in relation to the number of nodes of the original problem
            # [type: real, advanced: FALSE, range: [0,1], default: 0.1]
            'heuristics/locks/nodesquot': '0.1',
            # factor by which locks heuristic should at least improve the incumbent
            # [type: real, advanced: TRUE, range: [0,1], default: 0.01]
            'heuristics/locks/minimprove': '0.01',
            # should all active cuts from cutpool be copied to constraints in subproblem?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/locks/copycuts': 'TRUE',
            # should the locks be updated based on LP rows?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/locks/updatelocks': 'TRUE',
            # minimum fixing rate over all variables (including continuous) to solve LP
            # [type: real, advanced: TRUE, range: [0,1], default: 0]
            'heuristics/locks/minfixingratelp': '0',
            # priority of heuristic <lpface>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1104000]
            'heuristics/lpface/priority': '-1104000',
            # frequency for calling primal heuristic <lpface> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 15]
            'heuristics/lpface/freq': '15',
            # frequency offset for calling primal heuristic <lpface>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/lpface/freqofs': '0',
            # maximal depth level to call primal heuristic <lpface> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/lpface/maxdepth': '-1',
            # number of nodes added to the contingent of the total nodes
            # [type: longint, advanced: FALSE, range: [0,9223372036854775807], default: 200]
            'heuristics/lpface/nodesofs': '200',
            # maximum number of nodes to regard in the subproblem
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 5000]
            'heuristics/lpface/maxnodes': '5000',
            # minimum number of nodes required to start the subproblem
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 50]
            'heuristics/lpface/minnodes': '50',
            # contingent of sub problem nodes in relation to the number of nodes of the original problem
            # [type: real, advanced: FALSE, range: [0,1], default: 0.1]
            'heuristics/lpface/nodesquot': '0.1',
            # required percentage of fixed integer variables in sub-MIP to run
            # [type: real, advanced: FALSE, range: [0,1], default: 0.1]
            'heuristics/lpface/minfixingrate': '0.1',
            # factor by which the limit on the number of LP depends on the node limit
            # [type: real, advanced: TRUE, range: [1,1.79769313486232e+308], default: 2]
            'heuristics/lpface/lplimfac': '2',
            # should subproblem be created out of the rows in the LP rows?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/lpface/uselprows': 'TRUE',
            # should dually nonbasic rows be turned into equations?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/lpface/dualbasisequations': 'FALSE',
            # should the heuristic continue solving the same sub-SCIP?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/lpface/keepsubscip': 'FALSE',
            # if uselprows == FALSE, should all active cuts from cutpool be copied to constraints in subproblem?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/lpface/copycuts': 'TRUE',
            # objective function in the sub-SCIP: (z)ero, (r)oot-LP-difference, (i)nference, LP (f)ractionality, (o)riginal
            # [type: char, advanced: TRUE, range: {forzi}, default: z]
            'heuristics/lpface/subscipobjective': 'z',
            # the minimum active search tree path length along which lower bound hasn't changed before heuristic becomes active
            # [type: int, advanced: TRUE, range: [0,65531], default: 5]
            'heuristics/lpface/minpathlen': '5',
            # priority of heuristic <alns>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1100500]
            'heuristics/alns/priority': '-1100500',
            # frequency for calling primal heuristic <alns> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 20]
            'heuristics/alns/freq': '20',
            # frequency offset for calling primal heuristic <alns>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/alns/freqofs': '0',
            # maximal depth level to call primal heuristic <alns> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/alns/maxdepth': '-1',
            # minimum fixing rate for this neighborhood
            # [type: real, advanced: TRUE, range: [0,1], default: 0.3]
            'heuristics/alns/rens/minfixingrate': '0.3',
            # maximum fixing rate for this neighborhood
            # [type: real, advanced: TRUE, range: [0,1], default: 0.9]
            'heuristics/alns/rens/maxfixingrate': '0.9',
            # is this neighborhood active?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/alns/rens/active': 'TRUE',
            # positive call priority to initialize bandit algorithms
            # [type: real, advanced: TRUE, range: [0.01,1], default: 1]
            'heuristics/alns/rens/priority': '1',
            # minimum fixing rate for this neighborhood
            # [type: real, advanced: TRUE, range: [0,1], default: 0.3]
            'heuristics/alns/rins/minfixingrate': '0.3',
            # maximum fixing rate for this neighborhood
            # [type: real, advanced: TRUE, range: [0,1], default: 0.9]
            'heuristics/alns/rins/maxfixingrate': '0.9',
            # is this neighborhood active?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/alns/rins/active': 'TRUE',
            # positive call priority to initialize bandit algorithms
            # [type: real, advanced: TRUE, range: [0.01,1], default: 1]
            'heuristics/alns/rins/priority': '1',
            # minimum fixing rate for this neighborhood
            # [type: real, advanced: TRUE, range: [0,1], default: 0.3]
            'heuristics/alns/mutation/minfixingrate': '0.3',
            # maximum fixing rate for this neighborhood
            # [type: real, advanced: TRUE, range: [0,1], default: 0.9]
            'heuristics/alns/mutation/maxfixingrate': '0.9',
            # is this neighborhood active?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/alns/mutation/active': 'TRUE',
            # positive call priority to initialize bandit algorithms
            # [type: real, advanced: TRUE, range: [0.01,1], default: 1]
            'heuristics/alns/mutation/priority': '1',
            # minimum fixing rate for this neighborhood
            # [type: real, advanced: TRUE, range: [0,1], default: 0.3]
            'heuristics/alns/localbranching/minfixingrate': '0.3',
            # maximum fixing rate for this neighborhood
            # [type: real, advanced: TRUE, range: [0,1], default: 0.9]
            'heuristics/alns/localbranching/maxfixingrate': '0.9',
            # is this neighborhood active?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/alns/localbranching/active': 'TRUE',
            # positive call priority to initialize bandit algorithms
            # [type: real, advanced: TRUE, range: [0.01,1], default: 1]
            'heuristics/alns/localbranching/priority': '1',
            # minimum fixing rate for this neighborhood
            # [type: real, advanced: TRUE, range: [0,1], default: 0.3]
            'heuristics/alns/crossover/minfixingrate': '0.3',
            # maximum fixing rate for this neighborhood
            # [type: real, advanced: TRUE, range: [0,1], default: 0.9]
            'heuristics/alns/crossover/maxfixingrate': '0.9',
            # is this neighborhood active?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/alns/crossover/active': 'TRUE',
            # positive call priority to initialize bandit algorithms
            # [type: real, advanced: TRUE, range: [0.01,1], default: 1]
            'heuristics/alns/crossover/priority': '1',
            # the number of solutions that crossover should combine
            # [type: int, advanced: TRUE, range: [2,10], default: 2]
            'heuristics/alns/crossover/nsols': '2',
            # minimum fixing rate for this neighborhood
            # [type: real, advanced: TRUE, range: [0,1], default: 0.3]
            'heuristics/alns/proximity/minfixingrate': '0.3',
            # maximum fixing rate for this neighborhood
            # [type: real, advanced: TRUE, range: [0,1], default: 0.9]
            'heuristics/alns/proximity/maxfixingrate': '0.9',
            # is this neighborhood active?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/alns/proximity/active': 'TRUE',
            # positive call priority to initialize bandit algorithms
            # [type: real, advanced: TRUE, range: [0.01,1], default: 1]
            'heuristics/alns/proximity/priority': '1',
            # minimum fixing rate for this neighborhood
            # [type: real, advanced: TRUE, range: [0,1], default: 0.3]
            'heuristics/alns/zeroobjective/minfixingrate': '0.3',
            # maximum fixing rate for this neighborhood
            # [type: real, advanced: TRUE, range: [0,1], default: 0.9]
            'heuristics/alns/zeroobjective/maxfixingrate': '0.9',
            # is this neighborhood active?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/alns/zeroobjective/active': 'TRUE',
            # positive call priority to initialize bandit algorithms
            # [type: real, advanced: TRUE, range: [0.01,1], default: 1]
            'heuristics/alns/zeroobjective/priority': '1',
            # minimum fixing rate for this neighborhood
            # [type: real, advanced: TRUE, range: [0,1], default: 0.3]
            'heuristics/alns/dins/minfixingrate': '0.3',
            # maximum fixing rate for this neighborhood
            # [type: real, advanced: TRUE, range: [0,1], default: 0.9]
            'heuristics/alns/dins/maxfixingrate': '0.9',
            # is this neighborhood active?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/alns/dins/active': 'TRUE',
            # positive call priority to initialize bandit algorithms
            # [type: real, advanced: TRUE, range: [0.01,1], default: 1]
            'heuristics/alns/dins/priority': '1',
            # number of pool solutions where binary solution values must agree
            # [type: int, advanced: TRUE, range: [1,100], default: 5]
            'heuristics/alns/dins/npoolsols': '5',
            # minimum fixing rate for this neighborhood
            # [type: real, advanced: TRUE, range: [0,1], default: 0.3]
            'heuristics/alns/trustregion/minfixingrate': '0.3',
            # maximum fixing rate for this neighborhood
            # [type: real, advanced: TRUE, range: [0,1], default: 0.9]
            'heuristics/alns/trustregion/maxfixingrate': '0.9',
            # is this neighborhood active?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/alns/trustregion/active': 'FALSE',
            # positive call priority to initialize bandit algorithms
            # [type: real, advanced: TRUE, range: [0.01,1], default: 1]
            'heuristics/alns/trustregion/priority': '1',
            # the penalty for each change in the binary variables from the candidate solution
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 100]
            'heuristics/alns/trustregion/violpenalty': '100',
            # maximum number of nodes to regard in the subproblem
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 5000]
            'heuristics/alns/maxnodes': '5000',
            # offset added to the nodes budget
            # [type: longint, advanced: FALSE, range: [0,9223372036854775807], default: 500]
            'heuristics/alns/nodesofs': '500',
            # minimum number of nodes required to start a sub-SCIP
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 50]
            'heuristics/alns/minnodes': '50',
            # number of nodes since last incumbent solution that the heuristic should wait
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 25]
            'heuristics/alns/waitingnodes': '25',
            # fraction of nodes compared to the main SCIP for budget computation
            # [type: real, advanced: FALSE, range: [0,1], default: 0.1]
            'heuristics/alns/nodesquot': '0.1',
            # initial factor by which ALNS should at least improve the incumbent
            # [type: real, advanced: TRUE, range: [0,1], default: 0.01]
            'heuristics/alns/startminimprove': '0.01',
            # lower threshold for the minimal improvement over the incumbent
            # [type: real, advanced: TRUE, range: [0,1], default: 0.01]
            'heuristics/alns/minimprovelow': '0.01',
            # upper bound for the minimal improvement over the incumbent
            # [type: real, advanced: TRUE, range: [0,1], default: 0.01]
            'heuristics/alns/minimprovehigh': '0.01',
            # limit on the number of improving solutions in a sub-SCIP call
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 3]
            'heuristics/alns/nsolslim': '3',
            # the bandit algorithm: (u)pper confidence bounds, (e)xp.3, epsilon (g)reedy
            # [type: char, advanced: TRUE, range: {ueg}, default: u]
            'heuristics/alns/banditalgo': 'u',
            # weight between uniform (gamma ~ 1) and weight driven (gamma ~ 0) probability distribution for exp3
            # [type: real, advanced: TRUE, range: [0,1], default: 0.07041455]
            'heuristics/alns/gamma': '0.07041455',
            # reward offset between 0 and 1 at every observation for Exp.3
            # [type: real, advanced: TRUE, range: [0,1], default: 0]
            'heuristics/alns/beta': '0',
            # parameter to increase the confidence width in UCB
            # [type: real, advanced: TRUE, range: [0,100], default: 0.0016]
            'heuristics/alns/alpha': '0.0016',
            # distances from fixed variables be used for variable prioritization
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/alns/usedistances': 'TRUE',
            # should reduced cost scores be used for variable prioritization?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/alns/useredcost': 'TRUE',
            # should the ALNS heuristic do more fixings by itself based on variable prioritization until the target fixing rate is reached?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/alns/domorefixings': 'TRUE',
            # should the heuristic adjust the target fixing rate based on the success?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/alns/adjustfixingrate': 'TRUE',
            # should the heuristic activate other sub-SCIP heuristics during its search?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/alns/usesubscipheurs': 'FALSE',
            # reward control to increase the weight of the simple solution indicator and decrease the weight of the closed gap reward
            # [type: real, advanced: TRUE, range: [0,1], default: 0.8]
            'heuristics/alns/rewardcontrol': '0.8',
            # factor by which target node number is eventually increased
            # [type: real, advanced: TRUE, range: [1,100000], default: 1.05]
            'heuristics/alns/targetnodefactor': '1.05',
            # initial random seed for bandit algorithms and random decisions by neighborhoods
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 113]
            'heuristics/alns/seed': '113',
            # should the factor by which the minimum improvement is bound be dynamically updated?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/alns/adjustminimprove': 'FALSE',
            # should the target nodes be dynamically adjusted?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/alns/adjusttargetnodes': 'TRUE',
            # increase exploration in epsilon-greedy bandit algorithm
            # [type: real, advanced: TRUE, range: [0,1], default: 0.4685844]
            'heuristics/alns/eps': '0.4685844',
            # the reward baseline to separate successful and failed calls
            # [type: real, advanced: TRUE, range: [0,0.99], default: 0.5]
            'heuristics/alns/rewardbaseline': '0.5',
            # should the bandit algorithms be reset when a new problem is read?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/alns/resetweights': 'TRUE',
            # file name to store all rewards and the selection of the bandit
            # [type: string, advanced: TRUE, default: "-"]
            'heuristics/alns/rewardfilename': '"-"',
            # should random seeds of sub-SCIPs be altered to increase diversification?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/alns/subsciprandseeds': 'FALSE',
            # should the reward be scaled by the effort?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/alns/scalebyeffort': 'TRUE',
            # should cutting planes be copied to the sub-SCIP?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/alns/copycuts': 'FALSE',
            # tolerance by which the fixing rate may be missed without generic fixing
            # [type: real, advanced: TRUE, range: [0,1], default: 0.1]
            'heuristics/alns/fixtol': '0.1',
            # tolerance by which the fixing rate may be exceeded without generic unfixing
            # [type: real, advanced: TRUE, range: [0,1], default: 0.1]
            'heuristics/alns/unfixtol': '0.1',
            # should local reduced costs be used for generic (un)fixing?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/alns/uselocalredcost': 'FALSE',
            # should pseudo cost scores be used for variable priorization?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/alns/usepscost': 'TRUE',
            # is statistics table <neighborhood> active
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'table/neighborhood/active': 'TRUE',
            # priority of heuristic <nlpdiving>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1003000]
            'heuristics/nlpdiving/priority': '-1003000',
            # frequency for calling primal heuristic <nlpdiving> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 10]
            'heuristics/nlpdiving/freq': '10',
            # frequency offset for calling primal heuristic <nlpdiving>
            # [type: int, advanced: FALSE, range: [0,65534], default: 3]
            'heuristics/nlpdiving/freqofs': '3',
            # maximal depth level to call primal heuristic <nlpdiving> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/nlpdiving/maxdepth': '-1',
            # minimal relative depth to start diving
            # [type: real, advanced: TRUE, range: [0,1], default: 0]
            'heuristics/nlpdiving/minreldepth': '0',
            # maximal relative depth to start diving
            # [type: real, advanced: TRUE, range: [0,1], default: 1]
            'heuristics/nlpdiving/maxreldepth': '1',
            # minimial absolute number of allowed NLP iterations
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 200]
            'heuristics/nlpdiving/maxnlpiterabs': '200',
            # additional allowed number of NLP iterations relative to successfully found solutions
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 10]
            'heuristics/nlpdiving/maxnlpiterrel': '10',
            # maximal quotient (curlowerbound - lowerbound)/(cutoffbound - lowerbound) where diving is performed (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1], default: 0.8]
            'heuristics/nlpdiving/maxdiveubquot': '0.8',
            # maximal quotient (curlowerbound - lowerbound)/(avglowerbound - lowerbound) where diving is performed (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0]
            'heuristics/nlpdiving/maxdiveavgquot': '0',
            # maximal UBQUOT when no solution was found yet (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1], default: 0.1]
            'heuristics/nlpdiving/maxdiveubquotnosol': '0.1',
            # maximal AVGQUOT when no solution was found yet (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0]
            'heuristics/nlpdiving/maxdiveavgquotnosol': '0',
            # maximal number of NLPs with feasible solution to solve during one dive
            # [type: int, advanced: FALSE, range: [1,2147483647], default: 10]
            'heuristics/nlpdiving/maxfeasnlps': '10',
            # use one level of backtracking if infeasibility is encountered?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/nlpdiving/backtrack': 'TRUE',
            # should the LP relaxation be solved before the NLP relaxation?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/nlpdiving/lp': 'FALSE',
            # prefer variables that are also fractional in LP solution?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/nlpdiving/preferlpfracs': 'FALSE',
            # heuristic will not run if less then this percentage of calls succeeded (0.0: no limit)
            # [type: real, advanced: FALSE, range: [0,1], default: 0.1]
            'heuristics/nlpdiving/minsuccquot': '0.1',
            # percentage of fractional variables that should be fixed before the next NLP solve
            # [type: real, advanced: FALSE, range: [0,1], default: 0.2]
            'heuristics/nlpdiving/fixquot': '0.2',
            # should variables in a minimal cover be preferred?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/nlpdiving/prefercover': 'TRUE',
            # should a sub-MIP be solved if all cover variables are fixed?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/nlpdiving/solvesubmip': 'FALSE',
            # should the NLP solver stop early if it converges slow?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/nlpdiving/nlpfastfail': 'TRUE',
            # which point should be used as starting point for the NLP solver? ('n'one, last 'f'easible, from dive's'tart)
            # [type: char, advanced: TRUE, range: {fns}, default: s]
            'heuristics/nlpdiving/nlpstart': 's',
            # which variable selection should be used? ('f'ractionality, 'c'oefficient, 'p'seudocost, 'g'uided, 'd'ouble, 'v'eclen)
            # [type: char, advanced: FALSE, range: {fcpgdv}, default: d]
            'heuristics/nlpdiving/varselrule': 'd',
            # priority of heuristic <mutation>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1103000]
            'heuristics/mutation/priority': '-1103000',
            # frequency for calling primal heuristic <mutation> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'heuristics/mutation/freq': '-1',
            # frequency offset for calling primal heuristic <mutation>
            # [type: int, advanced: FALSE, range: [0,65534], default: 8]
            'heuristics/mutation/freqofs': '8',
            # maximal depth level to call primal heuristic <mutation> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/mutation/maxdepth': '-1',
            # number of nodes added to the contingent of the total nodes
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 500]
            'heuristics/mutation/nodesofs': '500',
            # maximum number of nodes to regard in the subproblem
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 5000]
            'heuristics/mutation/maxnodes': '5000',
            # minimum number of nodes required to start the subproblem
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 500]
            'heuristics/mutation/minnodes': '500',
            # number of nodes without incumbent change that heuristic should wait
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 200]
            'heuristics/mutation/nwaitingnodes': '200',
            # contingent of sub problem nodes in relation to the number of nodes of the original problem
            # [type: real, advanced: FALSE, range: [0,1], default: 0.1]
            'heuristics/mutation/nodesquot': '0.1',
            # percentage of integer variables that have to be fixed
            # [type: real, advanced: FALSE, range: [1e-06,0.999999], default: 0.8]
            'heuristics/mutation/minfixingrate': '0.8',
            # factor by which mutation should at least improve the incumbent
            # [type: real, advanced: TRUE, range: [0,1], default: 0.01]
            'heuristics/mutation/minimprove': '0.01',
            # should subproblem be created out of the rows in the LP rows?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/mutation/uselprows': 'FALSE',
            # if uselprows == FALSE, should all active cuts from cutpool be copied to constraints in subproblem?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/mutation/copycuts': 'TRUE',
            # limit on number of improving incumbent solutions in sub-CIP
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'heuristics/mutation/bestsollimit': '-1',
            # should uct node selection be used at the beginning of the search?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/mutation/useuct': 'FALSE',
            # priority of heuristic <multistart>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -2100000]
            'heuristics/multistart/priority': '-2100000',
            # frequency for calling primal heuristic <multistart> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 0]
            'heuristics/multistart/freq': '0',
            # frequency offset for calling primal heuristic <multistart>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/multistart/freqofs': '0',
            # maximal depth level to call primal heuristic <multistart> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/multistart/maxdepth': '-1',
            # number of random points generated per execution call
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 100]
            'heuristics/multistart/nrndpoints': '100',
            # maximum variable domain size for unbounded variables
            # [type: real, advanced: FALSE, range: [0,1e+20], default: 20000]
            'heuristics/multistart/maxboundsize': '20000',
            # number of iterations to reduce the maximum violation of a point
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 300]
            'heuristics/multistart/maxiter': '300',
            # minimum required improving factor to proceed in improvement of a single point
            # [type: real, advanced: FALSE, range: [-1e+20,1e+20], default: 0.05]
            'heuristics/multistart/minimprfac': '0.05',
            # number of iteration when checking the minimum improvement
            # [type: int, advanced: FALSE, range: [1,2147483647], default: 10]
            'heuristics/multistart/minimpriter': '10',
            # maximum distance between two points in the same cluster
            # [type: real, advanced: FALSE, range: [0,1e+20], default: 0.15]
            'heuristics/multistart/maxreldist': '0.15',
            # factor by which heuristic should at least improve the incumbent
            # [type: real, advanced: FALSE, range: [0,1e+20], default: 0]
            'heuristics/multistart/nlpminimpr': '0',
            # limit for gradient computations for all improvePoint() calls (0 for no limit)
            # [type: real, advanced: FALSE, range: [0,1e+20], default: 5000000]
            'heuristics/multistart/gradlimit': '5000000',
            # maximum number of considered clusters per heuristic call
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 3]
            'heuristics/multistart/maxncluster': '3',
            # should the heuristic run only on continuous problems?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/multistart/onlynlps': 'TRUE',
            # priority of heuristic <mpec>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -2050000]
            'heuristics/mpec/priority': '-2050000',
            # frequency for calling primal heuristic <mpec> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 50]
            'heuristics/mpec/freq': '50',
            # frequency offset for calling primal heuristic <mpec>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/mpec/freqofs': '0',
            # maximal depth level to call primal heuristic <mpec> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/mpec/maxdepth': '-1',
            # initial regularization right-hand side value
            # [type: real, advanced: FALSE, range: [0,0.25], default: 0.125]
            'heuristics/mpec/inittheta': '0.125',
            # regularization update factor
            # [type: real, advanced: FALSE, range: [0,1], default: 0.5]
            'heuristics/mpec/sigma': '0.5',
            # maximum number of NLP iterations per solve
            # [type: real, advanced: FALSE, range: [0,1], default: 0.001]
            'heuristics/mpec/subnlptrigger': '0.001',
            # maximum cost available for solving NLPs per call of the heuristic
            # [type: real, advanced: FALSE, range: [0,1e+20], default: 100000000]
            'heuristics/mpec/maxnlpcost': '100000000',
            # factor by which heuristic should at least improve the incumbent
            # [type: real, advanced: FALSE, range: [0,1], default: 0.01]
            'heuristics/mpec/minimprove': '0.01',
            # minimum amount of gap left in order to call the heuristic
            # [type: real, advanced: FALSE, range: [0,1e+20], default: 0.05]
            'heuristics/mpec/mingapleft': '0.05',
            # maximum number of iterations of the MPEC loop
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 100]
            'heuristics/mpec/maxiter': '100',
            # maximum number of NLP iterations per solve
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 500]
            'heuristics/mpec/maxnlpiter': '500',
            # maximum number of consecutive calls for which the heuristic did not find an improving solution
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 10]
            'heuristics/mpec/maxnunsucc': '10',
            # priority of heuristic <objpscostdiving>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1004000]
            'heuristics/objpscostdiving/priority': '-1004000',
            # frequency for calling primal heuristic <objpscostdiving> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 20]
            'heuristics/objpscostdiving/freq': '20',
            # frequency offset for calling primal heuristic <objpscostdiving>
            # [type: int, advanced: FALSE, range: [0,65534], default: 4]
            'heuristics/objpscostdiving/freqofs': '4',
            # maximal depth level to call primal heuristic <objpscostdiving> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/objpscostdiving/maxdepth': '-1',
            # minimal relative depth to start diving
            # [type: real, advanced: TRUE, range: [0,1], default: 0]
            'heuristics/objpscostdiving/minreldepth': '0',
            # maximal relative depth to start diving
            # [type: real, advanced: TRUE, range: [0,1], default: 1]
            'heuristics/objpscostdiving/maxreldepth': '1',
            # maximal fraction of diving LP iterations compared to total iteration number
            # [type: real, advanced: FALSE, range: [0,1], default: 0.01]
            'heuristics/objpscostdiving/maxlpiterquot': '0.01',
            # additional number of allowed LP iterations
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 1000]
            'heuristics/objpscostdiving/maxlpiterofs': '1000',
            # total number of feasible solutions found up to which heuristic is called (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'heuristics/objpscostdiving/maxsols': '-1',
            # maximal diving depth: number of binary/integer variables times depthfac
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0.5]
            'heuristics/objpscostdiving/depthfac': '0.5',
            # maximal diving depth factor if no feasible solution was found yet
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 2]
            'heuristics/objpscostdiving/depthfacnosol': '2',
            # priority of heuristic <octane>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1008000]
            'heuristics/octane/priority': '-1008000',
            # frequency for calling primal heuristic <octane> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'heuristics/octane/freq': '-1',
            # frequency offset for calling primal heuristic <octane>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/octane/freqofs': '0',
            # maximal depth level to call primal heuristic <octane> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/octane/maxdepth': '-1',
            # number of 0-1-points to be tested as possible solutions by OCTANE
            # [type: int, advanced: TRUE, range: [1,2147483647], default: 100]
            'heuristics/octane/fmax': '100',
            # number of 0-1-points to be tested at first whether they violate a common row
            # [type: int, advanced: TRUE, range: [1,2147483647], default: 10]
            'heuristics/octane/ffirst': '10',
            # execute OCTANE only in the space of fractional variables (TRUE) or in the full space?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/octane/usefracspace': 'TRUE',
            # should the inner normal of the objective be used as one ray direction?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/octane/useobjray': 'TRUE',
            # should the average of the basic cone be used as one ray direction?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/octane/useavgray': 'TRUE',
            # should the difference between the root solution and the current LP solution be used as one ray direction?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/octane/usediffray': 'FALSE',
            # should the weighted average of the basic cone be used as one ray direction?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/octane/useavgwgtray': 'TRUE',
            # should the weighted average of the nonbasic cone be used as one ray direction?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/octane/useavgnbray': 'TRUE',
            # priority of heuristic <ofins>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 60000]
            'heuristics/ofins/priority': '60000',
            # frequency for calling primal heuristic <ofins> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 0]
            'heuristics/ofins/freq': '0',
            # frequency offset for calling primal heuristic <ofins>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/ofins/freqofs': '0',
            # maximal depth level to call primal heuristic <ofins> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: 0]
            'heuristics/ofins/maxdepth': '0',
            # maximum number of nodes to regard in the subproblem
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 5000]
            'heuristics/ofins/maxnodes': '5000',
            # minimum number of nodes required to start the subproblem
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 50]
            'heuristics/ofins/minnodes': '50',
            # maximal rate of changed coefficients
            # [type: real, advanced: FALSE, range: [0,1], default: 0.5]
            'heuristics/ofins/maxchangerate': '0.5',
            # maximal rate of change per coefficient to get fixed
            # [type: real, advanced: FALSE, range: [0,1], default: 0.04]
            'heuristics/ofins/maxchange': '0.04',
            # should all active cuts from cutpool be copied to constraints in subproblem?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/ofins/copycuts': 'TRUE',
            # should all subproblem solutions be added to the original SCIP?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/ofins/addallsols': 'FALSE',
            # number of nodes added to the contingent of the total nodes
            # [type: longint, advanced: FALSE, range: [0,9223372036854775807], default: 500]
            'heuristics/ofins/nodesofs': '500',
            # contingent of sub problem nodes in relation to the number of nodes of the original problem
            # [type: real, advanced: FALSE, range: [0,1], default: 0.1]
            'heuristics/ofins/nodesquot': '0.1',
            # factor by which RENS should at least improve the incumbent
            # [type: real, advanced: TRUE, range: [0,1], default: 0.01]
            'heuristics/ofins/minimprove': '0.01',
            # factor by which the limit on the number of LP depends on the node limit
            # [type: real, advanced: TRUE, range: [1,1.79769313486232e+308], default: 2]
            'heuristics/ofins/lplimfac': '2',
            # priority of heuristic <oneopt>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -20000]
            'heuristics/oneopt/priority': '-20000',
            # frequency for calling primal heuristic <oneopt> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'heuristics/oneopt/freq': '1',
            # frequency offset for calling primal heuristic <oneopt>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/oneopt/freqofs': '0',
            # maximal depth level to call primal heuristic <oneopt> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/oneopt/maxdepth': '-1',
            # should the objective be weighted with the potential shifting value when sorting the shifting candidates?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/oneopt/weightedobj': 'TRUE',
            # should the heuristic be called before and during the root node?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/oneopt/duringroot': 'TRUE',
            # should the construction of the LP be forced even if LP solving is deactivated?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/oneopt/forcelpconstruction': 'FALSE',
            # should the heuristic be called before presolving?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/oneopt/beforepresol': 'FALSE',
            # should the heuristic continue to run as long as improvements are found?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/oneopt/useloop': 'TRUE',
            # priority of heuristic <padm>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 70000]
            'heuristics/padm/priority': '70000',
            # frequency for calling primal heuristic <padm> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 0]
            'heuristics/padm/freq': '0',
            # frequency offset for calling primal heuristic <padm>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/padm/freqofs': '0',
            # maximal depth level to call primal heuristic <padm> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/padm/maxdepth': '-1',
            # maximum number of nodes to regard in all subproblems
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 5000]
            'heuristics/padm/maxnodes': '5000',
            # minimum number of nodes to regard in one subproblem
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 50]
            'heuristics/padm/minnodes': '50',
            # factor to control nodelimits of subproblems
            # [type: real, advanced: TRUE, range: [0,0.99], default: 0.8]
            'heuristics/padm/nodefac': '0.8',
            # maximal number of ADM iterations in each penalty loop
            # [type: int, advanced: TRUE, range: [1,100], default: 4]
            'heuristics/padm/admiterations': '4',
            # maximal number of penalty iterations
            # [type: int, advanced: TRUE, range: [1,100000], default: 100]
            'heuristics/padm/penaltyiterations': '100',
            # mipgap at start
            # [type: real, advanced: TRUE, range: [0,16], default: 2]
            'heuristics/padm/gap': '2',
            # enable sigmoid rescaling of penalty parameters
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/padm/scaling': 'TRUE',
            # should linking constraints be assigned?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/padm/assignlinking': 'TRUE',
            # should the original problem be used?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/padm/original': 'FALSE',
            # should the heuristic run before or after the processing of the node? (0: before, 1: after, 2: both)
            # [type: int, advanced: FALSE, range: [0,2], default: 0]
            'heuristics/padm/timing': '0',
            # priority of heuristic <proximity>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -2000000]
            'heuristics/proximity/priority': '-2000000',
            # frequency for calling primal heuristic <proximity> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'heuristics/proximity/freq': '-1',
            # frequency offset for calling primal heuristic <proximity>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/proximity/freqofs': '0',
            # maximal depth level to call primal heuristic <proximity> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/proximity/maxdepth': '-1',
            # should subproblem be constructed based on LP row information?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/proximity/uselprows': 'FALSE',
            # should the heuristic immediately run again on its newly found solution?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/proximity/restart': 'TRUE',
            # should the heuristic solve a final LP in case of continuous objective variables?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/proximity/usefinallp': 'FALSE',
            # maximum number of nodes to regard in the subproblem
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 10000]
            'heuristics/proximity/maxnodes': '10000',
            # number of nodes added to the contingent of the total nodes
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 50]
            'heuristics/proximity/nodesofs': '50',
            # minimum number of nodes required to start the subproblem
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 1]
            'heuristics/proximity/minnodes': '1',
            # maximum number of LP iterations to be performed in the subproblem
            # [type: longint, advanced: TRUE, range: [-1,9223372036854775807], default: 100000]
            'heuristics/proximity/maxlpiters': '100000',
            # minimum number of LP iterations performed in subproblem
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 200]
            'heuristics/proximity/minlpiters': '200',
            # waiting nodes since last incumbent before heuristic is executed
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 100]
            'heuristics/proximity/waitingnodes': '100',
            # factor by which proximity should at least improve the incumbent
            # [type: real, advanced: TRUE, range: [0,1], default: 0.02]
            'heuristics/proximity/minimprove': '0.02',
            # sub-MIP node limit w.r.t number of original nodes
            # [type: real, advanced: TRUE, range: [0,1e+20], default: 0.1]
            'heuristics/proximity/nodesquot': '0.1',
            # threshold for percentage of binary variables required to start
            # [type: real, advanced: TRUE, range: [0,1], default: 0.1]
            'heuristics/proximity/binvarquot': '0.1',
            # quotient of sub-MIP LP iterations with respect to LP iterations so far
            # [type: real, advanced: TRUE, range: [0,1], default: 0.2]
            'heuristics/proximity/lpitersquot': '0.2',
            # minimum primal-dual gap for which the heuristic is executed
            # [type: real, advanced: TRUE, range: [0,1e+20], default: 0.01]
            'heuristics/proximity/mingap': '0.01',
            # should uct node selection be used at the beginning of the search?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/proximity/useuct': 'FALSE',
            # priority of heuristic <pscostdiving>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1002000]
            'heuristics/pscostdiving/priority': '-1002000',
            # frequency for calling primal heuristic <pscostdiving> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 10]
            'heuristics/pscostdiving/freq': '10',
            # frequency offset for calling primal heuristic <pscostdiving>
            # [type: int, advanced: FALSE, range: [0,65534], default: 2]
            'heuristics/pscostdiving/freqofs': '2',
            # maximal depth level to call primal heuristic <pscostdiving> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/pscostdiving/maxdepth': '-1',
            # minimal relative depth to start diving
            # [type: real, advanced: TRUE, range: [0,1], default: 0]
            'heuristics/pscostdiving/minreldepth': '0',
            # maximal relative depth to start diving
            # [type: real, advanced: TRUE, range: [0,1], default: 1]
            'heuristics/pscostdiving/maxreldepth': '1',
            # maximal fraction of diving LP iterations compared to node LP iterations
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 0.05]
            'heuristics/pscostdiving/maxlpiterquot': '0.05',
            # additional number of allowed LP iterations
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 1000]
            'heuristics/pscostdiving/maxlpiterofs': '1000',
            # maximal quotient (curlowerbound - lowerbound)/(cutoffbound - lowerbound) where diving is performed (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1], default: 0.8]
            'heuristics/pscostdiving/maxdiveubquot': '0.8',
            # maximal quotient (curlowerbound - lowerbound)/(avglowerbound - lowerbound) where diving is performed (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0]
            'heuristics/pscostdiving/maxdiveavgquot': '0',
            # maximal UBQUOT when no solution was found yet (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1], default: 0.1]
            'heuristics/pscostdiving/maxdiveubquotnosol': '0.1',
            # maximal AVGQUOT when no solution was found yet (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0]
            'heuristics/pscostdiving/maxdiveavgquotnosol': '0',
            # use one level of backtracking if infeasibility is encountered?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/pscostdiving/backtrack': 'TRUE',
            # percentage of immediate domain changes during probing to trigger LP resolve
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 0.15]
            'heuristics/pscostdiving/lpresolvedomchgquot': '0.15',
            # LP solve frequency for diving heuristics (0: only after enough domain changes have been found)
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 0]
            'heuristics/pscostdiving/lpsolvefreq': '0',
            # should only LP branching candidates be considered instead of the slower but more general constraint handler diving variable selection?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/pscostdiving/onlylpbranchcands': 'TRUE',
            # priority of heuristic <randrounding>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -200]
            'heuristics/randrounding/priority': '-200',
            # frequency for calling primal heuristic <randrounding> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 20]
            'heuristics/randrounding/freq': '20',
            # frequency offset for calling primal heuristic <randrounding>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/randrounding/freqofs': '0',
            # maximal depth level to call primal heuristic <randrounding> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/randrounding/maxdepth': '-1',
            # should the heuristic only be called once per node?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/randrounding/oncepernode': 'FALSE',
            # should the heuristic apply the variable lock strategy of simple rounding, if possible?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/randrounding/usesimplerounding': 'FALSE',
            # should the probing part of the heuristic be applied exclusively at the root node?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/randrounding/propagateonlyroot': 'TRUE',
            # limit of rounds for each propagation call
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 1]
            'heuristics/randrounding/maxproprounds': '1',
            # priority of heuristic <rens>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1100000]
            'heuristics/rens/priority': '-1100000',
            # frequency for calling primal heuristic <rens> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 0]
            'heuristics/rens/freq': '0',
            # frequency offset for calling primal heuristic <rens>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/rens/freqofs': '0',
            # maximal depth level to call primal heuristic <rens> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/rens/maxdepth': '-1',
            # minimum percentage of integer variables that have to be fixable
            # [type: real, advanced: FALSE, range: [0,1], default: 0.5]
            'heuristics/rens/minfixingrate': '0.5',
            # maximum number of nodes to regard in the subproblem
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 5000]
            'heuristics/rens/maxnodes': '5000',
            # number of nodes added to the contingent of the total nodes
            # [type: longint, advanced: FALSE, range: [0,9223372036854775807], default: 500]
            'heuristics/rens/nodesofs': '500',
            # minimum number of nodes required to start the subproblem
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 50]
            'heuristics/rens/minnodes': '50',
            # contingent of sub problem nodes in relation to the number of nodes of the original problem
            # [type: real, advanced: FALSE, range: [0,1], default: 0.1]
            'heuristics/rens/nodesquot': '0.1',
            # factor by which RENS should at least improve the incumbent
            # [type: real, advanced: TRUE, range: [0,1], default: 0.01]
            'heuristics/rens/minimprove': '0.01',
            # factor by which the limit on the number of LP depends on the node limit
            # [type: real, advanced: TRUE, range: [1,1.79769313486232e+308], default: 2]
            'heuristics/rens/lplimfac': '2',
            # solution that is used for fixing values ('l'p relaxation, 'n'lp relaxation)
            # [type: char, advanced: FALSE, range: {nl}, default: l]
            'heuristics/rens/startsol': 'l',
            # should general integers get binary bounds [floor(.),ceil(.)] ?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/rens/binarybounds': 'TRUE',
            # should subproblem be created out of the rows in the LP rows?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/rens/uselprows': 'FALSE',
            # if uselprows == FALSE, should all active cuts from cutpool be copied to constraints in subproblem?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/rens/copycuts': 'TRUE',
            # should the RENS sub-CIP get its own full time limit? This is only for testing and not recommended!
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/rens/extratime': 'FALSE',
            # should all subproblem solutions be added to the original SCIP?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/rens/addallsols': 'FALSE',
            # should the RENS sub-CIP be solved with cuts, conflicts, strong branching,... This is only for testing and not recommended!
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/rens/fullscale': 'FALSE',
            # limit on number of improving incumbent solutions in sub-CIP
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'heuristics/rens/bestsollimit': '-1',
            # should uct node selection be used at the beginning of the search?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/rens/useuct': 'FALSE',
            # priority of heuristic <reoptsols>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 40000]
            'heuristics/reoptsols/priority': '40000',
            # frequency for calling primal heuristic <reoptsols> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 0]
            'heuristics/reoptsols/freq': '0',
            # frequency offset for calling primal heuristic <reoptsols>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/reoptsols/freqofs': '0',
            # maximal depth level to call primal heuristic <reoptsols> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: 0]
            'heuristics/reoptsols/maxdepth': '0',
            # maximal number solutions which should be checked. (-1: all)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 1000]
            'heuristics/reoptsols/maxsols': '1000',
            # check solutions of the last k runs. (-1: all)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'heuristics/reoptsols/maxruns': '-1',
            # priority of heuristic <repair>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 0]
            'heuristics/repair/priority': '0',
            # frequency for calling primal heuristic <repair> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'heuristics/repair/freq': '-1',
            # frequency offset for calling primal heuristic <repair>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/repair/freqofs': '0',
            # maximal depth level to call primal heuristic <repair> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/repair/maxdepth': '-1',
            # file name of a solution to be used as infeasible starting point, [-] if not available
            # [type: string, advanced: FALSE, default: "-"]
            'heuristics/repair/filename': '"-"',
            # True : fractional variables which are not fractional in the given solution are rounded, FALSE : solving process of this heuristic is stopped.
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/repair/roundit': 'TRUE',
            # should a scaled objective function for original variables be used in repair subproblem?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/repair/useobjfactor': 'FALSE',
            # should variable fixings be used in repair subproblem?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/repair/usevarfix': 'TRUE',
            # should slack variables be used in repair subproblem?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/repair/useslackvars': 'FALSE',
            # factor for the potential of var fixings
            # [type: real, advanced: TRUE, range: [0,100], default: 2]
            'heuristics/repair/alpha': '2',
            # number of nodes added to the contingent of the total nodes
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 500]
            'heuristics/repair/nodesofs': '500',
            # maximum number of nodes to regard in the subproblem
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 5000]
            'heuristics/repair/maxnodes': '5000',
            # minimum number of nodes required to start the subproblem
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 50]
            'heuristics/repair/minnodes': '50',
            # contingent of sub problem nodes in relation to the number of nodes of the original problem
            # [type: real, advanced: FALSE, range: [0,1], default: 0.1]
            'heuristics/repair/nodesquot': '0.1',
            # minimum percentage of integer variables that have to be fixed
            # [type: real, advanced: FALSE, range: [0,1], default: 0.3]
            'heuristics/repair/minfixingrate': '0.3',
            # priority of heuristic <rins>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1101000]
            'heuristics/rins/priority': '-1101000',
            # frequency for calling primal heuristic <rins> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 25]
            'heuristics/rins/freq': '25',
            # frequency offset for calling primal heuristic <rins>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/rins/freqofs': '0',
            # maximal depth level to call primal heuristic <rins> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/rins/maxdepth': '-1',
            # number of nodes added to the contingent of the total nodes
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 500]
            'heuristics/rins/nodesofs': '500',
            # maximum number of nodes to regard in the subproblem
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 5000]
            'heuristics/rins/maxnodes': '5000',
            # minimum number of nodes required to start the subproblem
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 50]
            'heuristics/rins/minnodes': '50',
            # contingent of sub problem nodes in relation to the number of nodes of the original problem
            # [type: real, advanced: FALSE, range: [0,1], default: 0.3]
            'heuristics/rins/nodesquot': '0.3',
            # number of nodes without incumbent change that heuristic should wait
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 200]
            'heuristics/rins/nwaitingnodes': '200',
            # factor by which rins should at least improve the incumbent
            # [type: real, advanced: TRUE, range: [0,1], default: 0.01]
            'heuristics/rins/minimprove': '0.01',
            # minimum percentage of integer variables that have to be fixed
            # [type: real, advanced: FALSE, range: [0,1], default: 0.3]
            'heuristics/rins/minfixingrate': '0.3',
            # factor by which the limit on the number of LP depends on the node limit
            # [type: real, advanced: TRUE, range: [1,1.79769313486232e+308], default: 2]
            'heuristics/rins/lplimfac': '2',
            # should subproblem be created out of the rows in the LP rows?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/rins/uselprows': 'FALSE',
            # if uselprows == FALSE, should all active cuts from cutpool be copied to constraints in subproblem?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/rins/copycuts': 'TRUE',
            # should uct node selection be used at the beginning of the search?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/rins/useuct': 'FALSE',
            # priority of heuristic <rootsoldiving>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1005000]
            'heuristics/rootsoldiving/priority': '-1005000',
            # frequency for calling primal heuristic <rootsoldiving> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 20]
            'heuristics/rootsoldiving/freq': '20',
            # frequency offset for calling primal heuristic <rootsoldiving>
            # [type: int, advanced: FALSE, range: [0,65534], default: 5]
            'heuristics/rootsoldiving/freqofs': '5',
            # maximal depth level to call primal heuristic <rootsoldiving> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/rootsoldiving/maxdepth': '-1',
            # minimal relative depth to start diving
            # [type: real, advanced: TRUE, range: [0,1], default: 0]
            'heuristics/rootsoldiving/minreldepth': '0',
            # maximal relative depth to start diving
            # [type: real, advanced: TRUE, range: [0,1], default: 1]
            'heuristics/rootsoldiving/maxreldepth': '1',
            # maximal fraction of diving LP iterations compared to node LP iterations
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 0.01]
            'heuristics/rootsoldiving/maxlpiterquot': '0.01',
            # additional number of allowed LP iterations
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 1000]
            'heuristics/rootsoldiving/maxlpiterofs': '1000',
            # total number of feasible solutions found up to which heuristic is called (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'heuristics/rootsoldiving/maxsols': '-1',
            # maximal diving depth: number of binary/integer variables times depthfac
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0.5]
            'heuristics/rootsoldiving/depthfac': '0.5',
            # maximal diving depth factor if no feasible solution was found yet
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 2]
            'heuristics/rootsoldiving/depthfacnosol': '2',
            # soft rounding factor to fade out objective coefficients
            # [type: real, advanced: TRUE, range: [0,1], default: 0.9]
            'heuristics/rootsoldiving/alpha': '0.9',
            # priority of heuristic <rounding>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1000]
            'heuristics/rounding/priority': '-1000',
            # frequency for calling primal heuristic <rounding> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'heuristics/rounding/freq': '1',
            # frequency offset for calling primal heuristic <rounding>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/rounding/freqofs': '0',
            # maximal depth level to call primal heuristic <rounding> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/rounding/maxdepth': '-1',
            # number of calls per found solution that are considered as standard success, a higher factor causes the heuristic to be called more often
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 100]
            'heuristics/rounding/successfactor': '100',
            # should the heuristic only be called once per node?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/rounding/oncepernode': 'FALSE',
            # priority of heuristic <shiftandpropagate>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 1000]
            'heuristics/shiftandpropagate/priority': '1000',
            # frequency for calling primal heuristic <shiftandpropagate> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 0]
            'heuristics/shiftandpropagate/freq': '0',
            # frequency offset for calling primal heuristic <shiftandpropagate>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/shiftandpropagate/freqofs': '0',
            # maximal depth level to call primal heuristic <shiftandpropagate> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/shiftandpropagate/maxdepth': '-1',
            # The number of propagation rounds used for each propagation
            # [type: int, advanced: TRUE, range: [-1,1000], default: 10]
            'heuristics/shiftandpropagate/nproprounds': '10',
            # Should continuous variables be relaxed?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/shiftandpropagate/relax': 'TRUE',
            # Should domains be reduced by probing?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/shiftandpropagate/probing': 'TRUE',
            # Should heuristic only be executed if no primal solution was found, yet?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/shiftandpropagate/onlywithoutsol': 'TRUE',
            # The number of cutoffs before heuristic stops
            # [type: int, advanced: TRUE, range: [-1,1000000], default: 15]
            'heuristics/shiftandpropagate/cutoffbreaker': '15',
            # the key for variable sorting: (n)orms down, norms (u)p, (v)iolations down, viola(t)ions up, or (r)andom
            # [type: char, advanced: TRUE, range: {nrtuv}, default: v]
            'heuristics/shiftandpropagate/sortkey': 'v',
            # Should variables be sorted for the heuristic?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/shiftandpropagate/sortvars': 'TRUE',
            # should variable statistics be collected during probing?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/shiftandpropagate/collectstats': 'TRUE',
            # Should the heuristic stop calculating optimal shift values when no more rows are violated?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/shiftandpropagate/stopafterfeasible': 'TRUE',
            # Should binary variables be shifted first?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/shiftandpropagate/preferbinaries': 'TRUE',
            # should variables with a zero shifting value be delayed instead of being fixed?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/shiftandpropagate/nozerofixing': 'FALSE',
            # should binary variables with no locks in one direction be fixed to that direction?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/shiftandpropagate/fixbinlocks': 'TRUE',
            # should binary variables with no locks be preferred in the ordering?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/shiftandpropagate/binlocksfirst': 'FALSE',
            # should coefficients and left/right hand sides be normalized by max row coeff?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/shiftandpropagate/normalize': 'TRUE',
            # should row weight be increased every time the row is violated?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/shiftandpropagate/updateweights': 'FALSE',
            # should implicit integer variables be treated as continuous variables?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/shiftandpropagate/impliscontinuous': 'TRUE',
            # should the heuristic choose the best candidate in every round? (set to FALSE for static order)?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/shiftandpropagate/selectbest': 'FALSE',
            # maximum percentage of allowed cutoffs before stopping the heuristic
            # [type: real, advanced: TRUE, range: [0,2], default: 0]
            'heuristics/shiftandpropagate/maxcutoffquot': '0',
            # minimum fixing rate over all variables (including continuous) to solve LP
            # [type: real, advanced: TRUE, range: [0,1], default: 0]
            'heuristics/shiftandpropagate/minfixingratelp': '0',
            # priority of heuristic <shifting>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -5000]
            'heuristics/shifting/priority': '-5000',
            # frequency for calling primal heuristic <shifting> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 10]
            'heuristics/shifting/freq': '10',
            # frequency offset for calling primal heuristic <shifting>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/shifting/freqofs': '0',
            # maximal depth level to call primal heuristic <shifting> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/shifting/maxdepth': '-1',
            # priority of heuristic <simplerounding>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 0]
            'heuristics/simplerounding/priority': '0',
            # frequency for calling primal heuristic <simplerounding> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'heuristics/simplerounding/freq': '1',
            # frequency offset for calling primal heuristic <simplerounding>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/simplerounding/freqofs': '0',
            # maximal depth level to call primal heuristic <simplerounding> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/simplerounding/maxdepth': '-1',
            # should the heuristic only be called once per node?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/simplerounding/oncepernode': 'FALSE',
            # priority of heuristic <subnlp>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -2000000]
            'heuristics/subnlp/priority': '-2000000',
            # frequency for calling primal heuristic <subnlp> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'heuristics/subnlp/freq': '1',
            # frequency offset for calling primal heuristic <subnlp>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/subnlp/freqofs': '0',
            # maximal depth level to call primal heuristic <subnlp> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/subnlp/maxdepth': '-1',
            # verbosity level of NLP solver
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 0]
            'heuristics/subnlp/nlpverblevel': '0',
            # iteration limit of NLP solver; 0 to use solver default
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 0]
            'heuristics/subnlp/nlpiterlimit': '0',
            # time limit of NLP solver; 0 to use solver default
            # [type: real, advanced: FALSE, range: [0,1e+20], default: 0]
            'heuristics/subnlp/nlptimelimit': '0',
            # name of an NLP solver specific options file
            # [type: string, advanced: TRUE, default: ""]
            'heuristics/subnlp/nlpoptfile': '""',
            # if SCIP does not accept a NLP feasible solution, resolve NLP with feas. tolerance reduced by this factor (set to 1.0 to turn off resolve)
            # [type: real, advanced: TRUE, range: [0,1], default: 0.001]
            'heuristics/subnlp/resolvetolfactor': '0.001',
            # should the NLP resolve be started from the original starting point or the infeasible solution?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/subnlp/resolvefromscratch': 'TRUE',
            # number of iterations added to the contingent of the total number of iterations
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 500]
            'heuristics/subnlp/iteroffset': '500',
            # contingent of NLP iterations in relation to the number of nodes in SCIP
            # [type: real, advanced: FALSE, range: [0,1e+20], default: 0.1]
            'heuristics/subnlp/iterquotient': '0.1',
            # contingent of NLP iterations in relation to the number of nodes in SCIP
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 300]
            'heuristics/subnlp/itermin': '300',
            # whether to run NLP heuristic always if starting point available (does not use iteroffset,iterquot,itermin)
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/subnlp/runalways': 'FALSE',
            # factor by which NLP heuristic should at least improve the incumbent
            # [type: real, advanced: TRUE, range: [0,1], default: 0.01]
            'heuristics/subnlp/minimprove': '0.01',
            # limit on number of presolve rounds in sub-SCIP (-1 for unlimited, 0 for no presolve)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'heuristics/subnlp/maxpresolverounds': '-1',
            # whether to add constraints that forbid specific fixings that turned out to be infeasible
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/subnlp/forbidfixings': 'TRUE',
            # whether to keep SCIP copy or to create new copy each time heuristic is applied
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/subnlp/keepcopy': 'TRUE',
            # priority of heuristic <trivial>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 10000]
            'heuristics/trivial/priority': '10000',
            # frequency for calling primal heuristic <trivial> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 0]
            'heuristics/trivial/freq': '0',
            # frequency offset for calling primal heuristic <trivial>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/trivial/freqofs': '0',
            # maximal depth level to call primal heuristic <trivial> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/trivial/maxdepth': '-1',
            # priority of heuristic <trivialnegation>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 40000]
            'heuristics/trivialnegation/priority': '40000',
            # frequency for calling primal heuristic <trivialnegation> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 0]
            'heuristics/trivialnegation/freq': '0',
            # frequency offset for calling primal heuristic <trivialnegation>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/trivialnegation/freqofs': '0',
            # maximal depth level to call primal heuristic <trivialnegation> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: 0]
            'heuristics/trivialnegation/maxdepth': '0',
            # priority of heuristic <trustregion>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1102000]
            'heuristics/trustregion/priority': '-1102000',
            # frequency for calling primal heuristic <trustregion> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'heuristics/trustregion/freq': '-1',
            # frequency offset for calling primal heuristic <trustregion>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/trustregion/freqofs': '0',
            # maximal depth level to call primal heuristic <trustregion> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/trustregion/maxdepth': '-1',
            # number of nodes added to the contingent of the total nodes
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 1000]
            'heuristics/trustregion/nodesofs': '1000',
            # the number of binary variables necessary to run the heuristic
            # [type: int, advanced: FALSE, range: [1,2147483647], default: 10]
            'heuristics/trustregion/minbinvars': '10',
            # contingent of sub problem nodes in relation to the number of nodes of the original problem
            # [type: real, advanced: FALSE, range: [0,1], default: 0.05]
            'heuristics/trustregion/nodesquot': '0.05',
            # factor by which the limit on the number of LP depends on the node limit
            # [type: real, advanced: TRUE, range: [1,1.79769313486232e+308], default: 1.5]
            'heuristics/trustregion/lplimfac': '1.5',
            # minimum number of nodes required to start the subproblem
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 100]
            'heuristics/trustregion/minnodes': '100',
            # maximum number of nodes to regard in the subproblem
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 10000]
            'heuristics/trustregion/maxnodes': '10000',
            # number of nodes without incumbent change that heuristic should wait
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 1]
            'heuristics/trustregion/nwaitingnodes': '1',
            # should subproblem be created out of the rows in the LP rows?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/trustregion/uselprows': 'FALSE',
            # if uselprows == FALSE, should all active cuts from cutpool be copied to constraints in subproblem?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/trustregion/copycuts': 'TRUE',
            # limit on number of improving incumbent solutions in sub-CIP
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 3]
            'heuristics/trustregion/bestsollimit': '3',
            # the penalty for each change in the binary variables from the candidate solution
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 100]
            'heuristics/trustregion/violpenalty': '100',
            # the minimum absolute improvement in the objective function value
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 0.01]
            'heuristics/trustregion/objminimprove': '0.01',
            # priority of heuristic <trysol>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -3000000]
            'heuristics/trysol/priority': '-3000000',
            # frequency for calling primal heuristic <trysol> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'heuristics/trysol/freq': '1',
            # frequency offset for calling primal heuristic <trysol>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/trysol/freqofs': '0',
            # maximal depth level to call primal heuristic <trysol> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/trysol/maxdepth': '-1',
            # priority of heuristic <twoopt>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -20100]
            'heuristics/twoopt/priority': '-20100',
            # frequency for calling primal heuristic <twoopt> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'heuristics/twoopt/freq': '-1',
            # frequency offset for calling primal heuristic <twoopt>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/twoopt/freqofs': '0',
            # maximal depth level to call primal heuristic <twoopt> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/twoopt/maxdepth': '-1',
            #  Should Integer-2-Optimization be applied or not?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/twoopt/intopt': 'FALSE',
            # user parameter to determine number of nodes to wait after last best solution before calling heuristic
            # [type: int, advanced: TRUE, range: [0,10000], default: 0]
            'heuristics/twoopt/waitingnodes': '0',
            # maximum number of slaves for one master variable
            # [type: int, advanced: TRUE, range: [-1,1000000], default: 199]
            'heuristics/twoopt/maxnslaves': '199',
            # parameter to determine the percentage of rows two variables have to share before they are considered equal
            # [type: real, advanced: TRUE, range: [0,1], default: 0.5]
            'heuristics/twoopt/matchingrate': '0.5',
            # priority of heuristic <undercover>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1110000]
            'heuristics/undercover/priority': '-1110000',
            # frequency for calling primal heuristic <undercover> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 0]
            'heuristics/undercover/freq': '0',
            # frequency offset for calling primal heuristic <undercover>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/undercover/freqofs': '0',
            # maximal depth level to call primal heuristic <undercover> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/undercover/maxdepth': '-1',
            # prioritized sequence of fixing values used ('l'p relaxation, 'n'lp relaxation, 'i'ncumbent solution)
            # [type: string, advanced: FALSE, default: "li"]
            'heuristics/undercover/fixingalts': '"li"',
            # maximum number of nodes to regard in the subproblem
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 500]
            'heuristics/undercover/maxnodes': '500',
            # minimum number of nodes required to start the subproblem
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 500]
            'heuristics/undercover/minnodes': '500',
            # number of nodes added to the contingent of the total nodes
            # [type: longint, advanced: FALSE, range: [0,9223372036854775807], default: 500]
            'heuristics/undercover/nodesofs': '500',
            # weight for conflict score in fixing order
            # [type: real, advanced: TRUE, range: [-1.79769313486232e+308,1.79769313486232e+308], default: 1000]
            'heuristics/undercover/conflictweight': '1000',
            # weight for cutoff score in fixing order
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 1]
            'heuristics/undercover/cutoffweight': '1',
            # weight for inference score in fixing order
            # [type: real, advanced: TRUE, range: [-1.79769313486232e+308,1.79769313486232e+308], default: 1]
            'heuristics/undercover/inferenceweight': '1',
            # maximum coversize (as fraction of total number of variables)
            # [type: real, advanced: TRUE, range: [0,1], default: 1]
            'heuristics/undercover/maxcoversizevars': '1',
            # maximum coversize maximum coversize (as ratio to the percentage of non-affected constraints)
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 1.79769313486232e+308]
            'heuristics/undercover/maxcoversizeconss': '1.79769313486232e+308',
            # minimum percentage of nonlinear constraints in the original problem
            # [type: real, advanced: TRUE, range: [0,1], default: 0.15]
            'heuristics/undercover/mincoveredrel': '0.15',
            # factor by which the heuristic should at least improve the incumbent
            # [type: real, advanced: TRUE, range: [-1,1], default: 0]
            'heuristics/undercover/minimprove': '0',
            # contingent of sub problem nodes in relation to the number of nodes of the original problem
            # [type: real, advanced: FALSE, range: [0,1], default: 0.1]
            'heuristics/undercover/nodesquot': '0.1',
            # fraction of covering variables in the last cover which need to change their value when recovering
            # [type: real, advanced: TRUE, range: [0,1], default: 0.9]
            'heuristics/undercover/recoverdiv': '0.9',
            # minimum number of nonlinear constraints in the original problem
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 5]
            'heuristics/undercover/mincoveredabs': '5',
            # maximum number of backtracks in fix-and-propagate
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 6]
            'heuristics/undercover/maxbacktracks': '6',
            # maximum number of recoverings
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 0]
            'heuristics/undercover/maxrecovers': '0',
            # maximum number of reorderings of the fixing order
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 1]
            'heuristics/undercover/maxreorders': '1',
            # objective function of the covering problem (influenced nonlinear 'c'onstraints/'t'erms, 'd'omain size, 'l'ocks, 'm'in of up/down locks, 'u'nit penalties)
            # [type: char, advanced: TRUE, range: {cdlmtu}, default: u]
            'heuristics/undercover/coveringobj': 'u',
            # order in which variables should be fixed (increasing 'C'onflict score, decreasing 'c'onflict score, increasing 'V'ariable index, decreasing 'v'ariable index
            # [type: char, advanced: TRUE, range: {CcVv}, default: v]
            'heuristics/undercover/fixingorder': 'v',
            # should the heuristic be called at root node before cut separation?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/undercover/beforecuts': 'TRUE',
            # should integer variables in the cover be fixed first?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/undercover/fixintfirst': 'FALSE',
            # shall LP values for integer vars be rounded according to locks?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/undercover/locksrounding': 'TRUE',
            # should we only fix variables in order to obtain a convex problem?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/undercover/onlyconvexify': 'FALSE',
            # should the NLP heuristic be called to polish a feasible solution?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/undercover/postnlp': 'TRUE',
            # should bounddisjunction constraints be covered (or just copied)?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/undercover/coverbd': 'FALSE',
            # should all active cuts from cutpool be copied to constraints in subproblem?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/undercover/copycuts': 'TRUE',
            # shall the cover be reused if a conflict was added after an infeasible subproblem?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/undercover/reusecover': 'FALSE',
            # priority of heuristic <vbounds>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 2500]
            'heuristics/vbounds/priority': '2500',
            # frequency for calling primal heuristic <vbounds> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 0]
            'heuristics/vbounds/freq': '0',
            # frequency offset for calling primal heuristic <vbounds>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/vbounds/freqofs': '0',
            # maximal depth level to call primal heuristic <vbounds> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/vbounds/maxdepth': '-1',
            # minimum percentage of integer variables that have to be fixed
            # [type: real, advanced: FALSE, range: [0,1], default: 0.65]
            'heuristics/vbounds/minintfixingrate': '0.65',
            # minimum percentage of variables that have to be fixed within sub-SCIP (integer and continuous)
            # [type: real, advanced: FALSE, range: [0,1], default: 0.65]
            'heuristics/vbounds/minmipfixingrate': '0.65',
            # maximum number of nodes to regard in the subproblem
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 5000]
            'heuristics/vbounds/maxnodes': '5000',
            # number of nodes added to the contingent of the total nodes
            # [type: longint, advanced: FALSE, range: [0,9223372036854775807], default: 500]
            'heuristics/vbounds/nodesofs': '500',
            # minimum number of nodes required to start the subproblem
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 500]
            'heuristics/vbounds/minnodes': '500',
            # contingent of sub problem nodes in relation to the number of nodes of the original problem
            # [type: real, advanced: FALSE, range: [0,1], default: 0.1]
            'heuristics/vbounds/nodesquot': '0.1',
            # factor by which vbounds heuristic should at least improve the incumbent
            # [type: real, advanced: TRUE, range: [0,1], default: 0.01]
            'heuristics/vbounds/minimprove': '0.01',
            # maximum number of propagation rounds during probing (-1 infinity)
            # [type: int, advanced: TRUE, range: [-1,536870911], default: 2]
            'heuristics/vbounds/maxproprounds': '2',
            # should all active cuts from cutpool be copied to constraints in subproblem?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/vbounds/copycuts': 'TRUE',
            # should more variables be fixed based on variable locks if the fixing rate was not reached?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/vbounds/uselockfixings': 'FALSE',
            # maximum number of backtracks during the fixing process
            # [type: int, advanced: TRUE, range: [-1,536870911], default: 10]
            'heuristics/vbounds/maxbacktracks': '10',
            # which variants of the vbounds heuristic that try to stay feasible should be called? (0: off, 1: w/o looking at obj, 2: only fix to best bound, 4: only fix to worst bound
            # [type: int, advanced: TRUE, range: [0,7], default: 6]
            'heuristics/vbounds/feasvariant': '6',
            # which tightening variants of the vbounds heuristic should be called? (0: off, 1: w/o looking at obj, 2: only fix to best bound, 4: only fix to worst bound
            # [type: int, advanced: TRUE, range: [0,7], default: 7]
            'heuristics/vbounds/tightenvariant': '7',
            # priority of heuristic <veclendiving>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1003100]
            'heuristics/veclendiving/priority': '-1003100',
            # frequency for calling primal heuristic <veclendiving> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 10]
            'heuristics/veclendiving/freq': '10',
            # frequency offset for calling primal heuristic <veclendiving>
            # [type: int, advanced: FALSE, range: [0,65534], default: 4]
            'heuristics/veclendiving/freqofs': '4',
            # maximal depth level to call primal heuristic <veclendiving> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/veclendiving/maxdepth': '-1',
            # minimal relative depth to start diving
            # [type: real, advanced: TRUE, range: [0,1], default: 0]
            'heuristics/veclendiving/minreldepth': '0',
            # maximal relative depth to start diving
            # [type: real, advanced: TRUE, range: [0,1], default: 1]
            'heuristics/veclendiving/maxreldepth': '1',
            # maximal fraction of diving LP iterations compared to node LP iterations
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 0.05]
            'heuristics/veclendiving/maxlpiterquot': '0.05',
            # additional number of allowed LP iterations
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 1000]
            'heuristics/veclendiving/maxlpiterofs': '1000',
            # maximal quotient (curlowerbound - lowerbound)/(cutoffbound - lowerbound) where diving is performed (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1], default: 0.8]
            'heuristics/veclendiving/maxdiveubquot': '0.8',
            # maximal quotient (curlowerbound - lowerbound)/(avglowerbound - lowerbound) where diving is performed (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0]
            'heuristics/veclendiving/maxdiveavgquot': '0',
            # maximal UBQUOT when no solution was found yet (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1], default: 0.1]
            'heuristics/veclendiving/maxdiveubquotnosol': '0.1',
            # maximal AVGQUOT when no solution was found yet (0.0: no limit)
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0]
            'heuristics/veclendiving/maxdiveavgquotnosol': '0',
            # use one level of backtracking if infeasibility is encountered?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/veclendiving/backtrack': 'TRUE',
            # percentage of immediate domain changes during probing to trigger LP resolve
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 0.15]
            'heuristics/veclendiving/lpresolvedomchgquot': '0.15',
            # LP solve frequency for diving heuristics (0: only after enough domain changes have been found)
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 0]
            'heuristics/veclendiving/lpsolvefreq': '0',
            # should only LP branching candidates be considered instead of the slower but more general constraint handler diving variable selection?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'heuristics/veclendiving/onlylpbranchcands': 'FALSE',
            # priority of heuristic <zirounding>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -500]
            'heuristics/zirounding/priority': '-500',
            # frequency for calling primal heuristic <zirounding> (-1: never, 0: only at depth freqofs)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'heuristics/zirounding/freq': '1',
            # frequency offset for calling primal heuristic <zirounding>
            # [type: int, advanced: FALSE, range: [0,65534], default: 0]
            'heuristics/zirounding/freqofs': '0',
            # maximal depth level to call primal heuristic <zirounding> (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'heuristics/zirounding/maxdepth': '-1',
            # determines maximum number of rounding loops
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 2]
            'heuristics/zirounding/maxroundingloops': '2',
            # flag to determine if Zirounding is deactivated after a certain percentage of unsuccessful calls
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'heuristics/zirounding/stopziround': 'TRUE',
            # if percentage of found solutions falls below this parameter, Zirounding will be deactivated
            # [type: real, advanced: TRUE, range: [0,1], default: 0.02]
            'heuristics/zirounding/stoppercentage': '0.02',
            # determines the minimum number of calls before percentage-based deactivation of Zirounding is applied
            # [type: int, advanced: TRUE, range: [1,2147483647], default: 1000]
            'heuristics/zirounding/minstopncalls': '1000',
            # priority of propagator <dualfix>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 8000000]
            'propagating/dualfix/priority': '8000000',
            # frequency for calling propagator <dualfix> (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 0]
            'propagating/dualfix/freq': '0',
            # should propagator be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'propagating/dualfix/delay': 'FALSE',
            # timing when propagator should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS))
            # [type: int, advanced: TRUE, range: [1,15], default: 1]
            'propagating/dualfix/timingmask': '1',
            # presolving priority of propagator <dualfix>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 8000000]
            'propagating/dualfix/presolpriority': '8000000',
            # maximal number of presolving rounds the propagator participates in (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'propagating/dualfix/maxprerounds': '-1',
            # timing mask of the presolving method of propagator <dualfix> (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [2,60], default: 4]
            'propagating/dualfix/presoltiming': '4',
            # priority of propagator <genvbounds>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 3000000]
            'propagating/genvbounds/priority': '3000000',
            # frequency for calling propagator <genvbounds> (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'propagating/genvbounds/freq': '1',
            # should propagator be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'propagating/genvbounds/delay': 'FALSE',
            # timing when propagator should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS))
            # [type: int, advanced: TRUE, range: [1,15], default: 15]
            'propagating/genvbounds/timingmask': '15',
            # presolving priority of propagator <genvbounds>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -2000000]
            'propagating/genvbounds/presolpriority': '-2000000',
            # maximal number of presolving rounds the propagator participates in (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'propagating/genvbounds/maxprerounds': '-1',
            # timing mask of the presolving method of propagator <genvbounds> (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [2,60], default: 4]
            'propagating/genvbounds/presoltiming': '4',
            # apply global propagation?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'propagating/genvbounds/global': 'TRUE',
            # apply genvbounds in root node if no new incumbent was found?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'propagating/genvbounds/propinrootnode': 'TRUE',
            # sort genvbounds and wait for bound change events?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'propagating/genvbounds/sort': 'TRUE',
            # should genvbounds be transformed to (linear) constraints?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'propagating/genvbounds/propasconss': 'FALSE',
            # priority of propagator <obbt>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1000000]
            'propagating/obbt/priority': '-1000000',
            # frequency for calling propagator <obbt> (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 0]
            'propagating/obbt/freq': '0',
            # should propagator be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'propagating/obbt/delay': 'TRUE',
            # timing when propagator should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS))
            # [type: int, advanced: TRUE, range: [1,15], default: 4]
            'propagating/obbt/timingmask': '4',
            # presolving priority of propagator <obbt>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 0]
            'propagating/obbt/presolpriority': '0',
            # maximal number of presolving rounds the propagator participates in (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'propagating/obbt/maxprerounds': '-1',
            # timing mask of the presolving method of propagator <obbt> (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [2,60], default: 28]
            'propagating/obbt/presoltiming': '28',
            # should obbt try to provide genvbounds if possible?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'propagating/obbt/creategenvbounds': 'TRUE',
            # should coefficients in filtering be normalized w.r.t. the domains sizes?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'propagating/obbt/normalize': 'TRUE',
            # try to filter bounds in so-called filter rounds by solving auxiliary LPs?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'propagating/obbt/applyfilterrounds': 'FALSE',
            # try to filter bounds with the LP solution after each solve?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'propagating/obbt/applytrivialfilter': 'TRUE',
            # should we try to generate genvbounds during trivial and aggressive filtering?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'propagating/obbt/genvbdsduringfilter': 'TRUE',
            # try to create genvbounds during separation process?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'propagating/obbt/genvbdsduringsepa': 'TRUE',
            # minimal number of filtered bounds to apply another filter round
            # [type: int, advanced: TRUE, range: [1,2147483647], default: 2]
            'propagating/obbt/minfilter': '2',
            # multiple of root node LP iterations used as total LP iteration limit for obbt (<= 0: no limit )
            # [type: real, advanced: FALSE, range: [-1.79769313486232e+308,1.79769313486232e+308], default: 10]
            'propagating/obbt/itlimitfactor': '10',
            # multiple of OBBT LP limit used as total LP iteration limit for solving bilinear inequality LPs (< 0 for no limit)
            # [type: real, advanced: FALSE, range: [-1.79769313486232e+308,1.79769313486232e+308], default: 3]
            'propagating/obbt/itlimitfactorbilin': '3',
            # minimum absolute value of nonconvex eigenvalues for a bilinear term
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 0.1]
            'propagating/obbt/minnonconvexity': '0.1',
            # minimum LP iteration limit
            # [type: longint, advanced: FALSE, range: [0,9223372036854775807], default: 5000]
            'propagating/obbt/minitlimit': '5000',
            # feasibility tolerance for reduced costs used in obbt; this value is used if SCIP's dual feastol is greater
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 1e-09]
            'propagating/obbt/dualfeastol': '1e-09',
            # maximum condition limit used in LP solver (-1.0: no limit)
            # [type: real, advanced: FALSE, range: [-1,1.79769313486232e+308], default: -1]
            'propagating/obbt/conditionlimit': '-1',
            # minimal relative improve for strengthening bounds
            # [type: real, advanced: FALSE, range: [0,1], default: 0.001]
            'propagating/obbt/boundstreps': '0.001',
            # only apply obbt on non-convex variables
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'propagating/obbt/onlynonconvexvars': 'FALSE',
            # should integral bounds be tightened during the probing mode?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'propagating/obbt/tightintboundsprobing': 'TRUE',
            # should continuous bounds be tightened during the probing mode?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'propagating/obbt/tightcontboundsprobing': 'FALSE',
            # solve auxiliary LPs in order to find valid inequalities for bilinear terms?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'propagating/obbt/createbilinineqs': 'TRUE',
            # select the type of ordering algorithm which should be used (0: no special ordering, 1: greedy, 2: greedy reverse)
            # [type: int, advanced: TRUE, range: [0,2], default: 1]
            'propagating/obbt/orderingalgo': '1',
            # should the obbt LP solution be separated?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'propagating/obbt/separatesol': 'FALSE',
            # minimum number of iteration spend to separate an obbt LP solution
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 0]
            'propagating/obbt/sepaminiter': '0',
            # maximum number of iteration spend to separate an obbt LP solution
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 10]
            'propagating/obbt/sepamaxiter': '10',
            # trigger a propagation round after that many bound tightenings (0: no propagation)
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 0]
            'propagating/obbt/propagatefreq': '0',
            # priority of propagator <nlobbt>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1100000]
            'propagating/nlobbt/priority': '-1100000',
            # frequency for calling propagator <nlobbt> (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'propagating/nlobbt/freq': '-1',
            # should propagator be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'propagating/nlobbt/delay': 'TRUE',
            # timing when propagator should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS))
            # [type: int, advanced: TRUE, range: [1,15], default: 4]
            'propagating/nlobbt/timingmask': '4',
            # presolving priority of propagator <nlobbt>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 0]
            'propagating/nlobbt/presolpriority': '0',
            # maximal number of presolving rounds the propagator participates in (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'propagating/nlobbt/maxprerounds': '-1',
            # timing mask of the presolving method of propagator <nlobbt> (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [2,60], default: 28]
            'propagating/nlobbt/presoltiming': '28',
            # factor for NLP feasibility tolerance
            # [type: real, advanced: TRUE, range: [0,1], default: 0.01]
            'propagating/nlobbt/feastolfac': '0.01',
            # factor for NLP relative objective tolerance
            # [type: real, advanced: TRUE, range: [0,1], default: 0.01]
            'propagating/nlobbt/relobjtolfac': '0.01',
            # (#convex nlrows)/(#nonconvex nlrows) threshold to apply propagator
            # [type: real, advanced: TRUE, range: [0,1e+20], default: 0.2]
            'propagating/nlobbt/minnonconvexfrac': '0.2',
            # minimum (#convex nlrows)/(#linear nlrows) threshold to apply propagator
            # [type: real, advanced: TRUE, range: [0,1e+20], default: 0.02]
            'propagating/nlobbt/minlinearfrac': '0.02',
            # should non-initial LP rows be used?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'propagating/nlobbt/addlprows': 'TRUE',
            # iteration limit of NLP solver; 0 for no limit
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 500]
            'propagating/nlobbt/nlpiterlimit': '500',
            # time limit of NLP solver; 0.0 for no limit
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0]
            'propagating/nlobbt/nlptimelimit': '0',
            # verbosity level of NLP solver
            # [type: int, advanced: TRUE, range: [0,5], default: 0]
            'propagating/nlobbt/nlpverblevel': '0',
            # LP iteration limit for nlobbt will be this factor times total LP iterations in root node
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 2]
            'propagating/nlobbt/itlimitfactor': '2',
            # priority of propagator <probing>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -100000]
            'propagating/probing/priority': '-100000',
            # frequency for calling propagator <probing> (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'propagating/probing/freq': '-1',
            # should propagator be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'propagating/probing/delay': 'TRUE',
            # timing when propagator should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS))
            # [type: int, advanced: TRUE, range: [1,15], default: 4]
            'propagating/probing/timingmask': '4',
            # presolving priority of propagator <probing>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -100000]
            'propagating/probing/presolpriority': '-100000',
            # maximal number of presolving rounds the propagator participates in (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'propagating/probing/maxprerounds': '-1',
            # timing mask of the presolving method of propagator <probing> (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [2,60], default: 16]
            'propagating/probing/presoltiming': '16',
            # maximal number of runs, probing participates in (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 1]
            'propagating/probing/maxruns': '1',
            # maximal number of propagation rounds in probing subproblems (-1: no limit, 0: auto)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'propagating/probing/proprounds': '-1',
            # maximal number of fixings found, until probing is interrupted (0: don't iterrupt)
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 25]
            'propagating/probing/maxfixings': '25',
            # maximal number of successive probings without fixings, until probing is aborted (0: don't abort)
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 1000]
            'propagating/probing/maxuseless': '1000',
            # maximal number of successive probings without fixings, bound changes, and implications, until probing is aborted (0: don't abort)
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 50]
            'propagating/probing/maxtotaluseless': '50',
            # maximal number of probings without fixings, until probing is aborted (0: don't abort)
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 0]
            'propagating/probing/maxsumuseless': '0',
            # maximal depth until propagation is executed(-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'propagating/probing/maxdepth': '-1',
            # priority of propagator <pseudoobj>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 3000000]
            'propagating/pseudoobj/priority': '3000000',
            # frequency for calling propagator <pseudoobj> (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'propagating/pseudoobj/freq': '1',
            # should propagator be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'propagating/pseudoobj/delay': 'FALSE',
            # timing when propagator should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS))
            # [type: int, advanced: TRUE, range: [1,15], default: 7]
            'propagating/pseudoobj/timingmask': '7',
            # presolving priority of propagator <pseudoobj>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 6000000]
            'propagating/pseudoobj/presolpriority': '6000000',
            # maximal number of presolving rounds the propagator participates in (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'propagating/pseudoobj/maxprerounds': '-1',
            # timing mask of the presolving method of propagator <pseudoobj> (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [2,60], default: 4]
            'propagating/pseudoobj/presoltiming': '4',
            # minimal number of successive non-binary variable propagations without a bound reduction before aborted
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 100]
            'propagating/pseudoobj/minuseless': '100',
            # maximal fraction of non-binary variables with non-zero objective without a bound reduction before aborted
            # [type: real, advanced: TRUE, range: [0,1], default: 0.1]
            'propagating/pseudoobj/maxvarsfrac': '0.1',
            # whether to propagate all non-binary variables when we are propagating the root node
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'propagating/pseudoobj/propfullinroot': 'TRUE',
            # propagate new cutoff bound directly globally
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'propagating/pseudoobj/propcutoffbound': 'TRUE',
            # should the propagator be forced even if active pricer are present?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'propagating/pseudoobj/force': 'FALSE',
            # number of variables added after the propagator is reinitialized?
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 1000]
            'propagating/pseudoobj/maxnewvars': '1000',
            # use implications to strengthen the propagation of binary variable (increasing the objective change)?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'propagating/pseudoobj/propuseimplics': 'TRUE',
            # use implications to strengthen the resolve propagation of binary variable (increasing the objective change)?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'propagating/pseudoobj/respropuseimplics': 'TRUE',
            # maximum number of binary variables the implications are used if turned on (-1: unlimited)?
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 50000]
            'propagating/pseudoobj/maximplvars': '50000',
            # priority of propagator <redcost>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 1000000]
            'propagating/redcost/priority': '1000000',
            # frequency for calling propagator <redcost> (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'propagating/redcost/freq': '1',
            # should propagator be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'propagating/redcost/delay': 'FALSE',
            # timing when propagator should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS))
            # [type: int, advanced: TRUE, range: [1,15], default: 6]
            'propagating/redcost/timingmask': '6',
            # presolving priority of propagator <redcost>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 0]
            'propagating/redcost/presolpriority': '0',
            # maximal number of presolving rounds the propagator participates in (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'propagating/redcost/maxprerounds': '-1',
            # timing mask of the presolving method of propagator <redcost> (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [2,60], default: 28]
            'propagating/redcost/presoltiming': '28',
            # should reduced cost fixing be also applied to continuous variables?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'propagating/redcost/continuous': 'FALSE',
            # should implications be used to strength the reduced cost for binary variables?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'propagating/redcost/useimplics': 'FALSE',
            # should the propagator be forced even if active pricer are present?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'propagating/redcost/force': 'FALSE',
            # priority of propagator <rootredcost>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 10000000]
            'propagating/rootredcost/priority': '10000000',
            # frequency for calling propagator <rootredcost> (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'propagating/rootredcost/freq': '1',
            # should propagator be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'propagating/rootredcost/delay': 'FALSE',
            # timing when propagator should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS))
            # [type: int, advanced: TRUE, range: [1,15], default: 5]
            'propagating/rootredcost/timingmask': '5',
            # presolving priority of propagator <rootredcost>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 0]
            'propagating/rootredcost/presolpriority': '0',
            # maximal number of presolving rounds the propagator participates in (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'propagating/rootredcost/maxprerounds': '-1',
            # timing mask of the presolving method of propagator <rootredcost> (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [2,60], default: 28]
            'propagating/rootredcost/presoltiming': '28',
            # should only binary variables be propagated?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'propagating/rootredcost/onlybinary': 'FALSE',
            # should the propagator be forced even if active pricer are present?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'propagating/rootredcost/force': 'FALSE',
            # priority of propagator <symmetry>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1000000]
            'propagating/symmetry/priority': '-1000000',
            # frequency for calling propagator <symmetry> (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'propagating/symmetry/freq': '1',
            # should propagator be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'propagating/symmetry/delay': 'FALSE',
            # timing when propagator should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS))
            # [type: int, advanced: TRUE, range: [1,15], default: 1]
            'propagating/symmetry/timingmask': '1',
            # presolving priority of propagator <symmetry>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -10000000]
            'propagating/symmetry/presolpriority': '-10000000',
            # maximal number of presolving rounds the propagator participates in (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'propagating/symmetry/maxprerounds': '-1',
            # timing mask of the presolving method of propagator <symmetry> (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [2,60], default: 16]
            'propagating/symmetry/presoltiming': '16',
            # is statistics table <orbitalfixing> active
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'table/orbitalfixing/active': 'TRUE',
            # limit on the number of generators that should be produced within symmetry detection (0': 'no limit)',
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 1500]
            'propagating/symmetry/maxgenerators': '1500',
            # Should all symmetries be checked after computation?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'propagating/symmetry/checksymmetries': 'FALSE',
            # Should the number of variables affected by some symmetry be displayed?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'propagating/symmetry/displaynorbitvars': 'FALSE',
            # Double equations to positive/negative version?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'propagating/symmetry/doubleequations': 'FALSE',
            # Should the symmetry breaking constraints be added to the LP?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'propagating/symmetry/conssaddlp': 'TRUE',
            # Add inequalities for symresacks for each generator?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'propagating/symmetry/addsymresacks': 'TRUE',
            # Should we check whether the components of the symmetry group can be handled by orbitopes?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'propagating/symmetry/detectorbitopes': 'TRUE',
            # timing of adding constraints (0 = before presolving, 1 =' during presolving, 2': 'after presolving)',
            # [type: int, advanced: TRUE, range: [0,2], default: 2]
            'propagating/symmetry/addconsstiming': '2',
            # timing of symmetry computation for orbital fixing (0 = before presolving, 1 =' during presolving, 2': 'at first call)',
            # [type: int, advanced: TRUE, range: [0,2], default: 2]
            'propagating/symmetry/ofsymcomptiming': '2',
            # run orbital fixing during presolving?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'propagating/symmetry/performpresolving': 'FALSE',
            # recompute symmetries after a restart has occured?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'propagating/symmetry/recomputerestart': 'FALSE',
            # Should non-affected variables be removed from permutation to save memory?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'propagating/symmetry/compresssymmetries': 'TRUE',
            # Compression is used if percentage of moved vars is at most the threshold.
            # [type: real, advanced: TRUE, range: [0,1], default: 0.5]
            'propagating/symmetry/compressthreshold': '0.5',
            # Should the number of conss a variable is contained in be exploited in symmetry detection?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'propagating/symmetry/usecolumnsparsity': 'FALSE',
            # Shall orbital fixing be disabled if orbital fixing has found a reduction and a restart occurs?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'propagating/symmetry/disableofrestart': 'FALSE',
            # Whether all non-binary variables shall be not affected by symmetries if OF is active?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'propagating/symmetry/symfixnonbinaryvars': 'FALSE',
            # priority of propagator <vbounds>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 3000000]
            'propagating/vbounds/priority': '3000000',
            # frequency for calling propagator <vbounds> (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 1]
            'propagating/vbounds/freq': '1',
            # should propagator be delayed, if other propagators found reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'propagating/vbounds/delay': 'FALSE',
            # timing when propagator should be called (1:BEFORELP, 2:DURINGLPLOOP, 4:AFTERLPLOOP, 15:ALWAYS))
            # [type: int, advanced: TRUE, range: [1,15], default: 5]
            'propagating/vbounds/timingmask': '5',
            # presolving priority of propagator <vbounds>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -90000]
            'propagating/vbounds/presolpriority': '-90000',
            # maximal number of presolving rounds the propagator participates in (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'propagating/vbounds/maxprerounds': '-1',
            # timing mask of the presolving method of propagator <vbounds> (4:FAST, 8:MEDIUM, 16:EXHAUSTIVE, 32:FINAL)
            # [type: int, advanced: TRUE, range: [2,60], default: 24]
            'propagating/vbounds/presoltiming': '24',
            # should bound widening be used to initialize conflict analysis?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'propagating/vbounds/usebdwidening': 'TRUE',
            # should implications be propagated?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'propagating/vbounds/useimplics': 'FALSE',
            # should cliques be propagated?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'propagating/vbounds/usecliques': 'FALSE',
            # should vbounds be propagated?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'propagating/vbounds/usevbounds': 'TRUE',
            # should the bounds be topologically sorted in advance?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'propagating/vbounds/dotoposort': 'TRUE',
            # should cliques be regarded for the topological sort?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'propagating/vbounds/sortcliques': 'FALSE',
            # should cycles in the variable bound graph be identified?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'propagating/vbounds/detectcycles': 'FALSE',
            # minimum percentage of new cliques to trigger another clique table analysis
            # [type: real, advanced: FALSE, range: [0,1], default: 0.1]
            'propagating/vbounds/minnewcliques': '0.1',
            # maximum number of cliques per variable to run clique table analysis in medium presolving
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 50]
            'propagating/vbounds/maxcliquesmedium': '50',
            # maximum number of cliques per variable to run clique table analysis in exhaustive presolving
            # [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 100]
            'propagating/vbounds/maxcliquesexhaustive': '100',
            # priority of separator <cgmip>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1000]
            'separating/cgmip/priority': '-1000',
            # frequency for calling separator <cgmip> (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'separating/cgmip/freq': '-1',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying separator <cgmip> (0.0: only on current best node, 1.0: on all nodes)
            # [type: real, advanced: TRUE, range: [0,1], default: 0]
            'separating/cgmip/maxbounddist': '0',
            # should separator be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'separating/cgmip/delay': 'FALSE',
            # base for exponential increase of frequency at which separator <cgmip> is called (1: call at each multiple of frequency)
            # [type: int, advanced: TRUE, range: [1,100], default: 4]
            'separating/cgmip/expbackoff': '4',
            # maximal number of cgmip separation rounds per node (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 5]
            'separating/cgmip/maxrounds': '5',
            # maximal number of cgmip separation rounds in the root node (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 50]
            'separating/cgmip/maxroundsroot': '50',
            # maximal depth at which the separator is applied (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'separating/cgmip/maxdepth': '-1',
            # Use decision tree to turn separation on/off?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'separating/cgmip/decisiontree': 'FALSE',
            # time limit for sub-MIP
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 1e+20]
            'separating/cgmip/timelimit': '1e+20',
            # memory limit for sub-MIP
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 1e+20]
            'separating/cgmip/memorylimit': '1e+20',
            # minimum number of nodes considered for sub-MIP (-1: unlimited)
            # [type: longint, advanced: FALSE, range: [-1,9223372036854775807], default: 500]
            'separating/cgmip/minnodelimit': '500',
            # maximum number of nodes considered for sub-MIP (-1: unlimited)
            # [type: longint, advanced: FALSE, range: [-1,9223372036854775807], default: 5000]
            'separating/cgmip/maxnodelimit': '5000',
            # bounds on the values of the coefficients in the CG-cut
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 1000]
            'separating/cgmip/cutcoefbnd': '1000',
            # Use only active rows to generate cuts?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'separating/cgmip/onlyactiverows': 'FALSE',
            # maximal age of rows to consider if onlyactiverows is false
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'separating/cgmip/maxrowage': '-1',
            # Separate only rank 1 inequalities w.r.t. CG-MIP separator?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'separating/cgmip/onlyrankone': 'FALSE',
            # Generate cuts for problems with only integer variables?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'separating/cgmip/onlyintvars': 'FALSE',
            # Convert some integral variables to be continuous to reduce the size of the sub-MIP?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'separating/cgmip/contconvert': 'FALSE',
            # fraction of integral variables converted to be continuous (if contconvert)
            # [type: real, advanced: FALSE, range: [0,1], default: 0.1]
            'separating/cgmip/contconvfrac': '0.1',
            # minimum number of integral variables before some are converted to be continuous
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 100]
            'separating/cgmip/contconvmin': '100',
            # Convert some integral variables attaining fractional values to have integral value?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'separating/cgmip/intconvert': 'FALSE',
            # fraction of frac. integral variables converted to have integral value (if intconvert)
            # [type: real, advanced: FALSE, range: [0,1], default: 0.1]
            'separating/cgmip/intconvfrac': '0.1',
            # minimum number of integral variables before some are converted to have integral value
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 100]
            'separating/cgmip/intconvmin': '100',
            # Skip the upper bounds on the multipliers in the sub-MIP?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'separating/cgmip/skipmultbounds': 'TRUE',
            # Should the objective of the sub-MIP minimize the l1-norm of the multipliers?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'separating/cgmip/objlone': 'FALSE',
            # weight used for the row combination coefficient in the sub-MIP objective
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0.001]
            'separating/cgmip/objweight': '0.001',
            # Weight each row by its size?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'separating/cgmip/objweightsize': 'TRUE',
            # should generated cuts be removed from the LP if they are no longer tight?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'separating/cgmip/dynamiccuts': 'TRUE',
            # use CMIR-generator (otherwise add cut directly)?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'separating/cgmip/usecmir': 'TRUE',
            # use strong CG-function to strengthen cut?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'separating/cgmip/usestrongcg': 'FALSE',
            # tell CMIR-generator which bounds to used in rounding?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'separating/cgmip/cmirownbounds': 'FALSE',
            # use cutpool to store CG-cuts even if the are not efficient?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'separating/cgmip/usecutpool': 'TRUE',
            # only separate cuts that are tight for the best feasible solution?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'separating/cgmip/primalseparation': 'TRUE',
            # terminate separation if a violated (but possibly sub-optimal) cut has been found?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'separating/cgmip/earlyterm': 'TRUE',
            # add constraint to subscip that only allows violated cuts (otherwise add obj. limit)?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'separating/cgmip/addviolationcons': 'FALSE',
            # add constraint handler to filter out violated cuts?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'separating/cgmip/addviolconshdlr': 'FALSE',
            # should the violation constraint handler use the norm of a cut to check for feasibility?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'separating/cgmip/conshdlrusenorm': 'TRUE',
            # Use upper bound on objective function (via primal solution)?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'separating/cgmip/useobjub': 'FALSE',
            # Use lower bound on objective function (via primal solution)?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'separating/cgmip/useobjlb': 'FALSE',
            # Should the settings for the sub-MIP be optimized for speed?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'separating/cgmip/subscipfast': 'TRUE',
            # Should information about the sub-MIP and cuts be displayed?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'separating/cgmip/output': 'FALSE',
            # Try to generate primal solutions from Gomory cuts?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'separating/cgmip/genprimalsols': 'FALSE',
            # priority of separator <clique>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -5000]
            'separating/clique/priority': '-5000',
            # frequency for calling separator <clique> (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 0]
            'separating/clique/freq': '0',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying separator <clique> (0.0: only on current best node, 1.0: on all nodes)
            # [type: real, advanced: TRUE, range: [0,1], default: 0]
            'separating/clique/maxbounddist': '0',
            # should separator be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'separating/clique/delay': 'FALSE',
            # base for exponential increase of frequency at which separator <clique> is called (1: call at each multiple of frequency)
            # [type: int, advanced: TRUE, range: [1,100], default: 4]
            'separating/clique/expbackoff': '4',
            # factor for scaling weights
            # [type: real, advanced: TRUE, range: [1,1.79769313486232e+308], default: 1000]
            'separating/clique/scaleval': '1000',
            # maximal number of nodes in branch and bound tree (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 10000]
            'separating/clique/maxtreenodes': '10000',
            # frequency for premature backtracking up to tree level 1 (0: no backtracking)
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 1000]
            'separating/clique/backtrackfreq': '1000',
            # maximal number of clique cuts separated per separation round (-1: no limit)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 10]
            'separating/clique/maxsepacuts': '10',
            # maximal number of zero-valued variables extending the clique (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 1000]
            'separating/clique/maxzeroextensions': '1000',
            # maximal memory size of dense clique table (in kb)
            # [type: real, advanced: TRUE, range: [0,2097151.99902344], default: 20000]
            'separating/clique/cliquetablemem': '20000',
            # minimal density of cliques to use a dense clique table
            # [type: real, advanced: TRUE, range: [0,1], default: 0]
            'separating/clique/cliquedensity': '0',
            # priority of separator <closecuts>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 1000000]
            'separating/closecuts/priority': '1000000',
            # frequency for calling separator <closecuts> (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'separating/closecuts/freq': '-1',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying separator <closecuts> (0.0: only on current best node, 1.0: on all nodes)
            # [type: real, advanced: TRUE, range: [0,1], default: 1]
            'separating/closecuts/maxbounddist': '1',
            # should separator be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'separating/closecuts/delay': 'FALSE',
            # base for exponential increase of frequency at which separator <closecuts> is called (1: call at each multiple of frequency)
            # [type: int, advanced: TRUE, range: [1,100], default: 4]
            'separating/closecuts/expbackoff': '4',
            # generate close cuts w.r.t. relative interior point (best solution otherwise)?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'separating/closecuts/separelint': 'TRUE',
            # convex combination value for close cuts
            # [type: real, advanced: TRUE, range: [0,1], default: 0.3]
            'separating/closecuts/sepacombvalue': '0.3',
            # threshold on number of generated cuts below which the ordinary separation is started
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 50]
            'separating/closecuts/closethres': '50',
            # include an objective cutoff when computing the relative interior?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'separating/closecuts/inclobjcutoff': 'FALSE',
            # recompute relative interior point in each separation call?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'separating/closecuts/recomputerelint': 'FALSE',
            # turn off separation in current node after unsuccessful calls (-1 never turn off)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 0]
            'separating/closecuts/maxunsuccessful': '0',
            # factor for maximal LP iterations in relative interior computation compared to node LP iterations (negative for no limit)
            # [type: real, advanced: TRUE, range: [-1,1.79769313486232e+308], default: 10]
            'separating/closecuts/maxlpiterfactor': '10',
            # priority of separator <flowcover>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -100000]
            'separating/flowcover/priority': '-100000',
            # frequency for calling separator <flowcover> (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 10]
            'separating/flowcover/freq': '10',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying separator <flowcover> (0.0: only on current best node, 1.0: on all nodes)
            # [type: real, advanced: TRUE, range: [0,1], default: 0]
            'separating/flowcover/maxbounddist': '0',
            # should separator be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'separating/flowcover/delay': 'FALSE',
            # base for exponential increase of frequency at which separator <flowcover> is called (1: call at each multiple of frequency)
            # [type: int, advanced: TRUE, range: [1,100], default: 4]
            'separating/flowcover/expbackoff': '4',
            # priority of separator <cmir>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -100000]
            'separating/cmir/priority': '-100000',
            # frequency for calling separator <cmir> (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 10]
            'separating/cmir/freq': '10',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying separator <cmir> (0.0: only on current best node, 1.0: on all nodes)
            # [type: real, advanced: TRUE, range: [0,1], default: 0]
            'separating/cmir/maxbounddist': '0',
            # should separator be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'separating/cmir/delay': 'FALSE',
            # base for exponential increase of frequency at which separator <cmir> is called (1: call at each multiple of frequency)
            # [type: int, advanced: TRUE, range: [1,100], default: 4]
            'separating/cmir/expbackoff': '4',
            # priority of separator <aggregation>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -3000]
            'separating/aggregation/priority': '-3000',
            # frequency for calling separator <aggregation> (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 10]
            'separating/aggregation/freq': '10',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying separator <aggregation> (0.0: only on current best node, 1.0: on all nodes)
            # [type: real, advanced: TRUE, range: [0,1], default: 1]
            'separating/aggregation/maxbounddist': '1',
            # should separator be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'separating/aggregation/delay': 'FALSE',
            # base for exponential increase of frequency at which separator <aggregation> is called (1: call at each multiple of frequency)
            # [type: int, advanced: TRUE, range: [1,100], default: 4]
            'separating/aggregation/expbackoff': '4',
            # maximal number of cmir separation rounds per node (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'separating/aggregation/maxrounds': '-1',
            # maximal number of cmir separation rounds in the root node (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'separating/aggregation/maxroundsroot': '-1',
            # maximal number of rows to start aggregation with per separation round (-1: unlimited)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 200]
            'separating/aggregation/maxtries': '200',
            # maximal number of rows to start aggregation with per separation round in the root node (-1: unlimited)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'separating/aggregation/maxtriesroot': '-1',
            # maximal number of consecutive unsuccessful aggregation tries (-1: unlimited)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 20]
            'separating/aggregation/maxfails': '20',
            # maximal number of consecutive unsuccessful aggregation tries in the root node (-1: unlimited)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 100]
            'separating/aggregation/maxfailsroot': '100',
            # maximal number of aggregations for each row per separation round
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 3]
            'separating/aggregation/maxaggrs': '3',
            # maximal number of aggregations for each row per separation round in the root node
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 6]
            'separating/aggregation/maxaggrsroot': '6',
            # maximal number of cmir cuts separated per separation round
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 100]
            'separating/aggregation/maxsepacuts': '100',
            # maximal number of cmir cuts separated per separation round in the root node
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 500]
            'separating/aggregation/maxsepacutsroot': '500',
            # maximal slack of rows to be used in aggregation
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0]
            'separating/aggregation/maxslack': '0',
            # maximal slack of rows to be used in aggregation in the root node
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0.1]
            'separating/aggregation/maxslackroot': '0.1',
            # weight of row density in the aggregation scoring of the rows
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0.0001]
            'separating/aggregation/densityscore': '0.0001',
            # weight of slack in the aggregation scoring of the rows
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0.001]
            'separating/aggregation/slackscore': '0.001',
            # maximal density of aggregated row
            # [type: real, advanced: TRUE, range: [0,1], default: 0.2]
            'separating/aggregation/maxaggdensity': '0.2',
            # maximal density of row to be used in aggregation
            # [type: real, advanced: TRUE, range: [0,1], default: 0.05]
            'separating/aggregation/maxrowdensity': '0.05',
            # additional number of variables allowed in row on top of density
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 100]
            'separating/aggregation/densityoffset': '100',
            # maximal row aggregation factor
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 10000]
            'separating/aggregation/maxrowfac': '10000',
            # maximal number of different deltas to try (-1: unlimited)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'separating/aggregation/maxtestdelta': '-1',
            # tolerance for bound distances used to select continuous variable in current aggregated constraint to be eliminated
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0.01]
            'separating/aggregation/aggrtol': '0.01',
            # should negative values also be tested in scaling?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'separating/aggregation/trynegscaling': 'TRUE',
            # should an additional variable be complemented if f0': '0?',
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'separating/aggregation/fixintegralrhs': 'TRUE',
            # should generated cuts be removed from the LP if they are no longer tight?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'separating/aggregation/dynamiccuts': 'TRUE',
            # priority of separator <convexproj>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 0]
            'separating/convexproj/priority': '0',
            # frequency for calling separator <convexproj> (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'separating/convexproj/freq': '-1',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying separator <convexproj> (0.0: only on current best node, 1.0: on all nodes)
            # [type: real, advanced: TRUE, range: [0,1], default: 1]
            'separating/convexproj/maxbounddist': '1',
            # should separator be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'separating/convexproj/delay': 'TRUE',
            # base for exponential increase of frequency at which separator <convexproj> is called (1: call at each multiple of frequency)
            # [type: int, advanced: TRUE, range: [1,100], default: 4]
            'separating/convexproj/expbackoff': '4',
            # maximal depth at which the separator is applied (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'separating/convexproj/maxdepth': '-1',
            # iteration limit of NLP solver; 0 for no limit
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 250]
            'separating/convexproj/nlpiterlimit': '250',
            # time limit of NLP solver; 0.0 for no limit
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0]
            'separating/convexproj/nlptimelimit': '0',
            # priority of separator <disjunctive>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 10]
            'separating/disjunctive/priority': '10',
            # frequency for calling separator <disjunctive> (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 0]
            'separating/disjunctive/freq': '0',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying separator <disjunctive> (0.0: only on current best node, 1.0: on all nodes)
            # [type: real, advanced: TRUE, range: [0,1], default: 0]
            'separating/disjunctive/maxbounddist': '0',
            # should separator be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'separating/disjunctive/delay': 'TRUE',
            # base for exponential increase of frequency at which separator <disjunctive> is called (1: call at each multiple of frequency)
            # [type: int, advanced: TRUE, range: [1,100], default: 4]
            'separating/disjunctive/expbackoff': '4',
            # strengthen cut if integer variables are present.
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'separating/disjunctive/strengthen': 'TRUE',
            # node depth of separating bipartite disjunctive cuts (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'separating/disjunctive/maxdepth': '-1',
            # maximal number of separation rounds per iteration in a branching node (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 25]
            'separating/disjunctive/maxrounds': '25',
            # maximal number of separation rounds in the root node (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 100]
            'separating/disjunctive/maxroundsroot': '100',
            # maximal number of cuts investigated per iteration in a branching node
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 50]
            'separating/disjunctive/maxinvcuts': '50',
            # maximal number of cuts investigated per iteration in the root node
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 250]
            'separating/disjunctive/maxinvcutsroot': '250',
            # delay separation if number of conflict graph edges is larger than predefined value (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 100000]
            'separating/disjunctive/maxconfsdelay': '100000',
            # maximal rank of a disj. cut that could not be scaled to integral coefficients (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 20]
            'separating/disjunctive/maxrank': '20',
            # maximal rank of a disj. cut that could be scaled to integral coefficients (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'separating/disjunctive/maxrankintegral': '-1',
            # maximal valid range max(|weights|)/min(|weights|) of row weights
            # [type: real, advanced: TRUE, range: [1,1.79769313486232e+308], default: 1000]
            'separating/disjunctive/maxweightrange': '1000',
            # priority of separator <eccuts>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -13000]
            'separating/eccuts/priority': '-13000',
            # frequency for calling separator <eccuts> (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'separating/eccuts/freq': '-1',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying separator <eccuts> (0.0: only on current best node, 1.0: on all nodes)
            # [type: real, advanced: TRUE, range: [0,1], default: 1]
            'separating/eccuts/maxbounddist': '1',
            # should separator be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'separating/eccuts/delay': 'FALSE',
            # base for exponential increase of frequency at which separator <eccuts> is called (1: call at each multiple of frequency)
            # [type: int, advanced: TRUE, range: [1,100], default: 4]
            'separating/eccuts/expbackoff': '4',
            # should generated cuts be removed from the LP if they are no longer tight?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'separating/eccuts/dynamiccuts': 'TRUE',
            # maximal number of eccuts separation rounds per node (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 10]
            'separating/eccuts/maxrounds': '10',
            # maximal number of eccuts separation rounds in the root node (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 250]
            'separating/eccuts/maxroundsroot': '250',
            # maximal depth at which the separator is applied (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'separating/eccuts/maxdepth': '-1',
            # maximal number of edge-concave cuts separated per separation round
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 10]
            'separating/eccuts/maxsepacuts': '10',
            # maximal number of edge-concave cuts separated per separation round in the root node
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 50]
            'separating/eccuts/maxsepacutsroot': '50',
            # maximal coef. range of a cut (max coef. divided by min coef.) in order to be added to LP relaxation
            # [type: real, advanced: FALSE, range: [0,1e+20], default: 10000000]
            'separating/eccuts/cutmaxrange': '10000000',
            # minimal violation of an edge-concave cut to be separated
            # [type: real, advanced: FALSE, range: [0,0.5], default: 0.3]
            'separating/eccuts/minviolation': '0.3',
            # search for edge-concave aggregations of at least this size
            # [type: int, advanced: TRUE, range: [3,5], default: 3]
            'separating/eccuts/minaggrsize': '3',
            # search for edge-concave aggregations of at most this size
            # [type: int, advanced: TRUE, range: [3,5], default: 4]
            'separating/eccuts/maxaggrsize': '4',
            # maximum number of bilinear terms allowed to be in a quadratic constraint
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 500]
            'separating/eccuts/maxbilinterms': '500',
            # maximum number of unsuccessful rounds in the edge-concave aggregation search
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 5]
            'separating/eccuts/maxstallrounds': '5',
            # priority of separator <gauge>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 0]
            'separating/gauge/priority': '0',
            # frequency for calling separator <gauge> (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'separating/gauge/freq': '-1',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying separator <gauge> (0.0: only on current best node, 1.0: on all nodes)
            # [type: real, advanced: TRUE, range: [0,1], default: 1]
            'separating/gauge/maxbounddist': '1',
            # should separator be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'separating/gauge/delay': 'FALSE',
            # base for exponential increase of frequency at which separator <gauge> is called (1: call at each multiple of frequency)
            # [type: int, advanced: TRUE, range: [1,100], default: 4]
            'separating/gauge/expbackoff': '4',
            # iteration limit of NLP solver; 0 for no limit
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 1000]
            'separating/gauge/nlpiterlimit': '1000',
            # time limit of NLP solver; 0.0 for no limit
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0]
            'separating/gauge/nlptimelimit': '0',
            # priority of separator <gomory>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1000]
            'separating/gomory/priority': '-1000',
            # frequency for calling separator <gomory> (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 10]
            'separating/gomory/freq': '10',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying separator <gomory> (0.0: only on current best node, 1.0: on all nodes)
            # [type: real, advanced: TRUE, range: [0,1], default: 1]
            'separating/gomory/maxbounddist': '1',
            # should separator be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'separating/gomory/delay': 'FALSE',
            # base for exponential increase of frequency at which separator <gomory> is called (1: call at each multiple of frequency)
            # [type: int, advanced: TRUE, range: [1,100], default: 4]
            'separating/gomory/expbackoff': '4',
            # maximal number of gomory separation rounds per node (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 5]
            'separating/gomory/maxrounds': '5',
            # maximal number of gomory separation rounds in the root node (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 10]
            'separating/gomory/maxroundsroot': '10',
            # maximal number of gomory cuts separated per separation round
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 50]
            'separating/gomory/maxsepacuts': '50',
            # maximal number of gomory cuts separated per separation round in the root node
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 200]
            'separating/gomory/maxsepacutsroot': '200',
            # maximal rank of a gomory cut that could not be scaled to integral coefficients (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'separating/gomory/maxrank': '-1',
            # maximal rank of a gomory cut that could be scaled to integral coefficients (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: -1]
            'separating/gomory/maxrankintegral': '-1',
            # minimal integrality violation of a basis variable in order to try Gomory cut
            # [type: real, advanced: FALSE, range: [0.0001,0.5], default: 0.01]
            'separating/gomory/away': '0.01',
            # should generated cuts be removed from the LP if they are no longer tight?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'separating/gomory/dynamiccuts': 'TRUE',
            # try to scale cuts to integral coefficients
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'separating/gomory/makeintegral': 'FALSE',
            # if conversion to integral coefficients failed still consider the cut
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'separating/gomory/forcecuts': 'TRUE',
            # separate rows with integral slack
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'separating/gomory/separaterows': 'TRUE',
            # should cuts be added to the delayed cut pool?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'separating/gomory/delayedcuts': 'FALSE',
            # choose side types of row (lhs/rhs) based on basis information?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'separating/gomory/sidetypebasis': 'TRUE',
            # priority of separator <impliedbounds>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -50]
            'separating/impliedbounds/priority': '-50',
            # frequency for calling separator <impliedbounds> (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 10]
            'separating/impliedbounds/freq': '10',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying separator <impliedbounds> (0.0: only on current best node, 1.0: on all nodes)
            # [type: real, advanced: TRUE, range: [0,1], default: 1]
            'separating/impliedbounds/maxbounddist': '1',
            # should separator be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'separating/impliedbounds/delay': 'FALSE',
            # base for exponential increase of frequency at which separator <impliedbounds> is called (1: call at each multiple of frequency)
            # [type: int, advanced: TRUE, range: [1,100], default: 4]
            'separating/impliedbounds/expbackoff': '4',
            # should violated inequalities for cliques with 2 variables be separated?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'separating/impliedbounds/usetwosizecliques': 'TRUE',
            # priority of separator <intobj>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -100]
            'separating/intobj/priority': '-100',
            # frequency for calling separator <intobj> (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'separating/intobj/freq': '-1',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying separator <intobj> (0.0: only on current best node, 1.0: on all nodes)
            # [type: real, advanced: TRUE, range: [0,1], default: 0]
            'separating/intobj/maxbounddist': '0',
            # should separator be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'separating/intobj/delay': 'FALSE',
            # base for exponential increase of frequency at which separator <intobj> is called (1: call at each multiple of frequency)
            # [type: int, advanced: TRUE, range: [1,100], default: 4]
            'separating/intobj/expbackoff': '4',
            # priority of separator <mcf>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -10000]
            'separating/mcf/priority': '-10000',
            # frequency for calling separator <mcf> (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 0]
            'separating/mcf/freq': '0',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying separator <mcf> (0.0: only on current best node, 1.0: on all nodes)
            # [type: real, advanced: TRUE, range: [0,1], default: 0]
            'separating/mcf/maxbounddist': '0',
            # should separator be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'separating/mcf/delay': 'FALSE',
            # base for exponential increase of frequency at which separator <mcf> is called (1: call at each multiple of frequency)
            # [type: int, advanced: TRUE, range: [1,100], default: 4]
            'separating/mcf/expbackoff': '4',
            # number of clusters to generate in the shrunken network -- default separation
            # [type: int, advanced: TRUE, range: [2,32], default: 5]
            'separating/mcf/nclusters': '5',
            # maximal valid range max(|weights|)/min(|weights|) of row weights
            # [type: real, advanced: TRUE, range: [1,1.79769313486232e+308], default: 1000000]
            'separating/mcf/maxweightrange': '1000000',
            # maximal number of different deltas to try (-1: unlimited)  -- default separation
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 20]
            'separating/mcf/maxtestdelta': '20',
            # should negative values also be tested in scaling?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'separating/mcf/trynegscaling': 'FALSE',
            # should an additional variable be complemented if f0': '0?',
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'separating/mcf/fixintegralrhs': 'TRUE',
            # should generated cuts be removed from the LP if they are no longer tight?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'separating/mcf/dynamiccuts': 'TRUE',
            # model type of network (0: auto, 1:directed, 2:undirected)
            # [type: int, advanced: TRUE, range: [0,2], default: 0]
            'separating/mcf/modeltype': '0',
            # maximal number of mcf cuts separated per separation round
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 100]
            'separating/mcf/maxsepacuts': '100',
            # maximal number of mcf cuts separated per separation round in the root node  -- default separation
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 200]
            'separating/mcf/maxsepacutsroot': '200',
            # maximum inconsistency ratio for separation at all
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0.02]
            'separating/mcf/maxinconsistencyratio': '0.02',
            # maximum inconsistency ratio of arcs not to be deleted
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0.5]
            'separating/mcf/maxarcinconsistencyratio': '0.5',
            # should we separate only if the cuts shores are connected?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'separating/mcf/checkcutshoreconnectivity': 'TRUE',
            # should we separate inequalities based on single-node cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'separating/mcf/separatesinglenodecuts': 'TRUE',
            # should we separate flowcutset inequalities on the network cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'separating/mcf/separateflowcutset': 'TRUE',
            # should we separate knapsack cover inequalities on the network cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'separating/mcf/separateknapsack': 'TRUE',
            # priority of separator <oddcycle>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -15000]
            'separating/oddcycle/priority': '-15000',
            # frequency for calling separator <oddcycle> (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: -1]
            'separating/oddcycle/freq': '-1',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying separator <oddcycle> (0.0: only on current best node, 1.0: on all nodes)
            # [type: real, advanced: TRUE, range: [0,1], default: 1]
            'separating/oddcycle/maxbounddist': '1',
            # should separator be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'separating/oddcycle/delay': 'FALSE',
            # base for exponential increase of frequency at which separator <oddcycle> is called (1: call at each multiple of frequency)
            # [type: int, advanced: TRUE, range: [1,100], default: 4]
            'separating/oddcycle/expbackoff': '4',
            # Should the search method by Groetschel, Lovasz, Schrijver be used? Otherwise use levelgraph method by Hoffman, Padberg.
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'separating/oddcycle/usegls': 'TRUE',
            # Should odd cycle cuts be lifted?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'separating/oddcycle/liftoddcycles': 'FALSE',
            # maximal number of oddcycle cuts separated per separation round
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 5000]
            'separating/oddcycle/maxsepacuts': '5000',
            # maximal number of oddcycle cuts separated per separation round in the root node
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 5000]
            'separating/oddcycle/maxsepacutsroot': '5000',
            # maximal number of oddcycle separation rounds per node (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 10]
            'separating/oddcycle/maxrounds': '10',
            # maximal number of oddcycle separation rounds in the root node (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 10]
            'separating/oddcycle/maxroundsroot': '10',
            # factor for scaling of the arc-weights
            # [type: int, advanced: TRUE, range: [1,2147483647], default: 1000]
            'separating/oddcycle/scalingfactor': '1000',
            # add links between a variable and its negated
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'separating/oddcycle/addselfarcs': 'TRUE',
            # try to repair violated cycles with double appearance of a variable
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'separating/oddcycle/repaircycles': 'TRUE',
            # separate triangles found as 3-cycles or repaired larger cycles
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'separating/oddcycle/includetriangles': 'TRUE',
            # Even if a variable is already covered by a cut, still try it as start node for a cycle search?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'separating/oddcycle/multiplecuts': 'FALSE',
            # Even if a variable is already covered by a cut, still allow another cut to cover it too?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'separating/oddcycle/allowmultiplecuts': 'TRUE',
            # Choose lifting candidate by coef*lpvalue or only by coef?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'separating/oddcycle/lpliftcoef': 'FALSE',
            # Calculate lifting coefficient of every candidate in every step (or only if its chosen)?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'separating/oddcycle/recalcliftcoef': 'TRUE',
            # use sorted variable array (unsorted(0), maxlp(1), minlp(2), maxfrac(3), minfrac(4))
            # [type: int, advanced: TRUE, range: [0,4], default: 3]
            'separating/oddcycle/sortswitch': '3',
            # sort level of the root neighbors by fractionality (maxfrac)
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'separating/oddcycle/sortrootneighbors': 'TRUE',
            # percentage of variables to try the chosen method on [0-100]
            # [type: int, advanced: TRUE, range: [0,100], default: 0]
            'separating/oddcycle/percenttestvars': '0',
            # offset of variables to try the chosen method on (additional to the percentage of testvars)
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 100]
            'separating/oddcycle/offsettestvars': '100',
            # percentage of nodes allowed in the same level of the level graph [0-100]
            # [type: int, advanced: TRUE, range: [0,100], default: 100]
            'separating/oddcycle/maxpernodeslevel': '100',
            # offset of nodes allowed in the same level of the level graph (additional to the percentage of levelnodes)
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 10]
            'separating/oddcycle/offsetnodeslevel': '10',
            # maximal number of levels in level graph
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 20]
            'separating/oddcycle/maxnlevels': '20',
            # maximal number of oddcycle cuts generated per chosen variable as root of the level graph
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 1]
            'separating/oddcycle/maxcutsroot': '1',
            # maximal number of oddcycle cuts generated in every level of the level graph
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 50]
            'separating/oddcycle/maxcutslevel': '50',
            # minimal weight on an edge (in level graph or bipartite graph)
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 0]
            'separating/oddcycle/maxreference': '0',
            # number of unsuccessful calls at current node
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 3]
            'separating/oddcycle/maxunsucessfull': '3',
            # maximal number of other cuts s.t. separation is applied (-1 for direct call)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: -1]
            'separating/oddcycle/cutthreshold': '-1',
            # priority of separator <rapidlearning>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -1200000]
            'separating/rapidlearning/priority': '-1200000',
            # frequency for calling separator <rapidlearning> (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 5]
            'separating/rapidlearning/freq': '5',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying separator <rapidlearning> (0.0: only on current best node, 1.0: on all nodes)
            # [type: real, advanced: TRUE, range: [0,1], default: 1]
            'separating/rapidlearning/maxbounddist': '1',
            # should separator be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'separating/rapidlearning/delay': 'FALSE',
            # base for exponential increase of frequency at which separator <rapidlearning> is called (1: call at each multiple of frequency)
            # [type: int, advanced: TRUE, range: [1,100], default: 4]
            'separating/rapidlearning/expbackoff': '4',
            # should the found conflicts be applied in the original SCIP?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'separating/rapidlearning/applyconflicts': 'TRUE',
            # should the found global bound deductions be applied in the original SCIP?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'separating/rapidlearning/applybdchgs': 'TRUE',
            # should the inference values be used as initialization in the original SCIP?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'separating/rapidlearning/applyinfervals': 'TRUE',
            # should the inference values only be used when rapidlearning found other reductions?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'separating/rapidlearning/reducedinfer': 'FALSE',
            # should the incumbent solution be copied to the original SCIP?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'separating/rapidlearning/applyprimalsol': 'TRUE',
            # should a solved status be copied to the original SCIP?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'separating/rapidlearning/applysolved': 'TRUE',
            # should local LP degeneracy be checked?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'separating/rapidlearning/checkdegeneracy': 'TRUE',
            # should the progress on the dual bound be checked?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'separating/rapidlearning/checkdualbound': 'FALSE',
            # should the ratio of leaves proven to be infeasible and exceeding the cutoff bound be checked?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'separating/rapidlearning/checkleaves': 'FALSE',
            # check whether rapid learning should be executed
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'separating/rapidlearning/checkexec': 'TRUE',
            # should the (local) objective function be checked?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'separating/rapidlearning/checkobj': 'FALSE',
            # should the number of solutions found so far be checked?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'separating/rapidlearning/checknsols': 'TRUE',
            # should rapid learning be applied when there are continuous variables?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'separating/rapidlearning/contvars': 'FALSE',
            # maximal portion of continuous variables to apply rapid learning
            # [type: real, advanced: TRUE, range: [0,1], default: 0.3]
            'separating/rapidlearning/contvarsquot': '0.3',
            # maximal fraction of LP iterations compared to node LP iterations
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0.2]
            'separating/rapidlearning/lpiterquot': '0.2',
            # minimal degeneracy threshold to allow local rapid learning
            # [type: real, advanced: TRUE, range: [0,1], default: 0.7]
            'separating/rapidlearning/mindegeneracy': '0.7',
            # minimal threshold of inf/obj leaves to allow local rapid learning
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 10]
            'separating/rapidlearning/mininflpratio': '10',
            # minimal ratio of unfixed variables in relation to basis size to allow local rapid learning
            # [type: real, advanced: TRUE, range: [1,1.79769313486232e+308], default: 2]
            'separating/rapidlearning/minvarconsratio': '2',
            # maximum problem size (variables) for which rapid learning will be called
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 10000]
            'separating/rapidlearning/maxnvars': '10000',
            # maximum problem size (constraints) for which rapid learning will be called
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 10000]
            'separating/rapidlearning/maxnconss': '10000',
            # maximum number of overall calls
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 100]
            'separating/rapidlearning/maxcalls': '100',
            # maximum number of nodes considered in rapid learning run
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 5000]
            'separating/rapidlearning/maxnodes': '5000',
            # minimum number of nodes considered in rapid learning run
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 500]
            'separating/rapidlearning/minnodes': '500',
            # number of nodes that should be processed before rapid learning is executed locally based on the progress of the dualbound
            # [type: longint, advanced: TRUE, range: [0,9223372036854775807], default: 100]
            'separating/rapidlearning/nwaitingnodes': '100',
            # should all active cuts from cutpool be copied to constraints in subproblem?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: TRUE]
            'separating/rapidlearning/copycuts': 'TRUE',
            # priority of separator <strongcg>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -2000]
            'separating/strongcg/priority': '-2000',
            # frequency for calling separator <strongcg> (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 10]
            'separating/strongcg/freq': '10',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying separator <strongcg> (0.0: only on current best node, 1.0: on all nodes)
            # [type: real, advanced: TRUE, range: [0,1], default: 1]
            'separating/strongcg/maxbounddist': '1',
            # should separator be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'separating/strongcg/delay': 'FALSE',
            # base for exponential increase of frequency at which separator <strongcg> is called (1: call at each multiple of frequency)
            # [type: int, advanced: TRUE, range: [1,100], default: 4]
            'separating/strongcg/expbackoff': '4',
            # maximal number of strong CG separation rounds per node (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 5]
            'separating/strongcg/maxrounds': '5',
            # maximal number of strong CG separation rounds in the root node (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 20]
            'separating/strongcg/maxroundsroot': '20',
            # maximal number of strong CG cuts separated per separation round
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 20]
            'separating/strongcg/maxsepacuts': '20',
            # maximal number of strong CG cuts separated per separation round in the root node
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 500]
            'separating/strongcg/maxsepacutsroot': '500',
            # should generated cuts be removed from the LP if they are no longer tight?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'separating/strongcg/dynamiccuts': 'TRUE',
            # priority of separator <zerohalf>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: -6000]
            'separating/zerohalf/priority': '-6000',
            # frequency for calling separator <zerohalf> (-1: never, 0: only in root node)
            # [type: int, advanced: FALSE, range: [-1,65534], default: 10]
            'separating/zerohalf/freq': '10',
            # maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying separator <zerohalf> (0.0: only on current best node, 1.0: on all nodes)
            # [type: real, advanced: TRUE, range: [0,1], default: 1]
            'separating/zerohalf/maxbounddist': '1',
            # should separator be delayed, if other separators found cuts?
            # [type: bool, advanced: TRUE, range: {TRUE,FALSE}, default: FALSE]
            'separating/zerohalf/delay': 'FALSE',
            # base for exponential increase of frequency at which separator <zerohalf> is called (1: call at each multiple of frequency)
            # [type: int, advanced: TRUE, range: [1,100], default: 4]
            'separating/zerohalf/expbackoff': '4',
            # maximal number of zerohalf separation rounds per node (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 5]
            'separating/zerohalf/maxrounds': '5',
            # maximal number of zerohalf separation rounds in the root node (-1: unlimited)
            # [type: int, advanced: FALSE, range: [-1,2147483647], default: 20]
            'separating/zerohalf/maxroundsroot': '20',
            # maximal number of zerohalf cuts separated per separation round
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 20]
            'separating/zerohalf/maxsepacuts': '20',
            # initial seed used for random tie-breaking in cut selection
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 24301]
            'separating/zerohalf/initseed': '24301',
            # maximal number of zerohalf cuts separated per separation round in the root node
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 100]
            'separating/zerohalf/maxsepacutsroot': '100',
            # maximal number of zerohalf cuts considered per separation round
            # [type: int, advanced: FALSE, range: [0,2147483647], default: 2000]
            'separating/zerohalf/maxcutcands': '2000',
            # maximal slack of rows to be used in aggregation
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0]
            'separating/zerohalf/maxslack': '0',
            # maximal slack of rows to be used in aggregation in the root node
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0]
            'separating/zerohalf/maxslackroot': '0',
            # threshold for score of cut relative to best score to be considered good, so that less strict filtering is applied
            # [type: real, advanced: TRUE, range: [0,1], default: 1]
            'separating/zerohalf/goodscore': '1',
            # threshold for score of cut relative to best score to be discarded
            # [type: real, advanced: TRUE, range: [0,1], default: 0.5]
            'separating/zerohalf/badscore': '0.5',
            # weight of objective parallelism in cut score calculation
            # [type: real, advanced: TRUE, range: [0,1], default: 0]
            'separating/zerohalf/objparalweight': '0',
            # weight of efficacy in cut score calculation
            # [type: real, advanced: TRUE, range: [0,1], default: 1]
            'separating/zerohalf/efficacyweight': '1',
            # weight of directed cutoff distance in cut score calculation
            # [type: real, advanced: TRUE, range: [0,1], default: 0]
            'separating/zerohalf/dircutoffdistweight': '0',
            # maximum parallelism for good cuts
            # [type: real, advanced: TRUE, range: [0,1], default: 0.1]
            'separating/zerohalf/goodmaxparall': '0.1',
            # maximum parallelism for non-good cuts
            # [type: real, advanced: TRUE, range: [0,1], default: 0.1]
            'separating/zerohalf/maxparall': '0.1',
            # minimal violation to generate zerohalfcut for
            # [type: real, advanced: TRUE, range: [0,1.79769313486232e+308], default: 0.1]
            'separating/zerohalf/minviol': '0.1',
            # should generated cuts be removed from the LP if they are no longer tight?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'separating/zerohalf/dynamiccuts': 'TRUE',
            # maximal density of row to be used in aggregation
            # [type: real, advanced: TRUE, range: [0,1], default: 0.05]
            'separating/zerohalf/maxrowdensity': '0.05',
            # additional number of variables allowed in row on top of density
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 100]
            'separating/zerohalf/densityoffset': '100',
            # display activation status of display column <solfound> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/solfound/active': '1',
            # display activation status of display column <concsolfound> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/concsolfound/active': '1',
            # display activation status of display column <time> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/time/active': '1',
            # display activation status of display column <nnodes> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/nnodes/active': '1',
            # display activation status of display column <nodesleft> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/nodesleft/active': '1',
            # display activation status of display column <nobjleaves> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/nobjleaves/active': '1',
            # display activation status of display column <ninfeasleaves> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/ninfeasleaves/active': '1',
            # display activation status of display column <lpiterations> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/lpiterations/active': '1',
            # display activation status of display column <lpavgiterations> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/lpavgiterations/active': '1',
            # display activation status of display column <lpcond> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/lpcond/active': '1',
            # display activation status of display column <memused> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/memused/active': '1',
            # display activation status of display column <concmemused> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/concmemused/active': '1',
            # display activation status of display column <memtotal> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/memtotal/active': '1',
            # display activation status of display column <depth> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/depth/active': '1',
            # display activation status of display column <maxdepth> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/maxdepth/active': '1',
            # display activation status of display column <plungedepth> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/plungedepth/active': '1',
            # display activation status of display column <nfrac> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/nfrac/active': '1',
            # display activation status of display column <nexternbranchcands> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/nexternbranchcands/active': '1',
            # display activation status of display column <vars> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/vars/active': '1',
            # display activation status of display column <conss> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/conss/active': '1',
            # display activation status of display column <curconss> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/curconss/active': '1',
            # display activation status of display column <curcols> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/curcols/active': '1',
            # display activation status of display column <currows> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/currows/active': '1',
            # display activation status of display column <cuts> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/cuts/active': '1',
            # display activation status of display column <separounds> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/separounds/active': '1',
            # display activation status of display column <poolsize> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/poolsize/active': '1',
            # display activation status of display column <conflicts> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/conflicts/active': '1',
            # display activation status of display column <strongbranchs> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/strongbranchs/active': '1',
            # display activation status of display column <pseudoobj> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/pseudoobj/active': '1',
            # display activation status of display column <lpobj> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/lpobj/active': '1',
            # display activation status of display column <curdualbound> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/curdualbound/active': '1',
            # display activation status of display column <estimate> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/estimate/active': '1',
            # display activation status of display column <avgdualbound> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/avgdualbound/active': '1',
            # display activation status of display column <dualbound> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/dualbound/active': '1',
            # display activation status of display column <primalbound> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/primalbound/active': '1',
            # display activation status of display column <concdualbound> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/concdualbound/active': '1',
            # display activation status of display column <concprimalbound> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/concprimalbound/active': '1',
            # display activation status of display column <cutoffbound> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/cutoffbound/active': '1',
            # display activation status of display column <gap> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/gap/active': '1',
            # display activation status of display column <concgap> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/concgap/active': '1',
            # display activation status of display column <primalgap> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 0]
            'display/primalgap/active': '0',
            # display activation status of display column <nsols> (0: off, 1: auto, 2:on)
            # [type: int, advanced: FALSE, range: [0,2], default: 1]
            'display/nsols/active': '1',
            # is statistics table <status> active
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'table/status/active': 'TRUE',
            # is statistics table <timing> active
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'table/timing/active': 'TRUE',
            # is statistics table <origprob> active
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'table/origprob/active': 'TRUE',
            # is statistics table <presolvedprob> active
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'table/presolvedprob/active': 'TRUE',
            # is statistics table <presolver> active
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'table/presolver/active': 'TRUE',
            # is statistics table <constraint> active
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'table/constraint/active': 'TRUE',
            # is statistics table <constiming> active
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'table/constiming/active': 'TRUE',
            # is statistics table <propagator> active
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'table/propagator/active': 'TRUE',
            # is statistics table <conflict> active
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'table/conflict/active': 'TRUE',
            # is statistics table <separator> active
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'table/separator/active': 'TRUE',
            # is statistics table <pricer> active
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'table/pricer/active': 'TRUE',
            # is statistics table <branchrules> active
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'table/branchrules/active': 'TRUE',
            # is statistics table <heuristics> active
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'table/heuristics/active': 'TRUE',
            # is statistics table <compression> active
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'table/compression/active': 'TRUE',
            # is statistics table <benders> active
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'table/benders/active': 'TRUE',
            # is statistics table <lp> active
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'table/lp/active': 'TRUE',
            # is statistics table <nlp> active
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'table/nlp/active': 'TRUE',
            # is statistics table <relaxator> active
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'table/relaxator/active': 'TRUE',
            # is statistics table <tree> active
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'table/tree/active': 'TRUE',
            # is statistics table <root> active
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'table/root/active': 'TRUE',
            # is statistics table <solution> active
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'table/solution/active': 'TRUE',
            # is statistics table <concurrentsolver> active
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'table/concurrentsolver/active': 'TRUE',
            # soft time limit which should be applied after first solution was found (-1.0: disabled)
            # [type: real, advanced: FALSE, range: [-1,1.79769313486232e+308], default: -1]
            'limits/softtime': '-1',
            # the preferred number concurrent solvers of type <scip> with respect to the number of threads
            # [type: real, advanced: FALSE, range: [0,1], default: 1]
            'concurrent/scip/prefprio': '1',
            # the preferred number concurrent solvers of type <scip-default> with respect to the number of threads
            # [type: real, advanced: FALSE, range: [0,1], default: 0]
            'concurrent/scip-default/prefprio': '0',
            # the preferred number concurrent solvers of type <scip-cpsolver> with respect to the number of threads
            # [type: real, advanced: FALSE, range: [0,1], default: 0]
            'concurrent/scip-cpsolver/prefprio': '0',
            # the preferred number concurrent solvers of type <scip-easycip> with respect to the number of threads
            # [type: real, advanced: FALSE, range: [0,1], default: 0]
            'concurrent/scip-easycip/prefprio': '0',
            # the preferred number concurrent solvers of type <scip-feas> with respect to the number of threads
            # [type: real, advanced: FALSE, range: [0,1], default: 0]
            'concurrent/scip-feas/prefprio': '0',
            # the preferred number concurrent solvers of type <scip-hardlp> with respect to the number of threads
            # [type: real, advanced: FALSE, range: [0,1], default: 0]
            'concurrent/scip-hardlp/prefprio': '0',
            # the preferred number concurrent solvers of type <scip-opti> with respect to the number of threads
            # [type: real, advanced: FALSE, range: [0,1], default: 0]
            'concurrent/scip-opti/prefprio': '0',
            # the preferred number concurrent solvers of type <scip-counter> with respect to the number of threads
            # [type: real, advanced: FALSE, range: [0,1], default: 0]
            'concurrent/scip-counter/prefprio': '0',
            # priority of Benders' decomposition <default>
            # [type: int, advanced: FALSE, range: [-536870912,536870911], default: 0]
            'benders/default/priority': '0',
            # should Benders' cuts be generated for LP solutions?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'benders/default/cutlp': 'TRUE',
            # should Benders' cuts be generated for pseudo solutions?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'benders/default/cutpseudo': 'TRUE',
            # should Benders' cuts be generated for relaxation solutions?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'benders/default/cutrelax': 'TRUE',
            # should Benders' cuts from LNS heuristics be transferred to the main SCIP instance?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'benders/default/transfercuts': 'FALSE',
            # should Benders' decomposition be used in LNS heurisics?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'benders/default/lnscheck': 'TRUE',
            # maximum depth at which the LNS check is performed (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,65534], default: -1]
            'benders/default/lnsmaxdepth': '-1',
            # the maximum number of Benders' decomposition calls in LNS heuristics (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 10]
            'benders/default/lnsmaxcalls': '10',
            # the maximum number of root node Benders' decomposition calls in LNS heuristics (-1: no limit)
            # [type: int, advanced: TRUE, range: [-1,2147483647], default: 0]
            'benders/default/lnsmaxcallsroot': '0',
            # should the transferred cuts be added as constraints?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'benders/default/cutsasconss': 'TRUE',
            # fraction of subproblems that are solved in each iteration
            # [type: real, advanced: FALSE, range: [0,1], default: 1]
            'benders/default/subprobfrac': '1',
            # should the auxiliary variable bound be updated by solving the subproblem?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'benders/default/updateauxvarbound': 'FALSE',
            # if the subproblem objective is integer, then define the auxiliary variables as implied integers?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'benders/default/auxvarsimplint': 'FALSE',
            # should Benders' cuts be generated while checking solutions?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'benders/default/cutcheck': 'TRUE',
            # the convex combination multiplier for the cut strengthening
            # [type: real, advanced: FALSE, range: [0,1], default: 0.5]
            'benders/default/cutstrengthenmult': '0.5',
            # the maximum number of cut strengthening without improvement
            # [type: int, advanced: TRUE, range: [0,2147483647], default: 5]
            'benders/default/noimprovelimit': '5',
            # the constant use to perturb the cut strengthening core point
            # [type: real, advanced: FALSE, range: [0,1], default: 1e-06]
            'benders/default/corepointperturb': '1e-06',
            # should the core point cut strengthening be employed (only applied to fractional solutions or continuous subproblems)?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'benders/default/cutstrengthenenabled': 'FALSE',
            # where should the strengthening interior point be sourced from ('l'p relaxation, 'f'irst solution, 'i'ncumbent solution, 'r'elative interior point, vector of 'o'nes, vector of 'z'eros)
            # [type: char, advanced: FALSE, range: {lfiroz}, default: r]
            'benders/default/cutstrengthenintpoint': 'r',
            # the number of threads to use when solving the subproblems
            # [type: int, advanced: TRUE, range: [1,2147483647], default: 1]
            'benders/default/numthreads': '1',
            # should a feasibility phase be executed during the root node, i.e. adding slack variables to constraints to ensure feasibility
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'benders/default/execfeasphase': 'FALSE',
            # the objective coefficient of the slack variables in the subproblem
            # [type: real, advanced: FALSE, range: [0,1e+20], default: 1000000]
            'benders/default/slackvarcoef': '1000000',
            # should the constraints of the subproblems be checked for convexity?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'benders/default/checkconsconvexity': 'TRUE',
            # priority of Benders' cut <feas>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 10000]
            'benders/default/benderscut/feas/priority': '10000',
            # is this Benders' decomposition cut method used to generate cuts?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'benders/default/benderscut/feas/enabled': 'TRUE',
            # priority of Benders' cut <feasalt>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 10001]
            'benders/default/benderscut/feasalt/priority': '10001',
            # is this Benders' decomposition cut method used to generate cuts?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'benders/default/benderscut/feasalt/enabled': 'TRUE',
            # priority of Benders' cut <integer>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 0]
            'benders/default/benderscut/integer/priority': '0',
            # is this Benders' decomposition cut method used to generate cuts?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'benders/default/benderscut/integer/enabled': 'TRUE',
            # the constant term of the integer Benders' cuts.
            # [type: real, advanced: FALSE, range: [-1e+20,1e+20], default: -10000]
            'benders/default/benderscut/integer/cutsconstant': '-10000',
            # should cuts be generated and added to the cutpool instead of global constraints directly added to the problem.
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'benders/default/benderscut/integer/addcuts': 'FALSE',
            # priority of Benders' cut <nogood>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 500]
            'benders/default/benderscut/nogood/priority': '500',
            # is this Benders' decomposition cut method used to generate cuts?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'benders/default/benderscut/nogood/enabled': 'TRUE',
            # should cuts be generated and added to the cutpool instead of global constraints directly added to the problem.
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'benders/default/benderscut/nogood/addcuts': 'FALSE',
            # priority of Benders' cut <optimality>
            # [type: int, advanced: TRUE, range: [-536870912,536870911], default: 5000]
            'benders/default/benderscut/optimality/priority': '5000',
            # is this Benders' decomposition cut method used to generate cuts?
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: TRUE]
            'benders/default/benderscut/optimality/enabled': 'TRUE',
            # should cuts be generated and added to the cutpool instead of global constraints directly added to the problem.
            # [type: bool, advanced: FALSE, range: {TRUE,FALSE}, default: FALSE]
            'benders/default/benderscut/optimality/addcuts': 'FALSE',
            # priority of NLPI <ipopt>
            # [type: int, advanced: FALSE, range: [-536870912,536870911], default: 1000]
            'nlpi/ipopt/priority': '1000',
        }

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

        cmd.extend(['-s', self._options_file])
        # cmd.extend(['-c', 'set heuristics emphasis aggressive'])
        # cmd.extend(['-c', 'set presolving emphasis aggressive'])
        # cmd.extend(['-c', 'set separation emphasis aggressive'])
        cmd.extend(['-c', f'read {problem_files[0]}'])
        cmd.extend(['-c', 'optimize'])
        cmd.extend(['-c', f'write solution {self._soln_file}'])
        # cmd.extend(['-c', 'display statistics'])
        cmd.extend(['-c', 'quit'])

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

        #TODO: properly read the output file
        # with open(self._log_file, 'r') as output:
        return results

    def process_soln_file(self, results):
        results.problem.name = 'unknown'
        soln = results.solution.add()

        # see https://www.scipopt.org/doc/html/reader__sol_8h.php
        with open(self._soln_file, 'r') as f:
            line = next(f).strip()
            if line.startswith('solution status:'):
                solution_status = line[16:].strip()
                if solution_status == 'optimal solution found':
                    results.solver.termination_condition = TerminationCondition.optimal
                elif solution_status == 'infeasible':
                    results.solver.termination_condition = TerminationCondition.infeasible
                else:
                    logger.warning(f'Unexpected solution status: {solution_status}')
            else:
                logger.warning(f'Unexpected line in solution: {line}')
                
            line = next(f).strip()
            if line.startswith('objective value:'):
                objective_value = line[16:].strip()
                obj_val = float(objective_value)
                results.problem.lower_bound = obj_val
                results.problem.upper_bound = obj_val
                soln.objective['__default_objective__'] = {'Value': obj_val}
            else:
                logger.warning(f'Unexpected line in solution: {line}')

            p = re.compile(r'(\S*)\s*(\S*)\s*\(obj:(\S*)\)')
            for line in f:
                m = p.match(line)
                if m is not None:
                    name, val, obj = m.groups()
                    if name != 'ONE_VAR_CONSTANT':
                        soln.variable[name] = {"Value": float(val)}
                else:
                    logger.error(f'Unexpected line "{line}" while parsing {self._soln_file}')
