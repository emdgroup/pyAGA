import pickle
import logging

from mipsym.mip import find_permutations, Norm, Solver

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('integer_programming')
logger.setLevel(level=logging.INFO)  # set to logging.INFO for less, to logging.DEBUG for more verbosity


if __name__ == '__main__':
    filename = 'data/one_letter_words_5x5_integers_concurrence_matrix_100.pickle'
    logger.info(f'Loading file {filename}')
    with open(filename, 'rb') as f:
        A = pickle.load(f)

    find_permutations(
        A=A,
        norm=Norm.L_1,
        solver=Solver.SCIP,
        objective_bound=0.01,
        time_limit=None,
        known_entries=None
    )