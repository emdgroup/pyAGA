import os
import sys
from datetime import datetime

import local_import_paths
local_import_paths.import_paths()

import logging
import pickle
from itertools import count
import gzip
import shutil

import numpy as np
from matplotlib import pyplot as plt
from mipsym.mip import Norm
from mipsym.tools import to_matrix, matshow, deviation_value
from sympy.combinatorics import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup

from transformation_finder import find_trafos


# world_name = "two_letter_words_20x10"
world_name = "one_letter_words_10x5"
percentage = "98.0"
# percentage = "95.0"
integer_matrices = False
trafo_round_decimals = None
trafo_fault_tolerance_ratio = 0.25
kde_bandwidth = 1e-3
use_integer_programming = True
quiet = False
norm = Norm.L_INFINITY
error_value_limit = 0.01
log_to_file = False


log_filename = f'logs/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'
if log_to_file:
    os.makedirs("logs", exist_ok=True)
    handlers = [
        logging.FileHandler(log_filename, "w", "utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
else:
    handlers = [logging.StreamHandler(sys.stdout)]

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=handlers,
)
logger = logging.getLogger("trafofinder_presolving")
logger.setLevel(logging.DEBUG)


if integer_matrices:
    mat_filename = f"data/{world_name}_integers_concurrence_matrix_{percentage}.pickle"
else:
    mat_filename = f"data/{world_name}_concurrence_matrix_{percentage}.pickle"


with open(mat_filename, "rb") as correlation_matrix_file:
    logging.info(f"Loading matrix {mat_filename}")
    correlation_matrix = np.transpose(pickle.load(correlation_matrix_file))
    # trafos = find_trafos(correlation_matrix, trafo_accuracy)
    num_variables = correlation_matrix.shape[0]
    trafos, average_matchrate_per_trafo = find_trafos(
        correlation_matrix,
        fault_tolerance=int(
            trafo_fault_tolerance_ratio
            * num_variables),
        round_decimals=trafo_round_decimals,
        quiet=quiet,
        bandwidth=kde_bandwidth,
        casename=world_name,
        norm=norm,
        error_value_limit=error_value_limit,
        use_integer_programming=use_integer_programming
    )


    # Experimental
    # trafos, average_matchrate_per_trafo = find_trafos(
    #    correlation_matrix,
    #    fault_tolerance=int(
    #        trafo_fault_tolerance_ratio
    #        * num_variables),
    #    round_decimals=trafo_round_decimals,
    #    quiet=quiet,
    #    bandwidth=kde_bandwidth,
    #    casename=world_name, norm=norm,
    #    use_integer_programming=use_integer_programming,
    #    false_map_resistance_per_node=3
    # )

if not quiet:
    logger.info(f"Total number of found trafos {len(trafos)}")
    for i, trafo in enumerate(trafos):
        matrix = to_matrix(trafo)
        logger.info(f'Printing permutation number {i+1}')
        logger.debug(matshow(matrix))

# We try to find a small/minimal generating set for all valid-ish transformations as follows:
# For all found transformations p_i, we compute all powers p_i^k and see if |p_i^k A - A p_i^k|
# is larger than a pre-set error bound. If yes, we omit p, otherwise, we build a permutation
# group with all transformations found so far and see if p is already in there.
# If so, we skip p, otherwise we add it to the permutation group.
if not quiet:
    logging.info('Trying to compute a small/minimal generating set for the found transformations...')

id = np.eye(len(correlation_matrix))
permutation_group_generators = []
# during the filtering below, we will take note of the values of |p_i^k A - A p_i^k| to
# plot a histogram in the end, which allows for more convenient tuning of the error_value_limit
deviation_values = []

for trafo in trafos:
    generator = to_matrix(trafo)
    current_power = generator
    is_valid = True
    for power in count(0):
        if np.allclose(current_power, id):
            break  # went through the full cycle of generator

        deviation = deviation_value(norm, current_power, correlation_matrix)
        deviation_values.append(deviation)
        if deviation > error_value_limit:
            logging.info(
                f'Skipping a transformation due to deviation'
                f' {deviation} > {error_value_limit}' 
                f' for power {power}:\n{matshow(generator)}'
            )
            is_valid = False
            break

        current_power = current_power @ generator

    if is_valid:
        g = PermutationGroup(*permutation_group_generators)
        p = Permutation(trafo)
        if p not in g:
            permutation_group_generators.append(p)

if not quiet:
    logger.info(f'Found generating set with {len(permutation_group_generators)} '
               f'members:')
    for i, gen in enumerate(permutation_group_generators):
        logger.info(f'\nG_{i} =')
        logger.debug("\n" + matshow(to_matrix(gen.array_form)))

    plt.hist(deviation_values, bins=len(deviation_values))
    plt.title('Histogram of |p_i^k A - A p_i^k| for all identified permutations')
    plt.axvline(x=error_value_limit, color='red')
    plt.show()

if log_to_file:
    # Compress log file, and remove uncompressed original afterwards.
    with open(log_filename, 'rb') as f_in:
        with open(f'{log_filename}.gz', 'wb') as f_out:
            with gzip.GzipFile('file.txt', 'wb', fileobj=f_out) as gzfile:
                shutil.copyfileobj(f_in, gzfile)
    logging.shutdown()
    os.remove(log_filename)