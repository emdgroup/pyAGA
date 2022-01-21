import os
import sys
from datetime import datetime

import local_import_paths
local_import_paths.import_paths()

import logging
import pickle
import gzip
import shutil

import numpy as np
from mipsym.mip import Norm
from mipsym.tools import to_matrix, matshow

from transformation_finder import find_trafos
from permutation_group_utils import find_simple_generators


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
error_value_limit = 0.005
log_to_file = False


if log_to_file:
    log_filename = f'logs/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'
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
    trafos, average_matchrate_per_trafo, num_MIP_calls = find_trafos(
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
    # trafos, average_matchrate_per_trafo, num_MIP_calls = find_trafos(
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
    #for i, trafo in enumerate(trafos):
    #    matrix = to_matrix(trafo)
    #    logger.info(f'Printing permutation number {i+1}')
    #    logger.info('\n' + matshow(matrix))

if not quiet:
    logging.debug('Trying to compute a small/minimal generating set for the found transformations...')

simple_generators, permutation_group = find_simple_generators(trafos)

if not quiet:
    logger.info(f'Found generating set with {len(simple_generators)} members:')
    for i, gen in enumerate(simple_generators):
        logger.info(f'G_{i} =\n{matshow(to_matrix(gen.array_form))}')

    logger.info(f'Order of permutation group: {permutation_group.order()}')

if log_to_file:
    # Compress log file, and remove uncompressed original afterwards.
    with open(log_filename, 'rb') as f_in:
        with open(f'{log_filename}.gz', 'wb') as f_out:
            with gzip.GzipFile('file.txt', 'wb', fileobj=f_out) as gzfile:
                shutil.copyfileobj(f_in, gzfile)
    logging.shutdown()
    os.remove(log_filename)
