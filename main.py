import os
import sys
from datetime import datetime

import logging
import pickle
import gzip
import shutil

import numpy as np
from mipsym.mip import Norm
from mipsym.tools import to_matrix, matshow

from automorphism_finder import find_automorphisms
from permutation_group_utils import find_simple_generators


# Set the parameters of the calculation
# Choose which testcase to run. This testcase corresponds to the world with dimensions 15x15, where letters with
# horizontal or vertical axial symmetries have been removed.
world_name = "two_letter_words_20x10"
# This parameter corresponds to the percentage of unique observations present for the calculation of the concurrence matrix.
percentage = "75.0"
trafo_round_decimals = None
# The fault tolerance ratio corresponds to the maximum ratio of nodes for which we accept that no mapping exists.
trafo_fault_tolerance_ratio = 0.15
# The bandwidth is the parameter of the kernel density estimation, controlling the number and width of the bins.
kde_bandwidth = 3.3e-4
use_integer_programming = True
# With this parameter, you can mute some terminal outputs.
quiet = False
# Select the norm which will be used to measure the quality / the "error" of the found permutations.
# Depending on whether the calculated norm is bigger or smaller than the error_value_limit below, we decide whether
# the permutation will be included in the permutation group or not.
norm = Norm.L_INFINITY
error_value_limit = 0.01
# Choose whether to create a log file.
log_to_file = False


if log_to_file:
    log_filename = f'logs/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'
    os.makedirs("logs", exist_ok=True)
    handlers = [
        logging.FileHandler(log_filename, "w", "utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
else:
    handlers = [logging.StreamHandler(sys.stdout)]

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=handlers,
)
logger = logging.getLogger("pyAGA_presolving")
logger.setLevel(logging.DEBUG)

mat_filename = f"data/{world_name}_concurrence_matrix_{percentage}.pickle"
with open(mat_filename, "rb") as correlation_matrix_file:
    logging.info(f"Loading matrix {mat_filename}")
    correlation_matrix = np.transpose(pickle.load(correlation_matrix_file))
    # automorphisms = find_automorphisms(correlation_matrix, trafo_accuracy)
    num_variables = correlation_matrix.shape[0]
    trafos, num_MIP_calls = find_automorphisms(
        correlation_matrix,
        fault_tolerance=int(trafo_fault_tolerance_ratio * num_variables),
        round_decimals=trafo_round_decimals,
        quiet=quiet,
        bandwidth=kde_bandwidth,
        norm=norm,
        error_value_limit=error_value_limit,
        use_integer_programming=use_integer_programming,
    )

if not quiet:
    logger.info(f"Total number of found trafos {len(trafos)}")
    # for i, trafo in enumerate(automorphisms):
    #    matrix = to_matrix(trafo)
    #    logger.info(f'Printing permutation number {i+1}')
    #    logger.info('\n' + matshow(matrix))
    logging.debug(
        "Trying to compute a small/minimal generating set for the found transformations..."
    )

simple_generators, permutation_group = find_simple_generators(trafos)

if not quiet:
    logger.info(f"Found generating set with {len(simple_generators)} members:")
    for i, gen in enumerate(simple_generators):
        logger.info(f"G_{i} =\n{matshow(to_matrix(gen.array_form))}")

    logger.info(f"Order of permutation group: {permutation_group.order()}")

if log_to_file:
    # Compress log file, and remove uncompressed original afterwards.
    with open(log_filename, "rb") as f_in:
        with open(f"{log_filename}.gz", "wb") as f_out:
            with gzip.GzipFile("file.txt", "wb", fileobj=f_out) as gzfile:
                shutil.copyfileobj(f_in, gzfile)
    logging.shutdown()
    os.remove(log_filename)
