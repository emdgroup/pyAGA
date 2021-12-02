import logging
import math
import sys
import threading
import time

import numpy as np
import pandas as pd

import local_import_paths
local_import_paths.import_paths()
from transformation_finder import find_trafos
from mipsym.mip import Norm
from mipsym.tools import to_matrix, matshow, deviation_value
import pickle

logger = logging.getLogger("trafofinder_presolving")

world_name = "two_letter_words_20x10"
# world_name = "one_letter_words_10x5"
integer_matrices = False
trafo_round_decimals = 4
use_integer_programming = False
quiet = True
norm = Norm.L_INFINITY
error_value_limit = 0.006
time_per_iteration = 1000      # Maximum time a given iteration can process until it is
                               # terminated

handlers = [logging.StreamHandler(sys.stdout)]
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("trafofinder_presolving")
logger.setLevel(logging.DEBUG)


## Parameters of parameter study
percentages = ["100", "99.9", "98.0", "95.0"]
kde_bandwidths = np.arange(1e-3, 2e-3, 2e-4)
trafo_fault_tolerance_ratios = np.arange(0.2, 0.3, 0.05)
##

# short test parameters
# percentages = ["100"]
# kde_bandwidths = [1e-3]
# trafo_fault_tolerance_ratios = [0.2]


def find_trafos_wrapper(
    correlation_matrix,
    fault_tolerance,
    trafo_round_decimals,
    quiet,
    kde_bandwidth,
    world_name,
    norm,
    use_integer_programming,
    result,
    stop_thread,
):
    trafos, average_matchrate_per_trafo = find_trafos(
        correlation_matrix,
        fault_tolerance=fault_tolerance,
        round_decimals=trafo_round_decimals,
        quiet=quiet,
        bandwidth=kde_bandwidth,
        casename=world_name,
        norm=norm,
        use_integer_programming=use_integer_programming,
        stop_thread=stop_thread,
    )
    result[0] = trafos
    result[1] = average_matchrate_per_trafo


parameter_study_results = []
for percentage in percentages:
    if integer_matrices:
        mat_filename = f"data/{world_name}_integers_concurrence_matrix_" \
                       f"{percentage}.pickle"
    else:
        mat_filename = f"data/{world_name}_concurrence_matrix_{percentage}.pickle"
    with open(mat_filename, "rb") as correlation_matrix_file:
        logger.info(f"Loading matrix {mat_filename}")
        correlation_matrix = np.transpose(pickle.load(correlation_matrix_file))
        num_variables = correlation_matrix.shape[0]

        for kde_bandwidth in kde_bandwidths:
            for trafo_fault_tolerance_ratio in trafo_fault_tolerance_ratios:

                results = [None] * 2
                stop_thread = False
                thread = threading.Thread(
                    target=find_trafos_wrapper,
                    args=(
                        correlation_matrix,
                        int(trafo_fault_tolerance_ratio * num_variables),
                        trafo_round_decimals,
                        quiet,
                        kde_bandwidth,
                        world_name,
                        norm,
                        use_integer_programming,
                        results,
                        lambda: stop_thread,
                    ),
                )
                time_start = time.time()
                thread.start()
                # Wait until the timeout value has been reached (timeout value is given
                # in seconds).
                while thread.is_alive():
                    if time.time() - time_start < time_per_iteration:
                        time.sleep(1)
                    else:
                        break
                thread.join(timeout=1)

                if thread.is_alive():
                    logger.warning(f"Calculation of trafos timed out.")
                    parameters = (
                        percentage,
                        kde_bandwidth,
                        trafo_fault_tolerance_ratio,
                        "Timeout",
                        "Timeout",
                        round(time.time() - time_start, 2),
                    )
                    stop_thread = True  # Stop thread via lambda callback.
                    while thread.is_alive():
                        time.sleep(0.5)
                else:
                    trafos, average_matchrate_per_trafo = results
                    parameters = (
                        percentage,
                        kde_bandwidth,
                        trafo_fault_tolerance_ratio,
                        len(trafos),
                        average_matchrate_per_trafo,
                        round(time.time() - time_start, 2),
                    )
                    logger.info(
                        f"percentage_observations = {percentage}  ,",
                        f"kde_bandwidth = {round(kde_bandwidth, 12)},  "
                        f"trafo_fault_tolerance_ratio = {trafo_fault_tolerance_ratio},  ",
                        f"num_found_trafos = {len(trafos)},  ",
                        f"average_matchrate_per_trafo = {average_matchrate_per_trafo},  ",
                    )
                parameter_study_results.append(parameters)
results_dataframe = pd.DataFrame(
    parameter_study_results,
    columns=[
        "percentage_observation",
        "kde_bandwidth",
        "trafo_fault_tolerance_ratio",
        "num_found_trafos",
        "average_matchrate_per_trafo",
        "time for calculation"
    ],
)
results_dataframe.to_excel(f"{world_name}_results.xlsx", engine="xlsxwriter")
