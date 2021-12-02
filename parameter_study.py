import ast
import configparser
import logging
import platform

import sys
import threading
import time
from itertools import count
from typing import List, Callable, Tuple

import numpy as np
import pandas as pd
from sympy.combinatorics import PermutationGroup, Permutation

import local_import_paths

local_import_paths.import_paths()
from transformation_finder import find_trafos
from mipsym.mip import Norm
from mipsym.tools import to_matrix, matshow, deviation_value
import pickle

logger = logging.getLogger("trafofinder_presolving")



def run_parameter_study(
        parameters,
        global_timeout: Callable[[None], bool],
        parameter_study_results: List
) -> None:
    """The main function within this module, called from the if __name__ == "__main__"
    statement. After running through or after getting told to exit via global_timeout,
    it writes the previously computed values to an excel table and exits.
    :param parameters:
    :param parameter_study_results:
    :param global_timeout: This function passes along a lambda callback to tell this
    thread to terminate.
    """
    # world_name = "two_letter_words_20x10"
    world_name = parameters["world_name"]
    integer_matrices = parameters["integer_matrices"]
    trafo_round_decimals = parameters["trafo_round_decimals"]
    use_integer_programming = parameters["use_integer_programming"]
    quiet = parameters["quiet"]
    norm = parameters["norm"]
    error_value_limit = parameters["error_value_limit"]
    time_per_iteration = parameters["time_per_iteration"]

    percentages = parameters["percentages"]
    kde_bandwidths = parameters["kde_bandwidths"]
    fault_tolerance_ratios = parameters["fault_tolerance_ratios"]

    for percentage in percentages:
        if integer_matrices:
            mat_filename = (
                f"data/{world_name}_integers_concurrence_matrix_" f"{percentage}.pickle"
            )
        else:
            mat_filename = f"data/{world_name}_concurrence_matrix_{percentage}.pickle"
        with open(mat_filename, "rb") as correlation_matrix_file:
            logger.info(f"Loading matrix {mat_filename}")
            correlation_matrix = np.transpose(pickle.load(correlation_matrix_file))
            num_variables = correlation_matrix.shape[0]
            try_bandwidths_and_tolerance_ratios(
                percentage,
                kde_bandwidths,
                fault_tolerance_ratios,
                correlation_matrix,
                num_variables,
                trafo_round_decimals,
                quiet,
                world_name,
                norm,
                use_integer_programming,
                error_value_limit,
                parameter_study_results,
                time_per_iteration,
                global_timeout,
            )
            if global_timeout():
                break



def find_trafos_wrapper(
    correlation_matrix: np.ndarray,
    fault_tolerance: int,
    trafo_round_decimals: int,
    quiet: bool,
    kde_bandwidth: float,
    world_name: str,
    norm: Norm,
    use_integer_programming: bool,
    result: Tuple[List[List[int]], List[float]],
    stop_thread: Callable[[None], bool],
) -> None:
    """
    This function is a wrapper around the usual find_trafos s.t. the results are written
    to the out-parameter "results" instead of being returned.
    :param correlation_matrix: The adjacency matrix of the graph whose symmetries we
    want to find.
    :param fault_tolerance: The number of tolerated unmappable nodes.
    :param trafo_round_decimals: The number of positions which will be left after rounding
    the adjacency matrix values.
    :param quiet: A parameter to limit the number of console and log-outputs.
    :param kde_bandwidth: The bandwidth parameter for the kernel density estimation and
    subsequent bin calculation.
    :param world_name: The name of the testcase.
    :param norm: The norm to use for the integer program.
    :param use_integer_programming: Whether or not to use the integer programming
    routines to fill out partial transformations.
    :param result: A tuple containing the found transformations as its first entry,
    and the average matchrate over the found transformations as its second.
    Matchrates are the ratios of correctly mapped nodes to unmappable ones.
    :param stop_thread: This function passes along a lambda callback to tell this
    thread to terminate.
    """
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


def try_bandwidths_and_tolerance_ratios(
    current_percentage: str,
    kde_bandwidths: np.ndarray,
    fault_tolerance_ratios: np.ndarray,
    adjacency_matrix: np.ndarray,
    num_variables: int,
    trafo_round_decimals: int,
    quiet: bool,
    world_name: str,
    norm: Norm,
    use_integer_programming: bool,
    error_value_limit: float,
    parameter_study_results: List[Tuple],
    time_per_iteration: int,
    global_timeout: Callable[[None], bool],
) -> None:
    """This function iterates over all given kernel denstity estimation bandwidths
    and fault tolerance ratios, trying to calculate the number of transformations
    until it encounters a timeout. When it does, it cancels the calculations for all
    other bandwidths, as the next runs will be with bigger bandwidths, leading to bigger
    bins and are therefore sure to also time out.

    :param error_value_limit:
    :param current_percentage:
    :param kde_bandwidths: An sorted iterable of bandwidths.
    :param fault_tolerance_ratios: An iterable of fault_tolerance_ratios.
    :param adjacency_matrix: The adjacency matrix of the graph whose symmetries we
    want to find.
    :param num_variables:
    :param quiet: A parameter to limit the number of console and log-outputs.
    :param norm: The norm to use for the integer program.
    :param world_name: The name of the testcase.
    :param trafo_round_decimals:
    :param use_integer_programming:
    :param parameter_study_results: A list of tuples containing the result of the
    excel sheet. The individual tuples correspond to rows within the excel sheet.
    :param time_per_iteration: The time each individual transformation-finding run (
    including the integer program, if it is enabled)
    :param global_timeout:
    :return:
    """
    timed_out = False
    for kde_bandwidth in kde_bandwidths:
        for trafo_fault_tolerance_ratio in fault_tolerance_ratios:
            if not timed_out:
                results = [None] * 2
                stop_thread = False
                thread = threading.Thread(
                    target=find_trafos_wrapper,
                    args=(
                        adjacency_matrix,
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
                    time_spent = time.time() - time_start
                    if time_spent < time_per_iteration:
                        time.sleep(1)
                        logger.info(f"Time spent in current iteration: {time_spent} s.")
                    else:
                        break
                    if global_timeout():
                        break

                thread.join(timeout=1)

                if thread.is_alive():
                    logger.warning(f"Calculation of trafos timed out.")
                    timed_out = True
                    # If a given bandwidth has timed out, all other subsequent
                    # bandwidths (as we assume an ordered list) will time out as well.
                    # Set this flag in order to skip them.
                    parameters = (
                        current_percentage,
                        kde_bandwidth,
                        trafo_fault_tolerance_ratio,
                        "Timeout",
                        "Timeout",
                        "Timeout",
                        round(time.time() - time_start, 2),
                    )

                    stop_thread = True
                    while thread.is_alive():
                        time.sleep(0.5)
                else:
                    trafos, average_matchrate_per_trafo = results
                    num_generators = num_generators_contained(
                        trafos, norm, adjacency_matrix, error_value_limit
                    )
                    parameters = (
                        current_percentage,
                        kde_bandwidth,
                        trafo_fault_tolerance_ratio,
                        len(trafos),
                        average_matchrate_per_trafo,
                        num_generators,
                        round(time.time() - time_start, 2),
                    )
                    logger.info(
                        f"percentage_observations = {current_percentage},  "
                        f"kde_bandwidth = {round(kde_bandwidth, 12)},  "
                        f"trafo_fault_tolerance_ratio = {trafo_fault_tolerance_ratio},  "
                        f"num_found_trafos = {len(trafos)},  "
                        f"average_matchrate_per_trafo = {average_matchrate_per_trafo},  "
                        f"num_generators = {num_generators},  "
                    )
            else:
                # As a previous, smaller bandwidth already timed out, we can safely skip
                # this iteration.
                parameters = (
                    current_percentage,
                    kde_bandwidth,
                    trafo_fault_tolerance_ratio,
                    "skipped",
                    "skipped",
                    "skipped",
                    "skipped",
                )
                logger.info(
                    f"percentage_observations = {current_percentage},  "
                    f"kde_bandwidth = {round(kde_bandwidth, 12)},  "
                    f"trafo_fault_tolerance_ratio = {trafo_fault_tolerance_ratio},  "
                    f"num_found_trafos = skipped,  "
                    f"average_matchrate_per_trafo = skipped,  "
                )
            parameter_study_results.append(parameters)


def num_generators_contained(trafos, norm, adjacency_matrix, error_value_limit):
    """
    Calculate the permutation groups and return the number of the corresponding
    generators from the given transformations.
    :param trafos: The transformations from which to calculate the generators
    :param norm: The norm with which to compute the deviation value.
    :param adjacency_matrix: The adjacency matrix of the graph
    :param error_value_limit: The limit for the deviation value of each group element in
    order for the first one to be considered a valid generator.
    :return:
    """
    id = np.eye(len(adjacency_matrix))
    permutation_group_generators = []
    # during the filtering below, we will take note of the values of |p_i^k A - A
    # p_i^k| to
    # plot a histogram in the end, which allows for more convenient tuning of the
    # error_value_limit
    deviation_values = []
    for trafo in trafos:
        generator = to_matrix(trafo)
        current_power = generator
        is_valid = True
        for power in count(0):
            if np.allclose(current_power, id):
                break  # went through the full cycle of generator

            deviation = deviation_value(norm, current_power, adjacency_matrix)
            deviation_values.append(deviation)
            if deviation > error_value_limit:
                logging.info(
                    f"Skipping a transformation due to deviation"
                    f" {deviation} > {error_value_limit}"
                    f" for power {power}:\n{matshow(generator)}"
                )
                is_valid = False
                break

            current_power = current_power @ generator

        if is_valid:
            g = PermutationGroup(*permutation_group_generators)
            p = Permutation(trafo)
            if p not in g:
                permutation_group_generators.append(p)
    tmp = []
    for gen in permutation_group_generators:
        if not gen.is_Identity:
            tmp.append(gen)
    return len(tmp)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.DEBUG)
    config = configparser.ConfigParser()
    study_name = "10x5"
    config.read(f"parameter_study/parameter_study_{study_name}.ini")
    params = config._sections

    global_timeout = float(params["global_timeout"]["global_timeout"])
    global_stop_thread = False

    parameter_study_results = []
    for testcase, parameters_not_parsed in params.items():
        if testcase == "global_timeout":
            continue
        parameters_parsed = {}
        for name, value in parameters_not_parsed.items():
            if name != "world_name" and name != "norm":
                try:
                    parameters_parsed[name] = ast.literal_eval(value)
                except ValueError as e:
                    logger.error(f"Error when parsing value of variable {name}.")
                    logger.error(e)
                    logger.error(f"string was {value}")
                    sys.exit(1)
            elif name == "norm":
                if value == "Norm.L_INFINITY":
                    parameters_parsed[name] = Norm.L_INFINITY
                elif value == "Norm.L_1":
                    parameters_parsed[name] = Norm.L_1
                elif value == "Norm.L_2":
                    parameters_parsed[name] = Norm.L_2
                else:
                    raise ValueError(f"No valid norm set for testcase {testcase}.")
            elif name == "world_name":
                parameters_parsed[name] = value
        signal_queue = []  # One could use a queue.Queue here, but this is not necessary
        if not global_stop_thread:
            thread = threading.Thread(
                target=run_parameter_study, args=(
                    parameters_parsed,
                    lambda: global_stop_thread,
                    parameter_study_results
                )
            )

        time_start = time.time()
        thread.start()
        # Wait until the timeout value has been reached (timeout value is given
        # in seconds).
        while thread.is_alive():
            time_spent = time.time() - time_start
            if time_spent < global_timeout:
                time.sleep(1)
            else:
                logger.critical("-------------- GLOBAL TIMEOUT REACHED --------------")
                break
        thread.join(timeout=1)
        if thread.is_alive():
            logger.critical(
                "As stated above, global timeout has been reached.\n"
                "Cleanup process is underway."
            )
            global_stop_thread = True  # Stop thread via lambda callback (globally).
            while thread.is_alive():
                time.sleep(0.5)
    results_dataframe = pd.DataFrame(
        parameter_study_results,
        columns=[
            "percentage_observation",
            "kde_bandwidth",
            "trafo_fault_tolerance_ratio",
            "num_found_trafos",
            "average_matchrate_per_trafo",
            "num_generators",
            "time for calculation",
        ],
    )
    results_dataframe.to_excel(f"{study_name}_results.xlsx", engine="xlsxwriter")


