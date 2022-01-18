import ast
import configparser
import logging
import platform

import sys
import threading
import time
import uuid
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
    parameters: dict,
    global_timeout: Callable[[None], bool],
    parameter_study_results: List[Tuple],
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
            try_bandwidths_and_tolerance_ratios(
                percentage,
                kde_bandwidths,
                fault_tolerance_ratios,
                correlation_matrix,
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
    error_value_limit: float,
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
    :param trafo_round_decimals: The number of positions which will be left after
    rounding.
    the adjacency matrix values.
    :param quiet: A parameter to limit the number of console and log-outputs.
    :param kde_bandwidth: The bandwidth parameter for the kernel density estimation and
    subsequent bin calculation.
    :param world_name: The name of the testcase.
    :param norm: The norm to use for the integer program.
    :param error_value_limit:
    :param use_integer_programming: Whether or not to use the integer programming
    routines to fill out partial transformations.
    :param result: A tuple containing the found transformations as its first entry,
    and the average matchrate over the found transformations as its second.
    Matchrates are the ratios of correctly mapped nodes to unmappable ones.
    :param stop_thread: This function passes along a lambda callback to tell this
    thread to terminate.
    """
    trafos, average_matchrate_per_trafo, number_of_MIP_calls = find_trafos(
        correlation_matrix,
        fault_tolerance=fault_tolerance,
        round_decimals=trafo_round_decimals,
        quiet=quiet,
        bandwidth=kde_bandwidth,
        casename=world_name,
        norm=norm,
        error_value_limit=error_value_limit,
        use_integer_programming=use_integer_programming,
        stop_thread=stop_thread,
    )
    result[0] = trafos
    result[1] = average_matchrate_per_trafo
    result[2] = number_of_MIP_calls


def try_bandwidths_and_tolerance_ratios(
    current_percentage: str,
    kde_bandwidths: np.ndarray,
    fault_tolerance_ratios: np.ndarray,
    adjacency_matrix: np.ndarray,
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

    :param current_percentage: The percentage of (unique) observations in the
    :param kde_bandwidths: An sorted iterable of bandwidths.
    :param fault_tolerance_ratios: An iterable of fault_tolerance_ratios.
    :param adjacency_matrix: The adjacency matrix of the graph whose symmetries we
    want to find.
    number of rows or columns of the adjacency matrix.
    :param quiet: A parameter to limit the number of console and log-outputs.
    :param norm: The norm to use for the integer program.
    :param world_name: The name of the testcase.
    :param trafo_round_decimals:
    :param use_integer_programming:
    :param error_value_limit: The maximum
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
                results = [None] * 3
                stop_thread = False
                num_variables = adjacency_matrix.shape[0]
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
                        error_value_limit,
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
                    # If a given bandwidth has timed out, all other subsequent
                    # bandwidths (as we assume an ordered list) will time out as well.
                    # Set this flag in order to skip them.
                    timed_out = True
                    parameters = (
                        current_percentage,
                        kde_bandwidth,
                        trafo_fault_tolerance_ratio,
                        trafo_round_decimals
                        if trafo_round_decimals is not None
                        else "None",
                        error_value_limit,
                        "Timeout",
                        "Timeout",
                        "Timeout",
                        "Timeout",
                        "Timeout",
                        "Timeout",
                        "Timeout",
                        round(time.time() - time_start, 2),
                    )

                    stop_thread = True
                    while thread.is_alive():
                        time.sleep(0.5)
                else:
                    trafos, average_matchrate_per_trafo, number_of_MIP_calls = results
                    (
                        num_generators,
                        all_fundamentals_contained,
                        group_order,
                    ) = num_generators_contained(
                        trafos, norm, adjacency_matrix, error_value_limit
                    )
                    parameters = (
                        current_percentage,
                        kde_bandwidth,
                        trafo_fault_tolerance_ratio,
                        trafo_round_decimals
                        if trafo_round_decimals is not None
                        else "None",
                        error_value_limit,
                        len(trafos),
                        average_matchrate_per_trafo,
                        num_generators,
                        all_fundamentals_contained,
                        group_order,
                        number_of_MIP_calls.valid[0],
                        number_of_MIP_calls.invalid[0],
                        round(time.time() - time_start, 2),
                    )
                    assert len(parameters) == num_columns
                    logger.info(
                        f"percentage_observations = {current_percentage},  "
                        f"kde_bandwidth = {round(kde_bandwidth, 12)},  "
                        f"trafo_fault_tolerance_ratio = {trafo_fault_tolerance_ratio},  "
                        f"trafo_round_decimals = {trafo_round_decimals},   "
                        f"error_value_limit = {error_value_limit},  "
                        f"num_found_trafos = {len(trafos)},  "
                        f"average_matchrate_per_trafo = "
                        f"{average_matchrate_per_trafo},  "
                        f"num_generators = {num_generators},  "
                        f"all_fundamentals_contained = {all_fundamentals_contained},  "
                        f"group_order = {group_order},  "
                        f"number_of_MIP_calls.valid = {number_of_MIP_calls.valid[0]},   "
                        f"number_of_MIP_calls.invalid = "
                        f"{number_of_MIP_calls.invalid[0]},   "
                        f"time = {round(time.time() - time_start, 2)} s"
                    )
            else:
                # As a previous, smaller bandwidth already timed out, we can safely skip
                # this iteration.
                parameters = (
                    current_percentage,
                    kde_bandwidth,
                    trafo_fault_tolerance_ratio,
                    trafo_round_decimals
                    if trafo_round_decimals is not None
                    else "None",
                    error_value_limit,
                    "skipped",
                    "skipped",
                    "skipped",
                    "skipped",
                    "skipped",
                    "skipped",
                    "skipped",
                    "skipped",
                )
                assert len(parameters) == num_columns
                logger.info(
                    f"percentage_observations = {current_percentage},  "
                    f"kde_bandwidth = {round(kde_bandwidth, 12)},  "
                    f"trafo_fault_tolerance_ratio = {trafo_fault_tolerance_ratio},  "
                    f"error_value_limit = {error_value_limit},  "
                    f"num_found_trafos = skipped,  "
                    f"average_matchrate_per_trafo = skipped,  "
                )
            parameter_study_results.append(parameters)


def num_generators_contained(
    trafos: List[List[int]],
    norm: Norm,
    adjacency_matrix: np.ndarray,
    error_value_limit: float,
) -> Tuple[int, bool, int]:
    """
    Calculate the permutation groups and return the number of the corresponding
    generators from the given transformations.
    :param trafos: The transformations from which to calculate the generators
    :param norm: The norm with which to compute the deviation value.
    :param adjacency_matrix: The adjacency matrix of the graph
    :param error_value_limit: The limit for the deviation value of each group element in
    order for the first one to be considered a valid generator.
    :return: A value which is bigger or equal to the number of permutation generators of
    the permutation group defined by the variable "trafos", a boolean value indicating
    whether the fundamental shifts are all present and the order of the group defined by
    the permutations in "trafos".
    """
    id = np.eye(len(adjacency_matrix))
    permutation_group_generators = []
    # During the filtering below, we will take note of the values of
    # |p_i^k A - A p_i^k| to plot a histogram in the end, which allows for more
    # convenient tuning of the error_value_limit.
    # As it is currently implemented, this loop may yield unnecessarily many
    # generators (as there may be permutations which are elements of subgroups of the
    # horizontal or vertical shifts. For example, in a world which is 10 pixels wide,
    # finding a horizontal shift by 2 pixels will not disallow also finding and
    # adding the horizontal shift by 1 pixel to the list.
    deviation_values = []
    for trafo in trafos:
        fundamental_generator = to_matrix(trafo)
        current_power = fundamental_generator
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
                    f" for power {power}:\n{matshow(fundamental_generator)}"
                )
                is_valid = False
                break

            current_power = current_power @ fundamental_generator

        if is_valid:
            g = PermutationGroup(*permutation_group_generators)
            p = Permutation(trafo)
            if p not in g:
                permutation_group_generators.append(p)
    tmp = []
    for gen in permutation_group_generators:
        if not gen.is_Identity:
            logger.debug(f"Adding generator:")
            logger.debug(list(gen))
            tmp.append(gen)

    # Verify that all fundamental generators are present in the permutation group
    fundamental_generators = []
    num_horizontal_pixels, num_vertical_pixels = map(
        # study_name is the argument passed to parameter_study
        # e.g. if study_name == "10x5", then the first split just turns it into ["10x5"]
        # otherwise if the parameter is a longer study_name seperated by "_", the rest
        # is discarded
        lambda x: int(x),
        study_name.split("_")[0].split("x"),
    )
    with_colors = "colors" in study_name
    # Hardcoded for now. TODO: Change
    color_depth = 3 if with_colors else 1

    # Create horizontal shift by one pixel
    horizontal_shift = []
    column = np.arange(0, num_vertical_pixels * color_depth)
    for i in range(1, num_horizontal_pixels):
        horizontal_shift.extend(
            (column + i * num_vertical_pixels * color_depth).tolist()
        )
    horizontal_shift.extend(column.tolist())
    fundamental_generators.append(Permutation(horizontal_shift))

    # Create vertical shift by one pixel
    vertical_shift = []
    column = np.array(
        list((range(color_depth, num_vertical_pixels * color_depth)))
        + list(range(color_depth))
    )
    for i in range(0, num_horizontal_pixels):
        vertical_shift.extend((column + i * num_vertical_pixels * color_depth).tolist())
    fundamental_generators.append(Permutation(vertical_shift))
    with_rotations = "rotations" in study_name
    if "rotations" not in study_name:
        # Create flip
        flip_without_colors = list(
            range(num_horizontal_pixels * num_vertical_pixels - 1, -1, -1)
        )
        flip = []
        for f in flip_without_colors:
            temp = []
            for c in range(color_depth):
                temp.append(f * color_depth + c)
            flip.extend(temp)
        fundamental_generators.append(Permutation(flip))
    else:
        # Create 90 degree rotation
        assert num_horizontal_pixels == num_vertical_pixels
        rotation_without_colors = np.roll(
            np.arange(0, num_horizontal_pixels ** 2).reshape(
                num_horizontal_pixels, num_horizontal_pixels
            ),
            shift=-1,
            axis=0,
        ).T[:, ::-1].flatten()
        rotation = []
        for r in rotation_without_colors:
            temp = []
            for c in range(color_depth):
                temp.append(r * color_depth + c)
            rotation.extend(temp)
        fundamental_generators.append(Permutation(rotation))

    if with_colors:
        # Create symmetric color shift
        symmetric_color_shift = []
        symmetric_color_singlet = np.array(list(range(1, color_depth)) + [0])
        for i in range(num_horizontal_pixels):
            for j in range(num_vertical_pixels):
                symmetric_color_shift.extend(
                    (
                        symmetric_color_singlet
                        + j * color_depth
                        + i * num_vertical_pixels * color_depth
                    ).tolist()
                )
        fundamental_generators.append(Permutation(symmetric_color_shift))

        antisymmetric_color_shift = []
        antisymmetric_color_singlet = np.arange(color_depth - 1, -1, -1)
        for i in range(num_horizontal_pixels):
            for j in range(num_vertical_pixels):
                antisymmetric_color_shift.extend(
                    (
                        antisymmetric_color_singlet
                        + j * color_depth
                        + i * num_vertical_pixels * color_depth
                    ).tolist()
                )
        fundamental_generators.append(Permutation(antisymmetric_color_shift))

    all_fundamental_generators_present = True
    permutation_group = PermutationGroup(permutation_group_generators)
    for fundamental_generator in fundamental_generators:
        if fundamental_generator not in permutation_group:
            all_fundamental_generators_present = False
            break
        else:
            logger.debug(
                f"The fundamental generator {fundamental_generator} is present in the "
                f"generated permutation group."
            )
    group_order = permutation_group.order()
    return len(tmp), all_fundamental_generators_present, group_order


if __name__ == "__main__":
    num_columns = 13
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.DEBUG)
    config = configparser.ConfigParser()
    try:
        study_name = sys.argv[1]
    except IndexError:
        logger.error("Please provide name of testcase")
        sys.exit(1)
    filename_xlsx = f"{study_name}_results_{uuid.uuid4()}.xlsx"
    logger.info(f"Results table will be written to {filename_xlsx}")
    # config.read(f"parameter_study/parameter_study_{study_name}.ini")
    config_name = f"parameter_study/parameter_study_{study_name}.ini"
    with open(config_name, "r") as file:
        print(file.read())
    config.read(config_name)
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
                target=run_parameter_study,
                args=(
                    parameters_parsed,
                    lambda: global_stop_thread,
                    parameter_study_results,
                ),
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
    columns = [
        "percentage_observation",
        "kde_bandwidth",
        "trafo_fault_tolerance_ratio",
        "trafo_round_decimals",
        "error_value_limit",
        "num_found_trafos",
        "average_matchrate_per_trafo",
        "num_generators",
        "fundamental_generators_contained",
        "permutation_group_order",
        "number_of_MIP_calls.valid",
        "number_of_MIP_calls.invalid",
        "time for calculation",
    ]
    assert len(columns) == num_columns
    results_dataframe = pd.DataFrame(parameter_study_results, columns=columns)
    results_dataframe.to_excel(filename_xlsx, engine="xlsxwriter")
