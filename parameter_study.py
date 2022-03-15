import ast
import configparser
import datetime
import itertools
import logging
import os
import platform
import shutil

import sys
import threading
import time
import uuid
from itertools import count
from typing import List, Callable, Tuple

import numpy as np
import pandas as pd
from colorama import Fore, Style
from sympy.combinatorics import PermutationGroup, Permutation

from transformation_finder import find_trafos
from mipsym.mip import Norm
from mipsym.tools import to_matrix, matshow, deviation_value
import pickle

logger = logging.getLogger("trafofinder_presolving")


def run_parameter_study(
    parameters: dict,
    parameter_study_results: List[Tuple],
    global_timeout: Callable[[None], bool],
) -> None:
    """The main function within this module, called from the if __name__ == "__main__"
    statement (in a seperate thread). After running through or after getting told to
    exit via global_timeout, it returns the data computed up to this point such that
    it can be written to an excel sheet.
    :param parameters: The parameters for the parameter study, defined in the
    corresponding .ini-file.
    :param parameter_study_results: A list of tuples, where each tuple corresponds to a
    run of the transformation_finder, and its elements are the parameter values.
    :param global_timeout: This function passes along a lambda callback to tell this
    thread to terminate.
    """
    world_name = parameters["world_name"]
    integer_matrices = parameters["integer_matrices"]
    trafo_round_decimals = parameters["trafo_round_decimals"]
    use_integer_programming = parameters["use_integer_programming"]
    quiet = parameters["quiet"]
    norm = parameters["norm"]
    error_value_limits = parameters["error_value_limits"]
    time_per_iteration = parameters["time_per_iteration"]

    percentages = parameters["percentages"]
    kde_bandwidths = parameters["kde_bandwidths"]
    fault_tolerance_ratios = parameters["fault_tolerance_ratios"]

    for error_value_limit in error_value_limits:
        for percentage in percentages:
            if integer_matrices:
                mat_filename = (
                    f"data/{world_name}_integers_concurrence_matrix_"
                    f"{percentage}.pickle"
                )
            else:
                mat_filename = (
                    f"data/{world_name}_concurrence_matrix_{percentage}.pickle"
                )
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
                    return


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
    :param trafo_round_decimals: The number of decimals left after rounding the adjacency
    matrix values. If None, no rounding takes place.
    the adjacency matrix values.
    :param quiet: A parameter to limit the number of console and log-outputs.
    :param kde_bandwidth: The bandwidth parameter for the kernel density estimation and
    subsequent bin calculation.
    :param world_name: The name of the testcase.
    :param norm: The norm to use for the integer program.
    :param error_value_limit: The limit for the deviation value of each group element in
    order for the first one to be considered a valid generator.
    :param use_integer_programming: Whether or not to use the integer programming
    routines to fill out partial transformations.
    :param result: A tuple containing the found transformations as its first entry,
    and the average matchrate over the found transformations as its second.
    Matchrates are the ratios of correctly mapped nodes to unmappable ones.
    :param stop_thread: This function passes along a lambda callback to tell this
    thread to terminate.
    """
    trafos, number_of_MIP_calls = find_trafos(
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
    result[1] = number_of_MIP_calls


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
    :param trafo_round_decimals: The number of decimals left after rounding the adjacency
    matrix values. If None, no rounding takes place.
    :param use_integer_programming:
    :param error_value_limit: The limit a permutation can deviate from a perfect
    graph symmetry in order for it to be included in the permutation group.
    :param parameter_study_results: A list of tuples containing the result of the
    excel sheet. The individual tuples correspond to rows within the excel sheet.
    :param time_per_iteration: The time in seconds each individual transformation-finding run (
    including the integer program, if it is enabled) has before it reaches a timeout state.
    :param global_timeout:
    :return:
    """
    timed_out = False
    for kde_bandwidth in kde_bandwidths:
        for trafo_fault_tolerance_ratio in fault_tolerance_ratios:
            if not timed_out:
                results = [None] * 2
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
                last_printed_time_spent = 0
                while thread.is_alive():
                    time_spent = time.time() - time_start
                    if time_spent < time_per_iteration:
                        time.sleep(1)
                        if (
                            time_spent - last_printed_time_spent > 30
                        ):  # print approximately every 30 s
                            logger.info(
                                f"Time spent in current iteration: {time_spent} s."
                            )
                            last_printed_time_spent = time_spent
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
                    stop_thread = True
                    while thread.is_alive():
                        time.sleep(0.5)

                trafos, number_of_MIP_calls = results
                (
                    num_generators,
                    all_fundamentals_contained,
                    group_order,
                ) = num_generators_contained(
                    trafos, norm, adjacency_matrix, error_value_limit
                )
                if expected_permutation_group_order[study_name] < group_order:
                    is_group_order_correct = "too many"
                elif expected_permutation_group_order[study_name] == group_order:
                    is_group_order_correct = (
                        "exact" if not timed_out else "exact (Timeout)"
                    )
                else:
                    is_group_order_correct = (
                        "too few" if not timed_out else "too few (Timeout)"
                    )

                parameters = (
                    current_percentage,
                    kde_bandwidth,
                    trafo_fault_tolerance_ratio,
                    error_value_limit,
                    all_fundamentals_contained,
                    group_order,
                    expected_permutation_group_order[study_name],
                    is_group_order_correct,
                    round(time.time() - time_start, 2)
                    if not timed_out
                    else f">= {round(time.time() - time_start, 2)} (Timeout)",
                )
                assert len(parameters) == num_columns
                logger.info(
                    f"percentage_observations = {current_percentage},  "
                    f"kde_bandwidth = {round(kde_bandwidth, 12)},  "
                    f"trafo_fault_tolerance_ratio = {trafo_fault_tolerance_ratio},  "
                    f"trafo_round_decimals = {trafo_round_decimals},   "
                    f"error_value_limit = {error_value_limit},  "
                    f"num_found_trafos = {len(trafos)},  "
                    f"num_generators = {num_generators},  "
                    f"all_fundamentals_contained = {all_fundamentals_contained},  "
                    f"group_order = {group_order},  "
                    f"expected_group_order = {expected_permutation_group_order[study_name]},  "
                    f"is_group_order_correct = {is_group_order_correct},  "
                    f"number_of_MIP_calls.valid = {number_of_MIP_calls.valid[0]},   "
                    f"number_of_MIP_calls.invalid = "
                    f"{number_of_MIP_calls.invalid[0]},   "
                    +
                    (
                        f"time = {round(time.time() - time_start, 2)} s   "
                        if not timed_out else
                        f"time >= {round(time.time() - time_start, 2)} s (Timeout)  "
                    )
                )
            else:
                # As a previous, smaller bandwidth already timed out, we can safely skip
                # this iteration.
                parameters = (
                    current_percentage,
                    kde_bandwidth,
                    trafo_fault_tolerance_ratio,
                    error_value_limit,
                    "skipped",
                    "skipped",
                    expected_permutation_group_order[study_name],
                    "skipped",
                    "skipped",
                )
                assert len(parameters) == num_columns
                logger.info(
                    f"percentage_observations = {current_percentage},  "
                    f"kde_bandwidth = {round(kde_bandwidth, 12)},  "
                    f"trafo_fault_tolerance_ratio = {trafo_fault_tolerance_ratio},  "
                    f"error_value_limit = {error_value_limit},  "
                    f"all_fundamentals_contained = skipped,  "
                    f"group_order = skipped,  "
                    f"expected_group_order = {expected_permutation_group_order[study_name]},  "
                    f"is_group_order_correct = skipped,  "
                    f"time = skipped,  "
                )
            parameter_study_results.append(parameters)


def num_generators_contained(
    trafos: List[List[int]],
    norm: Norm,
    adjacency_matrix: np.ndarray,
    error_value_limit: float,
) -> Tuple[int, bool, int]:
    """
    Calculate an upper bound for the number of generators necessary to generate the
    permutation group of the given transformations. Also check if the "fundamental
    generators" of each transformation are present in the transformations.
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
    if dimensions is None:
        num_horizontal_pixels, num_vertical_pixels = map(
            # study_name is the argument passed to parameter_study
            # e.g. if study_name == "10x5", then the first split just turns it into ["10x5"]
            # otherwise if the parameter is a longer study_name seperated by "_", the rest
            # is discarded
            lambda x: int(x),
            study_name.split("_")[0].split("x"),
        )
    else:
        # dimensions[0] is always 1 in 2D testcases
        num_horizontal_pixels = dimensions[1]
        num_vertical_pixels = dimensions[2]
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
        rotation_without_colors = (
            np.roll(
                np.arange(0, num_horizontal_pixels**2).reshape(
                    num_horizontal_pixels, num_horizontal_pixels
                ),
                shift=-1,
                axis=0,
            )
            .T[:, ::-1]
            .flatten()
        )
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
    num_columns = 9
    expected_permutation_group_order = {
        "20x10": 400,  # 20*10*2
        "15x15_rotations": 900,  # 15*15*4
        "no_axsym_15x15_rotations": 900,
        "13x7_letters_indiv_colors": 1092,  # 13*7*6*2
    }
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
    job_array_id = (
        None  # if this is not None, then this current job is part of a job array
    )
    job_id_index = (
        None  # if this is not None, then this current job is part of a job array
    )
    try:
        job_array_id = int(sys.argv[2])
        job_id_index = int(sys.argv[3])
        array_task_min = int(sys.argv[4])
    except IndexError:
        pass
    except ValueError as e:
        logger.error(e)
        logger.error(
            "Failed to convert command line arguments to job_array_id, job_id_index and array_task_min."
        )
        sys.exit(1)
    if job_array_id is not None:
        assert job_id_index is not None
        jobarray_foldername = f"jobarray_{job_array_id}"
        if job_id_index == array_task_min:
            try:
                os.makedirs(f"parameter_study/results/{jobarray_foldername}")
            except FileExistsError as e:
                logger.error(e)
                logger.error(
                    "Tried creating a folder for a job array id which already exists. This can't happen on the cluster and"
                    " is therefore disallowed."
                )
                sys.exit(1)
        filename_xlsx = f"parameter_study/results/{jobarray_foldername}/{study_name}_results_{uuid.uuid4()}.xlsx"
    else:
        filename_xlsx = (
            f"parameter_study/results/{study_name}_results_{uuid.uuid4()}.xlsx"
        )
    logger.info(f"Results table will be written to {filename_xlsx}")
    # config.read(f"parameter_study/parameter_study_{study_name}.ini")
    config_name = f"parameter_study/parameter_study_{study_name}.ini"
    with open(config_name, "r") as ini_file:
        print(ini_file.read())
    if job_array_id is not None and job_array_id == array_task_min:
        shutil.copy(
            config_name,
            f"parameter_study/results/{jobarray_foldername}/parameter_study_{study_name}.ini",
        )
    config.read(config_name)
    params = config._sections

    global_timeout = float(params["global_timeout"]["global_timeout"])
    global_stop_thread = False

    iterable_parameters = [
        "error_value_limit",  # Keep both entries for clarity, even though the
        "error_value_limits",  # singular string is always replaced by the plural.
        "percentages",
        "kde_bandwidths",
        "fault_tolerance_ratios",
    ]
    dimensions = None
    parameter_study_results = []
    for testcase, parameters_not_parsed in params.items():
        if testcase == "global_timeout":
            continue
        parameters_parsed = {}
        for name, value in parameters_not_parsed.items():
            if name == "norm":
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
            elif name == "dimensions":
                dimensions = ast.literal_eval(value)
            else:
                if name == "error_value_limit":
                    name = "error_value_limits"
                try:
                    parsed_value = ast.literal_eval(value)
                    if name in iterable_parameters:
                        try:
                            _ = iter(parsed_value)
                        except TypeError:
                            # The parameter value should be iterable, but the value that
                            # resulted from parsing isn't.
                            parsed_value = (parsed_value,)
                    parameters_parsed[name] = parsed_value
                except ValueError as e:
                    logger.error(f"Error when parsing value of variable {name}.")
                    logger.error(e)
                    logger.error(f"string was {value}")
                    sys.exit(1)
        if not global_stop_thread:
            thread = threading.Thread(
                target=run_parameter_study,
                args=(
                    parameters_parsed,
                    parameter_study_results,
                    lambda: global_stop_thread,
                ),
            )
        if job_array_id is not None:
            # If script is part of job array, calculate the corresponding element of the cartesian product of parameters
            # and use it as the parameters instead.
            cartesian_product = tuple(
                itertools.product(
                    parameters_parsed["error_value_limits"],
                    parameters_parsed["percentages"],
                    parameters_parsed["kde_bandwidths"],
                    parameters_parsed["fault_tolerance_ratios"],
                )
            )
            if job_id_index == array_task_min:
                for element in cartesian_product:
                    with open(
                        f"parameter_study/results/{jobarray_foldername}/status_todo_{uuid.uuid4()}",
                        "w",
                    ) as file:
                        file.write(datetime.datetime.now().isoformat() + "\n")
                        file.write(str(time.time()) + "\n")
                        file.write(str(element) + "\n")
                        file.write("status: TODO\n")

            product_element = cartesian_product[job_id_index]
            with open(
                f"parameter_study/results/{jobarray_foldername}/status_started_{uuid.uuid4()}",
                "w",
            ) as file:
                file.write(datetime.datetime.now().isoformat() + "\n")
                file.write(str(time.time()) + "\n")
                file.write(str(product_element) + "\n")
                file.write("status: started\n")

            logger.debug(f"{Fore.RED}job_array_id: {job_array_id}{Style.RESET_ALL}")
            logger.debug(f"{Fore.RED}job_id_index: {job_id_index}{Style.RESET_ALL}")
            logger.debug(
                f"{Fore.RED}Element of cartesian product: {product_element}{Style.RESET_ALL}"
            )
            (
                parameters_parsed["error_value_limits"],
                parameters_parsed["percentages"],
                parameters_parsed["kde_bandwidths"],
                parameters_parsed["fault_tolerance_ratios"],
            ) = map(lambda x: (x,), product_element)
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
        "error_value_limit",
        "fundamental_generators_contained",
        "permutation_group_order",
        "expected_permutation_group_order",
        "is_group_order_correct",
        "time for calculation",
    ]
    assert len(columns) == num_columns
    results_dataframe = pd.DataFrame(parameter_study_results, columns=columns)
    results_dataframe.to_excel(filename_xlsx, engine="xlsxwriter")

    if job_array_id is not None:
        with open(
            f"parameter_study/results/{jobarray_foldername}/status_finished_{uuid.uuid4()}",
            "w",
        ) as file:
            file.write(datetime.datetime.now().isoformat() + "\n")
            file.write(str(time.time()) + "\n")
            file.write(str(product_element) + "\n")
            file.write("status: finished\n")
            try:
                file.write(f"result: {parameter_study_results[0][7]}\n")
            except IndexError:
                file.write(f"result: MIP Timeout without admissible solution")
