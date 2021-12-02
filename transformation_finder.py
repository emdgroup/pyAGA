import logging
from typing import List, Union, Set, Callable

import numpy as np
import sys

import kernel_density
import verify_transformations as vt

import local_import_paths
local_import_paths.import_paths()
from mipsym.mip import Norm, Solver
from mipsym.mip_reduced import create_reduced_mip_model
from mipsym.mip import create_mip_solver
from mipsym.tools import to_ndarray, to_list, to_matrix, matshow, deviation_value

logger = logging.getLogger("trafofinder_presolving")

def print_permutation(text: str, norm: Norm, a: np.ndarray, perm: List[int]):
    p = vt.to_matrix(perm)
    logger.info(text)
    logger.info(f'Error Norm  = {deviation_value(norm, p, a)}')
    logger.info(f'Permutation = {perm}')


def find_trafos(
    adjacency_matrix: np.ndarray,
    fault_tolerance: int,
    round_decimals: int,
    quiet: bool,
    bandwidth: float,
    casename: str,
    norm: Norm,
    use_integer_programming: bool,
    stop_thread: Callable[[None], bool] = None,
) -> List[List[Union[int, None]]]:
    """
    Find all transformations (i.e. graph symmetries) on a given graph.
    :param adjacency_matrix: The adjacency matrix of the graph.
    :param fault_tolerance: The number of tolerated unmappable nodes.
    :param round_decimals: The number of positions which will be left after rounding
    the adjacency matrix values.
    :param quiet: Whether to print debugging information to the terminal.
    :param bandwidth: The bandwidth parameter for the kernel density estimation and
    subsequent bin calculation.
    :param casename: The name of the testcase.
    :param use_integer_programming: Whether or not to use the integer programming
    routines to fill out partial transformations.
    :param stop_thread: This function passes along a lambda callback to tell this
    thread to terminate.
    :return: A tuple containing the found transformations as its first entry,
    and the average matchrate over the found transformations as its second.
    Matchrates are the ratios of correctly mapped nodes to unmappable ones.
    """
    if round_decimals is not None:
        adjacency_matrix = adjacency_matrix.round(round_decimals)
    bins = kernel_density.bins(adjacency_matrix, bandwidth=bandwidth, plot=not quiet)
    labels = np.digitize(adjacency_matrix, bins=bins)
    unique_values, _ = np.unique(labels, return_counts=True)
    equivalency_classes = []
    for value in unique_values:
        s = np.argwhere(labels == value)
        equivalency_classes.append([j for j in s if j[0] <= j[1]])

    n = np.shape(adjacency_matrix)[0]
    # possible_mappings: entry i is set of all possible mappings based on the
    # adjacency matrix
    possible_mappings = [set(range(n))] * n
    # this is where the algorithm starts
    matching_rates = []
    trafos = []
    calculate_trafos(
        adjacency_matrix,
        equivalency_classes,
        possible_mappings,
        quiet,
        fault_tolerance,
        matching_rates,
        casename,
        norm,
        use_integer_programming,
        trafos,
        stop_thread,
    )
    if stop_thread is not None and stop_thread():
        return [None, None]
    else:
        return trafos, sum(matching_rates) / len(matching_rates)


def calculate_trafos(
    adjacency_matrix: np.ndarray,
    equivalency_classes: List[List[np.ndarray]],
    possible_mappings: List[Set],
    quiet: bool,
    fault_tolerance: int,
    matching_rates: List[float],
    casename: str,
    norm: Norm,
    use_integer_programming: bool,
    result: List[List[Union[int, None]]],
    stop_thread,
) -> None:
    """Calculate the transformations with the given possible mappings. This function
    is called recursively, until all sets in the possible mappings have at most one
    entry.

    :param adjacency_matrix: The adjacency matrix of the graph.
    :param equivalency_classes: A list of lists of edges which can be mapped onto
    each other freely, as they are considered to be equivalent, i.e. their weights
    fall into the same bin.
    :param possible_mappings: A list containing all possible mappings of the node at
    each position.
    :param quiet: Whether to print debugging information to the terminal.
    :param fault_tolerance: The number of tolerated unmappable nodes.
    :param matching_rates: The list containing the matchrates of the found
    :param casename: The name of the testcase.
    :param use_integer_programming: Whether or not to use the integer programming
    routines to fill out partial transformations.
    :param result: The list containing all currently found transformations.
    :param stop_thread: This function passes along a lambda callback to tell this
    thread to terminate.
    :return: None
    """
    if stop_thread is not None and stop_thread():
        # If the stop_thread lambda callback is activated, exit early.
        return

    number_of_matches = {
        'impossible':  sum(1 for i in possible_mappings if len(i) == 0),
        'perfect': sum(1 for i in possible_mappings if len(i) == 1),
        'unsure': sum(1 for i in possible_mappings if len(i) > 1),
    }
    assert sum(number_of_matches.values()) == len(possible_mappings)

    if number_of_matches['unsure'] > 0:
        # search for the shortest remaining possible mapping...
        node_of_shortest = -1
        length_of_shortest = 2 * len(possible_mappings)
        for i, m in enumerate(possible_mappings):
            if 1 < len(m) < length_of_shortest:
                length_of_shortest = len(m)
                node_of_shortest = i
        # ... and just try all candidates in there via a recursive call
        for potential_target in possible_mappings[node_of_shortest]:
            new_poss = filter_perms(equivalency_classes, list(possible_mappings), node_of_shortest, potential_target)
            if new_poss == possible_mappings:
                logger.error("This should not happen. Please assure that all "
                             "self-concurrences have fallen into the same bin.")
                assert False
            calculate_trafos(
                adjacency_matrix,
                equivalency_classes,
                new_poss,
                quiet,
                fault_tolerance,
                matching_rates,
                casename,
                norm,
                use_integer_programming,
                result,
                stop_thread,
            )
    else:
        # Check if possible_mappings contains identical single_element entries
        perfectly_matched = [list(mapping)[0] for mapping in possible_mappings if len(mapping) == 1]
        if len(set(perfectly_matched)) < len(perfectly_matched):
            return  # supplied possible_mappings does not allow for a valid permutation

        if number_of_matches['impossible'] == 0:
            # for every node, we exactly have one target node left - this is a complete permutation
            permutation = [s.pop() for s in possible_mappings]
            result.append(permutation)
            matching_rates.append(1)
            if not quiet:
                print_permutation(
                    f'Permutation number {len(result)} matched all nodes.',
                    norm,
                    adjacency_matrix,
                    permutation
                )
                logger.debug("\n" + matshow(to_matrix(permutation)))
        else:
            # for at least one node there is no target node left

            if number_of_matches['impossible'] > fault_tolerance:
                return  # number of unmatched nodes exceeds fault tolerance

            # nodes either have exactly one match target or cannot be matched at all

            permutation = [s.pop() if len(s) > 0 else None for s in possible_mappings]
            matching_rate = number_of_matches['perfect'] / len(possible_mappings)

            if not quiet:
                logger.info(f'Incomplete permutation number {len(result)} matched '
                            f'{number_of_matches["perfect"]} nodes.')
                logger.info(permutation)

            if use_integer_programming:
                logger.info('Trying to fill missing entries in permutation using MIP')

                # Construct a reduced MIP only containing rows/cols that are still not resolved
                # Mappings for identifying the rows/cols of the reduced problem and corresponding matrices
                
                row_index_map = [i for i, p in enumerate(permutation) if p is None]
                col_index_map = [i for i, _ in enumerate(permutation) if i not in permutation]
                
                # Calculate a temporary complete permutation that assigns unmapped vertices in some way
                # Idea is to apply this to the adjacency matrix s.t. one obtains correctly permuted A_row, A_col
                
                tmp_filled_permutation = permutation[:]
                for i in range(len(col_index_map)):
                    tmp_filled_permutation[row_index_map[i]] = col_index_map[i]
                    
                tmp_p_matrix = to_matrix(tmp_filled_permutation)
                A_l = tmp_p_matrix@adjacency_matrix
                A_r = adjacency_matrix@tmp_p_matrix

                # Solve the reduced problem
                # Currently WIP, not fully integrated yet
                solver = Solver.SCIP
                model = create_reduced_mip_model(norm, A_l, A_r, col_index_map, row_index_map)
                ip_solver, solve_params = create_mip_solver(solver, norm)

                try:
                    logger.info(f'Solving using {solver}')
                    results = ip_solver.solve(
                        model,
                        tee=not quiet,
                        timelimit=None,
                        report_timing=True,
                        **solve_params)

                    reduced_p = to_ndarray(model.P, len(col_index_map), len(row_index_map))
                    reduced_permutation = to_list(reduced_p)

                    filled_permutation = permutation[:]
                    for i, p in enumerate(reduced_permutation):
                        assert filled_permutation[row_index_map[i]] is None
                        filled_permutation[row_index_map[i]] = col_index_map[p]

                    if not quiet:
                        logger.info('Solver Result:\n' + str(results))
                        print_permutation('Filled Permutation.', norm, adjacency_matrix, filled_permutation)

                    permutation = filled_permutation
                    matching_rate = 1

                except RuntimeError:
                    logger.warning(f'No solution found for {permutation}')
                    logger.warning('Solver Result:\n' + str(results))

            result.append(permutation)
            matching_rates.append(matching_rate)

            if not quiet:
                logger.debug("\n" + matshow(to_matrix(permutation)))


def filter_perms(
    equivalence_classes: List[List[np.ndarray]],
    possible_mappings: List[Set],
    start_node: int,
    target_node: int,
) -> List[Set]:
    """
    Filter the given list of possible mappings under the assumption that start_node is mapped
    to target_node.
    :param equivalence_classes: A list of lists of edges which can be mapped onto
    each other freely, as they are considered to be equivalent, i.e. their weights
    fall into the same bin.
    :param possible_mappings: A list containing all possible mappings of the node at
    each position.
    :param start_node: The original node of the assumed map.
    :param target_node: The image node of the assumed map.
    :return: The possible mappings under the given assumed map.
    """
    # Iterate over all possible weights w
    for equivalence_class in equivalence_classes:
        s_w = set()

        # Calculate set s_w of vertices i such that the edge (i, target_node) has weight w
        for edge in equivalence_class:
            if edge[0] == target_node:
                s_w.add(edge[1])
            elif edge[1] == target_node:
                s_w.add(edge[0])

        # Look for vertex z such that edge (start_node, z) also has weight w
        for edge in equivalence_class:
            if edge[0] == start_node:
                z = edge[1]
            elif edge[1] == start_node:
                z = edge[0]
            else:
                continue
            # Vertex z can only be mapped to vertices in s_w if start_node is mapped to target_node
            possible_mappings[z] = possible_mappings[z].intersection(s_w)
    return possible_mappings
