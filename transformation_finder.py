from typing import List, Union, Set

import numpy as np
import sys

import kernel_density
import verify_transformations as vt

sys.path.append(r"C:\Users\M305822\OneDrive - MerckGroup\PycharmProjects\integer_programming_for_transformations")
from mipsym import mip as ipt


def print_permutation(text: str, a: np.ndarray, perm: List[int]):
    p = vt.to_matrix(perm)
    print(text)
    print(f'Error Norm  = {np.linalg.norm( a @ p - p @ a)}')
    print(f'Permutation = {perm}')


def find_trafos(
    adjacency_matrix: np.ndarray,
    fault_tolerance: int,
    round_decimals: int,
    quiet: bool,
    bandwidth: float,
    casename: str,
    use_integer_programming,
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
    :return: A tuple containing the found transformations as its first entry,
    and the average matchrate over the found transformations as its second.
    Matchrates are the ratios of correctly mapped nodes to unmappable ones.
    """
    if round_decimals is not None:
        adjacency_matrix = adjacency_matrix.round(round_decimals)
    bins = kernel_density.bins(adjacency_matrix, bandwidth=bandwidth, plot=False)
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
        use_integer_programming,
        trafos,
    )
    return trafos, sum(matching_rates) / len(matching_rates)


def calculate_trafos(
    adjacency_matrix: np.ndarray,
    equivalency_classes: List[List[np.ndarray]],
    possible_mappings: List[Set],
    quiet: bool,
    fault_tolerance: int,
    matching_rates: List[float],
    casename: str,
    use_integer_programming: bool,
    result: List[List[Union[int, None]]],
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
    :return: None
    """
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
                print("This should not happen. Please assure that all self-concurrences "
                      "have fallen into the same bin.")
                assert False
            calculate_trafos(
                adjacency_matrix,
                equivalency_classes,
                new_poss,
                quiet,
                fault_tolerance,
                matching_rates,
                casename,
                use_integer_programming,
                result,
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
                    f'Permutation number {len(result)} correctly matched all nodes.',
                    adjacency_matrix,
                    permutation
                )
        else:
            # for at least one node there is no target node left

            if number_of_matches['impossible'] > fault_tolerance:
                return  # number of unmatched nodes exceeds fault tolerance

            # nodes either have exactly one match target or cannot be matched at all

            permutation = [s.pop() if len(s) > 0 else None for s in possible_mappings]
            result.append(permutation)
            matching_rates.append(number_of_matches['perfect'] / len(possible_mappings))

            if not quiet:
                print("---------------------------------------------------")
                # compute all nodes that actually participate in the known permutation,
                # i.e. those that are mapped onto and also map to another node

                complete_cycle_indices = [i for i, p in enumerate(permutation) if p is not None and i in permutation]

                # remove all rows and columns for nodes that are not in complete_cycles
                reduced_coeff = adjacency_matrix[complete_cycle_indices, :]
                reduced_coeff = reduced_coeff[:, complete_cycle_indices]

                # re-compute the permutation so that node indices are still valid after removing nodes
                index_map = [-1] * len(permutation)
                for new_index, old_index in enumerate(complete_cycle_indices):
                    index_map[old_index] = new_index

                reordered_permutation = [index_map[permutation[i]] for i in complete_cycle_indices]

                print_permutation(
                    f'Incomplete permutation number {len(result)} correctly matched {number_of_matches["perfect"]} nodes.',
                    reduced_coeff,
                    reordered_permutation
                )

            if use_integer_programming:
                try:
                    filled_permutation = ipt.find_permutations(
                        A=adjacency_matrix,
                        norm=ipt.Norm.L_1,
                        solver=ipt.Solver.SCIP,
                        objective_bound=1e9,
                        time_limit=None,
                        known_entries=permutation,
                    )

                    print_permutation(
                        'Filled Permutation.',
                        adjacency_matrix,
                        filled_permutation
                    )
                    # TODO: include in results (instead of original permutation above?)
                    print("---------------------------------------------------")
                except RuntimeError:
                    print(f'No solution found for {permutation}')


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
    :param x: The original node of the assumed map.
    :param y: The image node of the assumed map.
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
