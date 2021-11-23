from typing import List, Union, Set

import numpy as np
import sys

import combinatorics
import kernel_density
import verify_transformations as vt

sys.path.append(r"C:\Users\M305822\OneDrive - MerckGroup\PycharmProjects\integer_programming_for_transformations")
from mipsym import mip as ipt


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
    bins = kernel_density.bins(adjacency_matrix, bandwidth=bandwidth, plot=True)
    labels = np.digitize(adjacency_matrix, bins=bins)
    unique_values, counts = np.unique(labels, return_counts=True)
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
    if all(len(i) == 1 for i in possible_mappings):
        result.append([s.pop() for s in possible_mappings])
        matching_rates.append(1)
        if not quiet:
            permutation = result[-1]
            norm_val = np.linalg.norm(
                adjacency_matrix @ vt.to_matrix(permutation)
                - vt.to_matrix(permutation) @ adjacency_matrix
            )
            print(
                f"len(res) = {len(result)}, correctly matched all nodes. "
                f"Norm value = {norm_val}"
            )
        return
    elif any(len(i) == 0 for i in possible_mappings):
        num_unmatchable = len([i for i in possible_mappings if len(i) == 0])
        if num_unmatchable > fault_tolerance:
            return
        num_correctly_matched = len([i for i in possible_mappings if len(i) == 1])
        num_uncertain = len([i for i in possible_mappings if len(i) > 1])
        if num_uncertain == 0:
            if num_correctly_matched >= len(possible_mappings) - fault_tolerance:
                result.append(
                    [s.pop() if len(s) > 0 else None for s in possible_mappings]
                )
                matching_rates.append(num_correctly_matched / len(possible_mappings))
                permutation = result[-1]
                print("---------------------------------------------------")
                if not quiet:
                    cycles = combinatorics.cycles(permutation)
                    complete_cycle_indices = []
                    for cycle in cycles:
                        complete_cycle_indices.extend(cycle)

                    complete_cycle_indices.sort()
                    reduced_coeff = adjacency_matrix[complete_cycle_indices, :][
                        :, complete_cycle_indices
                    ]

                    reordered_permutation = reduced_coeff.shape[0] * [None]
                    for index, value in enumerate(permutation):
                        if index in complete_cycle_indices:
                            new_index = complete_cycle_indices.index(index)
                            entry = complete_cycle_indices.index(value)
                            reordered_permutation[new_index] = entry
                    norm_val = np.linalg.norm(
                        reduced_coeff @ vt.to_matrix(reordered_permutation)
                        - vt.to_matrix(reordered_permutation) @ reduced_coeff
                    )
                    print(
                        f"len(res) = {len(result)}, correctly matched"
                        f" {num_correctly_matched} nodes. "
                        f"Norm value = {norm_val}"
                    )
                    print("permutation")
                    print(permutation)
                    if vt.verify_one_transformation(permutation, casename=casename):
                        print("Permutation is legitimate")
                    else:
                        print("Permutation is NOT legitimate.")
                    if use_integer_programming:
                        filled_permutation = ipt.find_permutations(
                            A=adjacency_matrix,
                            norm=ipt.Norm.L_1,
                            solver=ipt.Solver.SCIP,
                            objective_bound=1e9,
                            time_limit=None,
                            prevent_diagonal=True,
                            known_entries=permutation,
                        )
                        print("filled permutation")
                        print(filled_permutation)
                        print("---------------------------------------------------")

                return
            else:
                assert False
        else:
            pass

    s = min(([i for i in possible_mappings if len(i) > 1]), key=len)
    x = possible_mappings.index(s)

    for y in s:
        new_poss = filter_perms(equivalency_classes, list(possible_mappings), x, y)
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
    return


def filter_perms(
    equivalence_classes: List[List[np.ndarray]],
    possible_mappings: List[Set],
    x: int,
    y: int,
) -> List[Set]:
    """
    Filter the given list of possible mappings under the assumption that x is mapped
    to y.
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
        s = set()

        # Calculate set S_w of vertices i such that (i, y) has weight w
        for edge in equivalence_class:
            if edge[0] == y:
                s.add(edge[1])
            elif edge[1] == y:
                s.add(edge[0])
            else:
                continue
        # Look for vertex z such that (x, z) has weight w
        for edge in equivalence_class:
            if edge[0] == x:
                z = edge[1]
            elif edge[1] == x:
                z = edge[0]
            else:
                continue
            # Vertex z can only be mapped to vertices in S_w if x is mapped to y
            possible_mappings[z] = possible_mappings[z].intersection(s)
    return possible_mappings
