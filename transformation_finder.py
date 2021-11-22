from enum import Enum

import numpy as np
import sys

import combinatorics
import kernel_density
import verify_transformations as vt

matching_rates = []

table_matched_nodes = []

sys.path.append(r"C:\Users\M305822\OneDrive - MerckGroup\PycharmProjects")
from integer_programming_for_transformations import main as ipt


class Norm(Enum):
    L_INFINITY = 0
    L_1 = 1
    L_2 = 2


class Solver(Enum):
    GLPK = 0
    IPOPT = 1
    HiGHS = 2
    SCIP = 3


def find_trafos(coeff, num_bins, fault_tolerance, round_decimals, quiet, bandwidth):
    # form set of all correlation coefficients
    if round_decimals is not None:
        coeff = coeff.round(round_decimals)
    bins = kernel_density.bins(coeff, bandwidth=bandwidth, plot=False)

    labels = np.digitize(coeff, bins=bins)

    uni, counts = np.unique(labels, return_counts=True)
    # print(f"Used labels before removal: {uni}, counts : {counts}")
    # # uni = uni[counts >= 200]
    # print(f"Used labels after removal: {uni}")
    coeff_perms = []
    for i in uni:
        s = np.argwhere(labels == i)
        coeff_perms.append([j for j in s if j[0] <= j[1]])
        # ## Binning method

    # # # isclose-method
    # # coeff_perms = []
    # # for i in coeff_vals:
    # #     s = np.argwhere(arr_2D_isclose(coeff, i, abs_tol=abs_tol, rel_tol=rel_tol)
    # #     s = np.argwhere(coeff == i)
    # #     coeff_perms.append([j for j in s if j[0] <= j[1]])#
    # # # isclose-method

    # coeff_perms = []
    # for i in coeff_vals:
    #     s = np.argwhere(coeff == i)
    #     coeff_perms.append([j for j in s if j[0] <= j[1]])

    # poss: entry i is set of all possible rho(i) based on coeff_perms
    n = np.shape(coeff)[0]
    poss = [set(range(n))] * n
    # this is where the algorithm starts
    matching_rates = []  # reset module-global list
    trafos = calculate_trafos(
        coeff_perms, poss, [], quiet, fault_tolerance, matching_rates, coeff
    )
    return trafos, sum(matching_rates) / len(matching_rates)


def calculate_trafos(perms, poss, res, quiet, fault_tolerance, matching_rates, coeff):
    if all(len(i) == 1 for i in poss):
        res.append([s.pop() for s in poss])
        matching_rates.append(1)
        if not quiet:
            permutation = res[-1]
            norm_val = np.linalg.norm(
                coeff @ vt.to_matrix(permutation) - vt.to_matrix(permutation) @ coeff
            )
            print(
                f"len(res) = {len(res)}, correctly matched all nodes. "
                f"Norm value = {norm_val}"
            )
        return res
    elif any(len(i) == 0 for i in poss):
        num_unmatchable = len([i for i in poss if len(i) == 0])
        if num_unmatchable > fault_tolerance:
            return res
        num_correctly_matched = len([i for i in poss if len(i) == 1])
        num_uncertain = len([i for i in poss if len(i) > 1])
        if num_uncertain == 0:
            if num_correctly_matched >= len(poss) - fault_tolerance:
                res.append([s.pop() if len(s) > 0 else None for s in poss])
                matching_rates.append(num_correctly_matched / len(poss))

                permutation = res[-1]
                print("---------------------------------------------------")
                if not quiet:
                    c = combinatorics.cycles(permutation)
                    complete_cycle_indices = []
                    for list_ in c:
                        complete_cycle_indices.extend(list_)

                    complete_cycle_indices.sort()
                    not_mapped_x = []
                    # for index, value in enumerate(permutation):
                    #     if value is None:
                    #         not_mapped_x.append(index)
                    # set_all_indices = set(list(range(len(permutation))))
                    # set_missing_y = set_all_indices - set(permutation)
                    # allowed_indices = set_all_indices - set(not_mapped_x) - \
                    #                   set_missing_y
                    # allowed_indices = list(allowed_indices)
                    reduced_coeff = coeff[
                        complete_cycle_indices, :][:, complete_cycle_indices
                    ]

                    reordered_permutation = reduced_coeff.shape[0] * [None]
                    for index, value in enumerate(permutation):
                        if index in complete_cycle_indices:
                            new_index = complete_cycle_indices.index(index)
                            entry = complete_cycle_indices.index(value)
                            reordered_permutation[new_index] = entry
                    # reordered_permutation = np.array(permutation)[allowed_indices]
                    # for index, value in enumerate(reordered_permutation):
                    #     reordered_permutation[index] = allowed_indices.index(value)
                    #
                    norm_val = np.linalg.norm(
                        reduced_coeff @ vt.to_matrix(reordered_permutation)
                        - vt.to_matrix(reordered_permutation) @ reduced_coeff
                    )
                    print(
                        f"len(res) = {len(res)}, correctly matched"
                        f" {num_correctly_matched} nodes. "
                        f"Norm value = {norm_val}"
                    )
                    print("permutation")
                    print(permutation)
                    if vt.verify_one_transformation(
                            permutation, casename="one_letter_words_10x5"
                    ):
                        print("Permutation is legitimate")
                    else:
                        print("Permutation is NOT legitimate.")
                    filled_permutation = ipt.find_permutations(
                        A=coeff,
                        norm=ipt.Norm.L_1,
                        solver=ipt.Solver.GLPK,
                        objective_bound=1e9,
                        time_limit=None,
                        prevent_diagonal=True,
                        known_entries=permutation
                    )
                    print("filled permutation")
                    print(filled_permutation)
                    print("---------------------------------------------------")

                return res
            else:
                assert False
                return res
        else:
            pass

    s = min(([i for i in poss if len(i) > 1]), key=len)
    x = poss.index(s)

    for y in s:
        new_poss = filter_perms(perms, list(poss), x, y)
        if new_poss == poss:
            print("Filtering left possibilities unchanged. Returning early.")
            return res
        res = calculate_trafos(
            perms, new_poss, res, quiet, fault_tolerance, matching_rates, coeff
        )
    return res


def filter_perms(equivalence_classes, poss, x, y):
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
            poss[z] = poss[z].intersection(s)
    return poss

