import numpy as np

matching_rates = []

table_matched_nodes = []


def find_trafos(coeff, num_bins, fault_tolerance, round_decimals, quiet):
    # form set of all correlation coefficients
    if round_decimals is not None:
        coeff = coeff.round(round_decimals)
    coeff_vals = np.unique(coeff)
    # coeff_perms: each entry contains the edges with same correlation
    # ## Binning method
    if num_bins is None:
        bins = np.histogram_bin_edges(coeff, bins="doane")
        if not quiet:
            print(bins)
    else:
        assert isinstance(num_bins, int)
        bins = np.linspace(coeff.min(), coeff.max(), num_bins + 1)

    labels = np.digitize(coeff, bins=bins)

    uni, counts = np.unique(labels, return_counts=True)
    print(f"Used labels before removal: {uni}, counts : {counts}")
    # uni = uni[counts >= 100]
    print(f"Used labels after removal: {uni}")
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
    trafos = calculate_trafos(coeff_perms, poss, [], quiet,
                              fault_tolerance, matching_rates)

    return trafos, sum(matching_rates) / len(matching_rates)


def calculate_trafos(perms, poss, res, quiet, fault_tolerance, matching_rates):
    if all(len(i) == 1 for i in poss):
        res.append([s.pop() for s in poss])
        matching_rates.append(1)
        if not quiet:
            print(f"len(res) = {len(res)}, correctly matched all nodes.")
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
                if not quiet:
                    print(f"len(res) = {len(res)}, correctly matched"
                          f" {num_correctly_matched} nodes.")
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
        res = calculate_trafos(perms, new_poss, res, quiet, fault_tolerance, matching_rates)
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
