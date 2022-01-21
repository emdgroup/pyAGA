from collections import defaultdict
import numpy as np
from sympy.combinatorics import Permutation, PermutationGroup


def find_simple_generators(trafos):
    # We try to find as simple-looking generator matrices as possible by
    # favoring permutations where nodes stay together, i.e.
    # the sum of differences of images of neighbouring nodes is small
    simpleness = defaultdict(list)
    for trafo in trafos:
        s = np.sum(np.abs(np.diff(trafo)))
        simpleness[s].append(trafo)

    simple_generators = []
    permutation_group = PermutationGroup()

    for _, t in sorted(simpleness.items()):
        for trafo in t:
            p = Permutation(trafo)
            if p not in permutation_group:
                simple_generators.append(p)
                permutation_group = PermutationGroup(*simple_generators)

                if permutation_group.order() == len(trafos):
                    return simple_generators, permutation_group

    assert False
