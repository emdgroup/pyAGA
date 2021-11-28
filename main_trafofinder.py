import numpy as np
from sympy.combinatorics import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup
from matplotlib import pyplot as plt

from transformation_finder import find_trafos
from mipsym.mip import Norm
from mipsym.tools import to_matrix, matshow, deviation_value
import pickle
from itertools import count

#world_name = "two_letter_words_20x10"
world_name = "one_letter_words_10x5"
percentage = "98.0"
integer_matrices = False
trafo_round_decimals = 4
trafo_fault_tolerance_ratio = 0.25
kde_bandwidth = 1e-3
use_integer_programming = True
quiet = False
norm = Norm.L_INFINITY
error_value_limit = 0.006


if integer_matrices:
    mat_filename = f"data/{world_name}_integers_concurrence_matrix_{percentage}.pickle"
else:
    mat_filename = f"data/{world_name}_concurrence_matrix_{percentage}.pickle"


with open(mat_filename, "rb") as correlation_matrix_file:
    print(f"Loading matrix {mat_filename}")
    correlation_matrix = np.transpose(pickle.load(correlation_matrix_file))
    # trafos = find_trafos(correlation_matrix, trafo_accuracy)
    num_variables = correlation_matrix.shape[0]
    trafos, average_matchrate_per_trafo = find_trafos(
        correlation_matrix,
        fault_tolerance=int(
            trafo_fault_tolerance_ratio
            * num_variables),
        round_decimals=trafo_round_decimals,
        quiet=False,
        bandwidth=kde_bandwidth,
        casename=world_name,
        norm=norm,
        use_integer_programming=use_integer_programming
    )

if not quiet:
    print(f"Total number of found trafos {len(trafos)}")
    for i, trafo in enumerate(trafos):
        matrix = to_matrix(trafo)
        print(f'Printing permutation number {i+1}')
        print(matshow(matrix))

# We try to find a small/minimal generating set for all valid-ish transformations as follows:
# For all found transformations p_i, we compute all powers p_i^k and see if |p_i^k A - A p_i^k|
# is larger than a pre-set error bound. If yes, we omit p, otherwise, we build a permutation
# group with all transformations found so far and see if p is already in there.
# If so, we skip p, otherwise we add it to the permutation group.
if not quiet:
    print('Trying to compute a small/minimal generating set for the found transformations...')

id = np.eye(len(correlation_matrix))
permutation_group_generators = []
# during the filtering below, we will take note of the values of |p_i^k A - A p_i^k| to
# plot a histogram in the end, which allows for more convenient tuning of the error_value_limit
deviation_values = []

for trafo in trafos:
    generator = to_matrix(trafo)
    current_power = generator
    is_valid = True
    for power in count(0):
        if np.allclose(current_power, id):
            break  # went through the full cycle of generator

        deviation = deviation_value(norm, current_power, correlation_matrix)
        deviation_values.append(deviation)
        if deviation > error_value_limit:
            print(f'Skipping a transformation due to deviation {deviation} > {error_value_limit}'
                  f' for power {power}:\n{matshow(generator)}')
            is_valid = False
            break

        current_power = current_power @ generator

    if is_valid:
        g = PermutationGroup(*permutation_group_generators)
        p = Permutation(trafo)
        if p not in g:
            permutation_group_generators.append(p)

if not quiet:
    print(f'Found generating set with {len(permutation_group_generators)} members:')
    for i, gen in enumerate(permutation_group_generators):
        print(f'\nG_{i} =')
        print(matshow(to_matrix(gen.array_form)))

    plt.hist(deviation_values, bins=len(deviation_values))
    plt.title('Histogram of |p_i^k A - A p_i^k| for all identified permutations')
    plt.axvline(x=error_value_limit, color='red')
    plt.show()
