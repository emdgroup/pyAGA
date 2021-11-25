import numpy as np
from transformation_finder import find_trafos
from mipsym.tools import to_matrix, matshow
import pickle

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
error_value_limit = 0.007


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