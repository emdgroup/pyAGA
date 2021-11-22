from pattern_finder import *
from transformation_finder import find_trafos
from verify_transformations import verify_transformations
import pickle

# world_name = "two_letter_words_20x10"
world_name = "one_letter_words_10x5"
percentage = "98.0"
integer_matrices = False
# world_name = "one_letter_words_10x5"
trafo_round_decimals = 4
trafo_fault_tolerance_ratio = 0.25
kde_bandwidth = 1e-3
use_integer_programming=True

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
        use_integer_programming=use_integer_programming
    )
    num_valid_trafos = verify_transformations(trafos, world_name)

    print(f"num_valid_trafos = {num_valid_trafos}")
    # pickle.dump(trafos, trafos_file)
