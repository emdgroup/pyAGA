from pattern_finder import *
from transformation_finder import find_trafos

import copy
import pickle

world_name = "12x7_a_and_other_letter"
trafo_num_bins = 26
trafo_round_decimals = 4
trafo_fault_tolerance_ratio = 0.0
pattern_size = 2
level_bound = 5

try:
    with open("trafos_" + world_name + ".pickle", "rb") as trafos_file:
        trafos = pickle.load(trafos_file)
except (FileNotFoundError, EOFError):
    try:
        with open("correlation_matrix_" + world_name + ".pickle", "rb") as correlation_matrix_file:
            with open("trafos_" + world_name + ".pickle", "wb") as trafos_file:
                correlation_matrix = np.transpose(pickle.load(correlation_matrix_file))
                # trafos = find_trafos(correlation_matrix, trafo_accuracy)
                fault_tolerance_ratio = 0.0
                num_variables = correlation_matrix.shape[0]
                trafos = find_trafos(
                    correlation_matrix,
                    num_bins=trafo_num_bins,
                    fault_tolerance=int(trafo_fault_tolerance_ratio*num_variables),
                    round_decimals=trafo_round_decimals,
                    quiet=False,
                )
                pickle.dump(trafos, trafos_file)
    except FileNotFoundError:
        print("Please provide file correlation_matrix_" + world_name + ".pickle or trafos_" + world_name + ".pickle!")
        exit(-1)

try:
    with open("unique_observations_mod_trafos_" + world_name + ".pickle", "rb") as observations_mod_trafos_file:
        original_observations = pickle.load(observations_mod_trafos_file)
except FileNotFoundError:
    try:
        with open("unique_observations_" + world_name + ".pickle", "rb") as observations_file:
            with open("unique_observations_mod_trafos_" + world_name + ".pickle", "wb") as observations_mod_trafos_file:
                observations = np.transpose(pickle.load(observations_file))
                original_observations = find_observation_representatives(set([tuple(o) for o in observations]), trafos)
                pickle.dump(original_observations, observations_mod_trafos_file)
    except FileNotFoundError:
        print("Please provide file unique_observations_" + world_name + ".pickle or unique_observations_mod_trafos_"
              + world_name + ".pickle!")
        exit(-1)

try:
    original_size = len(original_observations[0])
except IndexError:
    print("The list observations you provided is empty!")
    exit(-1)

basic_patterns = [{i} for i in range(original_size)]

observations = [frozenset([i for i, value in enumerate(observation) if value != 0]) for observation in original_observations]

find_true_patternset(pattern_size, observations, trafos, basic_patterns, original_size)

#level = 0
#while True:
#    print("level " + str(level) + "...")
#    with open("plotting_data_lvl_" + str(level) + ".txt", "w") as output:
#        output.write('dimensions = (1, 12, 7); color_depth = 3; columns = 3; mode = "given_data";\n')
#        basic_patterns = list(set().union(*find_basic_patterns(pattern_size, observations, trafos, output, basic_patterns, original_size)))
#    print("level completed")
#    if level == level_bound:
#        break
#    level += 1
