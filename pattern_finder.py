import copy

import numpy as np
import itertools

np.set_printoptions(threshold=np.inf)

# TODO: Perform conversion to index sets already here
# this just needs to be executed once before we do try to find basic patterns
def find_observation_representatives(observations, trafos):
    observation_representatives = []
    while observations != set():
        print(len(observations))
        observation = observations.pop()
        observation_representatives.append(observation)
        observation_indices = set([i for i, value in enumerate(observation) if value != 0])
        set_of_transformed_observation_indices = set(
            [tuple([trafo[i] for i in observation_indices]) for trafo in trafos])
        set_of_transformed_observation = set(
            [tuple([1 if i in observation_indices else 0 for i in range(len(observation))])
             for observation_indices in set_of_transformed_observation_indices])
        observations = observations - set_of_transformed_observation
    return [list(observation) for observation in observation_representatives]


# accomplishes two things at once: gives a list of all gaps and marks what's already been covered with -1
def find_gaps(observations, patterns, list_of_gaps):
    for i, observation in enumerate(observations):
        for pattern in patterns:
            if pattern.issubset(observation):
                list_of_gaps[i] = set(list_of_gaps[i]) - pattern


# find all subsets of size n of a given observation, return them as equivalence classes modulo transformations;
# if gap (an individual element) is provided, we enforce that it needs to be part of every subset
def find_subsets(size, observation, basic_patterns, trafos, gap=None):
    occurring_patterns = [p for p in basic_patterns if p.issubset(observation)]
    subsets = [set().union(*l) for l in itertools.combinations(occurring_patterns, size)]
    if gap is not None:
        subsets = [s for s in subsets if gap.intersection(s) != set()]
    equiv_classes = list(set([frozenset([frozenset([t[i] for i in s]) for t in trafos]) for s in subsets]))
    return equiv_classes


def find_best_pattern(equiv_classes, observations):
    # 1st step: count for all potential patterns how often they appear across all observations
    number_of_appearances = []
    for equiv_class in equiv_classes:
        counter = 0
        for observation in observations:
            for entity in equiv_class:
                if entity.issubset(observation):
                    counter += 1
        number_of_appearances.append(counter)
    # 2nd step: retrieve the pattern that appeared the most
    maximum = max(number_of_appearances)
    max_index = number_of_appearances.index(maximum)
    return equiv_classes[max_index]


# count number of appearances of patterns that cover at least one gap
def find_best_pattern_with_gaps(equiv_classes, observations, list_of_gaps):
    # as before: count how often patterns appear
    number_of_appearances = []
    for equiv_class in equiv_classes:
        counter = 0
        for i, observation in enumerate(observations):
            for entity in equiv_class:
                # only count when the pattern overlaps with at least one of the gaps
                if list_of_gaps[i].intersection(entity) != set() and entity.issubset(observation):
                    counter += 1
        number_of_appearances.append(counter)
    # 2nd step: retrieve the pattern that appeared the most
    maximum = max(number_of_appearances)
    max_index = number_of_appearances.index(maximum)
    return equiv_classes[max_index]


def convert_observation(observation, original_size):
    converted_observation = np.zeros(original_size)
    for i in observation:
        converted_observation[i] = 1
    return converted_observation


def find_basic_patterns(size, observations, trafos, basic_patterns, printing_stuff):
    original_size = printing_stuff[0]
    iteration = printing_stuff[1]
    level = printing_stuff[2]
    filename = "plotting_data_iteration_" + str(iteration) + "_lvl_" + str(level) + ".txt"
    with open(filename, "w") as output:
        output.write('dimensions = (1, 12, 7); color_depth = 3; columns = 3; mode = "given_data";\n')
    np.random.shuffle(observations)
    list_of_gaps = copy.deepcopy(observations)
    new_patterns = set()
    # find initial pattern
    equiv_classes = find_subsets(size, observations[0], basic_patterns, trafos)
    pattern = find_best_pattern(equiv_classes, observations)
    with open(filename, "a") as output:
        output.write(', '.join("%s" % i for i in convert_observation(list_of_gaps[0], original_size)) + "\n")
        output.write(', '.join("%s" % i for i in convert_observation(observations[0], original_size)) + "\n")
        output.write(', '.join("%s" % i for i in convert_observation(list(pattern)[0], original_size)) + "\n")
    new_patterns.add(pattern)
    while True:
        # shuffle observations
        c = list(zip(observations, list_of_gaps))
        np.random.shuffle(c)
        observations, list_of_gaps = zip(*c)
        list_of_gaps = list(list_of_gaps)
        # find observation with gap
        find_gaps(observations, pattern, list_of_gaps)
        gap_index = next((i for i, gaps in enumerate(list_of_gaps) if gaps != set()), None)
        if gap_index is None:
            break
        observation_with_gaps = list_of_gaps[gap_index]
        observation = observations[gap_index]
        gaps = list_of_gaps[gap_index]
        equiv_classes = find_subsets(size, observation, basic_patterns, trafos, gaps)
        pattern = find_best_pattern_with_gaps(equiv_classes, observations, list_of_gaps)
        with open(filename, "a") as output:
            output.write(', '.join("%s" % i for i in convert_observation(observation_with_gaps, original_size)) + "\n")
            output.write(', '.join("%s" % i for i in convert_observation(observation, original_size)) + "\n")
            output.write(', '.join("%s" % i for i in convert_observation(list(pattern)[0], original_size)) + "\n")
        new_patterns.add(pattern)
    return new_patterns


def find_true_patterns(equiv_classes, observations, trafos, original_size):
    true_patterns = []
    # for each potential true pattern (mod trafos)....
    for equiv_class in equiv_classes:
        # potential partners of pattern under transformations (which are not yet known)
        pattern_partners = []
        # have we found a reason why equiv_class cannot be a true pattern (mod trafos)? --> runtime optimizer
        abort = False
        # check for each observation which pattern_representatives appear
        for observation in observations:
            for pattern in equiv_class:
                # if pattern doesn't occur in observation, it's irrelevant
                if pattern.issubset(observation):
                    potential_partner = checker(observation, pattern, observations, trafos)
                    if potential_partner == set():
                        abort = True
                        break
                    pattern_partners.append(potential_partner)
            if abort:
                break
        if not abort:
            if set.intersection(*pattern_partners) != set() and len(pattern_partners) != 1:
                true_patterns.append(equiv_class)
    return true_patterns


def checker(observation, pattern, observations, trafos):
    subtracted_observation = observation - pattern
    # check if subtracted_observation exists in an observation different from 'observation'
    duplicated_subtracted_observation = [frozenset(trafo[i] for i in subtracted_observation) for trafo in trafos]
    potential_partners = set()
    for test_observation in observations:
        for subtracted_observation in duplicated_subtracted_observation:
            if subtracted_observation.issubset(test_observation) and test_observation != observation:
                potential_partners.add(test_observation - subtracted_observation)
    return potential_partners


def remove_true_patterns(observations, true_patterns):
    for i, observation in enumerate(observations):
        new_observation = observation
        for equiv_class in true_patterns:
            for pattern in equiv_class:
                if pattern.issubset(observation):
                    new_observation = new_observation - pattern
        observations[i] = new_observation



def find_true_patternset(size, observations, trafos, basic_patterns, original_size):
    true_patterns = []
    iteration = 0
    with open("true_patterns.txt", "w") as output:
        output.write('dimensions = (1, 12, 7); color_depth = 3; columns = 4; mode = "given_data"; height = 1000;\n')
    while not all([i == set() for i in observations]):
        print("ITERATION: " + str(iteration))
        level = 0
        while True:
            print("level " + str(level) + "...")
            new_patterns = find_basic_patterns(size, observations, trafos, basic_patterns, (original_size, iteration, level))
            true_patterns = find_true_patterns(new_patterns, observations, trafos, original_size)
            if true_patterns != []:
                print(str(len(true_patterns)) + " true pattern(s) found!")
                with open("true_patterns.txt", "a") as output:
                    for equiv_class in true_patterns:
                        output.write(', '.join("%s" % i for i in convert_observation(list(equiv_class)[0], original_size)) + "\n")
                remove_true_patterns(observations, true_patterns)
                basic_patterns = [{i} for i in range(original_size)]
                iteration += 1
                break
            else:
                level += 1
                basic_patterns = list(set().union(*new_patterns))

