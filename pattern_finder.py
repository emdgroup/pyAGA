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


def find_basic_patterns(size, observations, trafos, output, basic_patterns, original_size):
    np.random.shuffle(observations)
    list_of_gaps = copy.deepcopy(observations)
    new_patterns = set()
    # find initial pattern
    equiv_classes = find_subsets(size, observations[0], basic_patterns, trafos)
    pattern = find_best_pattern(equiv_classes, observations)
    output.write(', '.join("%s" % i for i in convert_observation(list_of_gaps[0], original_size)) + "\n")
    output.write(', '.join("%s" % i for i in convert_observation(observations[0], original_size)) + "\n")
    output.write(', '.join("%s" % i for i in convert_observation(list(pattern)[0], original_size)) + "\n")
    print("finished first 3")
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
        print("another iteration")
        observation_with_gaps = list_of_gaps[gap_index]
        observation = observations[gap_index]
        gaps = list_of_gaps[gap_index]
        equiv_classes = find_subsets(size, observation, basic_patterns, trafos, gaps)
        pattern = find_best_pattern_with_gaps(equiv_classes, observations, list_of_gaps)
        output.write(', '.join("%s" % i for i in convert_observation(observation_with_gaps, original_size)) + "\n")
        output.write(', '.join("%s" % i for i in convert_observation(observation, original_size)) + "\n")
        output.write(', '.join("%s" % i for i in convert_observation(list(pattern)[0], original_size)) + "\n")
        new_patterns.add(pattern)
    new_basic_patterns = list(set().union(*new_patterns))
    return new_basic_patterns
