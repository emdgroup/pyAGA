import numpy as np
import itertools

np.set_printoptions(threshold=np.inf)

# find all subsets of size n of a given observation, return them as equivalence classes modulo transformations
# If gap is provided, we also enforce that it needs to be in every subset
def find_subsets(size, observation, basic_patterns, trafos, gap = None):
    occurring_patterns = [p for p in basic_patterns if all(observation[i] == 1 for i in p)]
    subsets = [set().union(*l) for l in itertools.combinations(occurring_patterns, size)]
    if gap is not None:
        subsets = [s for s in subsets if gap in s]
    equiv_classes = list(set([frozenset([frozenset([t[i] for i in s]) for t in trafos]) for s in subsets]))
    return equiv_classes


def find_best_pattern(equiv_classes, observations):
    # 1st step: count for all potential patterns how often they appear across all observations
    number_of_appearances = []
    for equiv_class in equiv_classes:
        counter = 0
        for observation in observations:
            for entity in equiv_class:
                if all(observation[i] for i in entity):
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
                if list_of_gaps[i].intersection(entity) != set() and all(observation[i] for i in entity):
                    counter += 1
        number_of_appearances.append(counter)
    # same as before: retrieve the best pattern
    maximum = max(number_of_appearances)
    max_index = number_of_appearances.index(maximum)
    return equiv_classes[max_index]


# accomplishes two things at once: gives a list of all gaps and marks what's already been covered with -1
def find_gaps(observations, patterns):
    list_of_gaps = []
    for observation in observations:
        for pattern in patterns:
            if all(observation[i] for i in pattern):
                for i in pattern:
                    observation[i] = -1
        gaps = set()
        for i in range(len(observation)):
            if observation[i] == 1:
                gaps.add(i)
        list_of_gaps.append(gaps)
    return list_of_gaps


# this just needs to be executed once before we do try to find basic patterns
def find_observation_representatives(observations, trafos):
    observation_representatives = []
    while observations != set():
        print(len(observations))
        observation = observations.pop()
        observation_representatives.append(observation)
        observation_indices = set([i for i, value in enumerate(observation) if value != 0])
        set_of_transformed_observation_indices = set([tuple([trafo[i] for i in observation_indices]) for trafo in trafos])
        set_of_transformed_observation = set([tuple([1 if i in observation_indices else 0 for i in range(len(observation))])
                                           for observation_indices in set_of_transformed_observation_indices])
        observations = observations - set_of_transformed_observation
    return [list(observation) for observation in observation_representatives]


def find_basic_patterns(size, observations, trafos, output, basic_patterns, original_size):
    np.random.shuffle(observations)
    new_patterns = set()
    # find initial pattern
    output.write(', '.join("%s" % i for i in observations[0]) + "\n")
    output.write(', '.join("%s" % i for i in observations[0]) + "\n")
    equiv_classes = find_subsets(size, observations[0], basic_patterns, trafos)
    pattern = find_best_pattern(equiv_classes, observations)
    pattern_to_entity = [1 if i in list(pattern)[0] else 0 for i in range(len(observations[0]))]
    output.write(', '.join("%s" % i for i in pattern_to_entity) + "\n")
    print("finished first 3")
    new_patterns.add(pattern)
    #
    while True:
        np.random.shuffle(observations)
        list_of_gaps = find_gaps(observations, pattern)
        gap_index = next((i for i, gaps in enumerate(list_of_gaps) if gaps != set()), None)
        if gap_index is None:
            break
        print("another iteration")
        observation = observations[gap_index]
        original_observation = [1 if i != 0 else 0 for i in observation]
        # choose a gap
        gap = list(list_of_gaps[gap_index])[0]
        output.write(', '.join("%s" % i for i in original_observation) + "\n")
        output.write(', '.join("%s" % i for i in observation) + "\n")
        equiv_classes = find_subsets(size, original_observation, basic_patterns, trafos, gap)
        pattern = find_best_pattern_with_gaps(equiv_classes, observations, list_of_gaps)
        pattern_to_entity = [1 if i in list(pattern)[0] else 0 for i in range(len(observations[0]))]
        output.write(', '.join("%s" % i for i in pattern_to_entity) + "\n")
        new_patterns.add(pattern)
    new_basic_patterns = list(set().union(*new_patterns))
    return new_basic_patterns
