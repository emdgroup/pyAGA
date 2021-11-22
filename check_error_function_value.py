import numpy as np
from copy import deepcopy
import pickle
import os
from collections import defaultdict
import logging

from main import Norm
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)  # set to logging.INFO for less, to logging.DEBUG for more verbosity


def to_matrix(trafo):
    matrix = np.zeros((len(trafo), len(trafo)))
    for index, value in enumerate(trafo):
        if value is not None:
            matrix[index, value] = 1
        else:
            # Setting this to 0 is unnecessary, but I will leave
            # this line in case one wants to set it to some other value for
            # visualization.
            matrix[index, :] = 0
    return matrix


def error_value(norm, P, A):
    if norm == Norm.L_INFINITY:
        return np.max(np.abs(P@A - A@P))
    elif norm == Norm.L_1:
        return np.sum(np.abs(P@A - A@P))
    elif norm == Norm.L_2:
        return np.sum(np.square(P@A - A@P))


percentages = ["100", "99.0", "98.0", "75.0", "40.0", "20.0", "10.0", "5.0"]
worlds = ["one_letter_words_5x5", "one_letter_words_10x5", "two_letter_words_20x10"]
max_trafo = 50

for world in worlds:
    test_transformations_file = f"data/test_transformations_{world}.pickle"

    with open(test_transformations_file, "rb") as file:
        test_transformations = pickle.load(file)

    fig, axes = plt.subplots(ncols=3, figsize=(14, 6))
    fig.suptitle(world)

    concurrence_matrices = {}
    for percentage in percentages:
        concurrence_matrix_file = f"data/{world}_integers_concurrence_matrix_{percentage}.pickle"

        if not os.path.isfile(concurrence_matrix_file):
            continue

        with open(concurrence_matrix_file, "rb") as file:
            concurrence_matrices[percentage] = pickle.load(file)

    logger.info(f'==== {world} ====')

    correct_x = []
    correct_y = defaultdict(list)
    logger.info('Actual transformations...')
    for percentage, concurrence_matrix in concurrence_matrices.items():
        correct_x.extend([float(percentage)-0.25, ]*len(test_transformations))
        for norm in (Norm.L_INFINITY, Norm.L_1, Norm.L_2):
            correct_y[norm].extend([error_value(norm, to_matrix(trafo), concurrence_matrix) for trafo in test_transformations])

    almost_x = []
    almost_y = defaultdict(list)
    almost2_x = []
    almost2_y = defaultdict(list)

    for x, y, dx, trafos, title in zip(
        (almost_x, almost2_x),
        (almost_y, almost2_y),
        (0, 0.25),
        ((test_transformations[0], ), test_transformations[1:5]),
        ('identity', 'other')
    ):
        logger.info(f'Disturbed {title} transformations...')
        for trafo in trafos:
            for i in range(len(trafo)) if len(trafo) <= max_trafo else np.random.choice(len(trafo), size=max_trafo, replace=False):
                for j in range(i) if i <= max_trafo else np.random.choice(i, size=max_trafo, replace=False):
                    new_trafo = deepcopy(trafo)
                    new_trafo[i], new_trafo[j] = new_trafo[j], new_trafo[i]
                    if new_trafo not in test_transformations:
                        mat = to_matrix(new_trafo)

                        for percentage, concurrence_matrix in concurrence_matrices.items():
                            x.append(float(percentage) + dx)
                            for norm in (Norm.L_INFINITY, Norm.L_1, Norm.L_2):
                                y[norm].append(error_value(norm, mat, concurrence_matrix))
                    else:
                        logger.info('generated known trafo, skipping')

    for norm, ax in zip((Norm.L_INFINITY, Norm.L_1, Norm.L_2), axes):
        ax.scatter(correct_x, correct_y[norm], color="red", s=1, label="correct transformations")
        ax.scatter(almost_x, almost_y[norm], color="green", s=1, label="almost identity")
        ax.scatter(almost2_x, almost2_y[norm], color="blue", s=1, label="almost correct")

    for norm, ax in zip((Norm.L_INFINITY, Norm.L_1, Norm.L_2), axes):
        ax.set_title(norm)
        ax.set_xlabel('Percentage of Full Observation Set')
        ax.set_ylabel('||PA - AP||')
        ax.set_yscale('log')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + 0.2*box.height, box.width, 0.8*box.height])
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35))

    os.makedirs("plots", exist_ok=True)
    fig.savefig(f"plots/{world}.png")
    plt.close(fig)
