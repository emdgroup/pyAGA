import numpy as np
import pickle
import os

from main import Norm
import matplotlib.pyplot as plt


def to_matrix(trafo):
    matrix = np.zeros((len(trafo), len(trafo)))
    for index, value in enumerate(trafo):
        if value is not None:
            matrix[index, value] = 1
        else:
            matrix[index, :] = 0  # Setting this to 0 is unnecessary, but I will leave
                                  # this line in case one wants to set it to some other value for
                                  # visualization.
    return matrix

def error_value(norm, P, A):
    if norm == Norm.L_INFINITY:
        return np.max(np.abs(P@A - A@P))
    elif norm == Norm.L_1:
        return np.sum(np.abs(P@A - A@P))
    elif norm == Norm.L_2:
        return np.sum(np.square(P@A - A@P))

percentages = ["100", "99.0", "98.0", "75.0"]
worlds = ["one_letter_words_5x5", "one_letter_words_10x5", "two_letter_words_20x10"]

for world in worlds:
    fig, axes = plt.subplots(ncols=3, figsize=(16, 6))
    fig.suptitle(world)

    for norm, ax in zip((Norm.L_INFINITY, Norm.L_1, Norm.L_2), axes):
        print(world, norm)
        ax.set_title(norm)

        for percentage in percentages:
            concurrence_matrix_file = f"data/{world}_integers_concurrence_matrix_{percentage}.pickle"
            test_transformations_file = f"data/test_transformations_{world}.pickle"

            with open(test_transformations_file, "rb") as file:
                test_transformations = pickle.load(file)

            with open(concurrence_matrix_file, "rb") as file:
                concurrence_matrix = pickle.load(file)

            points_y_correct = [error_value(norm, to_matrix(trafo), concurrence_matrix) for trafo in test_transformations]
            ax.scatter([float(percentage)-0.2, ]*len(points_y_correct), points_y_correct, color="red", s=1, label="correct transformations")

            points_almost_correct = []
            for trafo in (test_transformations[0], ):  # could also loop over all trafors here instead of only the first one (which is identity)
                for i in range(len(trafo)):
                    for j in range(i):
                        mat = to_matrix(trafo)
                        mat[[i, j]] = mat[[j, i]]

                        points_almost_correct.append(error_value(norm, mat, concurrence_matrix))

            ax.scatter([float(percentage), ]*len(points_almost_correct), points_almost_correct, color="orange", s=1, label="almost correct transformations")

            rng = np.random.default_rng()
            points_y_wrong = [error_value(norm, to_matrix(rng.permutation(np.arange(len(test_transformations[0])))), concurrence_matrix) for _ in range(len(test_transformations))]
            ax.scatter([float(percentage)+0.2, ]*len(points_y_wrong), points_y_wrong, color="blue", s=1, label="random permutations")
            #ax.legend()

        os.makedirs("plots", exist_ok=True)
        fig.savefig(f"plots/{world}.png")
