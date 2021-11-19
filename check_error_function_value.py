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

percentages = ["100", "99.0", "98.0", "75.0", "40.0", "20.0", "10.0", "5.0"]
worlds = ["one_letter_words_5x5", "one_letter_words_10x5", "two_letter_words_20x10"]

for world in worlds:
    test_transformations_file = f"data/test_transformations_{world}.pickle"

    with open(test_transformations_file, "rb") as file:
        test_transformations = pickle.load(file)

    fig, axes = plt.subplots(ncols=3, figsize=(16, 6))
    fig.suptitle(world)

    concurrence_matrices = {}
    for percentage in percentages:
        concurrence_matrix_file = f"data/{world}_integers_concurrence_matrix_{percentage}.pickle"

        if not os.path.isfile(concurrence_matrix_file):
            continue

        with open(concurrence_matrix_file, "rb") as file:
            concurrence_matrices[percentage] = pickle.load(file)

    for norm, ax in zip((Norm.L_INFINITY, Norm.L_1, Norm.L_2), axes):
        print(world, norm)
        ax.set_title(norm)

        correct_x = []
        correct_y = []
        almost_x = []
        almost_y = []
        almost2_x = []
        almost2_y = []

        for percentage, concurrence_matrix in concurrence_matrices.items():
            correct_x.extend([float(percentage)-0.25, ]*len(test_transformations))
            correct_y.extend([error_value(norm, to_matrix(trafo), concurrence_matrix) for trafo in test_transformations])

        for x, y, dx, trafos in zip(
            (almost_x, almost2_x),
            (almost_y, almost2_y),
            (0, 0.25),
            ((test_transformations[0], ), test_transformations[1:])):
            for trafo in trafos:
                for i in range(len(trafo)):
                    mat_trafo = to_matrix(trafo)
                    for j in range(i):
                        mat = mat_trafo.copy()
                        mat[[i, j]] = mat[[j, i]]
                        for percentage, concurrence_matrix in concurrence_matrices.items():
                            x.append(float(percentage) + dx)
                            y.append(error_value(norm, mat, concurrence_matrix))

        ax.scatter(correct_x, correct_y, color="red", s=1, label="correct transformations")
        ax.scatter(almost_x, almost_y, color="green", s=1, label="almost identity transformations")
        ax.scatter(almost2_x, almost2_y, color="blue", s=1, label="almost correct transformations")

        ax.set_xlabel('Percentage of full observations')
        ax.set_ylabel('|PA - AP|')
        ax.legend()

    os.makedirs("plots", exist_ok=True)
    fig.savefig(f"plots/{world}.png")
