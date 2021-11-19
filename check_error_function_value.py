import copy
import numpy as np

import pandas
import pickle
import os

import matplotlib.pyplot as plt

def to_matrix(trafo):
    matrix = np.zeros((len(trafo), len(trafo)))
    for index, value in enumerate(trafo):
        if value is not None:
            matrix[index, value] = 1
        else:
            matrix[index, :] = 0 # Setting this to 0 is unnecessary, but I will leave
                                # this line in case one wants to set it to some other value for
                                # visualization.
    return matrix


percentages = ["100", "99.0", "98.0", "75.0"]
worlds = ["two_letter_words_20x10", "one_letter_words_10x5", "one_letter_words_5x5"]

for world in worlds:
    for percentage in percentages:
        concurrence_matrix_file = f"data/{world}_integers_concurrence_matrix_{percentage}.pickle"
        test_transformations_file = f"data/test_transformations_{world}.pickle"

        with open(test_transformations_file, "rb") as file:
            test_transformations = pickle.load(file)

        with open(concurrence_matrix_file, "rb") as file:
            concurrence_matrix = pickle.load(file)

        pd_list = []
        fig, ax = plt.subplots()
        points_x_correct = []
        points_y_correct = []
        for index, trafo in enumerate(test_transformations):
            mat = to_matrix(trafo)
            # error_value = np.linalg.norm(mat @ concurrence_matrix - concurrence_matrix @ mat)
            error_value = np.sum(np.abs(mat @ concurrence_matrix - concurrence_matrix @ mat))
            points_x_correct.append(index)
            points_y_correct.append(error_value)
        ax.scatter(points_x_correct, points_y_correct, color="red", s = 3, label = "correct transformations")

        rng = np.random.default_rng()
        almost_correct_transformations = []
        points_y_almost_correct = []
        for index, trafo in enumerate(test_transformations):
            a, b = rng.choice(np.arange(len(trafo)), size=2)
            almost_correct_trafo = copy.deepcopy(trafo)
            almost_correct_trafo[a] = trafo[b]
            almost_correct_trafo[b] = trafo[a]
            mat = to_matrix(almost_correct_trafo)
            # error_value = np.linalg.norm(mat @ concurrence_matrix - concurrence_matrix @ mat)
            error_value = np.sum(np.abs(mat @ concurrence_matrix - concurrence_matrix @ mat))
            points_y_almost_correct.append(error_value)

        ax.scatter(points_x_correct, points_y_almost_correct, color="green", s = 3, label = "almost correct transformations")


        almost_identity = []
        points_y_almost_identity = []
        n = len(test_transformations[0])
        print("got here")
        for i in range(n):
            for j in range(i):
                identity = test_transformations[0]
                # a, b = rng.choice(np.arange(len(trafo)), size=2)
                almost_correct_trafo = copy.deepcopy(identity)
                almost_correct_trafo[i] = identity[j]
                almost_correct_trafo[j] = identity[i]
                mat = to_matrix(almost_correct_trafo)
                # error_value = np.linalg.norm(mat @ concurrence_matrix - concurrence_matrix @ mat)
                error_value = np.sum(np.abs(mat @ concurrence_matrix - concurrence_matrix @ mat))
                points_y_almost_identity.append(error_value)
        print("and here")
        ax.scatter(np.arange(len(points_y_almost_identity)), points_y_almost_identity, color="orange", s = 3, label = "almost identity transformations")

        random_permutations = []
        points_y_wrong = []
        for i in range(len(test_transformations)):
            perm = rng.permutation(np.arange(len(test_transformations[0])))
            mat = to_matrix(perm)
            # error_value = np.linalg.norm(mat @ concurrence_matrix - concurrence_matrix @ mat)
            error_value = np.sum(np.abs(mat @ concurrence_matrix - concurrence_matrix @ mat))
            points_y_wrong.append(error_value)
        print(perm)


        ax.scatter(points_x_correct, points_y_wrong, color="blue", s = 3, label = "random permutations")
        ax.legend()
        os.makedirs("plots", exist_ok=True)
        fig.savefig(f"plots/{world}_{percentage}.pdf")
