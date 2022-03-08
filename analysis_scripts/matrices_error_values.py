import glob
import pickle

import numpy as np
import matplotlib.pyplot as plt

from mipsym.mip import Norm


def calculate_error_values(testcase):
    perfect_data_filename = f"../data/{testcase}_concurrence_matrix_100.pickle"
    with open(perfect_data_filename, "rb") as file:
        perfect_data: np.ndarray = pickle.load(file)
    num_entries = perfect_data.shape[0]
    error_values = {}
    for norm in Norm:
        error_values_norm = []
        for percentage in percentages:
            matrix_filename = f"../data/{testcase}_concurrence_matrix_{percentage}.pickle"
            with open(matrix_filename, "rb") as file:
                matrix: np.ndarray = pickle.load(file)
            if norm == Norm.L_INFINITY:
                error = np.max(np.abs(matrix - perfect_data))
                color = "red"
            elif norm == Norm.L_1:
                error = np.sum(np.abs(matrix - perfect_data)) / num_entries**2
                color = "green"
            elif norm == Norm.L_2:
                error = np.sum((matrix - perfect_data) ** 2) / num_entries**2
                color = "blue"

            error_values_norm.append((percentage, error))
        error_values[norm] = error_values_norm
    return error_values



if __name__ == "__main__":
    fig, axes = plt.subplots(nrows=4, figsize=(10, 20))
    percentages = ["100", "99.9", "98.0", "95.0", "90.0", "85.0", "80.0", "75.0", "70.0", "65.0", "60.0", "55.0",
                   "50.0", "40.0", "30.0", "20.0", "10.0", "5.0"]
    testcases = ["two_letter_words_20x10", "two_letter_words_15x15_rotations",
                 "two_letter_words_no_axsym_15x15_rotations",
                 "two_letter_words_no_axsym_13x7_letters_indiv_colors"]

    error_values = {}
    for i in range(4):
        error_values[testcases[i]] = calculate_error_values(testcases[i])

    for i, testcase in enumerate(testcases):
        ax = axes[i]
        ax.set_title(testcase)
        ax.set_xlabel("percentages")
        ax.set_ylabel("error values (scaled)")
        for norm in Norm:
            graph = np.array(error_values[testcase][norm]).astype(float)
            if norm == Norm.L_1:
                scale_factor = 10
            elif norm == Norm.L_2:
                scale_factor = 1e4
            else:
                scale_factor = 1
            ax.plot(graph[:, 0], scale_factor*graph[:, 1], label = str(norm), marker = "o")
        ax.legend()
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.7)
    # plt.show()
    fig.savefig("error_values_all.pdf")
