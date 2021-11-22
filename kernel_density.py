import numpy as np
from sklearn.neighbors import KernelDensity
import pickle
import matplotlib.pyplot as plt
import scipy.signal as spsig


def coalesce_values(matrix):
    """
    Coalesce values of a given matrix in order to remove spread.
    :param matrix: Input matrix
    :return: The output matrix with coalesced values.
    """

    values, counts = np.unique(concurrence_matrix, return_counts=True)
    maxima = spsig.argrelmax(counts)[0]
    maxima_values = values[maxima]

    for index, value in enumerate(values):
        if index not in maxima:
            diffs = np.abs(maxima_values - value)
            pass


def bins(matrix, bandwidth, plot=False):

    bin_between_maxima = True
    # If this is True, bins are halfway-points between maxima. If this parameter is
    # False, then the bins are the minima locations

    ignore_zero = True
    ignore_self_concurrence = True

    # cutoff = np.mean(concurrence_matrix) + 4*np.std(concurrence_matrix)
    # concurrence_matrix = concurrence_matrix[concurrence_matrix < cutoff]
    #

    # coalesce_values(concurrence_matrix)

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    if ignore_zero:
        kde.fit(matrix[matrix > 0].reshape(-1, 1))
    else:
        kde.fit(matrix.reshape(-1, 1))
    #
    #
    # Multiply matrix.max() by 1.01 such that the peak for the self-concurrence is
    # well pronounced.
    plot_x = np.linspace(0, matrix.max() * 1.01, 1000)
    y = np.exp(kde.score_samples(plot_x.reshape(-1, 1)))
    minima_indices = spsig.argrelmin(y)
    minima_locations = plot_x[minima_indices]
    maxima_indices = spsig.argrelmax(y)
    maxima_locations = plot_x[maxima_indices]
    maxima_diffs = np.diff(maxima_locations)

    # for index, value in enumerate(maxima_locations):

    bins_with_maxima = maxima_locations[:-1] + maxima_diffs / 2

    if bin_between_maxima:
        bins = [1e-10] + bins_with_maxima.tolist()
    else:
        bins = [1e-10] + minima_locations.tolist() + [bins_with_maxima[-1]]
        # bins = [1e-10] + minima_locations.tolist()
    # bins = plot_x[minima_indices]
    print(f"bins = {bins}")

    if plot:
        fig, ax = plt.subplots(nrows=2, sharex=True)
        ax0, ax1 = ax
        ax0.plot(plot_x, y)
        for bin in bins:
            ax1.axvline(bin, color="red")
        # for mini in minima_locations:
        #     ax1.axvline(mini, color="green")
        if ignore_zero:
            ax1.hist(
                matrix[matrix > 0],
                bins=300,
            )
        else:
            ax1.hist(matrix, bins=300)

        plt.show()

    return bins

if __name__ == "__main__":
    with open(
        "data/two_letter_words_20x10_integers_concurrence_matrix_75.0.pickle", "rb"
    ) as file:
        concurrence_matrix = pickle.load(file)

    bins(concurrence_matrix, bandwidth=40)
