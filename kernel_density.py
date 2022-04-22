import logging
from typing import List

import numpy as np
from sklearn.neighbors import KernelDensity
import pickle
import matplotlib.pyplot as plt
import scipy.signal as spsig

logger = logging.getLogger("pyAGA_presolving")


def bins(matrix: np.ndarray, bandwidth: float, plot=False) -> List[float]:
    """
    Calculate the transformation-finder bins using kernel density estimation. For
    further details, see https://scikit-learn.org/stable/modules/generated/sklearn
    .neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity.
    :param matrix: The adjacency matrix of the graph.
    :param bandwidth: The bandwidth parameter controlling the regularity of the
    estimation.
    :param plot: Whether or not to show a plot.
    :return: The bin edges to use to bin the graph weights.
    """
    ignore_zero = True
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    if ignore_zero:
        kde.fit(matrix[matrix > 0].reshape(-1, 1))
    else:
        kde.fit(matrix.reshape(-1, 1))

    # Multiply matrix.max() by 1.01 such that the peak for the self-concurrence is
    # well pronounced.
    plot_x = np.linspace(0, matrix.max() * 1.01, 1000)
    y = np.exp(kde.score_samples(plot_x.reshape(-1, 1)))
    minima_indices = spsig.argrelmin(y)
    minima_locations = plot_x[minima_indices]
    maxima_indices = spsig.argrelmax(y)
    maxima_locations = plot_x[maxima_indices]
    maxima_diffs = np.diff(maxima_locations)

    bins_with_maxima = maxima_locations[:-1] + maxima_diffs / 2

    bins = [1e-10] + bins_with_maxima.tolist()
    # bins = plot_x[minima_indices]
    logger.debug(f"bins = {bins}")

    if plot:
        fig, ax = plt.subplots(nrows=2, sharex=True)
        ax0, ax1 = ax
        ax0.set_xlabel("concurrence $\langle x_1 x_2 \\rangle$")
        ax1.set_xlabel("concurrence $\langle x_1 x_2 \\rangle$")
        ax0.set_ylabel("value density $\\rho$")
        ax1.set_ylabel("value frequency")
        ax0.plot(plot_x, y)
        for bin in bins:
            ax1.axvline(bin, color="red")
        if ignore_zero:
            ax1.hist(
                matrix[matrix > 0],
                bins=300,
            )
        else:
            ax1.hist(matrix, bins=300)
        plt.show()
    return bins
