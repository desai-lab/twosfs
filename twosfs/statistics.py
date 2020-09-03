"""Functions for running statistical tests on twosfs."""
import numpy as np


def conditional_sfs(twosfs):
    """Compute the conditional cumulative distributions from a 2SFS."""
    F = np.cumsum(twosfs, axis=1)
    F /= F[:, -1][:, None]
    return F


def distance(F1, F2):
    """Compute the KS distance between two distribution functions."""
    return np.max(np.abs(F1 - F2), axis=1)


def resample_distance(sampling_dist, comparison_dist, n_obs: int, n_reps: int):
    """
    Compute distances between a distribution and random empirical CDF.

    Parameters
    ----------
    sampling_dist : ndarray
        Discrete distribution to take multinomial samples from
    comparison_dist : ndarray
        Distribution to compute distances from
    n_obs : int
        Number of samples in the multinomial distribution
    n_reps : int
        Number of multinomial samples to take

    Returns
    -------
    ndarray
        The distances between the random draws and comparison_dist
    """
    F_exp = conditional_sfs(comparison_dist)
    D = np.zeros((n_reps, sampling_dist.shape[0]))
    for rep in range(n_reps):
        rand_counts = np.random.multinomial(n_obs, sampling_dist.ravel()).reshape(
            sampling_dist.shape
        )
        rand_counts = (rand_counts + rand_counts.T) / 2
        cumcounts = np.cumsum(rand_counts, axis=1)
        n_row = cumcounts[:, -1]
        F_obs = cumcounts / n_row[:, None]
        D[rep] = distance(F_exp, F_obs) * np.sqrt(n_row)
    return D


def rank(value, comparisons):
    """Compute the rank of a value in a list of comparisons."""
    return np.sum(value[:, None] > comparisons[None, :], axis=0)
