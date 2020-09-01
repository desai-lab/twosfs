import numpy as np


def conditional_sfs(twosfs):
    F = np.cumsum(twosfs, axis=1)
    F /= F[:, -1][:, None]
    return F


def distance(F1, F2):
    return np.max(np.abs(F1 - F2), axis=1)


def resample_distance(sampling_dist, comparison_dist, n_obs, n_reps):
    F_exp = conditional_sfs(comparison_dist)
    D = np.zeros((n_reps, sampling_dist.shape[0]))
    for rep in range(n_reps):
        rand_counts = np.random.multinomial(
            n_obs, sampling_dist.ravel()).reshape(sampling_dist.shape)
        rand_counts = (rand_counts + rand_counts.T) / 2
        cumcounts = np.cumsum(rand_counts, axis=1)
        n_row = cumcounts[:, -1]
        F_obs = cumcounts / n_row[:, None]
        D[rep] = distance(F_exp, F_obs) * np.sqrt(n_row)
    return D


def rank(value, comparisons):
    return np.sum(value[:, None] > comparisons[None, :], axis=0)
