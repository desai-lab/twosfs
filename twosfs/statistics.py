"""Functions for running statistical tests on twosfs."""
import numpy as np

from twosfs.spectra import Spectra, lump_twosfs


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


def compare(
    spectra_data: Spectra,
    spectra_fitted: Spectra,
    d1: int,
    d2: int,
    max_k: int,
    n_reps: int,
    folded: bool,
):
    """Compare two 2-SFS by resampling the distance statistic.

    Parameters
    ----------
    spectra_data : Spectra
        The observed spectra.
    spectra_fitted : Spectra
        The simulated spectra of the fitted model.
    d1 : int
        The distance between pairs of sites in the data.
    d2 : int
        The distance between pairs of sites in the fitted model.
    max_k : int
        The largest allele frequency to consider.
    n_reps : int
        The number of times to resample from the simulated 2-SFS.
    folded : bool
        If True, fold the 2-SFS

    Returns
    -------
    D : np.ndarray
        The observed normalized KS distances as a function of allele frequency.
    resamples : np.ndarray
        The observed normalized KS distances for each resample

    """
    twosfs_data = spectra_data.normalized_twosfs(folded=folded)[d1, 1:max_k, 1:max_k]
    twosfs_fitted = spectra_fitted.normalized_twosfs(folded=folded)[
        d2, 1:max_k, 1:max_k
    ]
    F_data = conditional_sfs(twosfs_data)
    F_fitted = conditional_sfs(twosfs_fitted)
    n_obs = np.sum(spectra_data.twosfs[d1, 1:max_k, 1:max_k], axis=1) / 2
    D = distance(F_data, F_fitted)
    resamples = resample_distance(
        twosfs_fitted,
        twosfs_fitted,
        spectra_data.num_pairs[d1],
        n_reps,
    )
    return D * np.sqrt(n_obs), resamples


def compare2(
    spectra_comparison: Spectra,
    spectra_null: Spectra,
    d1: int,
    d2: int,
    max_k: int,
    n_reps: int,
    num_pairs: int,
    folded: bool,
):
    """Compare two simulated 2-SFS by resampling the distance statistic in both.

    Parameters
    ----------
    spectra_comparison : Spectra
        The spectra for comparison.
    spectra_null : Spectra
        The null spectra.
    d1 : int
        The distance between pairs of sites in the comparison spectra.
    d2 : int
        The distance between pairs of sites in the null spectra.
    max_k : int
        The largest allele frequency to consider.
    n_reps : int
        The number of times to resample from the simulated 2-SFS.
    num_pairs : int
        The number of pairs of sites to resample.
    folded : bool
        If True, fold the 2-SFS

    Returns
    -------
    resamples_comparison : np.ndarray
        The observed normalized KS distances between comparison and null
    resamples_null : np.ndarray
        The observed normalized KS distances between null and null

    """
    twosfs_comparison = lump_twosfs(
        spectra_comparison.normalized_twosfs(folded=folded), max_k
    )[d1, 1:, 1:]
    twosfs_null = lump_twosfs(spectra_null.normalized_twosfs(folded=folded), max_k)[
        d1, 1:, 1:
    ]
    resamples_comparison = resample_distance(
        twosfs_comparison,
        twosfs_null,
        num_pairs,
        n_reps,
    )
    resamples_null = resample_distance(
        twosfs_null,
        twosfs_null,
        num_pairs,
        n_reps,
    )
    return resamples_comparison, resamples_null
