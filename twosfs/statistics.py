"""Functions for running statistical tests on twosfs."""
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.stats import cauchy, chi2

from twosfs.spectra import Spectra, lump_twosfs


def twosfs_pdf(
    spectra: Spectra, dist: Union[int, List[int]], max_k: int, folded: bool
) -> np.ndarray:
    """Get the twosfs for segregating sites as a normalized 2D pdf."""
    return lump_twosfs(spectra.normalized_twosfs(folded=folded), max_k)[dist, 1:, 1:]


def fisher_test(p_values: np.ndarray) -> Tuple[float, float]:
    """Compute the Fisher's test statistic and p-value for an array of p-values."""
    fisher_stat = -2 * np.sum(np.log(p_values))
    df = 2 * len(p_values)
    fisher_p = chi2.sf(fisher_stat, df=df)
    return fisher_stat, fisher_p


def cauchy_test(p_values: np.ndarray) -> Tuple[float, float]:
    """Compute the Cauchy combination test stat and p-value for an array of p-values."""
    cauchy_stat = np.mean(np.tan(0.5 - p_values) - np.pi)
    cauchy_p = cauchy.sf(cauchy_stat)
    return cauchy_stat, cauchy_p


def ks_distance(cdf1: np.ndarray, cdf2: np.ndarray) -> float:
    """Compute the KS distance between two CDFs."""
    return np.max(np.abs(cdf1 - cdf2))


def max_ks_distance(pdf1: np.ndarray, pdf2: np.ndarray) -> float:
    """Compute the maximum KS distance between two (multidimensional) PDFs."""
    return max(
        ks_distance(cdf1, cdf2) for cdf1, cdf2 in zip(_all_cdfs(pdf1), _all_cdfs(pdf2))
    )


def empirical_pvals(values: np.ndarray, comparisons: List[np.ndarray]):
    """Compute the rank of a value in an array of comparisons with pseudocounts."""
    return (1 + np.sum(comparisons > values, axis=0)) / (2 + len(comparisons))


def resample_twosfs_pdf(input_twosfs_pdf: np.ndarray, n_obs: np.ndarray) -> np.ndarray:
    """Multinomial sample the twosfs pdf, symmetrize, and normalize."""
    sampled_pdf = np.zeros_like(input_twosfs_pdf)
    for i, pdf in enumerate(input_twosfs_pdf):
        rand_counts = np.random.multinomial(n_obs[i], pdf.ravel()).reshape(pdf.shape)
        sampled_pdf[i] = (rand_counts + rand_counts.T) / (2 * np.sum(rand_counts))
    return sampled_pdf


def twosfs_test(
    spectra_comp: Spectra,
    spectra_null: Spectra,
    d_comp: List[int],
    d_null: List[int],
    max_k: int,
    folded: bool,
    n_reps: int,
    resample_comp: bool,
    num_pairs: Optional[List[int]] = None,
) -> np.ndarray:
    """Return array of p-values from the twosfs test. For power calculations."""
    twosfs_comp = twosfs_pdf(spectra_comp, d_comp, max_k, folded)
    twosfs_null = twosfs_pdf(spectra_null, d_null, max_k, folded)
    if not num_pairs:
        num_pairs = spectra_comp.num_pairs[d_comp]
    if resample_comp:
        samples_comp = [
            resample_twosfs_pdf(twosfs_comp, num_pairs) for rep in range(n_reps)
        ]
    else:
        samples_comp = [twosfs_comp]
    samples_null = [
        resample_twosfs_pdf(twosfs_null, num_pairs) for rep in range(n_reps)
    ]
    ks_comp = [
        np.array([max_ks_distance(s, n) for s, n in zip(sample, twosfs_null)])
        for sample in samples_comp
    ]
    ks_null = [
        np.array([max_ks_distance(s, n) for s, n in zip(sample, twosfs_null)])
        for sample in samples_null
    ]
    p_comp = [empirical_pvals(samp, ks_null) for samp in ks_comp]
    return np.array([fisher_test(samp)[1] for samp in p_comp])


def _axis_combinations(n_dims: int) -> List[Tuple]:
    if n_dims <= 0:
        return [()]
    else:
        sublist = _axis_combinations(n_dims - 1)
        return sublist + [(*t, n_dims - 1) for t in sublist]


def _cumsum_all_axes(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    if axis is None:
        axis = x.ndim - 1
    if axis < 0:
        return x
    else:
        return np.cumsum(_cumsum_all_axes(x, axis=axis - 1), axis=axis)


def _all_cdfs(pdf: np.ndarray) -> List[np.ndarray]:
    flips = [partial(np.flip, axis=axes) for axes in _axis_combinations(pdf.ndim)]
    return [flip(_cumsum_all_axes(flip(pdf))) for flip in flips]


def scan_parameters(
    spectra_comp: Spectra,
    spectra_null: Spectra,
    pair_densities: List[int],
    max_ds: List[int],
    max_k: int,
    n_reps: int,
) -> List[Dict[str, Any]]:
    """Scan parameters and compute `n_reps` pvalues for each."""
    results = []
    for folded in [True, False]:
        for pair_density in pair_densities:
            for max_d in max_ds:
                d = np.arange(1, max_d)
                num_pairs = [pair_density] * len(d)
                p_vals = twosfs_test(
                    spectra_comp,
                    spectra_null,
                    d,
                    d,
                    max_k,
                    folded,
                    n_reps,
                    True,
                    num_pairs,
                )
                results.append(
                    {
                        "folded": folded,
                        "max_d": max_d,
                        "pair_density": pair_density,
                        "p_vals": p_vals.tolist(),
                    }
                )
    return results
