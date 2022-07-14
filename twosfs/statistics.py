"""Functions for running statistical tests on twosfs."""
from functools import partial
from typing import Callable, Iterable, Iterator, Optional, Union

import h5py
import numpy as np
from scipy.constants import golden

from twosfs.simulations import simulate_spectra
from twosfs.spectra import Spectra, lump_twosfs, spectra_to_hdf5


def search_recombination_rates(
    spectra: Spectra,
    k_max: int,
    folded: bool,
    sim_kwargs: dict,
    r_low: float,
    r_high: float,
    num_iters: int,
) -> tuple[tuple[float, float, Spectra], tuple[float, float, Spectra]]:
    """Use golden section search to find the r that minimizes ks distance."""
    (r_l, r_u), ((ks_l, spec_l), (ks_u, spec_u)) = golden_section_search(
        simulate_ks, r_low, r_high, num_iters, spectra, k_max, folded, **sim_kwargs
    )
    return (r_l, ks_l, spec_l), (r_u, ks_u, spec_u)


def search_recombination_rates_save(
    output_file,
    spectra: Spectra,
    k_max: int,
    folded: bool,
    sim_kwargs: dict,
    r_low: float,
    r_high: float,
    num_iters: int,
) -> None:
    """
    Use golden section search to find the r that minimizes ks distance.

    Save output to a file in hdf5 format.
    """
    (r_l, ks_l, spec_l), (r_h, ks_h, spec_h) = search_recombination_rates(
        spectra, k_max, folded, sim_kwargs, r_low, r_high, num_iters
    )
    with h5py.File(output_file, "w") as f:
        spectra_to_hdf5(
            spectra,
            f,
            "spectra_target",
        )
        spectra_to_hdf5(
            spec_l,
            f,
            "spectra_low",
            attrs={"recombination_rate": r_l, "ks_distance": ks_l},
        )
        spectra_to_hdf5(
            spec_h,
            f,
            "spectra_high",
            attrs={"recombination_rate": r_h, "ks_distance": ks_h},
        )


def simulate_ks(
    r: float, spectra: Spectra, k_max: int, folded: bool, **simulation_kwargs
) -> tuple[float, Spectra]:
    """Simulate a Spectra and compute its KS distance to the supplied Spectra."""
    spectra_sim = simulate_spectra(scaled_recombination_rate=r, **simulation_kwargs)
    twosfs_orig = reweight_and_symmetrize(
        twosfs_pdf(spectra, k_max, folded)[: len(spectra_sim.num_pairs)],
        spectra.num_pairs,
    )
    twosfs_sim = reweight_and_symmetrize(
        twosfs_pdf(spectra_sim, k_max, folded),
        spectra.num_pairs,
    )
    return max_ks_distance(twosfs_orig, twosfs_sim), spectra_sim


def sample_onesfs(spectra: Spectra, num_sites: int, rng: Optional[np.random.Generator]):
    """Take a random sample of num_sites from the onesfs."""
    if rng:
        gen = rng
    else:
        gen = np.random.default_rng()
    return gen.binomial(num_sites, spectra.normalized_onesfs())


def sample_twosfs(
    spectra: Spectra, num_pairs: np.ndarray, rng: Optional[np.random.Generator]
):
    """Take a random sample of num_pairs from the twosfs."""
    if rng:
        gen = rng
    else:
        gen = np.random.default_rng()
    twosfs_sampled = np.zeros_like(spectra.twosfs)
    for i, (n, p) in enumerate(zip(num_pairs, spectra.normalized_twosfs())):
        twosfs_sampled[i] = gen.binomial(n, p)
    return twosfs_sampled


def sample_spectra(
    spectra: Spectra,
    num_sites: Optional[int] = None,
    num_pairs: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> Spectra:
    """Resample the one- and twosfs from a spectra. Return a new spectra."""
    if num_sites is None:
        num_sites_new = spectra.num_sites
        onesfs_new = spectra.onesfs.copy()
    else:
        num_sites_new = num_sites
        onesfs_new = sample_onesfs(spectra, num_sites, rng)
    if num_pairs is None:
        num_pairs_new = spectra.num_pairs.copy()
        twosfs_new = spectra.twosfs.copy()
    else:
        num_pairs_new = num_pairs
        twosfs_new = sample_twosfs(spectra, num_pairs, rng)
    return Spectra(
        num_samples=spectra.num_samples,
        windows=spectra.windows,
        recombination_rate=spectra.recombination_rate,
        num_sites=num_sites_new,
        num_pairs=num_pairs_new,
        onesfs=onesfs_new,
        twosfs=twosfs_new,
    )


def golden_section_search(
    f: Callable, a: float, b: float, num_iters: int, *args, **kwargs
):
    """Minimize a scalar function by golden section search."""
    print(a, b)
    lamb = 1 / golden
    x_l = a + (b - a) * (1 - lamb)
    f_l = f(x_l, *args, **kwargs)
    x_u = a + (b - a) * lamb
    f_u = f(x_u, *args, **kwargs)
    for i in range(num_iters):
        if f_l <= f_u:
            b = x_u
            x_u = x_l
            f_u = f_l
            x_l = a + (b - a) * (1 - lamb)
            f_l = f(x_l, *args, **kwargs)
        else:
            a = x_l
            x_l = x_u
            f_l = f_u
            x_u = a + (b - a) * lamb
            f_u = f(x_u, *args, **kwargs)
        print(a, b)
    return (x_l, x_u), (f_l, f_u)


def ks_distance(cdf1: np.ndarray, cdf2: np.ndarray) -> float:
    """Compute the KS distance between two CDFs."""
    return np.max(np.abs(cdf1 - cdf2))


def max_ks_distance(pdf1: np.ndarray, pdf2: np.ndarray) -> float:
    """Compute the maximum KS distance between two (multidimensional) PDFs."""
    return max(
        ks_distance(cdf1, cdf2) for cdf1, cdf2 in zip(_all_cdfs(pdf1), _all_cdfs(pdf2))
    )


def empirical_pvals(values: np.ndarray, comparisons: list[np.ndarray]):
    """Compute the rank of a value in an array of comparisons with pseudocounts."""
    return (1 + np.sum(comparisons > values, axis=0)) / (2 + len(comparisons))


def resample_pdf(pdf: np.ndarray, n_obs: int) -> np.ndarray:
    """Multinomial resample discrete pdf."""
    rand_counts = np.random.multinomial(n_obs, pdf.ravel()).reshape(pdf.shape)
    return rand_counts / np.sum(rand_counts)


def symmetrize(pdf: np.ndarray) -> np.ndarray:
    """Return a symmetrized version of a 2D pdf."""
    return (pdf + pdf.T) / 2


def twosfs_pdf(spectra: Spectra, k_max: int, folded: bool) -> np.ndarray:
    """Get the twosfs for segregating sites as a normalized 2D pdf."""
    ret = lump_twosfs(spectra.normalized_twosfs(folded=folded), k_max)[:, 1:, 1:]
    return ret / np.sum(ret)


def resample_marginal_pdfs(pdfs: np.ndarray, n_obs: Iterable[int]) -> np.ndarray:
    """Resample 2D PDFs along the first axis of a 3D array."""
    return np.array([resample_pdf(pdf / np.sum(pdf), n) for pdf, n in zip(pdfs, n_obs)])


def reweight_and_symmetrize(pdf: np.ndarray, weights: Iterable[float]) -> np.ndarray:
    """Reweight 3D pdf along first dimension by weights and symmetrize."""
    ret = np.array([symmetrize(p) * w for p, w in zip(pdf, weights)])
    return ret / np.sum(ret)


def degenerate_pairs(spectra: Spectra, max_distance: int) -> np.ndarray:
    """Return an array with ones at 4-fold degenerate distances up to max_distance."""
    ret = np.zeros(len(spectra.windows) - 1)
    for i in range(3, max_distance + 1, 3):
        ret[i] = 1
    return ret


def sample_ks_statistics(
    spectra_comp: Spectra,
    spectra_null: Spectra,
    k_max: int,
    folded: bool,
    n_reps: int,
    num_pairs: np.ndarray,
) -> np.ndarray:
    """Sample 2-SFS KS statistics between spectra_comp and spectra_null."""
    nonzero = num_pairs > 0
    np_nz = num_pairs[nonzero]
    max_d = num_pairs.shape[0]
    twosfs_comp = reweight_and_symmetrize(
        twosfs_pdf(spectra_comp, k_max, folded)[:max_d][nonzero], np_nz
    )
    twosfs_null = reweight_and_symmetrize(
        twosfs_pdf(spectra_null, k_max, folded)[:max_d][nonzero], np_nz
    )
    ks_values = np.zeros(n_reps)
    for i in range(n_reps):
        resampled = reweight_and_symmetrize(
            resample_marginal_pdfs(twosfs_comp, np_nz), np_nz
        )
        ks_values[i] = max_ks_distance(resampled, twosfs_null)
    return ks_values # * np.sqrt(sum(np_nz))


def sample_ks_statistics_save(
    spectra_null: Spectra,
    k_max: int,
    folded: bool,
    n_reps: int,
    num_pairs: np.ndarray,
    output_file,
) -> np.ndarray:
    """Save sampled 2-SFS KS statistics from spectra_comp and spectra_null."""
    ks_null = sample_ks_statistics(
        spectra_null,
        spectra_null,
        k_max,
        folded,
        n_reps,
        num_pairs,
    )
    """
    ks_comp = sample_ks_statistics(
        spectra_comp,
        spectra_null,
        k_max,
        folded,
        n_reps,
        num_pairs,
    )
    """
    with h5py.File(output_file, "w") as hf:
        data_null = hf.create_dataset("ks_null", data = ks_null)
        # data_comp = hf.create_dataset("ks_comp", data = ks_comp)


def _axis_combinations(n_dims: int) -> list[tuple]:
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


def _all_cdfs(pdf: np.ndarray) -> list[np.ndarray]:
    flips = [partial(np.flip, axis=axes) for axes in _axis_combinations(pdf.ndim)]
    return [flip(_cumsum_all_axes(flip(pdf))) for flip in flips]


def scan_parameters(
    spectra_comp: Spectra,
    spectra_null: Spectra,
    pair_densities: Iterable[int],
    max_distances: Iterable[int],
    k_max: int,
    folded: bool,
    n_reps: int,
) -> Iterator[dict[str, Union[int, list[float]]]]:
    """Compute resampled KS stats scanning over pair densities and max distances."""
    for pd in pair_densities:
        for md in max_distances:
            num_pairs = pd * degenerate_pairs(spectra_comp, md)
            ks = sample_ks_statistics(
                spectra_comp, spectra_null, k_max, folded, n_reps, num_pairs
            )
            yield {
                "pair_density": pd,
                "max_distance": md,
                "ks_stats": list(ks),
            }
