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
    distances: np.ndarray,
    folded: bool,
    sim_kwargs: dict,
    r_low: float,
    r_high: float,
    num_iters: int,
    n_reps: int,
    pair_density: np.ndarray,
) -> tuple[tuple[float, float, Spectra], tuple[float, float, Spectra]]:
    """Use golden section search to find the r that maximizes the p-value. In reality,
       we minimize 1 - p, which is equivalent."""
    # (r_l, r_u), ((ks_l, spec_l), (ks_u, spec_u)) = golden_section_search(
    #     simulate_ks, r_low, r_high, num_iters, spectra, k_max, distances, folded, **sim_kwargs
    # )
    (r_l, r_u), ((p_l, ks_l, ks_dist_l, spec_l), (p_u, ks_u, ks_dist_u, spec_u)) = golden_section_search(
        simulate_p_value, r_low, r_high, num_iters, spectra, k_max,
        distances, folded, n_reps, pair_density, **sim_kwargs
    )
    # return (r_l, ks_l, spec_l), (r_u, ks_u, spec_u)
    return (r_l, p_l, ks_l, ks_dist_l, spec_l), (r_u, p_u, ks_u, ks_dist_u, spec_u)


def search_recombination_rates_save(
    output_file,
    spectra: Spectra,
    k_max: int,
    distances: np.ndarray,
    folded: bool,
    sim_kwargs: dict,
    r_low: float,
    r_high: float,
    num_iters: int,
    n_reps: int,
    pair_density: np.ndarray,
) -> None:
    """
    Use golden section search to find the r that minimizes ks distance.

    Save output to a file in hdf5 format.
    """
    (r_l, p_l, ks_l, ks_dist_l, spec_l), (r_h, p_h, ks_h, ks_dist_h, spec_h) = search_recombination_rates(
        spectra, k_max, distances, folded, sim_kwargs, r_low, r_high, num_iters, n_reps, pair_density,
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
            attrs={"recombination_rate": r_l, "p_value": 1 - p_l, "ks_distance": ks_l},
        )
        spectra_to_hdf5(
            spec_h,
            f,
            "spectra_high",
            attrs={"recombination_rate": r_h, "p_value": 1 - p_h, "ks_distance": ks_h},
        )
    if (p_l, ks_l) > (p_h, ks_h):
        return r_h, 1 - p_h, ks_h, ks_dist_h, "high"
    else:
        return r_l, 1 - p_l, ks_l, ks_dist_l, "low"


def simulate_p_value(
    r: float, spectra: Spectra, k_max: int, distances: np.ndarray, folded: bool, n_reps: int, 
    num_pairs: np.ndarray, **simulation_kwargs
) -> tuple[float, Spectra]:
    """Simulate a Spectra and compute the p-value its 2-SFS came from the supplied Spectra."""
    ks, spectra_sim = simulate_ks(r, spectra, k_max, distances, folded, **simulation_kwargs)
    print(num_pairs)
    print(distances)
    ks_values = sample_ks_statistics(spectra_sim, spectra_sim, k_max, distances, folded, n_reps, num_pairs)
    p_value = 1 - sum(ks < ks_values) / n_reps
    return p_value, ks, ks_values, spectra_sim


def simulate_ks(
    r: float, spectra: Spectra, k_max: int, distances: np.ndarray, folded: bool, **simulation_kwargs
) -> tuple[float, Spectra]:
    """Simulate a Spectra and compute its KS distance to the supplied Spectra."""
    spectra_sim = simulate_spectra(scaled_recombination_rate=r, **simulation_kwargs)
    print("simulate ks")
    for i in range(5):
        print(np.round(spectra.twosfs[i, 1:5, 1:5],3))
        print(np.sum(spectra.twosfs[i]))
    twosfs_orig = reweight(
        twosfs_pdf(spectra, k_max, folded)[: len(spectra_sim.num_pairs)][distances],
        spectra.num_pairs[distances],
    )
    """
    print(distances)
    print("orig")
    for i in range(6):
        print(np.round(twosfs_orig[i, :4, :4], 3))
        print(np.sum(twosfs_orig[i]))\
    """
    twosfs_sim = reweight(
        twosfs_pdf(spectra_sim, k_max, folded)[distances],
        spectra.num_pairs[distances],
    )
    """
    print()
    print("sim")
    for i in range(6):
        print(np.round(twosfs_sim[i, :4, :4], 3))
        print(np.sum(twosfs_orig[i]))
    """
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
    lamb = 1 / golden
    x_l = a + (b - a) * (1 - lamb)
    f_l = f(x_l, *args, **kwargs)
    x_u = a + (b - a) * lamb
    f_u = f(x_u, *args, **kwargs)
    for i in range(num_iters):
        # if f_l[0] >= f_u[0]:
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
    return (x_l, x_u), (f_l, f_u)


def noise_spectra(spec, noise_level):
    """Add sequencing noise to a spectra object"""
    spec.onesfs[1] = spec.onesfs[1] * (1+noise_level)
    spec.twosfs[1,:] = spec.twosfs[1,:] * (1+noise_level)
    return spec

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
    return ret / np.sum(ret, axis = (1, 2))[:,None,None]


def resample_marginal_pdfs(pdfs: np.ndarray, n_obs: Iterable[int]) -> np.ndarray:
    """Resample 2D PDFs along the first axis of a 3D array."""
    return np.array([resample_pdf(pdf / np.sum(pdf), n) for pdf, n in zip(pdfs, n_obs)])


def reweight_and_symmetrize(pdf: np.ndarray, weights: Iterable[float]) -> np.ndarray:
    """Reweight 3D pdf along first dimension by weights and symmetrize."""
    ret = np.array([symmetrize(p) * w for p, w in zip(pdf, weights)])
    return ret / np.sum(ret)


def reweight(pdf: np.ndarray, weights: Iterable[float]) -> np.ndarray:
    """Reweight already symmetric 3D pdf along first dimension by weights"""
    return np.array([p * w for p, w in zip(pdf, weights)])


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
    distances: np.ndarray,
    folded: bool,
    n_reps: int,
    num_pairs: np.ndarray,
) -> np.ndarray:
    """Sample 2-SFS KS statistics between spectra_comp and spectra_null."""
    twosfs_comp = reweight(
        twosfs_pdf(spectra_comp, k_max, folded)[distances], num_pairs[distances]
    )
    twosfs_null = reweight(
        twosfs_pdf(spectra_null, k_max, folded)[distances], num_pairs[distances]
    )
    ks_values = np.zeros(n_reps)
    for i in range(n_reps):
        if i == 0:
            print(num_pairs)
        resampled = reweight(
            resample_marginal_pdfs(twosfs_comp, num_pairs[distances]), num_pairs[distances]
        )
        ks_values[i] = max_ks_distance(resampled, twosfs_null)
    return ks_values


def sample_ks_statistics_save(
    spectra_null: Spectra,
    k_max: int,
    distances: np.ndarray,
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
        distances,
        folded,
        n_reps,
        num_pairs,
    )
    with h5py.File(output_file, "w") as hf:
        data_null = hf.create_dataset("ks_null", data = ks_null)


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
            # num_pairs = pd * degenerate_pairs(spectra_comp, md)
            nonzero_dist = np.zeros(25)
            nonzero_dist[2:] = 1
            num_pairs = pd * nonzero_dist
            ks = sample_ks_statistics(
                spectra_comp, spectra_null, k_max, folded, n_reps, num_pairs
            )
            yield {
                "pair_density": pd,
                "max_distance": md,
                "ks_stats": list(ks),
            }
