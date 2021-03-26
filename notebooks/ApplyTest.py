# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 1
# %aimport twosfs.demographicmodel, twosfs.spectra, twosfs.statistics

import matplotlib.pyplot as plt
import numpy as np

from twosfs.demographicmodel import DemographicModel
from twosfs.spectra import Spectra, load_spectra
from twosfs.statistics import twosfs_test


def power(pvals, alpha=0.05):
    """Power at alpha=alpha."""
    return np.mean(pvals < alpha)


def compute_pvals(
    spectra_comp: Spectra,
    spectra_null: Spectra,
    pair_density: int,
    max_d: int,
    max_k: int,
    folded: bool,
    offset: int = 0,
    n_reps: int = 1000,
):
    """Compute the p-vals for a set of parameter."""
    if offset >= 0:
        d_comp = np.arange(1, max_d)
        d_null = np.arange(1, max_d) + offset
    else:
        d_comp = np.arange(1, max_d) + offset
        d_null = np.arange(1, max_d)
    num_pairs = [pair_density] * len(d_comp)
    return twosfs_test(
        spectra_comp,
        spectra_null,
        d_comp,
        d_null,
        max_k,
        folded,
        n_reps,
        True,
        num_pairs,
    )


# +
alphas = [f"{a:0.2f}" for a in np.arange(1.05, 2.0, 0.05)]
growth_rates = [f"{g:.1f}" for g in [0.5, 1.0, 2.0]]
growth_times = [f"{t:.1f}" for t in [0.5, 1.0, 2.0]]

sims_alpha = [f"xibeta-alpha={alpha}" for alpha in alphas]
sims_exp = [f"expgrowth-g={g}-t={t}" for g in growth_rates for t in growth_times]
sims = sims_alpha + sims_exp

models = {}
spectra_comps = {}
spectra_nulls = {}
for sim in sims:
    modelfn = f"../simulations/fastNeutrino/{sim}.3Epoch.txt"
    comparisonfn = f"../simulations/msprime/{sim}.npz"
    nullfn = f"../simulations/msprime/fastNeutrino.{sim}.3Epoch.npz"

    dm = DemographicModel(modelfn)
    dm.rescale()
    models[sim] = dm
    spectra_comps[sim] = load_spectra(comparisonfn)
    spectra_nulls[sim] = load_spectra(nullfn)

# +
max_k = 20
folded = False
pair_densities = [100, 1000, 10000]
max_ds = [2, 6, 11, 16]
offsets = [-3, -1, 0, 1, 3]

pvals = {}
for sim in sims:
    print(sim)
    for pair_density in pair_densities:
        for max_d in max_ds:
            for offset in offsets:
                params = (sim, pair_density, max_d, offset)
                pvals[params] = compute_pvals(
                    spectra_comps[sim],
                    spectra_nulls[sim],
                    pair_density,
                    max_d,
                    max_k,
                    folded,
                    offset,
                )
# -

for pair_density in pair_densities:
    for i, max_d in enumerate(max_ds):
        for alpha in alphas:
            params = (f"xibeta-alpha={alpha}", pair_density, max_d, 0)
            plt.plot(alpha, power(pvals[params]), ".", color=f"C{i}")
    plt.ylim([0, 1])
    plt.hlines(0.05, 0, 18, "k")
    plt.title(f"{pair_density}")
    plt.show()

for pair_density in pair_densities:
    for i, max_d in enumerate(max_ds):
        for j, g in enumerate(growth_rates):
            for k, t in enumerate(growth_times):
                params = (f"expgrowth-g={g}-t={t}", pair_density, max_d, 0)
                plt.plot(j + 3 * k, power(pvals[params]), ".", color=f"C{i}")
    plt.ylim([0, 1])
    plt.hlines(0.05, 0, 8, "k")
    plt.title(f"{pair_density}")
    plt.show()
