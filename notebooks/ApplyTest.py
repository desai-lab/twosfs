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

from twosfs.spectra import load_spectra
from twosfs.statistics import twosfs_test

# +
alphas = [f"{a:0.2f}" for a in np.arange(1.05, 2.0, 0.05)]
growth_rates = [f"{g:.1f}" for g in [0.5, 1.0, 2.0]]
growth_times = [f"{t:.1f}" for t in [0.5, 1.0, 2.0]]

sims_alpha = [f"xibeta-alpha={alpha}" for alpha in alphas]
sims_exp = [f"expgrowth-g={g}-t={t}" for g in growth_rates for t in growth_times]
sims = sims_alpha + sims_exp

spectra_comps = {}
spectra_nulls = {}
for sim in sims:
    modelfn = f"../simulations/fastNeutrino/{sim}.3Epoch.txt"
    comparisonfn = f"../simulations/msprime/{sim}.npz"
    nullfn = f"../simulations/msprime/fastNeutrino.{sim}.3Epoch.npz"

    spectra_comps[sim] = load_spectra(comparisonfn)
    spectra_nulls[sim] = load_spectra(nullfn)

# +
d_comp = [1, 4, 7]
d_null = [1, 4, 7]
max_k = 20
folded = False
resample_comp = True

n_reps = 1000
num_pairs = [10000, 10000, 10000]
# -

pvals = {}
for sim in sims:
    pvals[sim] = twosfs_test(
        spectra_comps[sim],
        spectra_nulls[sim],
        d_comp,
        d_null,
        max_k,
        folded,
        n_reps,
        resample_comp,
        num_pairs,
    )


def power(pvals, alpha=0.05):
    """Power at alpha=alpha."""
    return np.mean(pvals < alpha)


for alpha in alphas:
    plt.plot(alpha, power(pvals[f"xibeta-alpha={alpha}"]), ".k")
plt.ylim([0, 1])
