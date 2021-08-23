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

import json
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from twosfs.demographicmodel import DemographicModel
from twosfs.spectra import Spectra, load_spectra
from twosfs.statistics import twosfs_test


def power(pvals, alpha=0.05):
    """Power at alpha=alpha."""
    return np.mean(pvals < alpha)


def filt(data: Dict, folded: bool, max_d: int, pair_density: int):
    return filter(
        lambda x: x["folded"] == folded
        and x["max_d"] == max_d
        and x["pair_density"] == pair_density,
        data,
    )


rec_rates = [f"{r:.2f}" for r in np.logspace(-1, 1, 4, base=2)] + ["1.00"]
alphas = [f"{a:0.2f}" for a in np.arange(1.05, 2.0, 0.05)]
growth_rates = [f"{g:.1f}" for g in [0.5, 1.0, 2.0]]
growth_times = [f"{t:.1f}" for t in [0.5, 1.0, 2.0]]
sims_alpha = [f"xibeta-alpha={alpha}" for alpha in alphas]
sims_exp = [f"expgrowth-g={g}-t={t}" for g in growth_rates for t in growth_times]
sims = sims_alpha + sims_exp

max_k = 20
folded = False
pair_densities = [100, 1000, 10000]
max_ds = [2, 6, 11, 16]

data: Dict[Tuple[str, Optional[str]], Dict] = {}
for sim in sims:
    with open(f"../simulations/pvalues/{sim}.3Epoch.json") as datafile:
        data[(sim, None)] = json.load(datafile)
    for r in rec_rates:
        with open(f"../simulations/pvalues/{sim}.3Epoch.rec={r}.json") as datafile:
            data[(sim, r)] = json.load(datafile)


for sim in sims:
    nonrec_data = data[(sim, None)]
    for i, pd in enumerate(pair_densities):
        for line in filt(nonrec_data, False, 16, pd):
            plt.hlines(power(np.array(line["p_vals"])), 0.5, 2.0, color=f"C{i}")

    for r in rec_rates:
        rec_data = data[(sim, r)]
        for i, pd in enumerate(pair_densities):
            for line in filt(rec_data, False, 16, pd):
                p = power(np.array(line["p_vals"]))
                plt.semilogx(float(r), p, ".", color=f"C{i}")
    plt.title(sim)
    plt.ylim([0, 1.05])
    plt.show()


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


spectra_dmel_data = load_spectra("../data/DPGP3/AllChroms.spectra.npz")
spectra_dmel_fitted = load_spectra("../data/DPGP3/msprime/AllChroms.3Epoch.npz")
d_dmel = [3, 6, 9, 12, 15]
folded_dmel = True
pvals_dmel = twosfs_test(
    spectra_dmel_data,
    spectra_dmel_fitted,
    d_dmel,
    d_dmel,
    max_k=20,
    folded=True,
    n_reps=1000,
    resample_comp=False,
)
print(pvals_dmel)

for offset in offsets:
    for pair_density in pair_densities:
        for i, max_d in enumerate(max_ds):
            for j, g in enumerate(growth_rates):
                for k, t in enumerate(growth_times):
                    params = (f"expgrowth-g={g}-t={t}", pair_density, max_d, offset)
                    plt.plot(j + 3 * k, power(pvals[params]), ".", color=f"C{i}")
        plt.ylim([0, 1])
        plt.hlines(0.05, 0, 8, "k")
        plt.title(f"{pair_density} {offset}")
        plt.show()


with open("output.json", "w") as outfile:
    json.dump([(key, pv.tolist()) for key, pv in pvals.items()], outfile)
