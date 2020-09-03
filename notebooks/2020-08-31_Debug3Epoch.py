# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 1
# %aimport twosfs.demographicmodel, twosfs.simulations, twosfs.twosfs

import matplotlib.pyplot as plt
import msprime
import numpy as np
from scipy.special import kl_div

from twosfs.demographicmodel import DemographicModel
from twosfs.simulations import beta_timescale
from twosfs.twosfs import lump_sfs, sfs2pi, sims2pi, sims2sfs


def read_fastNeutrino_log(fn):
    onesfs_fitted = None
    onesfs_observed = None
    with open(fn, "r") as data:
        for line in data:
            if line.startswith("Expected  spectrum"):
                spectrum = data.readline()
                onesfs_fitted = np.array(spectrum.split(), dtype=float)
            if line.startswith("Observed  spectrum"):
                spectrum = data.readline()
                onesfs_observed = np.array(spectrum.split(), dtype=float)
    return onesfs_fitted, onesfs_observed


model = "3Epoch"
alphas = np.arange(1.5, 2.0, 0.05)
maxb = 10
modelfn = "../simulations/fastNeutrino/xibeta-alpha={0:.2f}.{1}.txt"
betafn = "../simulations/msprime/xibeta-alpha={0:.2f}.npz"
logfn = "../log/fastNeutrino.xibeta-alpha={0:.2f}.{1}.log"
models = [DemographicModel(modelfn.format(alpha, model)) for alpha in alphas]
sfs_beta = [np.load(betafn.format(alpha))["onesfs"] for alpha in alphas]
sfs_lumped = [lump_sfs(sfs, maxb) for sfs in sfs_beta]
sfs_fitted = [read_fastNeutrino_log(logfn.format(alpha, model))[0] for alpha in alphas]

for alpha, dm in zip(alphas, models):
    t = np.logspace(np.log10(dm.times[1] / 2), np.log10(2 * dm.times[-1]), 100)
    print(dm.t2())
    plt.loglog(t, dm.population_size(t), label=f"{alpha:.2f}")
plt.legend()

num_replicates = 1000
for alpha, dm in zip(alphas, models):
    demographic_events = dm.get_demographic_events()
    sims = msprime.simulate(
        sample_size=2,
        random_seed=1,
        num_replicates=num_replicates,
        demographic_events=demographic_events,
    )
    t2_sim = sims2pi(sims, num_replicates)
    t2_exp = dm.t2()
    rel_err = (t2_sim - t2_exp) / t2_exp
    print(alpha, t2_exp, t2_sim, rel_err, sep="\t")

k = np.arange(1, maxb + 1)
for alpha, beta, fitted in zip(alphas, sfs_lumped, sfs_fitted):
    plt.semilogy(k, beta[1:] / np.sum(beta), "xk")
    plt.semilogy(k, fitted / np.sum(fitted), ".")
    plt.title(f"{alpha:.2f}")
    plt.show()
    print(beta[1:] / np.sum(beta))
    print(fitted)

for alpha, sfs in zip(alphas, sfs_beta):
    print(alpha)
    print(f"Expected T2:\t{4*beta_timescale(alpha)}\nObserved:\t{sfs2pi(sfs)}")

num_replicates = 10000
sample_size = 100
sfs_sim = []
for alpha, dm in zip(alphas, models):
    demographic_events = dm.get_demographic_events()
    sims = msprime.simulate(
        sample_size=sample_size,
        num_replicates=num_replicates,
        demographic_events=demographic_events,
    )
    sfs_sim.append(sims2sfs(sims, sample_size, 1)[0])

k = np.arange(1, maxb + 1)
for alpha, simmed, fitted in zip(alphas, sfs_sim, sfs_fitted):
    simmed = lump_sfs(simmed, maxb)[1:]
    simmed /= np.sum(simmed)
    plt.semilogy(k, simmed, ".")
    plt.semilogy(k, fitted, "xk")
    plt.title(f"{alpha:.2f}")
    plt.show()
    print(np.sum(kl_div(simmed, fitted)))
