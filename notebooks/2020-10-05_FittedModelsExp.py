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
# %aimport twosfs.demographicmodel, twosfs.simulations, twosfs.spectra, twosfs.statistics # noqa

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import rel_entr

import twosfs.statistics as stats
from twosfs.demographicmodel import DemographicModel
from twosfs.spectra import load_spectra

# ## Import data

GS = [f"{g:.1f}" for g in [0.5, 1.0, 2.0]]
TS = [f"{t:.1f}" for t in [0.5, 1.0, 2.0]]
model = "3Epoch"
sample_size = 100
kmax = 20

params = [(g, t) for g in GS for t in TS]
print(params)

modelfn = "../simulations/fastNeutrino/expgrowth-g={0}-t={1}.{2}.txt"
expfn = "../simulations/msprime/expgrowth-g={0}-t={1}.npz"
fittedfn = "../simulations/msprime/fastNeutrino.expgrowth-g={0}-t={1}.{2}.npz"

data_kingman = load_spectra("../simulations/msprime/kingman.npz")
data_kingman.normalize()

demo_models = []
data_exp = []
data_fitted = []
for g, t in params:
    dm = DemographicModel(modelfn.format(g, t, model))
    dm.rescale()
    demo_models.append(dm)

    spectra = load_spectra(expfn.format(g, t))
    spectra.normalize()
    data_exp.append(spectra)
    spectra = load_spectra(fittedfn.format(g, t, model))
    spectra.normalize()
    data_fitted.append(spectra)

lumped_kingman = data_kingman.lumped_twosfs(kmax)
lumped_exp = [s.lumped_twosfs(kmax) for s in data_exp]
lumped_fitted = [s.lumped_twosfs(kmax) for s in data_fitted]

# ## Fitted demographic models

time = np.logspace(-2, 2, 100)
for (g, t), dm in zip(params, demo_models):
    plt.semilogx(time, dm.population_size(time), label=(g, t))
plt.legend(title="G, T")
plt.ylim([0.0, 3.5])
plt.ylabel("Population size (coal. units)")
plt.xlabel("Time (coal. units)")

# ## SFS comparisons

for param, dexp, dfitted in zip(params, data_exp, data_fitted):
    plt.loglog(data_kingman.sfs / data_kingman.sfs[1], "-k", label="Const-N")
    plt.loglog(dexp.sfs / dexp.sfs[1], "x", label="exp")
    plt.loglog(dfitted.sfs / dfitted.sfs[1], ".", label="fitted")
    plt.vlines(kmax, 5e-3, 1, color="0.5", linestyle="dashed", label="kmax")
    plt.title(param)
    plt.legend()
    plt.show()

# ## 2-SFS comparisons

# First, look at no recombination

d = 0
for param, dexp, dfitted in zip(params, lumped_exp, lumped_fitted):
    twosfs_exp = dexp[d, 1:, 1:]
    twosfs_fitted = dfitted[d, 1:, 1:]
    plt.pcolormesh(
        np.log2(twosfs_exp / twosfs_fitted), vmin=-0.5, vmax=0.5, cmap="PuOr_r"
    )
    plt.colorbar()
    plt.title(param)
    plt.show()

    print(np.sum(rel_entr(twosfs_fitted, twosfs_exp)))

for param, dexp, dfitted in zip(params, lumped_exp, lumped_fitted):
    twosfs_exp = dexp[d, 1:, 1:]
    F_exp = stats.conditional_sfs(twosfs_exp)
    twosfs_fitted = dfitted[d, 1:, 1:]
    F_fitted = stats.conditional_sfs(twosfs_fitted)
    D = stats.distance(F_exp, F_fitted)
    plt.plot(D)
    print(np.sum(D))

npairs = 10000
nresample = 1000
D_kingman = []
D_exp = []
for param, dexp, dfitted in zip(params, lumped_exp, lumped_fitted):
    twosfs_exp = dexp[d, 1:, 1:]
    twosfs_fitted = dfitted[d, 1:, 1:]
    D_kingman.append(
        stats.resample_distance(twosfs_fitted, twosfs_fitted, npairs, nresample)
    )
    D_exp.append(stats.resample_distance(twosfs_exp, twosfs_fitted, npairs, nresample))

bins = np.linspace(7, 23, 100)
for param, d_k, d_b in zip(params, D_kingman, D_exp):
    total_k = np.sum(d_k, axis=1)
    total_b = np.sum(d_b, axis=1)
    power = np.mean(stats.rank(total_b, total_k) > 0.95 * nresample)
    plt.hist(total_k, histtype="step", bins=bins)
    plt.hist(total_b, histtype="step", bins=bins)
    plt.title(param)
    plt.show()
    print(power)

power = []
for param, d_k, d_b in zip(params, D_kingman, D_exp):
    total_k = np.sum(d_k, axis=1)
    total_b = np.sum(d_b, axis=1)
    power.append(np.mean(stats.rank(total_b, total_k) > 0.95 * nresample))
print(power)
