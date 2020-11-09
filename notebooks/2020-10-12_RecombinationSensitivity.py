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
# %aimport twosfs.spectra, twosfs.statistics

import matplotlib.pyplot as plt
import numpy as np

import twosfs.statistics as stats
from twosfs.spectra import load_spectra

# ## Import data

model = "3Epoch"
sample_size = 100
kmax = 20
r = 0.1  # Per-site rec rate

data_kingman = load_spectra("../simulations/msprime/kingman.npz")
data_kingman.normalize()

# +
GS = [f"{g:.1f}" for g in [0.5, 1.0, 2.0]]
TS = [f"{t:.1f}" for t in [0.5, 1.0, 2.0]]
params = [(g, t) for g in GS for t in TS]

expfn = "../simulations/msprime/expgrowth-g={0}-t={1}.npz"
expfittedfn = "../simulations/msprime/fastNeutrino.expgrowth-g={0}-t={1}.{2}.npz"

data_exp = []
data_exp_fitted = []
for g, t in params:

    spectra = load_spectra(expfn.format(g, t))
    spectra.normalize()
    data_exp.append(spectra)
    spectra = load_spectra(expfittedfn.format(g, t, model))
    spectra.normalize()
    data_exp_fitted.append(spectra)

# +
alphas = [f"{a:.2f}" for a in np.arange(1.05, 2.0, 0.05)]

betafn = "../simulations/msprime/xibeta-alpha={0}.npz"
betafittedfn = "../simulations/msprime/fastNeutrino.xibeta-alpha={0}.{1}.npz"

data_beta = []
data_beta_fitted = []
for alpha in alphas:
    spectra = load_spectra(betafn.format(alpha))
    spectra.normalize()
    data_beta.append(spectra)
    spectra = load_spectra(betafittedfn.format(alpha, model))
    spectra.normalize()
    data_beta_fitted.append(spectra)
# -

lumped_kingman = data_kingman.lumped_twosfs(kmax)
lumped_exp = [s.lumped_twosfs(kmax) for s in data_exp]
lumped_exp_fitted = [s.lumped_twosfs(kmax) for s in data_exp_fitted]
lumped_beta = [s.lumped_twosfs(kmax) for s in data_beta]
lumped_beta_fitted = [s.lumped_twosfs(kmax) for s in data_beta_fitted]

# ## 2-SFS comparisons

# ### Kingman at different recombination rates

# First, look at no recombination

num_windows = 50  # lumped_kingman.shape[0]
w = np.arange(num_windows)
npairs = 10000
nresample = 1000

power = np.zeros((num_windows, num_windows))
for d1 in w:
    for d2 in w:
        twosfs1 = lumped_kingman[d1, 1:, 1:]
        twosfs2 = lumped_kingman[d2, 1:, 1:]
        D1 = stats.resample_distance(twosfs1, twosfs1, npairs, nresample)
        total_1 = np.sum(D1, axis=1)
        D2 = stats.resample_distance(twosfs2, twosfs1, npairs, nresample)
        total_2 = np.sum(D2, axis=1)
        power[d1, d2] = np.mean(stats.rank(total_2, total_1) > 0.95 * nresample)

plt.pcolormesh(r * w, r * w, power, vmin=0, vmax=0.5, shading="auto")
plt.xlabel(r"Reference distance, $rT_2$")
plt.ylabel(r"Comparison distance, $rT_2$")
plt.colorbar()

plt.contour(r * w, r * w, power, levels=np.arange(0, 0.6, 0.1))
plt.xlabel(r"Reference distance, $rT_2$")
plt.ylabel(r"Comparison distance, $rT_2$")
plt.colorbar()

plt.plot(power.diagonal())

# ### Exponential at different recombination rates

# +
num_windows = 10
w = np.arange(num_windows)

for param, dexp, dfitted in zip(params, lumped_exp, lumped_exp_fitted):
    power_exp = np.zeros((num_windows, num_windows))
    for d1 in w:
        for d2 in w:
            twosfs1 = dfitted[d1, 1:, 1:]
            twosfs2 = dexp[d2, 1:, 1:]
            D1 = stats.resample_distance(twosfs1, twosfs1, npairs, nresample)
            total_1 = np.sum(D1, axis=1)
            D2 = stats.resample_distance(twosfs2, twosfs1, npairs, nresample)
            total_2 = np.sum(D2, axis=1)
            power_exp[d1, d2] = np.mean(stats.rank(total_2, total_1) > 0.95 * nresample)

    plt.contour(r * w, r * w, power_exp, levels=np.arange(0, 0.55, 0.05))
    plt.xlabel(r"Reference distance, $rT_2$")
    plt.ylabel(r"Comparison distance, $rT_2$")
    plt.title(param)
    plt.colorbar()
    plt.show()

# +
num_windows = 20
w = np.arange(num_windows)

for param, dexp, dfitted in zip(params, lumped_exp, lumped_exp_fitted):
    power_exp = np.zeros((num_windows, num_windows))
    for d1 in w:
        for d2 in w:
            twosfs1 = dfitted[d1, 1:, 1:]
            twosfs2 = dexp[d2, 1:, 1:]
            D1 = stats.resample_distance(twosfs1, twosfs1, npairs, nresample)
            total_1 = np.sum(D1, axis=1)
            D2 = stats.resample_distance(twosfs2, twosfs1, npairs, nresample)
            total_2 = np.sum(D2, axis=1)
            power_exp[d1, d2] = np.mean(stats.rank(total_2, total_1) > 0.95 * nresample)

    plt.contour(r * w, r * w, power_exp, levels=np.arange(0, 0.55, 0.05))
    plt.xlabel(r"Reference distance, $rT_2$")
    plt.ylabel(r"Comparison distance, $rT_2$")
    plt.title(param)
    plt.colorbar()
    plt.show()
# -

# ### Beta comparisons

# +
num_windows = 10
w = np.arange(num_windows)

for alpha, dbeta, dfitted in zip(alphas, lumped_beta, lumped_beta_fitted):
    power_beta = np.zeros((num_windows, num_windows))
    for d1 in w:
        for d2 in w:
            twosfs1 = dfitted[d1, 1:, 1:]
            twosfs2 = dbeta[d2, 1:, 1:]
            D1 = stats.resample_distance(twosfs1, twosfs1, npairs, nresample)
            total_1 = np.sum(D1, axis=1)
            D2 = stats.resample_distance(twosfs2, twosfs1, npairs, nresample)
            total_2 = np.sum(D2, axis=1)
            power_beta[d1, d2] = np.mean(
                stats.rank(total_2, total_1) > 0.95 * nresample
            )

    plt.pcolor(
        r * w, r * w, power_beta, shading="auto", vmin=0.0, vmax=1.0
    )  # , levels=np.arange(0, 0.55, 0.05))
    plt.xlabel(r"Reference distance, $rT_2$")
    plt.ylabel(r"Comparison distance, $rT_2$")
    plt.title(alpha)
    plt.colorbar()
    plt.show()

    plt.plot(r * w, power_beta.diagonal())
    plt.ylim([0, 1.05])
    plt.title(alpha)
    plt.show()
