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

import twosfs.statistics as stats
from twosfs import load_spectra

spectra_data = load_spectra("../data/DPGP3/AllChroms.spectra.npz")
spectra_fitted = load_spectra("../data/DPGP3/msprime/AllChroms.3Epoch.npz")

plt.loglog(spectra_data.normalized_onesfs(folded=True))
plt.loglog(spectra_fitted.normalized_onesfs(folded=True))

spectra_data.num_pairs

spectra_fitted.num_pairs

d = 3
max_folded = spectra_data.num_samples // 2 + 1
n_reps = 1000
obs, resamples = stats.compare(spectra_data, spectra_fitted, d, d, max_folded, n_reps)

plt.figure(figsize=(10, 3))
plt.violinplot(resamples, showextrema=False, positions=range(max_folded - 1))
plt.plot(obs, ".")

bins = np.linspace(20, 35, 100)
plt.hist(np.sum(resamples, axis=1), histtype="step", bins=bins)
plt.plot(np.sum(obs), 1, "vk")

bins = np.linspace(20, 35, 100)
fig = plt.figure(figsize=(15, 15))
for i, d1 in enumerate(range(3, 16, 3)):
    for j, d2 in enumerate(range(3, 16, 3)):
        obs, resamples = stats.compare(
            spectra_data, spectra_fitted, d1, d2, max_folded, n_reps
        )

        ax = fig.add_subplot(5, 5, i * 5 + j + 1)
        ax.hist(np.sum(resamples, axis=1), histtype="step", bins=bins)
        ax.plot(np.sum(obs), 2, "vk")
        ax.set_xlim([20, 45])
        ax.set_ylim([0, 60])
        ax.set_title(f"{d1}; {d2}")
