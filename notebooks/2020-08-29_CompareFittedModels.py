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

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import kl_div

from twosfs.demographicmodel import DemographicModel
from twosfs.twosfs import lump_sfs, sfs2pi

model = '3Epoch'
alphas = np.arange(1.5, 2.0, 0.05)
kmax = 10

for alpha in alphas:
    modelfn = f"../simulations/fastNeutrino/xibeta-alpha={alpha:.2f}.{model}.txt"
    dm = DemographicModel(modelfn)
    t = np.logspace(np.log10(dm.times[1] / 2), np.log10(2 * dm.times[-1]), 100)
    plt.loglog(t, dm.population_size(t), label=f"{alpha:.2f}")
plt.legend()

for alpha in alphas:
    simfn = f"../simulations/msprime/fastNeutrino.xibeta-alpha={alpha:.2f}.{model}.npz"
    data = np.load(simfn)
    onesfs = data['onesfs']
    plt.loglog(onesfs / onesfs[1], '.', label=f"{alpha:.2f}")
plt.legend()

for alpha in alphas:
    simfn = f"../simulations/msprime/xibeta-alpha={alpha:.2f}.npz"
    data = np.load(simfn)
    onesfs = data['onesfs']
    plt.loglog(onesfs / onesfs[1], '.', label=f"{alpha:.2f}")
plt.legend()

# ## `msprime` simulation comparisons

model = '3Epoch'
for alpha in alphas:
    simfn = f"../simulations/msprime/xibeta-alpha={alpha:.2f}.npz"
    data = np.load(simfn)
    onesfs_beta = data['onesfs']

    simfn = f"../simulations/msprime/fastNeutrino.xibeta-alpha={alpha:.2f}.{model}.npz"
    data = np.load(simfn)
    onesfs_fitted = data['onesfs']

    beta_lumped = lump_sfs(onesfs_beta, kmax) / np.sum(onesfs_beta)
    fitted_lumped = lump_sfs(onesfs_fitted, kmax) / np.sum(onesfs_fitted)
    plt.semilogy(beta_lumped, 'xk')
    plt.semilogy(fitted_lumped, '.')
    plt.title(f"{alpha:.2f}")
    plt.show()

    print(np.sum(kl_div(fitted_lumped, beta_lumped)))

# ## `fastNeutrino` expectations

model = '3Epoch'
for alpha in alphas:
    fn = f"../log/fastNeutrino.xibeta-alpha={alpha:.2f}.{model}.log"
    with open(fn, 'r') as data:
        for line in data:
            if line.startswith('Expected  spectrum'):
                spectrum = data.readline()
                onesfs_fitted = np.array(spectrum.split(), dtype=float)
            if line.startswith('Observed  spectrum'):
                spectrum = data.readline()
                onesfs_beta = np.array(spectrum.split(), dtype=float)
    plt.semilogy(onesfs_beta / np.sum(onesfs_beta), 'xk')
    plt.semilogy(onesfs_fitted / np.sum(onesfs_fitted), '.')
    plt.title(f"{alpha:.2f}")
    plt.show()

# `fastNeutrino` is expecting better fits than our simulations produce. Will need to troubleshoot.
