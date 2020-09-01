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
import numpy as np
from scipy.special import rel_entr

import twosfs.twosfs as twosfs
from twosfs.demographicmodel import DemographicModel

# ## Import data

alphas = [f'{a:.2f}' for a in np.arange(1.5, 2.0, 0.05)]
model = '3Epoch'
sample_size = 100
kmax = 20

modelfn = "../simulations/fastNeutrino/xibeta-alpha={0}.{1}.txt"
betafn = "../simulations/msprime/xibeta-alpha={0}.npz"
fittedfn = "../simulations/msprime/fastNeutrino.xibeta-alpha={0}.{1}.npz"

data_kingman = twosfs.load_spectra('../simulations/msprime/kingman.npz')
print(twosfs.sfs2pi(data_kingman[0]))

demo_models = []
data_beta = []
data_fitted = []
for alpha in alphas:
    dm = DemographicModel(modelfn.format(alpha, model))
    dm.rescale()
    demo_models.append(dm)

    data_beta.append(twosfs.load_spectra(betafn.format(alpha)))
    data_fitted.append(twosfs.load_spectra(fittedfn.format(alpha, model)))

lumped_kingman = twosfs.lump_spectra(*data_kingman, kmax=kmax)
lumped_beta = [twosfs.lump_spectra(*s, kmax=kmax) for s in data_beta]
lumped_fitted = [twosfs.lump_spectra(*s, kmax=kmax) for s in data_fitted]

# ## Fitted demographic models

t = np.logspace(-2, 2, 100)
for alpha, dm in zip(alphas, demo_models):
    plt.semilogx(t, dm.population_size(t), label=alpha)
plt.legend(title='alpha')
plt.ylim([0, 5])
plt.ylabel('Population size (coal. units)')
plt.xlabel('Time (coal. units)')

# ## SFS comparisons

for alpha, dbeta, dfitted in zip(alphas, data_beta, data_fitted):
    plt.loglog(data_kingman[0] / data_kingman[0][1], '-k', label='Const-N')
    plt.loglog(dbeta[0] / dbeta[0][1], 'x', label='beta')
    plt.loglog(dfitted[0] / dfitted[0][1], '.', label='fitted')
    plt.vlines(kmax, 5e-3, 1, color='0.5', linestyle='dashed', label='kmax')
    plt.title("alpha = " + alpha)
    plt.legend()
    plt.show()

# # Old


def read_fastNeutrino_log(fn):
    onesfs_fitted = None
    onesfs_observed = None
    with open(fn, 'r') as data:
        for line in data:
            if line.startswith('Expected  spectrum'):
                spectrum = data.readline()
                onesfs_fitted = np.array(spectrum.split(), dtype=float)
            if line.startswith('Observed  spectrum'):
                spectrum = data.readline()
                onesfs_observed = np.array(spectrum.split(), dtype=float)
    return onesfs_fitted, onesfs_observed


fitted, observed = read_fastNeutrino_log('../log/fastNeutrino.test.3Epoch.log')

k = np.arange(1, 21)
plt.loglog(k, fitted, '.')
plt.loglog(k, observed, '.')
