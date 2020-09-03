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

from twosfs.demographicmodel import DemographicModel
from twosfs.twosfs import sfs2pi, sims2sfs


def read_fastNeutrino(fn):
    with open(fn, "r") as data:
        for line in data:
            if line.startswith("Expected SFS"):
                spectrum = data.readline()
                return np.array(spectrum.split(), dtype=float)


model_file = "../troubleshoot/model.txt"
sfs_file = "../troubleshoot/fn_sfs.txt"

sfs_fn = read_fastNeutrino(sfs_file)
dm = DemographicModel(model_file)

pi_exact = dm.t2()
print(pi_exact)

pi_fn = sfs2pi(sfs_fn)
print(pi_fn / pi_exact)

sample_size = 100
k = np.arange(sample_size + 1)
plt.loglog(k, sfs_fn)
plt.loglog(k, 1 / k * sfs_fn[1])

num_replicates = 10000
# dm.sizes = [n/2 for n in dm.sizes]
demographic_events = dm.get_demographic_events()
sims = msprime.simulate(
    sample_size=sample_size,
    num_replicates=num_replicates,
    demographic_events=demographic_events,
)
sfs_msp = sims2sfs(sims, sample_size, 1)[0]

sfs2pi(sfs_msp)

sample_size = 100
k = np.arange(sample_size + 1)
plt.loglog(k, 1 / k, "-k")
plt.loglog(k, sfs_fn / pi_exact)
plt.loglog(k, sfs_msp / pi_exact, ".")
