# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import matplotlib.pyplot as plt
import numpy as np

import msprime
from src.twosfs import sims2sfs

# ## Import test simulated data

data = np.load('simulations/msprime/test.npz')
onesfs_import = data['onesfs']
twosfs_import = data['twosfs']

plt.plot(twosfs_import[:, 1, 1] - onesfs_import[1]**2, label=r'$a=1, b=1$')
plt.plot(twosfs_import[:, 1, 2] - onesfs_import[1] * onesfs_import[2],
         label=r'$a=1, b=2$')
plt.plot(twosfs_import[:, 2, 1] - onesfs_import[2] * onesfs_import[1],
         label=r'$a=2, b=1$')
plt.plot(twosfs_import[:, 2, 2] - onesfs_import[2]**2, label=r'$a=2, b=2$')
plt.xlabel("Distance between sites, $d$")
plt.ylabel(
    r"$\left< T_a^0 T_b^d \right> - \left< T_a \right> \left< T_b \right>$")
plt.legend()

# ## Generate what should be the same data locally.

onesfs_local = 0.0
twosfs_local = 0.0
for rep in range(10):
    parameters = {
        'sample_size': 4,
        'length': 100,
        'recombination_rate': 0.1,
        'random_seed': 1 + int(rep),
        'num_replicates': 100,
    }
    sims = msprime.simulate(**parameters)
    onesfs, twosfs = sims2sfs(sims, parameters['sample_size'],
                              parameters['length'])
    onesfs_local += onesfs
    twosfs_local += twosfs
onesfs_local /= 10
twosfs_local /= 10

# ## Test whether they match

np.allclose(onesfs_import, onesfs_local)

np.allclose(twosfs_import, twosfs_local)
