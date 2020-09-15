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

import matplotlib.pyplot as plt
import msprime
import numpy as np

from twosfs.demographicmodel import DemographicModel
from twosfs.spectra import sims2pi

dm = DemographicModel()
dm.add_epoch(0.0, 2.0)
dm.add_epoch(2.0, 1.0)

dm.t2()

nreps = 100000
sims = msprime.simulate(
    demographic_events=dm.get_demographic_events(),
    sample_size=2,
    num_replicates=nreps,
    random_seed=100,
)
sims2pi(sims, num_replicates=nreps)

dm = DemographicModel()
dm.add_epoch(0.0, 2.0, 1.0)
dm.add_epoch(2.0, 2.0 * np.exp(-2.0))
dm.t2()

nreps = 100000
sims = msprime.simulate(
    demographic_events=dm.get_demographic_events(),
    sample_size=2,
    num_replicates=nreps,
    random_seed=100,
)
sims2pi(sims, num_replicates=nreps)

dm = DemographicModel()
dm.add_epoch(0.0, 2.0, 2.0)
dm.add_epoch(2.0, 2.0 * np.exp(-4.0))
dm.t2()

nreps = 100000
sims = msprime.simulate(
    demographic_events=dm.get_demographic_events(),
    sample_size=2,
    num_replicates=nreps,
    random_seed=100,
)
sims2pi(sims, num_replicates=nreps)

T = np.arange(0, 5, 0.01)
plt.semilogy(T, dm.population_size(T))

T = np.arange(0, 4, 0.01)
init_size = 1.0
rates = [0.5, 1.0, 2.0]
times = [0.5, 1.0, 2.0]
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)
for i, t in enumerate(times):
    for r in rates:
        dm = DemographicModel()
        dm.add_epoch(0.0, init_size, r)
        dm.add_epoch(t, init_size * np.exp(-r * t))
        ax1.semilogy(T, dm.population_size(T))
        ax2.plot(r, dm.t2(), ".", c=f"C{i}")

T = np.arange(0, 8, 0.01)
init_size = 1.0
rates = [0.5, 1.0, 2.0]
times = [0.5, 1.0, 2.0]
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)
for i, t in enumerate(times):
    for r in rates:
        dm = DemographicModel()
        dm.add_epoch(0.0, init_size, r)
        dm.add_epoch(t, init_size * np.exp(-r * t))
        dm.rescale()
        ax1.semilogy(T, dm.population_size(T))
        print(dm.t2())
