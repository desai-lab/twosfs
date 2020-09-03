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
# %aimport twosfs.demographicmodel, twosfs.simulations

import matplotlib.pyplot as plt
import msprime
import numpy as np

from twosfs.demographicmodel import DemographicModel
from twosfs.simulations import beta_timescale


def sims2pi(sims, num_replicates):
    """Compute pairwise diversity (T_2) for a generator of tree sequences."""
    pi = sum(tseq.diversity(mode="branch") for tseq in sims)
    pi /= num_replicates
    return pi


alpha = 1.55
for model in ["3Epoch", "FixedTimes", "FixedTimesDense"]:
    modelfn = f"../simulations/fastNeutrino/xibeta-alpha={alpha:.2f}.{model}.txt"
    dm = DemographicModel(modelfn)
    t = np.logspace(np.log10(dm.times[1] / 2), np.log10(2 * dm.times[-1]), 100)
    plt.loglog(t, dm.population_size(t))

model = "3Epoch"
alphas = np.arange(1.5, 2.0, 0.05)
for alpha in alphas:
    modelfn = f"../simulations/fastNeutrino/xibeta-alpha={alpha:.2f}.{model}.txt"
    dm = DemographicModel(modelfn)
    t = np.logspace(np.log10(dm.times[1] / 2), np.log10(2 * dm.times[-1]), 100)
    plt.loglog(t, dm.population_size(t), label=f"{alpha:.2f}")
plt.legend()

model = "FixedTimesDense"
alphas = np.arange(1.5, 2.0, 0.05)
for alpha in alphas:
    modelfn = f"../simulations/fastNeutrino/xibeta-alpha={alpha:.2f}.{model}.txt"
    dm = DemographicModel(modelfn)
    t = np.logspace(np.log10(dm.times[1] / 2), np.log10(2 * dm.times[-1]), 100)
    plt.loglog(t, dm.population_size(t), label=f"{alpha:.2f}")
plt.legend()

model = "FixedTimesDense"
alphas = np.arange(1.5, 2.0, 0.05)
for alpha in alphas:
    modelfn = f"../simulations/fastNeutrino/xibeta-alpha={alpha:.2f}.{model}.txt"
    dm = DemographicModel(modelfn)
    t = np.logspace(np.log10(dm.times[1] / 2), np.log10(2 * dm.times[-1]), 100)
    plt.loglog(t, dm.population_size(t), label=f"{alpha:.2f}")
plt.legend()

demographic_events = dm.get_demographic_events()
sample_size = 2
num_replicates = 10000
sims = msprime.simulate(
    sample_size=sample_size,
    random_seed=1,
    num_replicates=num_replicates,
    demographic_events=demographic_events,
)
T2 = sims2pi(sims, num_replicates)
print(T2)
print(4 * beta_timescale(alpha))

dm.rescale(T2 / 4)
t = np.logspace(np.log10(dm.times[1] / 10), np.log10(10 * dm.times[-1]), 100)
plt.semilogx(t, dm.population_size(t))

demographic_events = dm.get_demographic_events()
sample_size = 2
num_replicates = 10000
sims = msprime.simulate(
    sample_size=sample_size,
    random_seed=1,
    num_replicates=num_replicates,
    demographic_events=demographic_events,
)
T2_scaled = sims2pi(sims, num_replicates)
print(T2_scaled)

sample_size = 2
num_replicates = 10000
sims = msprime.simulate(
    sample_size=sample_size, random_seed=1, num_replicates=num_replicates
)
print(sims2pi(sims, num_replicates))

# +
num_replicates = 10000

for alpha in np.arange(1.5, 2.0, 0.05):
    print(alpha)
    modelfn = f"../simulations/fastNeutrino/xibeta-alpha={alpha:.2f}.3Epoch.txt"
    dm = DemographicModel(modelfn)

    t = np.logspace(np.log10(dm.times[1] / 10), np.log10(10 * dm.times[-1]), 100)
    # plt.semilogx(t, dm.population_size(t))
    # plt.show()

    demographic_events = dm.get_demographic_events()

    sims = msprime.simulate(
        sample_size=2,
        random_seed=1,
        num_replicates=num_replicates,
        demographic_events=demographic_events,
    )
    T2 = sims2pi(sims, num_replicates)
    print(f"$T_2$ = {T2}")

    dm.rescale(T2 / 4)
    t = np.logspace(np.log10(dm.times[1] / 10), np.log10(10 * dm.times[-1]), 100)
    plt.semilogx(t, dm.population_size(t))
    plt.show()
# -
