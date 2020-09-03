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
import msprime
import numpy as np
from scipy.special import betaln


def beta_multiple_merger_example():
    ts = msprime.sim_ancestry(
        sample_size=5,
        ploidy=2,
        random_seed=1,
        model=msprime.BetaCoalescent(alpha=1.001, truncation_point=1),
    )
    tree = ts.first()
    print(tree.draw(format="unicode"))


beta_multiple_merger_example()

ts = msprime.simulate(
    sample_size=10,
    random_seed=1,
    model=msprime.BetaCoalescent(alpha=1.01, truncation_point=1),
)
tree = ts.first()
print(tree.draw(format="unicode"))


def sfs2pi(sfs):
    """Compute the average pairwise diversity from an SFS."""
    n = len(sfs) - 1
    k = np.arange(n + 1)
    weights = 2 * k * (n - k) / (n * (n - 1))
    return np.dot(sfs, weights)


# ## Comparing `msprime.simulate` with `msprime.sim_ancestry`

# +
sample_size = 100
num_replicates = 10000
model = msprime.BetaCoalescent(alpha=1.5, truncation_point=1)
sims = msprime.simulate(
    sample_size=sample_size, random_seed=1, num_replicates=num_replicates, model=model
)

afs = np.zeros(sample_size + 1)
x = np.arange(0, sample_size)
for tseq in sims:
    afs += tseq.allele_frequency_spectrum(mode="branch", polarised=True)
afs /= num_replicates
plt.semilogy(afs / afs[1])
plt.semilogy(x, 1 / x)
print(sfs2pi(afs))
# -

# `msprime.sim_ancestry` handles ploidy and sample size differently

# +
sample_size = 50
num_replicates = 10000
model = msprime.BetaCoalescent(alpha=1.5, truncation_point=1)
sims = msprime.sim_ancestry(
    sample_size=sample_size,
    random_seed=1,
    num_replicates=num_replicates,
    model=model,
    ploidy=2,
)

afs = np.zeros(sample_size * 2 + 1)
x = np.arange(0, sample_size * 2)
for tseq in sims:
    afs += tseq.allele_frequency_spectrum(mode="branch", polarised=True)
afs /= num_replicates
plt.semilogy(afs / afs[1])
plt.semilogy(x, 1 / x)
print(sfs2pi(afs))
# -

# ## More extreme $\alpha$

# +
sample_size = 100
num_replicates = 10000
model = msprime.BetaCoalescent(alpha=1.01, truncation_point=1)
sims = msprime.simulate(
    sample_size=sample_size,
    random_seed=1,
    num_replicates=num_replicates,
    model=model,
    Ne=100,
)

afs = np.zeros(sample_size + 1)
x = np.arange(0, sample_size)
for tseq in sims:
    afs += tseq.allele_frequency_spectrum(mode="branch", polarised=True)
afs /= num_replicates
plt.semilogy(afs / afs[1])
plt.semilogy(x, 1 / x)
print(sfs2pi(afs))
# -

# +
sample_size = 100
num_replicates = 10000
model = msprime.BetaCoalescent(alpha=1.99, truncation_point=1)
sims = msprime.simulate(
    sample_size=sample_size, random_seed=1, num_replicates=num_replicates, model=model
)

afs = np.zeros(sample_size + 1)
x = np.arange(0, sample_size)
for tseq in sims:
    afs += tseq.allele_frequency_spectrum(mode="branch", polarised=True)
afs /= num_replicates
plt.semilogy(afs / afs[1])
plt.semilogy(x, 1 / x)
print(sfs2pi(afs))
# -

# ## Scaling of the timescale with $\alpha$

# ?msprime.ancestry.BetaCoalescent


def timescale(alpha, pop_size):
    m = 2 + np.exp(alpha * np.log(2) + (1 - alpha) * np.log(3) - np.log(alpha - 1))
    N = pop_size / 2
    return np.exp(
        alpha * np.log(m)
        + (alpha - 1) * np.log(N)
        - np.log(alpha)
        - betaln(2 - alpha, alpha)
    )


timescale(1.99, 1)

timescale(1.5, 1)

dx = 0.2
alphas = np.arange(1.0 + dx, 2.0, dx)

ne = 1
sample_size = 2
num_replicates = 10000
T2 = np.zeros_like(alphas)
for i, a in enumerate(alphas):
    model = msprime.BetaCoalescent(alpha=a, truncation_point=1)
    sims = msprime.simulate(
        sample_size=sample_size,
        random_seed=1,
        num_replicates=num_replicates,
        model=model,
        Ne=ne,
    )
    afs = np.zeros(sample_size + 1)
    for tseq in sims:
        afs += tseq.allele_frequency_spectrum(mode="branch", polarised=True)
    afs /= num_replicates
    T2[i] = afs[1]
plt.plot(alphas, T2)

# +
plt.plot(alphas, 8 * timescale(alphas, ne))
plt.plot(alphas, T2)
plt.show()

plt.plot(alphas, T2 / timescale(alphas, ne))
# -

# __Conclusion__: $T_2 = 8 \text{timescale}(\alpha, N_e)$.

# ## Check $T_2$ is proportional to $\pi$

ne = 1
sample_size = 100
num_replicates = 10000
Pi = np.zeros_like(alphas)
for i, a in enumerate(alphas):
    model = msprime.BetaCoalescent(alpha=a, truncation_point=1)
    sims = msprime.simulate(
        sample_size=sample_size,
        random_seed=1,
        num_replicates=num_replicates,
        model=model,
        Ne=ne,
    )
    afs = np.zeros(sample_size + 1)
    for tseq in sims:
        afs += tseq.allele_frequency_spectrum(mode="branch", polarised=True)
    afs /= num_replicates
    pi = sfs2pi(afs)
    Pi[i] = pi

plt.plot(alphas, T2)
plt.plot(alphas, Pi)

plt.plot(alphas, Pi / T2)

# ## Compare with T2 of standard Kingman

ne = 1
sample_size = 2
num_replicates = 10000
sims = msprime.simulate(
    sample_size=sample_size, random_seed=1, num_replicates=num_replicates, Ne=ne
)
afs = np.zeros(sample_size + 1)
for tseq in sims:
    afs += tseq.allele_frequency_spectrum(mode="branch", polarised=True)
afs /= num_replicates
T2_king = afs[1]
print(T2_king)
