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
import msprime
import numpy as np

# ## Testing the tskit `allele_frequency_spectrum` method.

random_seed = 100
sample_size = 100
num_replicates = 1000
sims = msprime.simulate(
    sample_size=sample_size, num_replicates=num_replicates, random_seed=random_seed
)
afs = np.zeros(sample_size + 1)
for tseq in sims:
    afs += tseq.allele_frequency_spectrum(mode="branch")
afs /= num_replicates
plt.loglog(afs)

# Find T2. It should be half the branch length of singletons when n=2

sample_size = 2
sims = msprime.simulate(
    sample_size=sample_size, num_replicates=num_replicates, random_seed=random_seed
)
afs = np.zeros(sample_size + 1)
for tseq in sims:
    afs += tseq.allele_frequency_spectrum(mode="branch")
afs /= num_replicates
pi = afs[1]
T2 = pi / 2
print(T2)

# Simulate 100bp. Choose $r=0.1$ so that $1 / rT_2 = 10$:

params = {
    "sample_size": 2,
    "length": 100,
    "recombination_rate": 0.1,
    "random_seed": 100,
    "num_replicates": 10000,
}
d_c = 1.0 / params["recombination_rate"]
sims = msprime.simulate(**params)

# Compute the branch length correlation as a function of distance.

windows = np.arange(params["length"] + 1)
corr = np.zeros(params["length"])
for tseq in sims:
    afs = tseq.allele_frequency_spectrum(mode="branch", windows=windows)
    corr += afs[0, 1] * afs[:, 1]
corr /= params["num_replicates"]

plt.plot(corr)
plt.xlabel("Distance between sites")
plt.ylabel(r"$\left< T_2^j T_2^k \right>$")
plt.vlines([d_c], 4.0, 7.5, linestyle="dashed", label="$d_c$")
plt.legend()
plt.title("Decay in $T_2$ correlation")

# Expand to larger samples


def get_sfs(sims, sample_size, length):
    """Compute the twosfs compared to left end of a sequence."""
    windows = np.arange(length + 1)
    onesfs = np.zeros((sample_size + 1))
    twosfs = np.zeros((length, sample_size + 1, sample_size + 1))
    n_sims = 0
    for tseq in sims:
        afs = tseq.allele_frequency_spectrum(mode="branch", windows=windows)
        onesfs += np.mean(afs, axis=0)
        twosfs += afs[0, :, None] * afs[:, None, :]
        n_sims += 1
    return onesfs / n_sims, twosfs / n_sims


params = {
    "sample_size": 4,
    "length": 100,
    "recombination_rate": 0.1,
    "random_seed": 100,
    "num_replicates": 10000,
}
d_c = 1.0 / params["recombination_rate"]
sims = msprime.simulate(**params)
onesfs, twosfs = get_sfs(sims, params["sample_size"], params["length"])

plt.plot(twosfs[:, 1, 1] - onesfs[1] ** 2, label=r"$a=1, b=1$")
plt.plot(twosfs[:, 1, 2] - onesfs[1] * onesfs[2], label=r"$a=1, b=2$")
plt.plot(twosfs[:, 2, 1] - onesfs[2] * onesfs[1], label=r"$a=2, b=1$")
plt.plot(twosfs[:, 2, 2] - onesfs[2] ** 2, label=r"$a=2, b=2$")
plt.xlabel("Distance between sites, $d$")
plt.ylabel(r"$\left< T_a^0 T_b^d \right> - \left< T_a \right> \left< T_b \right>$")
plt.legend()

# ## Multiple mergers

# ### Beta coalescent

# Test whether the times have been rescaled:

# +
params = {
    "sample_size": 2,
    "length": 1,
    "recombination_rate": 0,
    "random_seed": 100,
    "num_replicates": 1000,
}

for alpha in np.arange(2.0, 1.0, -0.1):
    params["model"] = msprime.BetaCoalescent(alpha=alpha)
    sims = msprime.simulate(**params)
    onesfs, twosfs = get_sfs(sims, params["sample_size"], params["length"])
    T2 = onesfs[1] / 2
    print(f"a = {alpha:.2f}\tT_2 = {T2:.4f}")
# -

# It looks like they have been.
#
# Now, test that the 2SFS still works.

params = {
    "sample_size": 4,
    "length": 100,
    "recombination_rate": 0.1,
    "model": msprime.BetaCoalescent(alpha=1.75),
    "random_seed": 100,
    "num_replicates": 1000,
}
sims = msprime.simulate(**params)
onesfs, twosfs = get_sfs(sims, params["sample_size"], params["length"])

plt.plot(twosfs[:, 1, 1] - onesfs[1] ** 2, label=r"$a=1, b=1$")
plt.plot(twosfs[:, 1, 2] - onesfs[1] * onesfs[2], label=r"$a=1, b=2$")
plt.plot(twosfs[:, 2, 1] - onesfs[2] * onesfs[1], label=r"$a=2, b=1$")
plt.plot(twosfs[:, 2, 2] - onesfs[2] ** 2, label=r"$a=2, b=2$")
plt.xlabel("Distance between sites, $d$")
plt.ylabel(r"$\left< T_a^0 T_b^d \right> - \left< T_a \right> \left< T_b \right>$")
plt.legend()

# Make sure $\pi$ doesn't scale with sample size.


def sfs2pi(sfs):
    """Compute the average pairwise diversity from the SFS."""
    n = len(sfs) - 1
    k = np.arange(n + 1)
    weights = 2 * k * (n - k) / (n * (n - 1))
    return np.dot(sfs, weights)


# +
params = {
    "length": 1,
    "recombination_rate": 0,
    "random_seed": 100,
    "num_replicates": 1000,
}

for alpha in np.arange(1.9, 1.0, -0.1):
    for n in [2, 4, 10]:
        params["model"] = msprime.BetaCoalescent(alpha=alpha)
        params["sample_size"] = n
        sims = msprime.simulate(**params)
        onesfs, twosfs = get_sfs(sims, params["sample_size"], params["length"])
        T2 = sfs2pi(onesfs) / 2
        print(f"a = {alpha:.2f}\tn = {n}\tT_2 = {T2:.4f}")
# -

# ### Dirac coalescent

# Parameters:
# $0 < \psi < 1$, $c >= 0$

# +
params = {
    "sample_size": 2,
    "length": 1,
    "recombination_rate": 0,
    "random_seed": 100,
    "num_replicates": 1000,
}

for psi in [0.1, 0.5, 0.9]:
    for c in [0, 1, 5, 10, 100]:
        params["model"] = msprime.DiracCoalescent(psi=psi, c=c)
        sims = msprime.simulate(**params)
        onesfs, twosfs = get_sfs(sims, params["sample_size"], params["length"])
        T2 = onesfs[1] / 2
        print(f"psi = {psi:.2f}\tc = {c:.2f}\tT_2 = {T2:.4f}")
# -

# Unlike the Beta coalescent, the Dirac coalescent is not scaled to have
# $T_2 = N$.

# +
params = {
    "length": 1,
    "recombination_rate": 0,
    "random_seed": 100,
    "num_replicates": 1000,
}

for psi in [0.1, 0.5, 0.9]:
    for c in [0, 1, 5, 10, 100]:
        params["model"] = msprime.DiracCoalescent(psi=psi, c=c)
        for n in [2, 4, 10]:
            params["sample_size"] = n
            sims = msprime.simulate(**params)
            onesfs, twosfs = get_sfs(sims, params["sample_size"], params["length"])
            T2 = sfs2pi(onesfs) / 2
            print(f"psi = {psi:.2f}\tc = {c:.2f}\tn = {n}\tT_2 = {T2:.4f}")
# -

# Looking at the issues on GitHub, it seems like the Dirac coalescent isn't
# fully implemented.

# ## Testing the recombination in the Xi-Beta coalescent

params = {
    "sample_size": 10,
    "length": 100,
    "recombination_rate": 0.1,
    "model": msprime.BetaCoalescent(alpha=1.75),
    "random_seed": 100,
    "num_replicates": 10000,
}
sims = msprime.simulate(**params)
onesfs, twosfs = get_sfs(sims, params["sample_size"], params["length"])

plt.figure(figsize=(10, 10))
for a in range(1, 6):
    for b in range(1, 6):
        ax = plt.subplot(5, 5, a + 5 * (b - 1))
        ax.plot(
            np.arange(1, params["length"]), twosfs[1:, a, b] - onesfs[a] * onesfs[b]
        )
        ax.set_title(f"$a={a}, b={b}$")

# Repeat with Kingman

params = {
    "sample_size": 10,
    "length": 100,
    "recombination_rate": 0.1,
    "random_seed": 100,
    "num_replicates": 10000,
}
sims = msprime.simulate(**params)
onesfs, twosfs = get_sfs(sims, params["sample_size"], params["length"])

plt.figure(figsize=(10, 10))
for a in range(1, 6):
    for b in range(1, 6):
        ax = plt.subplot(5, 5, a + 5 * (b - 1))
        ax.plot(
            np.arange(1, params["length"]), twosfs[1:, a, b] - onesfs[a] * onesfs[b]
        )
        ax.set_title(f"$a={a}, b={b}$")
