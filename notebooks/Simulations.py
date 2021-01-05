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
from matplotlib.cm import viridis

import twosfs.statistics as stats
from twosfs import foldonesfs, load_spectra, lump_onesfs, lump_twosfs
from twosfs.demographicmodel import DemographicModel

# +


def power(resamples_comparison, resamples_null, alpha):
    n_resample = resamples_null.shape[0]
    total_comparison = np.sum(resamples_comparison, axis=1)
    total_null = np.sum(resamples_null, axis=1)
    return np.mean(stats.rank(total_comparison, total_null) > (1 - alpha) * n_resample)


def plot_onesfs(ax, spectra_comparison, spectra_null, folded, max_k):
    n = spectra_comparison.num_samples
    k = np.arange(n + 1)
    neutral_sfs = np.zeros(n + 1)
    neutral_sfs[1:n] = 1 / (k[1:n] * np.sum(1 / k[1:n]))
    if folded:
        neutral_sfs = foldonesfs(neutral_sfs)
    neutral_sfs = lump_onesfs(neutral_sfs, max_k)

    onesfs_comparison = lump_onesfs(
        spectra_comparison.normalized_onesfs(folded=folded), max_k
    )
    onesfs_null = lump_onesfs(spectra_null.normalized_onesfs(folded=folded), max_k)

    ax.loglog(k[1 : max_k + 1], onesfs_null[1:], "x")
    ax.loglog(k[1 : max_k + 1], onesfs_comparison[1:], ".")
    ax.loglog(k[1 : max_k + 1], neutral_sfs[1:], ".", c="0.5")
    return ax


def violin(ax, resamples_comparison, resamples_null):
    ax.violinplot(resamples_null, showextrema=False)
    ax.violinplot(resamples_comparison, showextrema=False)
    return ax


def histogram(ax, resamples_comparison, resamples_null, bins):
    ax.hist(np.sum(resamples_null, axis=1), histtype="step", bins=bins)
    ax.hist(np.sum(resamples_comparison, axis=1), histtype="step", bins=bins)
    return ax


# +
alphas = [f"{a:0.2f}" for a in np.arange(1.05, 2.0, 0.05)]
growth_rates = [f"{g:.1f}" for g in [0.5, 1.0, 2.0]]
growth_times = [f"{t:.1f}" for t in [0.5, 1.0, 2.0]]

sims_alpha = [f"xibeta-alpha={alpha}" for alpha in alphas]
sims_exp = [f"expgrowth-g={g}-t={t}" for g in growth_rates for t in growth_times]
sims = sims_alpha + sims_exp

models = {}
spectra_comparisons = {}
spectra_nulls = {}
for sim in sims:
    modelfn = f"../simulations/fastNeutrino/{sim}.3Epoch.txt"
    comparisonfn = f"../simulations/msprime/{sim}.npz"
    nullfn = f"../simulations/msprime/fastNeutrino.{sim}.3Epoch.npz"

    dm = DemographicModel(modelfn)
    dm.rescale()
    models[sim] = dm
    spectra_comparisons[sim] = load_spectra(comparisonfn)
    spectra_nulls[sim] = load_spectra(nullfn)

# +
ds = [0, 10]
max_k = 20
n_reps = 1000
num_pairs = 10000

resamples_comparison = {}
resamples_null = {}
resamples_comparison_folded = {}
resamples_null_folded = {}
for sim in sims:
    for d in ds:
        rc, rn = stats.compare2(
            spectra_comparisons[sim],
            spectra_nulls[sim],
            d,
            d,
            max_k,
            n_reps,
            num_pairs,
            folded=False,
        )
        resamples_comparison[(sim, d)] = rc
        resamples_null[(sim, d)] = rn

        rc, rn = stats.compare2(
            spectra_comparisons[sim],
            spectra_nulls[sim],
            d,
            d,
            max_k,
            n_reps,
            num_pairs,
            folded=True,
        )
        resamples_comparison_folded[(sim, d)] = rc
        resamples_null_folded[(sim, d)] = rn

# +
time = np.logspace(-2, 2, 100)
max_k = 20

for sim in sims:
    print(sim)
    fig = plt.figure(figsize=(6, 3))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    plot_onesfs(ax1, spectra_comparisons[sim], spectra_nulls[sim], False, max_k)
    plot_onesfs(ax2, spectra_comparisons[sim], spectra_nulls[sim], True, max_k)
    plt.show()
# -

for sim in sims_exp:
    plt.semilogx(time, models[sim].population_size(time))

for sim in sims:
    bins = np.arange(7, 45)
    ax1 = plt.subplot(121, title=sim)
    histogram(ax1, resamples_comparison[(sim, 0)], resamples_null[(sim, 0)], bins=bins)
    ax2 = plt.subplot(122, title=sim + " folded")
    histogram(
        ax2,
        resamples_comparison_folded[(sim, 0)],
        resamples_null_folded[(sim, 0)],
        bins=bins,
    )
    plt.show()

for sim in sims:
    bins = np.arange(7, 45)
    ax1 = plt.subplot(121, title=sim)
    violin(ax1, resamples_comparison[(sim, 0)], resamples_null[(sim, 0)])
    ax2 = plt.subplot(122, title=sim + " folded")
    violin(ax2, resamples_comparison_folded[(sim, 0)], resamples_null_folded[(sim, 0)])
    plt.show()

for sim in sims_alpha:
    alpha = float(sim.split("=")[1])
    p = power(resamples_comparison[(sim, 0)], resamples_null[(sim, 0)], alpha=0.05)
    plt.plot(alpha, p, ".k")

    p_folded = power(
        resamples_comparison_folded[(sim, 0)],
        resamples_null_folded[(sim, 0)],
        alpha=0.05,
    )
    plt.plot(alpha, p_folded, "xk")
plt.ylim([0, 1.05])

# +

d1 = 1
d2 = 3
d3 = 20

for sim in sims:
    print(sim)
    fig = plt.figure(figsize=(10, 3))
    axes = [fig.add_subplot(1, 3, i) for i in [1, 2, 3]]

    twosfs_beta = lump_twosfs(spectra_comparisons[sim].normalized_twosfs(), max_k)
    twosfs_null = lump_twosfs(spectra_nulls[sim].normalized_twosfs(), max_k)
    for d, ax in zip([d1, d2, d3], fig.axes):
        ax.pcolormesh(
            np.log2(twosfs_beta[d][1:, 1:] / twosfs_null[d][1:, 1:]),
            cmap="PuOr_r",
            vmin=-0.5,
            vmax=0.5,
        )
        ax.set_title(d)
    plt.show()


# +
def plot_nt(ax):
    time = np.logspace(-2, 2, 100)
    for sim in sims_alpha:
        alpha = float(sim.split("=")[1])
        ax.loglog(time, models[sim].population_size(time), color=viridis(alpha - 1))
    return ax


ax = plt.subplot(
    ylabel=r"Population size (scaled by $T_2$)",
    xlabel=r"Time in the past (scaled by $T_2$)",
)
plot_nt(ax)

# +
max_k = 20


def singleton_tail(spectra, max_k, folded=False):
    onesfs = lump_onesfs(spectra.normalized_onesfs(folded=folded), max_k)
    return tuple(onesfs[[1, -1]])


def plot_fits(ax):
    for sim in sims:
        singleton_comp, tail_comp = singleton_tail(spectra_comparisons[sim], max_k)
        singleton_null, tail_null = singleton_tail(spectra_nulls[sim], max_k)
        if sim.startswith("xibeta"):
            alpha = float(sim.split("=")[-1])
            color = viridis(alpha - 1)
        else:
            color = "0.5"

        (l1,) = ax.plot(singleton_comp, tail_comp, "o", color=color, ms=10)
        (l2,) = ax.plot(singleton_null, tail_null, "o", mec="w", mfc="k")

    ax.legend([l1, l2], ["Actual", "Fitted"])
    return ax


ax = plt.subplot(
    ylabel="Tail fraction",
    xlabel="Singleton fraction",
    title="Demographic model fits to SFS",
)
plot_fits(ax)


# +
def plot_test_statistic(ax):
    w = 0.025
    for i, sim in enumerate(sims_alpha):
        alpha = float(sim.split("=")[-1])

        d_comp = np.sum(resamples_comparison[(sim, 0)], axis=1)
        d_null = np.sum(resamples_null[(sim, 0)], axis=1)

        pc = ax.violinplot(d_null, showextrema=False, positions=[alpha], widths=w)
        pc["bodies"][0].set_facecolor("C0")
        pc = ax.violinplot(d_comp, showextrema=False, positions=[alpha], widths=w)
        pc["bodies"][0].set_facecolor("C1")

    for i, sim in enumerate(sims_exp):
        pos = 2.05 + 0.05 * i
        d_comp = np.sum(resamples_comparison[(sim, 0)], axis=1)
        d_null = np.sum(resamples_null[(sim, 0)], axis=1)

        pc = ax.violinplot(d_null, showextrema=False, positions=[pos], widths=w)
        pc["bodies"][0].set_facecolor("C0")
        pc = ax.violinplot(d_comp, showextrema=False, positions=[pos], widths=w)
        pc["bodies"][0].set_facecolor("C1")

    ax.vlines(2, 5, 35, linestyle="dashed", color="k")
    ax.text(1.3, -3, "Xi-Beta alpha")
    ax.text(2.3, -3, "Exponential\ngrowth", ha="center")
    ax.set_xticks(np.arange(1, 2.1, 0.2))
    return ax


ax = plt.subplot(ylabel="Test statistic")
plot_test_statistic(ax)

# +
fig = plt.figure(figsize=(8, 8))

ax1 = plt.subplot(
    221,
    ylabel=r"Population size (scaled by $T_2$)",
    xlabel=r"Time in the past (scaled by $T_2$)",
    title="Demographic models",
)
plot_nt(ax1)

ax2 = plt.subplot(
    222,
    ylabel="Tail fraction",
    xlabel="Singleton fraction",
    title="Demographic model fits to SFS",
    yticks=[0.1, 0.2, 0.3],
    ylim=[0.08, 0.32],
)
plot_fits(ax2)

ax3 = plt.subplot(212, ylabel="Test statistic")
plot_test_statistic(ax3)


# +
def plot_twosfs_ratio(ax, twosfs_beta, twosfs_null, d):
    ax.pcolormesh(
        np.log2(twosfs_beta[d][1:, 1:] / twosfs_null[d][1:, 1:]),
        cmap="PuOr_r",
        vmin=-0.5,
        vmax=0.5,
    )
    return ax


d1 = 0
d2 = 1
d3 = 10

sims_to_plot = ["xibeta-alpha=1.20", "xibeta-alpha=1.60", "expgrowth-g=2.0-t=1.0"]

fig = plt.figure(figsize=(10, 10))

for i, sim in enumerate(sims_to_plot):
    twosfs_beta = lump_twosfs(spectra_comparisons[sim].normalized_twosfs(), max_k)
    twosfs_null = lump_twosfs(spectra_nulls[sim].normalized_twosfs(), max_k)
    for j, d in enumerate([d1, d2, d3]):
        ax = fig.add_subplot(3, 3, 1 + j + 3 * i)
        plot_twosfs_ratio(ax, twosfs_beta, twosfs_null, d)
        if i == 0:
            ax.set_title(f"$drT_2$ = {d*0.1}")
        if j == 0:
            ax.text(-10, 10, "\n".join(sim.split("-")))
# -

for d in ds:
    for sim in sims_alpha:
        alpha = float(sim.split("=")[1])
        p = power(resamples_comparison[(sim, d)], resamples_null[(sim, d)], alpha=0.05)
        plt.plot(alpha, p, ".k")

        p_folded = power(
            resamples_comparison_folded[(sim, d)],
            resamples_null_folded[(sim, d)],
            alpha=0.05,
        )
        plt.plot(alpha, p_folded, "xk")
plt.ylim([0, 1.05])
plt.xlabel("Alpha")
plt.ylabel("Power")
