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

import helpers as h
import matplotlib.pyplot as plt
import numpy as np
from demographicmodel import DemographicModel


def readsim(fn):
    times = None
    sizes = None
    with open(fn) as infile:
        while True:
            line = infile.readline()
            if not line.startswith("#"):
                break
            elif line.startswith("#DEMOCHANGE_TIMES"):
                times = np.array(line.split("=")[1].split(","), dtype=float)
            elif line.startswith("#DEMOCHANGE_SIZES"):
                sizes = np.array(line.split("=")[1].split(","), dtype=float)

        sfs = np.array(line.split(), dtype=float)
        twosfs = np.zeros((len(sfs), len(sfs)))
        twosfs[np.triu_indices_from(twosfs)] = np.array(
            infile.readline().split(), dtype=float
        )

    twosfs[np.diag_indices_from(twosfs)] /= 2
    twosfs += np.transpose(twosfs)

    return sfs, twosfs, times, sizes


# +
def lump_sfs(sfs, bmax):
    sfs_lumped = np.zeros(bmax + 1)
    sfs_lumped[:bmax] = sfs[:bmax]
    sfs_lumped[bmax] = sum(sfs[bmax:])
    return sfs_lumped


def normalize_sfs(sfs):
    return sfs / np.sum(sfs)


def lump_2sfs(twosfs, bmax):
    lumped = np.zeros((bmax + 1, bmax + 1))
    lumped[:bmax, :bmax] = twosfs[:bmax, :bmax]
    lumped[bmax, :bmax] = np.sum(twosfs[bmax:, :bmax], axis=0)
    lumped[:bmax, bmax] = np.sum(twosfs[:bmax:, bmax:], axis=1)
    lumped[bmax, bmax] = np.sum(twosfs[bmax:, bmax:])
    return lumped


# def kldiv(p, q):
#     return - np.sum(p*np.log2(q/p))


# +
ALPHAS = [1.975, 1.95, 1.9, 1.85, 1.8, 1.75, 1.625, 1.5]
RS = [0.0, 0.5, 1.0, 2.5, 5.0]
# RS = [1.0]

n = 100
model = "ConstFixedTimes"
maxb = 10

# +
sfs_fitted = {}
twosfs_fitted = {}
sfs_xibeta = {}
twosfs_xibeta = {}

for alpha in ALPHAS:
    for r in RS:
        if r == 0:
            pattern_fitted = "../simulations/\
                msprime/jsfs_fastNeutrino-xibeta_n-{}_alpha-{}_{}_r-{}.txt"
            pattern_xibeta = "../simulations/msprime/jsfs_n-{}_xibeta-{}_r-{}.txt"
        else:
            pattern_fitted = "../simulations/\
                msprime/jsfs_fastNeutrino-xibeta_n-{}_alpha-{}_{}_r-{}.scaled.txt"
            pattern_xibeta = (
                "../simulations/msprime/jsfs_n-{}_xibeta-{}_r-{}.scaled.txt"
            )

        # Get fitted simulations
        fn = pattern_fitted.format(n, alpha, model, r)
        sfs, twosfs, _, _ = readsim(fn)
        sfs_fitted[(alpha, r)] = sfs
        twosfs_fitted[(alpha, r)] = twosfs

        # Get XiBeta simulations
        fn = pattern_xibeta.format(n, alpha, r)
        sfs, twosfs, _, _ = readsim(fn)
        sfs_xibeta[(alpha, r)] = sfs
        twosfs_xibeta[(alpha, r)] = twosfs
# -

# ## Site Frequency Spectrum plots

r = 0.0

# +
daf = np.arange(1, n)

for alpha in ALPHAS:
    plt.loglog(
        daf, sfs_xibeta[(alpha, r)] / np.sum(sfs_xibeta[(alpha, r)]), label=alpha
    )
plt.legend()
# -

daf = np.arange(1, n)
for alpha in ALPHAS:
    plt.plot(
        daf, sfs_xibeta[(alpha, r)] / np.sum(sfs_xibeta[(alpha, r)]) * daf, label=alpha
    )
plt.legend()

# +
x = np.arange(1, 12)

for alpha in ALPHAS:
    plt.semilogy(
        x, lump_sfs(sfs_xibeta[(alpha, r)] / np.sum(sfs_xibeta[(alpha, r)]), maxb), "o"
    )
plt.semilogy(x, lump_sfs(1 / np.arange(1, n), maxb) / np.sum(1 / np.arange(1, n)), ":k")

# -

# ## Fitted demographic models


def plot_model(ax, model, scale=None, **kwargs):
    t = np.append(model.times, model.times[-1] * 2)
    y = np.append(model.sizes, model.sizes[-1])
    if scale is not None:
        t /= scale
        y /= scale
    ax.plot(t, y, drawstyle="steps-post", **kwargs)
    return ax


model_dict = {}
for alpha in ALPHAS:
    fn = "../fastNeutrino/fitted_params/n-{}_xibeta-{}.{}.txt".format(n, alpha, model)
    dm = DemographicModel(fn)
    dm.rescale()
    model_dict[alpha] = dm

ax = plt.subplot(111)
for alpha in ALPHAS:
    plot_model(ax, model_dict[alpha], label=alpha)
ax.set_yscale("log")
ax.set_xscale("log")
ax.legend()
# ax.set_ylim([0.08, (5/4)*100])

ax = plt.subplot(111)
for alpha in ALPHAS:
    plot_model(ax, model_dict[alpha], label=alpha)
ax.set_yscale("log")
ax.set_xscale("log")
ax.legend()
ax.set_ylim([0.08, (5 / 4) * 100])

# ## Compare fitted to Xi-Beta SFS

# +
x = np.arange(1, maxb + 2)

for alpha in ALPHAS:
    print(alpha)
    fig = plt.figure()

    sfs = normalize_sfs(sfs_xibeta[(alpha, r)])
    plt.plot(x, lump_sfs(sfs, 10), "ok", label="Xi-Beta")
    plt.plot(
        x,
        lump_sfs(1 / np.arange(1, n), maxb) / np.sum(1 / np.arange(1, n)),
        ":k",
        label="Kingman",
    )

    try:
        sfs = normalize_sfs(sfs_fitted[(alpha, r)])
    except KeyError:
        print("not found")
        plt.show()
        continue
    plt.semilogy(x, lump_sfs(sfs, 10), "-x", label="fitted")  # , label=model)

    plt.legend()
    plt.show()


# +
x = np.arange(1, n)

for alpha in ALPHAS:
    print(alpha)
    fig = plt.figure()

    sfs = normalize_sfs(sfs_xibeta[(alpha, r)])
    plt.loglog(x, sfs, label="Xi-Beta")
    plt.loglog(x, (1 / x) / np.sum(1 / x), ":k", label="Kingman")

    try:
        sfs = normalize_sfs(sfs_fitted[(alpha, r)])
    except KeyError:
        print("not found")
        plt.show()
        continue
    plt.loglog(x, sfs, "-", label="fitted")  # , label=model)

    plt.legend()
    plt.ylabel("Fraction of segregating sites")
    plt.xlabel("Derived allele count")
    plt.show()


# +
pi_xibeta = {}
pi_fitted = {}

for alpha in ALPHAS:
    sfs = sfs_xibeta[(alpha, r)]
    sfs = (sfs + sfs[::-1])[: n // 2]
    pi_xibeta[alpha] = h.sfs2pi(sfs, n)

    sfs = sfs_fitted[(alpha, r)]
    sfs = (sfs + sfs[::-1])[: n // 2]
    pi_fitted[alpha] = h.sfs2pi(sfs, n)

    plt.plot(alpha, pi_fitted[alpha], "o", c="C0")
    plt.plot(alpha, pi_xibeta[alpha], "o", c="C1")

    print("'{}':{},".format(alpha, pi_xibeta[alpha] / 2))
    print("('{}','{}'):{},".format(alpha, model, pi_fitted[alpha] / 2))


plt.ylabel(r"$\pi$")
plt.xlabel("Alpha")
# plt.ylim([-0.0195,0.0195])
#     plt.plot(alpha, pi_fitted, 'ob')
# -

ax = plt.subplot(111)
for alpha in ALPHAS:
    plot_model(
        ax, model_dict[alpha], label=alpha, scale=pi_fitted[alpha] / 2
    )  # , color=str(1-alpha))
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_ylim([0.05, 200])
ax.legend()

ax = plt.subplot(111)
for alpha in ALPHAS[:-1]:
    plot_model(
        ax, model_dict[alpha], label=alpha, scale=pi_fitted[alpha] / 2
    )  # , color=str(1-alpha))
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_ylim([0.05, 200])
ax.legend()

print(ALPHAS)
A_PLOT = [1.9, 1.8, 1.625]

# +
x = np.arange(1, maxb + 2)

fig = plt.figure(figsize=(6, 3))

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.plot(
    x,
    lump_sfs(1 / np.arange(1, n), maxb) / np.sum(1 / np.arange(1, n)),
    ":k",
    label="Kingman",
)

for i, alpha in enumerate(A_PLOT):
    color = "C" + str(i)

    sfs = normalize_sfs(sfs_xibeta[(alpha, r)])
    ax1.plot(x, lump_sfs(sfs, 10), ".", color=color)  # , label="Xi-Beta")
    sfs = normalize_sfs(sfs_fitted[(alpha, r)])
    ax1.semilogy(
        x, lump_sfs(sfs, 10), "-x", color=color
    )  # , label="fitted")#, label=model)

    plot_model(
        ax2,
        model_dict[alpha],
        label=r"$\alpha={}$".format(alpha),
        scale=pi_fitted[alpha] / 2,
        color=color,
    )  # , color=str(1-alpha))

ax1.plot(0, 0, marker="x", color="0.5", label="fitted")
ax1.plot(0, 0, marker=".", color="0.5", label="Xi-Beta")


ax1.legend()
ax1.set_ylabel("Normalized SFS")
ax1.set_yticks([])
ax1.set_xticks([1, 5, 9, 11])
ax1.set_xlabel("Minor allele count")
ax1.set_xticklabels([1, 5, 9, ">10"])
ax2.set_yscale("log")
ax2.set_xscale("log")
# ax2.set_ylim([0.05,200])
ax2.legend()
ax2.set_ylabel("Scaled population size")
ax2.set_xlabel("Scaled time")

plt.tight_layout()
plt.show()
fig.savefig("fitted_models.pdf")

# -

# ## 2-SFS


def conditional_sfs(twosfs):
    F = np.cumsum(twosfs, axis=1)
    F /= F[:, -1][:, None]
    return F


def fold_twosfs(twosfs):
    n_samples = twosfs.shape[0] + 1
    folded = (twosfs + twosfs[::-1, :] + twosfs[:, ::-1] + twosfs[::-1, ::-1])[
        : n_samples // 2, : n_samples // 2
    ]
    if n_samples % 2 == 0:
        folded[:, -1] /= 2
        folded[-1, :] /= 2
    return folded


def distance(F1, F2):
    return np.max(np.abs(F1 - F2), axis=1)


def resample_distance(sampling_dist, comparison_dist, n_obs, n_reps):
    F_exp = conditional_sfs(comparison_dist)

    D = np.zeros((n_reps, sampling_dist.shape[0]))
    for rep in range(n_reps):
        rand_counts = np.random.multinomial(n_obs, sampling_dist.ravel()).reshape(
            sampling_dist.shape
        )
        rand_counts = (rand_counts + rand_counts.T) / 2

        cumcounts = np.cumsum(rand_counts, axis=1)
        n_row = cumcounts[:, -1]
        F_obs = cumcounts / n_row[:, None]

        D[rep] = distance(F_exp, F_obs) * np.sqrt(n_row)

    return D


def rank(value, comparisons):
    return np.sum(value[:, None] > comparisons[None, :], axis=0)


def violin_with_iqrange(ax, data, color):
    x = np.arange(1, data.shape[1] + 1)
    lb = np.percentile(data, 25, axis=0)
    ub = np.percentile(data, 75, axis=0)
    ax.violinplot(data, showextrema=False)
    ax.plot(x, lb, linestyle="", marker="_", color=color)
    ax.plot(x, ub, linestyle="", marker="_", color=color)
    ax.vlines(x, ub, lb, color=color)
    return ax


# +
x = np.arange(1, maxb + 2)

fig = plt.figure(figsize=(6, 6))

# ax1 = fig.add_subplot(121)
# ax2 = fig.add_subplot(122)

# ax1.plot(x, lump_sfs(1/np.arange(1,n), maxb)/np.sum(1/np.arange(1,n)), \
#   ':k', label='Kingman')

for i, alpha in enumerate(A_PLOT):
    for j, r in enumerate([0, 0.5, 1.0]):
        f_xibeta = normalize_sfs(fold_twosfs(twosfs_xibeta[(alpha, r)]))
        f_fitted = normalize_sfs(fold_twosfs(twosfs_fitted[(alpha, r)]))

        ax = fig.add_subplot(3, 3, 1 + j + i * 3)
        ax.pcolormesh(
            np.arange(1, 51),
            np.arange(1, 51),
            np.log2(f_xibeta / f_fitted),
            vmin=-2,
            vmax=2,
        )
        ax.set_yscale("log")
        ax.set_xscale("log")

# +
R_PLOT = [0, 0.5, 1.0]

fig = plt.figure(figsize=(8, 6))

for i, alpha in enumerate(A_PLOT):
    for j, r in enumerate(R_PLOT):
        f_xibeta = lump_2sfs(
            normalize_sfs(fold_twosfs(twosfs_xibeta[(alpha, r)])), maxb
        )
        f_fitted = lump_2sfs(
            normalize_sfs(fold_twosfs(twosfs_fitted[(alpha, r)])), maxb
        )

        ax = fig.add_subplot(3, 4, 2 + j + i * 4)
        ax.pcolormesh(
            np.arange(1, maxb + 3),
            np.arange(1, maxb + 3),
            np.log2(f_xibeta / f_fitted),
            vmin=-2,
            vmax=2,
            cmap=r"PuOr",
        )

        if i == 0:
            ax.set_title(r"$r T_2 = {}$".format(r))
        if i == 1 and j == 0:
            ax.set_yticks([1.5, 6.5, 11.5])
            ax.set_yticklabels([1, 6, ">10"])
            ax.set_ylabel("Minor allele count at position $k+d$")
        else:
            ax.set_yticks([])

        if i == 2 and j == 1:
            ax.set_xticks([1.5, 6.5, 11.5])
            ax.set_xticklabels([1, 6, ">10"])
            ax.set_xlabel("Minor allele count at position $k$")
        else:
            ax.set_xticks([])
        if j == 0:
            ax.text(-9, 5.5, r"$\alpha={}$".format(alpha))

fig.savefig("relative_2sfs.pdf")  # , bboxinches="tight")
# -

NPAIRS = [1000, 5000, 10000]

# +
n_resample = 1000
D_kingman = {}
D_xibeta = {}

for alpha in ALPHAS:
    print(alpha)
    for r in RS:
        try:
            f_xibeta = normalize_sfs(fold_twosfs(twosfs_xibeta[(alpha, r)]))
            f_fitted = normalize_sfs(fold_twosfs(twosfs_fitted[(alpha, r)]))
        except KeyError:
            continue

        if f_xibeta.size == 0 or f_fitted.size == 0:
            continue

        F_xibeta = conditional_sfs(f_xibeta)
        F_fitted = conditional_sfs(f_fitted)

        for n_pairs in NPAIRS:
            D_kingman[(alpha, r, n_pairs)] = resample_distance(
                f_fitted, f_fitted, n_pairs, n_resample
            )
            D_xibeta[(alpha, r, n_pairs)] = resample_distance(
                f_xibeta, f_fitted, n_pairs, n_resample
            )
# -

n_pairs = 10000
for alpha in ALPHAS:
    print(alpha)
    fig = plt.figure(figsize=(15, 2))
    for i, r in enumerate(RS):
        ax = fig.add_subplot(1, len(RS), i + 1)
        ax.set_title("$r T_2 = {}$".format(r))
        try:
            violin_with_iqrange(ax, D_kingman[(alpha, r, n_pairs)], "C0")
            violin_with_iqrange(ax, D_xibeta[(alpha, r, n_pairs)], "C1")
        except KeyError:
            continue
    plt.show()


# bins=np.arange(0,400)
r = 1.0
n_pairs = 10000
for alpha in ALPHAS:
    print(alpha)
    fig = plt.figure(figsize=(15, 2))
    for i, r in enumerate(RS):
        ax = fig.add_subplot(1, len(RS), i + 1)
        ax.set_title("$r T_2 = {}$".format(r))
        try:
            d_xibeta = np.sum(D_xibeta[(alpha, r, n_pairs)], axis=1)
            d_fitted = np.sum(D_kingman[(alpha, r, n_pairs)], axis=1)
        except KeyError:
            continue

        lb = np.floor(np.min(np.hstack([d_fitted, d_xibeta])))
        ub = np.ceil(np.max(np.hstack([d_fitted, d_xibeta])))
        bins = np.arange(lb, ub + 1, 1)
        plt.hist(d_xibeta, histtype="step", lw=2, bins=bins)
        plt.hist(d_fitted, histtype="step", lw=2, bins=bins)
    #         plt.xlim([lb, ub])
    plt.show()

# +
fig = plt.figure(figsize=(8, 6))
ymax = 350
xlims = [[20, 40], [0, 200], [0, 200]]

n_pairs = 5000
for i, alpha in enumerate(A_PLOT):
    for j, r in enumerate(R_PLOT):
        ax = fig.add_subplot(3, 4, 2 + j + i * 4)

        d_xibeta = np.sum(D_xibeta[(alpha, r, n_pairs)], axis=1)
        d_fitted = np.sum(D_kingman[(alpha, r, n_pairs)], axis=1)

        lb = np.floor(np.min(np.hstack([d_fitted, d_xibeta])))
        ub = np.ceil(np.max(np.hstack([d_fitted, d_xibeta])))
        bins = np.arange(lb, ub + 1, 1)
        ax.hist(d_fitted, histtype="step", lw=2, bins=bins)
        ax.hist(d_xibeta, histtype="step", lw=2, bins=bins)

        ax.set_xlim(xlims[i])
        ax.set_ylim([0, ymax])
        ax.set_yticks([])
        #         ax.set_xscale('log')
        if i == 0:
            ax.set_title(r"$r T_2 = {}$".format(r))

        if i == 1 and j == 0:
            ax.set_yticks([0, n_resample * 0.15, n_resample * 0.30])
            ax.set_yticklabels([0, "15%", "30%"])
            ax.set_ylabel("Fraction of of simulations")
        #         else:
        #             ax.set_yticks([])

        if i == 2 and j == 1:
            #             ax.set_xticks([1.5, 6.5, 11.5])
            #             ax.set_xticklabels([1, 6, ">10"])
            ax.set_xlabel("Cumulative KS statistic")
        #         else:
        #             ax.set_xticks([])
        if j == 0:
            ax.text(
                xlims[i][0] - 0.95 * (xlims[i][1] - xlims[i][0]),
                350 / 2,
                r"$\alpha={}$".format(alpha),
            )

        if i == 0 and j == 0:
            ax.text(21, 300, "Kingman", color="C0")
            ax.text(39, 300, "Xi-Beta", color="C1", ha="right")
#             ax.plot(-5,5, color="C0", label="Kingman")
#             ax.plot(-5,5, color="C1", label="Xi-Beta")
#             ax.legend()
fig.savefig("test_statistic_histograms.pdf")
# -

power = {}
for r in RS:
    for n_pairs in NPAIRS:
        power[(r, n_pairs)] = np.zeros(len(ALPHAS))
        for i, alpha in enumerate(ALPHAS):
            try:
                total_xibeta = np.sum(D_xibeta[(alpha, r, n_pairs)], axis=1)
                total_fitted = np.sum(D_kingman[(alpha, r, n_pairs)], axis=1)
            except KeyError:
                continue
            #                 power[(r,n_pairs)][i] = np.nan
            power[(r, n_pairs)][i] = np.mean(
                rank(total_xibeta, total_fitted) > 0.95 * n_resample
            )


for r in RS:
    for n_pairs in NPAIRS:
        plt.plot(ALPHAS, power[(r, n_pairs)], ".-", label=n_pairs)
    plt.ylim([-0.05, 1.05])
    plt.title("$r T_2 = {}$".format(r))
    plt.legend(title="Pairs of sites")
    plt.ylabel("Power at 0.05")
    plt.xlabel(r"Coalescent $\alpha$")
    plt.show()

fig = plt.figure(figsize=(8, 2))
for i, r in enumerate(R_PLOT):
    ax = fig.add_subplot(1, 3, i + 1)
    for n_pairs in NPAIRS:
        ax.plot(ALPHAS, power[(r, n_pairs)], ".-", label=n_pairs)
    ax.set_ylim([-0.05, 1.05])
    ax.set_title("$r T_2 = {}$".format(r))
    if i == 0:
        ax.set_ylabel("Power to reject Kingman")
    else:
        ax.set_yticks([])
    if i == 0:
        ax.legend(title="Pairs of sites")
    if i == 1:
        ax.set_xlabel(r"Xi-Beta $\alpha$")
fig.savefig("power.pdf", bbox_inches="tight")

# ## Repeat on lumped 2-SFS

# +
n_resample = 1000
D_kingman = {}
D_xibeta = {}

for alpha in ALPHAS:
    for r in RS:
        f_xibeta = lump_2sfs(
            normalize_sfs(fold_twosfs(twosfs_xibeta[(alpha, r)])), maxb
        )
        f_fitted = lump_2sfs(
            normalize_sfs(fold_twosfs(twosfs_fitted[(alpha, r)])), maxb
        )
        F_xibeta = conditional_sfs(f_xibeta)
        F_fitted = conditional_sfs(f_fitted)

        for n_pairs in NPAIRS:
            D_kingman[(alpha, r, n_pairs)] = resample_distance(
                f_fitted, f_fitted, n_pairs, n_resample
            )
            D_xibeta[(alpha, r, n_pairs)] = resample_distance(
                f_xibeta, f_fitted, n_pairs, n_resample
            )
# -

n_pairs = 10000
for alpha in ALPHAS:
    print(alpha)
    fig = plt.figure(figsize=(15, 2))
    for i, r in enumerate(RS):
        ax = fig.add_subplot(1, len(RS), i + 1)
        ax.set_title("$r T_2 = {}$".format(r))
        try:
            violin_with_iqrange(ax, D_kingman[(alpha, r, n_pairs)], "C0")
            violin_with_iqrange(ax, D_xibeta[(alpha, r, n_pairs)], "C1")
        except KeyError:
            continue
    plt.show()


bins = np.arange(1, 100)
r = 1.0
n_pairs = 1000
for alpha in ALPHAS:
    print(alpha)
    try:
        plt.hist(
            np.sum(D_xibeta[(alpha, r, n_pairs)], axis=1),
            histtype="step",
            lw=2,
            bins=bins,
        )
        plt.hist(
            np.sum(D_kingman[(alpha, r, n_pairs)], axis=1),
            histtype="step",
            lw=2,
            bins=bins,
        )
    except KeyError:
        continue
    plt.xlim([1, np.ceil(np.max(np.sum(D_xibeta[(alpha, r, n_pairs)], axis=1)))])
    plt.show()

power = {}
for r in RS:
    for n_pairs in NPAIRS:
        power[(r, n_pairs)] = np.zeros(len(ALPHAS))
        for i, alpha in enumerate(ALPHAS):
            try:
                total_xibeta = np.sum(D_xibeta[(alpha, r, n_pairs)], axis=1)
                total_fitted = np.sum(D_kingman[(alpha, r, n_pairs)], axis=1)
            except KeyError:
                continue
            power[(r, n_pairs)][i] = np.mean(
                rank(total_xibeta, total_fitted) > 0.95 * n_resample
            )


for r in RS:
    for n_pairs in NPAIRS:
        try:
            plt.plot(ALPHAS, power[(r, n_pairs)], ".-", label=n_pairs)
        except KeyError:
            continue
    plt.ylim([-0.05, 1.05])
    plt.title("$r T_2 = {}$".format(r))
    plt.legend(title="Pairs of sites")
    plt.ylabel("Power at 0.05")
    plt.xlabel(r"Coalescent $\alpha$")
    plt.show()
