# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 1
# %aimport twosfs.config, twosfs.spectra, twosfs.statistics

import json
from dataclasses import dataclass

import matplotlib.pyplot as plt
import msprime
import numpy as np
from scipy.special import rel_entr

import twosfs.statistics as stats
from twosfs.config import configuration_from_json
from twosfs.spectra import (
    Spectra,
    foldonesfs,
    foldtwosfs,
    load_spectra,
    lump_onesfs,
    lump_twosfs,
)


@dataclass
class Data:
    model: str
    params: dict
    spectra: Spectra
    demography: dict
    spectra_fitted: Spectra


def kl_div(x, y, axis=None):
    return np.sum(rel_entr(x, y), axis=axis)


configuration = configuration_from_json("../simulation_parameters.json")
kmax = configuration.k_max
num_samples = (
    configuration.msprime_parameters["samples"]
    * configuration.msprime_parameters["ploidy"]
)
k = np.arange(num_samples + 1)
const_onesfs = np.zeros_like(k, dtype=float)
const_onesfs[1:-1] = 1 / k[1:-1]
const_onesfs /= np.sum(const_onesfs)

data = []
root = "../"
folded = True
rec_factor = 1.0
for model, params in configuration.iter_models():
    with open(
        root + configuration.format_fitted_demography_file(model, params, folded)
    ) as f:
        demography = json.load(f)
    data.append(
        Data(
            model,
            params,
            load_spectra(
                root + configuration.format_initial_spectra_file(model, params)
            ),
            demography,
            load_spectra(
                root
                + configuration.format_fitted_spectra_file(
                    model, params, folded, rec_factor
                )
            ),
        )
    )


d.spectra.normalized_onesfs(folded=False)

d.spectra.onesfs / np.sum(d.spectra.onesfs)

type(np.array([True, False])[0])

type(bool(np.array([True, False])[0]))

# ?np.all

for d in data:
    print(
        d.model,
        d.params,
        sep="\t",
    )
    print(
        round(d.spectra.scaled_recombination_rate(), ndigits=4),
        round(d.spectra_fitted.scaled_recombination_rate(), ndigits=4),
        sep="\t",
    )
    sfs_sim = d.spectra.normalized_onesfs(folded=folded, kmax=kmax)[1:]
    sfs_sim_fit = d.spectra_fitted.normalized_onesfs(folded=folded, kmax=kmax)[1:]
    sfs_exp = d.demography["sfs_exp"]
    sfs_obs = d.demography["sfs_obs"]
    print(d.demography["kl_div"])
    print(kl_div(sfs_obs, sfs_exp))
    print(kl_div(sfs_obs, sfs_sim))
    print(kl_div(sfs_obs, sfs_sim_fit))


for d in data:
    onesfs = lump_onesfs(d.spectra.normalized_onesfs(folded=folded), kmax=kmax)
    onesfs_fitted = lump_onesfs(
        d.spectra_fitted.normalized_onesfs(folded=folded), kmax=kmax
    )
    twosfs = lump_twosfs(d.spectra.normalized_twosfs(folded=folded), kmax=kmax)
    twosfs_fitted = lump_twosfs(
        d.spectra_fitted.normalized_twosfs(folded=folded), kmax=kmax
    )
    print(
        d.model,
        d.params,
        kl_div(onesfs, onesfs_fitted),
        sep="\t",
    )

    sfs_sim = lump_onesfs(d.spectra.normalized_onesfs(folded=folded), kmax=kmax)[1:]
    sfs_sim_fit = lump_onesfs(
        d.spectra_fitted.normalized_onesfs(folded=folded), kmax=kmax
    )[1:]
    sfs_exp = d.demography["sfs_exp"]
    sfs_obs = d.demography["sfs_obs"]
    print(d.demography["kl_div"])
    print(kl_div(sfs_obs, sfs_exp))
    print(kl_div(sfs_obs, sfs_sim))
    print(kl_div(sfs_obs, sfs_sim_fit))

    plt.plot(onesfs, "o")
    plt.plot(onesfs_fitted, "xk")
    plt.show()

for d in data:
    twosfs = lump_twosfs(d.spectra.normalized_twosfs(folded=folded), kmax=kmax)
    twosfs_fitted = lump_twosfs(
        d.spectra_fitted.normalized_twosfs(folded=folded), kmax=kmax
    )

    if d.model == "beta":
        color = "C0"
    else:
        color = "C1"
    plt.semilogy(
        d.spectra.windows[:-1],
        kl_div(twosfs, twosfs_fitted, axis=(1, 2)),
        ".",
        color=color,
    )
    plt.ylim([1e-6, 1e-1])
    plt.ylabel("2-SFS KL divergence")
    plt.xlabel("Distance")
plt.show()

for dist in [0, 1, 5, 10]:
    for d in data:
        onesfs = lump_onesfs(d.spectra.normalized_onesfs(folded=folded), kmax=kmax)
        onesfs_fitted = lump_onesfs(
            d.spectra_fitted.normalized_onesfs(folded=folded), kmax=kmax
        )
        twosfs = lump_twosfs(d.spectra.normalized_twosfs(folded=folded), kmax=kmax)
        twosfs_fitted = lump_twosfs(
            d.spectra_fitted.normalized_twosfs(folded=folded), kmax=kmax
        )

        if d.model == "beta":
            color = "C0"
        else:
            color = "C1"

        # kld_1 = kl_div(lump_onesfs(const_onesfs, kmax=kmax), onesfs)
        kld_1 = kl_div(onesfs, onesfs_fitted)
        kld_2 = kl_div(twosfs[dist], twosfs_fitted[dist])
        plt.loglog(kld_1, kld_2, ".", color=color)
        plt.ylabel(r"2-SFS KL divergence bw. real and fitted")
        plt.xlabel(r"1-SFS KL divergence bw. real and fitted")
        plt.title(f"Distance = {dist}")
        # plt.ylim([-0.001, 0.015])
    plt.show()

for dist in [0, 1, 5, 10]:
    for d in data:
        onesfs = lump_onesfs(d.spectra.normalized_onesfs(folded=folded), kmax=kmax)
        onesfs_fitted = lump_onesfs(
            d.spectra_fitted.normalized_onesfs(folded=folded), kmax=kmax
        )
        twosfs = lump_twosfs(d.spectra.normalized_twosfs(folded=folded), kmax=kmax)
        twosfs_fitted = lump_twosfs(
            d.spectra_fitted.normalized_twosfs(folded=folded), kmax=kmax
        )

        if d.model == "beta":
            color = "C0"
        else:
            color = "C1"

        if folded:
            kld_1 = kl_div(lump_onesfs(foldonesfs(const_onesfs), kmax=kmax), onesfs)
        else:
            kld_1 = kl_div(lump_onesfs(const_onesfs, kmax=kmax), onesfs)
        kld_2 = kl_div(twosfs[dist], twosfs_fitted[dist])
        plt.plot(kld_1, kld_2, ".", color=color)
        plt.ylabel(r"2-SFS KL divergence bw. real and fitted")
        plt.xlabel(r"1-SFS KL divergence from constant-$N$")
        plt.title(f"Distance = {dist}")
        plt.ylim([-0.001, 0.015])
    plt.show()

for dist in [0, 1, 5, 10, 20, 40]:
    for d in data:
        onesfs = lump_onesfs(d.spectra.normalized_onesfs(), kmax=kmax)
        onesfs_fitted = lump_onesfs(d.spectra_fitted.normalized_onesfs(), kmax=kmax)
        twosfs = lump_twosfs(d.spectra.normalized_twosfs(), kmax=kmax)
        twosfs_fitted = lump_twosfs(d.spectra_fitted.normalized_twosfs(), kmax=kmax)

        if d.model == "beta":
            color = "C0"
        else:
            color = "C1"

        if folded:
            ksd_1 = stats.max_ks_distance(
                lump_onesfs(foldonesfs(const_onesfs), kmax=kmax), onesfs
            )
        else:
            ksd_1 = stats.max_ks_distance(lump_onesfs(const_onesfs, kmax=kmax), onesfs)
        ksd_2 = stats.max_ks_distance(twosfs[dist], twosfs_fitted[dist])
        plt.plot(ksd_1, ksd_2, ".", color=color)
        plt.ylabel(r"2-SFS KS distance bw. real and fitted")
        plt.xlabel(r"1-SFS KS distance from constant-$N$")
        plt.title(f"Distance = {dist}")
        plt.ylim([-0.001, 0.05])
    plt.show()

for max_dist in [1, 5, 10, 20, 40]:
    for d in data:
        onesfs = lump_onesfs(d.spectra.normalized_onesfs(), kmax=kmax)
        onesfs_fitted = lump_onesfs(d.spectra_fitted.normalized_onesfs(), kmax=kmax)
        twosfs = lump_twosfs(d.spectra.normalized_twosfs(), kmax=kmax)
        twosfs_fitted = lump_twosfs(d.spectra_fitted.normalized_twosfs(), kmax=kmax)

        if d.model == "beta":
            color = "C0"
        else:
            color = "C1"

        twosfs_adj = twosfs[:max_dist] / max_dist
        twosfs_adj_fitted = twosfs_fitted[:max_dist] / max_dist
        ksd_1 = stats.max_ks_distance(lump_onesfs(const_onesfs, kmax=kmax), onesfs)
        ksd_2 = stats.max_ks_distance(twosfs_adj, twosfs_adj_fitted)
        plt.plot(ksd_1, ksd_2, ".", color=color)
        plt.ylabel(r"2-SFS KS distance bw. real and fitted")
        plt.xlabel(r"1-SFS KS distance from constant-$N$")
        plt.title(f"Max distance = {max_dist}")
        plt.ylim([-0.001, 0.05])
    plt.show()

for max_dist in [1, 5, 10, 20, 40]:
    for d in data:
        onesfs = lump_onesfs(d.spectra.normalized_onesfs(), kmax=kmax)
        onesfs_fitted = lump_onesfs(d.spectra_fitted.normalized_onesfs(), kmax=kmax)
        twosfs = lump_twosfs(d.spectra.normalized_twosfs(), kmax=kmax)
        twosfs_fitted = lump_twosfs(d.spectra_fitted.normalized_twosfs(), kmax=kmax)

        if d.model == "beta":
            color = "C0"
        else:
            color = "C1"

        twosfs_adj = twosfs[:max_dist] / max_dist
        twosfs_adj_fitted = twosfs_fitted[:max_dist] / max_dist
        kld_1 = kl_div(lump_onesfs(const_onesfs, kmax=kmax), onesfs)
        kld_2 = kl_div(twosfs_adj, twosfs_adj_fitted)
        plt.plot(kld_1, kld_2, ".", color=color)
        plt.ylabel(r"2-SFS KL divergence bw. real and fitted")
        plt.xlabel(r"1-SFS KL divergence from constant-$N$")
        plt.title(f"Max distance = {max_dist}")
        plt.ylim([-0.001, 0.015])
    plt.show()

# +
d = data[15]
print(d.model, d.params)

n_reps = 1000
pair_density = 5000
bins = np.arange(0, 4, 0.05)

twosfs = lump_twosfs(d.spectra.normalized_twosfs(folded=folded), kmax=kmax)
twosfs_fitted = lump_twosfs(
    d.spectra_fitted.normalized_twosfs(folded=folded), kmax=kmax
)


for max_dist in np.arange(4, 26, 3):
    num_pairs = np.zeros(twosfs.shape[0], dtype=int)
    for i in range(3, max_dist, 3):
        num_pairs[i] = pair_density

    nonzero = num_pairs > 0

    n_total = np.sum(num_pairs)
    ks_comp = np.zeros(n_reps)
    ks_null = np.zeros(n_reps)
    for i in range(n_reps):
        twosfs_resampled = stats.resample_twosfs_pdf(
            twosfs[nonzero], num_pairs[nonzero]
        )
        twosfs_fitted_resampled = stats.resample_twosfs_pdf(
            twosfs_fitted[nonzero], num_pairs[nonzero]
        )
        twosfs_null = twosfs_fitted[nonzero] * num_pairs[nonzero, None, None] / n_total

        ks_comp[i] = stats.max_ks_distance(twosfs_resampled, twosfs_null) * np.sqrt(
            n_total
        )
        ks_null[i] = stats.max_ks_distance(
            twosfs_fitted_resampled, twosfs_null
        ) * np.sqrt(n_total)

    power = np.mean(np.mean(ks_comp[:, None] <= ks_null[None, :], axis=1) < 0.05)

    plt.hist(ks_comp, histtype="step", bins=bins)
    plt.hist(ks_null, histtype="step", bins=bins)
    plt.title(max_dist - 1)
    print(f"Power = {power}")
    plt.show()

# +
n_reps = 1000
pair_density = 5000
max_dists = np.arange(4, 26, 3)

for d in data:
    twosfs = lump_twosfs(d.spectra.normalized_twosfs(folded=folded), kmax=kmax)
    twosfs_fitted = lump_twosfs(
        d.spectra_fitted.normalized_twosfs(folded=folded), kmax=kmax
    )

    power = []
    for max_dist in max_dists:
        num_pairs = np.zeros(twosfs.shape[0], dtype=int)
        for i in range(3, max_dist, 3):
            num_pairs[i] = pair_density

        nonzero = num_pairs > 0

        n_total = np.sum(num_pairs)
        ks_comp = np.zeros(n_reps)
        ks_null = np.zeros(n_reps)
        for i in range(n_reps):
            twosfs_resampled = stats.resample_twosfs_pdf(
                twosfs[nonzero], num_pairs[nonzero]
            )
            twosfs_fitted_resampled = stats.resample_twosfs_pdf(
                twosfs_fitted[nonzero], num_pairs[nonzero]
            )
            twosfs_null = (
                twosfs_fitted[nonzero] * num_pairs[nonzero, None, None] / n_total
            )

            ks_comp[i] = stats.max_ks_distance(twosfs_resampled, twosfs_null) * np.sqrt(
                n_total
            )
            ks_null[i] = stats.max_ks_distance(
                twosfs_fitted_resampled, twosfs_null
            ) * np.sqrt(n_total)

        power.append(
            np.mean(np.mean(ks_comp[:, None] <= ks_null[None, :], axis=1) < 0.05)
        )

    if d.model == "beta":
        color = "C0"
    else:
        color = "C1"
    plt.plot(max_dists, power, color=color)
plt.show()

# +
n_reps = 1000
pair_density = 20000
max_dists = np.arange(4, 26, 3)

for d in data[10:]:
    twosfs = lump_twosfs(d.spectra.normalized_twosfs(folded=folded), kmax=kmax)
    twosfs_fitted = lump_twosfs(
        d.spectra_fitted.normalized_twosfs(folded=folded), kmax=kmax
    )

    power = []
    for max_dist in max_dists:
        num_pairs = np.zeros(twosfs.shape[0], dtype=int)
        for i in range(3, max_dist, 3):
            num_pairs[i] = pair_density

        nonzero = num_pairs > 0

        n_total = np.sum(num_pairs)
        ks_comp = np.zeros(n_reps)
        ks_null = np.zeros(n_reps)
        for i in range(n_reps):
            twosfs_resampled = stats.resample_twosfs_pdf(
                twosfs[nonzero], num_pairs[nonzero]
            )
            twosfs_fitted_resampled = stats.resample_twosfs_pdf(
                twosfs_fitted[nonzero], num_pairs[nonzero]
            )
            twosfs_null = (
                twosfs_fitted[nonzero] * num_pairs[nonzero, None, None] / n_total
            )

            ks_comp[i] = stats.max_ks_distance(twosfs_resampled, twosfs_null) * np.sqrt(
                n_total
            )
            ks_null[i] = stats.max_ks_distance(
                twosfs_fitted_resampled, twosfs_null
            ) * np.sqrt(n_total)

        power.append(
            np.mean(np.mean(ks_comp[:, None] <= ks_null[None, :], axis=1) < 0.05)
        )

    if d.model == "beta":
        color = "C0"
    else:
        color = "C1"
    plt.plot(max_dists, power, color=color)
plt.show()
