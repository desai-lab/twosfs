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
import gzip

import matplotlib.pyplot as plt
import numpy as np

import twosfs.statistics as stats
from twosfs.config import configuration_from_json
from twosfs.spectra import Spectra, foldonesfs, load_spectra, lump_onesfs, lump_twosfs

configuration = configuration_from_json("../simulation_parameters.json")

print(configuration)

# ls ../simulations

root = "../"
for fn1, fn2 in zip(configuration.fitted_ks_distance_files(), configuration.initial_ks_distance_files()):
    print(fn1)
    print(fn2)

root = "../"
folded = True
data = dict()
for model, params in configuration.iter_models():
    for rec_factor in configuration.rec_factors:
        fn = root + configuration.format_fitted_ks_distance_file(model=model, params=params, folded=folded, rec_factor=rec_factor)
        with gzip.open(fn) as f:
            data[(model, str(params), rec_factor)] = json.load(f)


def get_ks_stats(data, pair_density, max_distance):
    return next(filter(lambda x: x["pair_density"] == pair_density and x["max_distance"] == max_distance, data))["ks_stats"]


model = "beta"
params = {"alpha": 1.5}
pair_density = 1000
max_distance = 10
ks_null = np.array(get_ks_stats(data[(model, str(params), 1.0)], pair_density, max_distance))
for rec_factor in configuration.rec_factors:
    print(rec_factor)
    ks_comp = np.array(get_ks_stats(data[(model, str(params), rec_factor)], pair_density, max_distance))
    p_vals = np.sum(ks_comp[:,None] > ks_null[None,:], axis=1) / configuration.n_reps
    plt.hist(p_vals)
    print(np.mean(p_vals > 0.95))
    plt.show()

model = "beta"
params = {"alpha": 1.25}
pair_density = 1000
max_distance = 10
ks_null = np.array(get_ks_stats(data[(model, str(params), 1.0)], pair_density, max_distance))
for rec_factor in configuration.rec_factors:
    print(rec_factor)
    ks_comp = np.array(get_ks_stats(data[(model, str(params), rec_factor)], pair_density, max_distance))
    p_vals = np.sum(ks_comp[:,None] > ks_null[None,:], axis=1) / configuration.n_reps
    plt.hist(p_vals)
    print(np.mean(p_vals > 0.95))
    plt.show()

model = "const"
params = {}
pair_density = 1000
max_distance = 10
ks_null = np.array(get_ks_stats(data[(model, str(params), 1.0)], pair_density, max_distance))
for rec_factor in configuration.rec_factors:
    print(rec_factor)
    ks_comp = np.array(get_ks_stats(data[(model, str(params), rec_factor)], pair_density, max_distance))
    p_vals = np.sum(ks_comp[:,None] > ks_null[None,:], axis=1) / configuration.n_reps
    plt.hist(p_vals)
    print(np.mean(p_vals > 0.95))
    plt.show()

model = "exp"
params = {"end_time":1.0, "growth_rate":1.0}
pair_density = 1000
max_distance = 10
ks_null = np.array(get_ks_stats(data[(model, str(params), 1.0)], pair_density, max_distance))
for rec_factor in configuration.rec_factors:
    print(rec_factor)
    ks_comp = np.array(get_ks_stats(data[(model, str(params), rec_factor)], pair_density, max_distance))
    p_vals = np.sum(ks_comp[:,None] > ks_null[None,:], axis=1) / configuration.n_reps
    plt.hist(p_vals)
    print(np.mean(p_vals > 0.95))
    plt.show()

# ls ../simulations/fitted_ks_distances/

model = "beta"
pair_density = 10000
max_distance = 19
for alpha in configuration.alphas:
    print(alpha)
    params = {"alpha": alpha}
    ks_null = np.array(get_ks_stats(data[(model, str(params), 1.0)], pair_density, max_distance))
    for rec_factor in configuration.rec_factors:
        ks_comp = np.array(get_ks_stats(data[(model, str(params), rec_factor)], pair_density, max_distance))
        fpr = np.mean(np.mean(ks_comp[:, None] <= ks_null[None, :], axis=1) < 0.05)
        plt.semilogx(rec_factor, fpr, 'ok')
    plt.ylim([0,1])
    plt.show()

model = "exp"
pair_density = 10000
max_distance = 19
for growth_rate in configuration.growth_rates:
    for end_time in configuration.end_times:
        print(growth_rate, end_time)
        params = {"end_time": end_time, "growth_rate": growth_rate}
        ks_null = np.array(get_ks_stats(data[(model, str(params), 1.0)], pair_density, max_distance))
        for rec_factor in configuration.rec_factors:
            ks_comp = np.array(get_ks_stats(data[(model, str(params), rec_factor)], pair_density, max_distance))
            fpr = np.mean(np.mean(ks_comp[:, None] <= ks_null[None, :], axis=1) < 0.05)
            plt.semilogx(rec_factor, fpr, 'ok')
        plt.ylim([0,1])
        plt.show()


