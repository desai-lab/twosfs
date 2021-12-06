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
# %autoreload 2

import json

import numpy as np

from twosfs import config, spectra, statistics

seed = 1000
max_distance = 25
num_replicates = 200
pair_density = 10000
model = "beta"
alpha = 1.6
params = {"alpha": alpha}
folded = False

configuration = config.configuration_from_json("../simulation_parameters.json")
root_dir = "../"
spectra_file = root_dir + configuration.format_initial_spectra_file(model, params)
demo_file = root_dir + configuration.format_fitted_demography_file(
    model=model, params=params, folded=folded
)
k_max = configuration.k_max
msprime_parameters = configuration.msprime_parameters
msprime_parameters["sequence_length"] = max_distance
msprime_parameters["num_replicates"] = num_replicates

raw_spectra = spectra.load_spectra(spectra_file)
with open(demo_file) as f:
    model_parameters = json.load(f)

rng = np.random.default_rng(seed)

num_pairs = pair_density * statistics.degenerate_pairs(raw_spectra, max_distance)
spectra_samp = statistics.sample_spectra(raw_spectra, num_pairs=num_pairs, rng=rng)

sim_kwargs = dict(
    model="pwc",
    model_parameters=model_parameters,
    msprime_parameters=msprime_parameters,
    random_seed=rng,
)
(x_l, x_u), (f_l, f_u) = statistics.golden_section_search(
    statistics.simulate_ks, 0.9, 1.4, 4, spectra_samp, k_max, folded, **sim_kwargs
)

print(x_l, x_u)

print(f_l[0], f_u[0])
