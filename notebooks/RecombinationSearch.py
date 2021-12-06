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

model = "beta"
alpha = 1.6
params = {"alpha": alpha}
folded = False
configuration = config.configuration_from_json("../simulation_parameters.json")
root_dir = "../"
raw_spectra = spectra.load_spectra(
    root_dir + configuration.format_initial_spectra_file(model, params)
)
with open(
    root_dir
    + configuration.format_fitted_demography_file(
        model=model, params=params, folded=folded
    )
) as f:
    model_parameters = json.load(f)
true_r = configuration.scaled_recombination_rate

max_distance = 25
pair_density = 10000
num_pairs = pair_density * statistics.degenerate_pairs(raw_spectra, max_distance)
spectra_samp = statistics.sample_spectra(raw_spectra, num_pairs=num_pairs)

k_max = configuration.k_max
msprime_parameters = {
    "samples": 50,
    "ploidy": 2,
    "sequence_length": max_distance,
    "num_replicates": 200,
}
sim_kwargs = dict(
    model="pwc",
    model_parameters=model_parameters,
    msprime_parameters=msprime_parameters,
    random_seed=np.random.randint(0, 10000),
)
(x_l, x_u), (f_l, f_u) = statistics.golden_section_search(
    statistics.simulate_ks, 0.9, 1.4, 4, spectra_samp, k_max, folded, **sim_kwargs
)

print(x_l, x_u)

print(f_l[0], f_u[0])
