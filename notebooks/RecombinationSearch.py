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
import tempfile

import h5py
import numpy as np

from twosfs import config, spectra, statistics

# +
seed = 1000
output_file = tempfile.TemporaryFile()

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

pair_density = configuration.power_pair_densities[0]
sequence_length = configuration.power_sequence_lengths[0]
# -

rng = np.random.default_rng(seed)
raw_spectra = spectra.load_spectra(spectra_file)
num_pairs = pair_density * statistics.degenerate_pairs(raw_spectra, sequence_length)
spectra_samp = statistics.sample_spectra(raw_spectra, num_pairs=num_pairs, rng=rng)

with open(demo_file) as f:
    model_parameters = json.load(f)
msprime_parameters = configuration.msprime_parameters
msprime_parameters["sequence_length"] = sequence_length
msprime_parameters["num_replicates"] = configuration.search_num_replicates
sim_kwargs = dict(
    model="pwc",
    model_parameters=model_parameters,
    msprime_parameters=msprime_parameters,
    random_seed=rng,
)


statistics.search_recombination_rates_save(
    output_file,
    spectra_samp,
    configuration.k_max,
    folded,
    sim_kwargs,
    configuration.search_r_low,
    configuration.search_r_high,
    configuration.search_iters,
)
with h5py.File(output_file, "r") as f:
    for x in f:
        print(x)
        print(dict(f[x].attrs))  # type: ignore
        # print(spectra.spectra_from_hdf5(f[x]))  # type: ignore
