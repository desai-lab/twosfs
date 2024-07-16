import h5py
import numpy as np
import matplotlib.pyplot as plt
from twosfs.spectra import Spectra, spectra_from_hdf5, load_spectra
from twosfs.config import configuration_from_json, parse_parameter_string, make_parameter_string
import json

config = configuration_from_json("../simulation_parameters.json")
reps = 100

alphas = [1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0]
growth_rates = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
end_times = [0.5, 1.0, 2.0]
mut_rates = [1e-10, 1e-11]
s_vals = [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08]

beta_params = '"alpha":{}'
exp_params = '"end_time":{},"growth_rate":{}'
sel_params = '"s":{},"rec_rate":1e-05,"mut_rate":{}'

demo_file = '../simulations/fitted_demographies/model={}.params={{{}}}.folded=True.txt'
rec_file = '../simulations/recombination_search/model={}.params={{{}}}.folded=True.pair_density=10000.sequence_length=25.power_rep={}.hdf5'
sim_file = '../simulations/initial_spectra/model={}.params={{{}}}.rep=all.hdf5'
ks_file = '../simulations/ks_distances/model={}.params={{{}}}.folded=True.pair_density=10000.sequence_length=25.power_rep={}.hdf5'

### Beta coalescent ###
power_beta = []
tajd_beta = []
for a in alphas:
    p_vals = []
    if a == 2.0:
        params = ("const", "")
    else:
        params = ("beta", beta_params.format(a))
    for rep in range(reps):
        with h5py.File(rec_file.format(*params, rep)) as hf:
            ks_h = hf.get("spectra_high").attrs["ks_distance"]
            ks_l = hf.get("spectra_low").attrs["ks_distance"]
        with h5py.File(ks_file.format(*params, rep)) as hf:
            if ks_h > ks_l:
                p_vals.append(sum( ks_l < np.array(hf.get("ks_null")[:])) / len (hf.get("ks_null")[:]))
            else:
                p_vals.append(sum( ks_h < np.array(hf.get("ks_null")[:])) / len (hf.get("ks_null")[:]))

    power_beta.append(sum(0.05 > np.array(p_vals)) / reps)
    tajd_beta.append(load_spectra(sim_file.format(*params)).tajimas_d())

np.save("power_files/beta_power.npy", power_beta)
np.save("power_files/beta_tajd.npy", tajd_beta)

### Selective ###
power_sel = []
tajd_sel = []
for mu in mut_rates:
    for s in s_vals:
        p_vals = []
        if s == 0:
            params = ("sel", sel_params.format(0, "1e-09"))
        else:
            params = ("sel", sel_params.format(s, mu))
        for rep in range(reps):
            with h5py.File(rec_file.format(*params, rep)) as hf:
                ks_h = hf.get("spectra_high").attrs["ks_distance"]
                ks_l = hf.get("spectra_low").attrs["ks_distance"]
            with h5py.File(ks_file.format(*params, rep)) as hf:
                if ks_h > ks_l:
                    p_vals.append(sum( ks_l < np.array(hf.get("ks_null")[:])) / len (hf.get("ks_null")[:]))
                else:
                    p_vals.append(sum( ks_h < np.array(hf.get("ks_null")[:])) / len (hf.get("ks_null")[:]))

        power_sel.append(sum(0.05 > np.array(p_vals)) / reps)
        tajd_sel.append(load_spectra(sim_file.format(*params)).tajimas_d())

np.save("power_files/sel_power.npy", power_sel)
np.save("power_files/sel_tajd.npy", tajd_sel)

### Exponential ###
power_exp = []
tajd_exp = []
for t in end_times:
    for g in growth_rates:
        p_vals = []
        if g == 0:
            params = ("const", "")
        else:
            params = ("exp", exp_params.format(t, g))
        for rep in range(reps):
            with h5py.File(rec_file.format(*params, rep)) as hf:
                ks_h = hf.get("spectra_high").attrs["ks_distance"]
                ks_l = hf.get("spectra_low").attrs["ks_distance"]
            with h5py.File(ks_file.format(*params, rep)) as hf:
                if ks_h > ks_l:
                    p_vals.append(sum( ks_l < np.array(hf.get("ks_null")[:])) / len (hf.get("ks_null")[:]))
                else:
                    p_vals.append(sum( ks_h < np.array(hf.get("ks_null")[:])) / len (hf.get("ks_null")[:]))

        power_exp.append(sum(0.05 > np.array(p_vals)) / reps)
        tajd_exp.append(load_spectra(sim_file.format(*params)).tajimas_d())

np.save("power_files/exp_power.npy", power_exp)
np.save("power_files/exp_tajd.npy", tajd_exp)

