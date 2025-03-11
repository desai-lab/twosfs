import h5py
import numpy as np
import matplotlib.pyplot as plt
from twosfs.spectra import Spectra, spectra_from_hdf5, load_spectra
from twosfs.config import configuration_from_json, parse_parameter_string, make_parameter_string
from twosfs.analysis import get_p_value, get_power
import json

config = configuration_from_json("../simulation_parameters.json", root = "../")
save_file = "power.txt"

# Load the pair density and sequence length
pd = config.power_pair_densities[0]
sl = config.power_sequence_lengths[0]

# Clear the power file and write the header
with open(save_file, "w") as sf:
    sf.write("Model\tp-val\tTajima's D\n")

# Calculate and write power for the backwards-time models
for model in config.iter_models():
    p = get_power(model[0], model[1], True, pd, sl)
    t_d = load_spectra(
                config.initial_spectra_file.format(model=model[0],
                params=make_parameter_string(model[1]), rep="all")).tajimas_d()
    with open(save_file, "a") as sf:
        sf.write(f"{model[0]}, {make_parameter_string(model[1])}\t{p}\t{t_d}\n")

# Calculate and write power for the forwards-time models
for model in config.iter_forward_models():
    p = get_power(model[0], model[1], True, pd, sl)
    t_d = load_spectra(
                config.initial_spectra_file.format(model=model[0],
                params=make_parameter_string(model[1]), rep="all")).tajimas_d()
    with open(save_file, "a") as sf:
        sf.write(f"{model[0]}, {make_parameter_string(model[1])}\t{p}\t{t_d}\n")

"""
### Beta coalescent ###
for a in alphas:
    p_vals = []
    if a == 2.0:
        params = ("const", "")
    else:
        params = ("beta", beta_params.format(a))
    for rep in range(reps):
        p_vals.append(get_p_value(rec_file.format(*params, rep), ks_file.format(*params, rep)))

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
            p_vals.append(get_p_value(rec_file.format(*params, rep), ks_file.format(*params, rep)))
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
            p_vals.append(get_p_value(rec_file.format(*params, rep), ks_file.format(*params, rep)))

        power_exp.append(sum(0.05 > np.array(p_vals)) / reps)
        tajd_exp.append(load_spectra(sim_file.format(*params)).tajimas_d())

np.save("power_files/exp_power.npy", power_exp)
np.save("power_files/exp_tajd.npy", tajd_exp)
"""
