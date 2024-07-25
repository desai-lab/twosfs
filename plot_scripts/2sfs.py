import h5py
import numpy as np
import matplotlib.pyplot as plt
from twosfs.spectra import Spectra, spectra_from_hdf5, load_spectra
from twosfs.config import configuration_from_json, parse_parameter_string, make_parameter_string
import json

config = configuration_from_json("../simulation_parameters.json", root = "../")
save_path = "figures/"

######## Parameters in the example plots for each model ########
beta_params = {"alpha": 1.3}
sel_params = {"s": 0.055, "rec_rate": 1e-05, "mut_rate": 1e-10}
exp_params = {"end_time": 0.5, "growth_rate": 2.0}

######## Loading ########

# This is used to fill in file strings, eg: print(config.recombination_search_file.format(**params))
# will give a string to a recombination search file with the parameter values in the dict below
params = {"model": "const",
          "params": make_parameter_string({}),
          "pair_density": 10000,
          "sequence_length": 25,
          "power_rep": 0,
          "folded": True,
          "rep": "all"}
# First load the null 2-SFS for the constant model and save it as const_null
with h5py.File(config.recombination_search_file.format(**params)) as hf:
    if dict(hf.get("spectra_high").attrs)["ks_distance"] < dict(hf.get("spectra_low").attrs)["ks_distance"]:
        spec = spectra_from_hdf5(hf.get("spectra_high"))
        const_null = spec.normalized_twosfs(folded=True, k_max=20)[0,1:,1:]
    else:
        spec = spectra_from_hdf5(hf.get("spectra_low"))
        const_null = spec.normalized_twosfs(folded=True, k_max=20)[0,1:,1:]
# Then load the simulated (data) constant population 2-SFS
spec = load_spectra(config.initial_spectra_file.format(**params))
const_data = spec.normalized_twosfs(folded=True, k_max=20)[0,1:,1:]

# Edit the parameter string for the beta coalescent
params["model"] = "beta"
params["params"] = make_parameter_string(beta_params)
# Load the null 2-SFS for the beta coalescent
with h5py.File(config.recombination_search_file.format(**params)) as hf:
    if dict(hf.get("spectra_high").attrs)["ks_distance"] < dict(hf.get("spectra_low").attrs)["ks_distance"]:
        spec = spectra_from_hdf5(hf.get("spectra_high"))
        beta_null = spec.normalized_twosfs(folded=True, k_max=20)[0,1:,1:]
    else:
        spec = spectra_from_hdf5(hf.get("spectra_low"))
        beta_null = spec.normalized_twosfs(folded=True, k_max=20)[0,1:,1:]
# Load the simulated beta coalescent 2-SFS
spec = load_spectra(config.initial_spectra_file.format(**params))
beta_data = spec.normalized_twosfs(folded=True, k_max=20)[0,1:,1:]

# Do the same with the selective model
params["model"] = "sel"
params["params"] = make_parameter_string(sel_params)
with h5py.File(config.recombination_search_file.format(**params)) as hf:
    if dict(hf.get("spectra_high").attrs)["ks_distance"] < dict(hf.get("spectra_low").attrs)["ks_distance"]:
        spec = spectra_from_hdf5(hf.get("spectra_high"))
        sel_null = spec.normalized_twosfs(folded=True, k_max=20)[0,1:,1:]
    else:
        spec = spectra_from_hdf5(hf.get("spectra_low"))
        sel_null = spec.normalized_twosfs(folded=True, k_max=20)[0,1:,1:]
spec = load_spectra(config.initial_spectra_file.format(**params))
sel_data = spec.normalized_twosfs(folded=True, k_max=20)[0,1:,1:]

# Do the same with the exponential model
params["model"] = "exp"
params["params"] = make_parameter_string(exp_params)
with h5py.File(config.recombination_search_file.format(**params)) as hf:
    if dict(hf.get("spectra_high").attrs)["ks_distance"] < dict(hf.get("spectra_low").attrs)["ks_distance"]:
        spec = spectra_from_hdf5(hf.get("spectra_high"))
        exp_null = spec.normalized_twosfs(folded=True, k_max=20)[0,1:,1:]
    else:
        spec = spectra_from_hdf5(hf.get("spectra_low"))
        exp_null = spec.normalized_twosfs(folded=True, k_max=20)[0,1:,1:]
spec = load_spectra(config.initial_spectra_file.format(**params))
exp_data = spec.normalized_twosfs(folded=True, k_max=20)[0,1:,1:]


######## Plotting ########

fig, axs = plt.subplots(2, 2, figsize = (3.46,3.1))

# Plot all 2-SFS
axs[0,0].pcolormesh( np.log2(const_data / const_null), vmin = -0.5, vmax = 0.5, cmap = "PuOr_r")
# This is plotted with different code to help create the colorbar
exp = axs[0,1].pcolormesh( np.log2(exp_data / exp_null), vmin = -0.5, vmax = 0.5, cmap = "PuOr_r")
axs[1,0].pcolormesh( np.log2(beta_data / beta_null), vmin = -0.5, vmax = 0.5, cmap = "PuOr_r")
axs[1,1].pcolormesh( np.log2(sel_data / sel_null), vmin = -0.5, vmax = 0.5, cmap = "PuOr_r")

######## Formatting ########

# Add titles to all plots
axs[0,0].set_title(r"Const. Kingman", fontsize=7)
axs[0,1].set_title(r"Exponential growth", fontsize=7)
axs[1,0].set_title(r"Beta coalescent", fontsize=7)
axs[1,1].set_title(r"Positive selection", fontsize=7)

# Remove all tick labels
for ax in axs.ravel():
    ax.set_xticks(ticks=[0.5, 9.5, 19.5], labels=[])
    ax.set_yticks(ticks=[0.5, 9.5, 19.5], labels=[])
# Add tick labels to outer axes
axs[0,0].set_yticklabels(["1","10","20+"], fontsize=6)
axs[1,0].set_xticklabels(["1","10","20+"], fontsize=6)
axs[1,0].set_yticklabels(["1","10","20+"], fontsize=6)
axs[1,1].set_xticklabels(["1","10","20+"], fontsize=6)

# Add axis labels
fig.supxlabel(r"Allele frequency site 1", fontsize=7)
fig.supylabel(r"Allele frequency site 2", fontsize=7)

# Initial spacing for the plots
fig.tight_layout(pad=0.3, h_pad=0.5, w_pad=0.5)

# Adjust the spacing slightly to fit the colorbar without distortion
fig.subplots_adjust(right=0.8)
x0 = axs[1,0].get_position().bounds[1]
xt = axs[0,0].get_position().bounds[1] + axs[0,0].get_position().bounds[3]
h = xt - x0
cbar_ax = fig.add_axes([0.85, x0, 0.03, h])
fig.colorbar(exp, cax=cbar_ax, ticks=[-0.5, -0.25, 0, 0.25, 0.5])
cbar_ax.tick_params(labelsize=6)

# Add subplot identifiers
axs[0,0].text(-0.12, 1.09, "a.", fontweight = "bold", transform = axs[0,0].transAxes, fontsize=7)
axs[0,1].text(-0.12, 1.09, "b.", fontweight = "bold", transform = axs[0,1].transAxes, fontsize=7)
axs[1,0].text(-0.12, 1.09, "c.", fontweight = "bold", transform = axs[1,0].transAxes, fontsize=7)
axs[1,1].text(-0.12, 1.09, "d.", fontweight = "bold", transform = axs[1,1].transAxes, fontsize=7)

# Save and close
fig.savefig(save_path + "two_sfs.pdf")
plt.close()

