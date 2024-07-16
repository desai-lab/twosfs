import h5py
import numpy as np
import matplotlib.pyplot as plt
from twosfs.spectra import Spectra, spectra_from_hdf5, load_spectra
from twosfs.config import configuration_from_json, parse_parameter_string, make_parameter_string
import json

config = configuration_from_json("../simulation_parameters.json")
save_path = "figures/"

beta_params = '"alpha":{}'
exp_params = '"end_time":{},"growth_rate":{}'
sel_params = '"s":{},"rec_rate":1e-05,"mut_rate":{}'

demo_file = '../simulations/fitted_demographies/model={}.params={{{}}}.folded=True.txt'
rec_file = '../simulations/recombination_search/model={}.params={{{}}}.folded=True.pair_density=10000.sequence_length=25.power_rep={}.hdf5'
sim_file = '../simulations/initial_spectra/model={}.params={{{}}}.rep=all.hdf5'
ks_file = '../simulations/ks_distances/model={}.params={{{}}}.folded=True.pair_density=10000.sequence_length=25.power_rep={}.hdf5'


######## Loading ########

params = ("const", "", 0)
with h5py.File(rec_file.format(*params)) as hf:
    if dict(hf.get("spectra_high").attrs)["ks_distance"] < dict(hf.get("spectra_low").attrs)["ks_distance"]:
        const_null = spectra_from_hdf5(hf.get("spectra_high")).normalized_twosfs(folded=True, k_max=20)[0,1:,1:]
    else:
        const_null = spectra_from_hdf5(hf.get("spectra_low")).normalized_twosfs(folded=True, k_max=20)[0,1:,1:]
spec_const = load_spectra(sim_file.format(*params[:-1]))
const_data = load_spectra(sim_file.format(*params[:-1])).normalized_twosfs(folded=True, k_max=20)[0,1:,1:]


alpha = 1.3
params = ("beta", beta_params.format(alpha), 0)
with h5py.File(rec_file.format(*params)) as hf:
    if dict(hf.get("spectra_high").attrs)["ks_distance"] < dict(hf.get("spectra_low").attrs)["ks_distance"]:
        beta_null = spectra_from_hdf5(hf.get("spectra_high")).normalized_twosfs(folded=True, k_max=20)[0,1:,1:]
    else:
        beta_null = spectra_from_hdf5(hf.get("spectra_low")).normalized_twosfs(folded=True, k_max=20)[0,1:,1:]
beta_data = load_spectra(sim_file.format(*params[:-1])).normalized_twosfs(folded=True, k_max=20)[0,1:,1:]
print("beta")
print(load_spectra(sim_file.format(*params[:-1])).tajimas_d())

s = 0.055
mu = 1e-10
params = ("sel", sel_params.format(s, mu), 0)
with h5py.File(rec_file.format(*params)) as hf:
    if dict(hf.get("spectra_high").attrs)["ks_distance"] < dict(hf.get("spectra_low").attrs)["ks_distance"]:
        sel_null = spectra_from_hdf5(hf.get("spectra_high")).normalized_twosfs(folded=True, k_max=20)[0,1:,1:]
    else:
        sel_null = spectra_from_hdf5(hf.get("spectra_low")).normalized_twosfs(folded=True, k_max=20)[0,1:,1:]
sel_data = load_spectra(sim_file.format(*params[:-1])).normalized_twosfs(folded=True, k_max=20)[0,1:,1:]
print("sel")
print(load_spectra(sim_file.format(*params[:-1])).tajimas_d())

g = 2.0
t = 0.5
params = ("exp", exp_params.format(t, g), 0)
with h5py.File(rec_file.format(*params)) as hf:
    if dict(hf.get("spectra_high").attrs)["ks_distance"] < dict(hf.get("spectra_low").attrs)["ks_distance"]:
        exp_null = spectra_from_hdf5(hf.get("spectra_high")).normalized_twosfs(folded=True, k_max=20)[0,1:,1:]
    else:
        exp_null = spectra_from_hdf5(hf.get("spectra_low")).normalized_twosfs(folded=True, k_max=20)[0,1:,1:]
exp_data = load_spectra(sim_file.format(*params[:-1])).normalized_twosfs(folded=True, k_max=20)[0,1:,1:]
print("exp")
print(load_spectra(sim_file.format(*params[:-1])).tajimas_d())


######## Plotting ########

fig, axs = plt.subplots(2, 2, figsize = (3.46,3.1))
# 2-SFS #
axs[0,0].pcolormesh( np.log2(const_data / const_null), vmin = -0.5, vmax = 0.5, cmap = "PuOr_r")
exp = axs[0,1].pcolormesh( np.log2(exp_data / exp_null), vmin = -0.5, vmax = 0.5, cmap = "PuOr_r")
axs[1,0].pcolormesh( np.log2(beta_data / beta_null), vmin = -0.5, vmax = 0.5, cmap = "PuOr_r")
axs[1,1].pcolormesh( np.log2(sel_data / sel_null), vmin = -0.5, vmax = 0.5, cmap = "PuOr_r")

#### Formatting ####
axs[0,0].set_title(r"Const. Kingman", fontsize=7)
axs[0,1].set_title(r"Exponential growth", fontsize=7)
# gamma = 1.0, t_0 = 1.0
axs[1,0].set_title(r"Beta coalescent", fontsize=7)
axs[1,1].set_title(r"Positive selection", fontsize=7)

for ax in axs.ravel():
    # ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True)
    # ax.set_xticks(ticks=[0.5, 9.5, 19.5], labels=["1", "10", "20+"], fontsize=6)
    ax.set_xticks(ticks=[0.5, 9.5, 19.5], labels=[])
    # ax.tick_params(axis="y", which="both", left=True, labelleft=True)
    # ax.set_yticks(ticks=[0.5, 9.5, 19.5], labels=["1", "10", "20+"], fontsize=6)
    ax.set_yticks(ticks=[0.5, 9.5, 19.5], labels=[])
axs[0,0].set_yticklabels(["1","10","20+"], fontsize=6)
axs[1,0].set_xticklabels(["1","10","20+"], fontsize=6)
axs[1,0].set_yticklabels(["1","10","20+"], fontsize=6)
axs[1,1].set_xticklabels(["1","10","20+"], fontsize=6)


fig.supxlabel(r"Allele frequency site 1", fontsize=7)
fig.supylabel(r"Allele frequency site 2", fontsize=7)

fig.tight_layout(pad=0.3, h_pad=0.5, w_pad=0.5)

fig.subplots_adjust(right=0.8)
x0 = axs[1,0].get_position().bounds[1]
xt = axs[0,0].get_position().bounds[1] + axs[0,0].get_position().bounds[3]
h = xt - x0
cbar_ax = fig.add_axes([0.85, x0, 0.03, h])
fig.colorbar(exp, cax=cbar_ax, ticks=[-0.5, -0.25, 0, 0.25, 0.5])
cbar_ax.tick_params(labelsize=6)

axs[0,0].text(-0.12, 1.09, "a.", fontweight = "bold", transform = axs[0,0].transAxes, fontsize=7)
axs[0,1].text(-0.12, 1.09, "b.", fontweight = "bold", transform = axs[0,1].transAxes, fontsize=7)
axs[1,0].text(-0.12, 1.09, "c.", fontweight = "bold", transform = axs[1,0].transAxes, fontsize=7)
axs[1,1].text(-0.12, 1.09, "d.", fontweight = "bold", transform = axs[1,1].transAxes, fontsize=7)

fig.savefig(save_path + "two_sfs.pdf")
plt.close()

