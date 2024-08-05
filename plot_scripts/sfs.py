import h5py
import numpy as np
import matplotlib.pyplot as plt
from twosfs.spectra import Spectra, spectra_from_hdf5, load_spectra
from twosfs.config import configuration_from_json, parse_parameter_string, make_parameter_string
import json
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from twosfs.analysis import demo_to_plot

config = configuration_from_json("../simulation_parameters.json", root = "../")
save_path = "figures/"

######## Parameters in the example plots for each model ########
beta_params = {"alpha": 1.3}
sel_params = {"s": 0.055, "rec_rate": 1e-05, "mut_rate": 1e-10}
exp_params = {"end_time": 0.5, "growth_rate": 2.0}

# This is used to fill in file strings, eg: print(config.recombination_search_file.format(**params))
# will give a string to a recombination search file with the parameter values in the dict below
params = {"model": "const",
          "params": make_parameter_string({}),
          "pair_density": 10000,
          "sequence_length": 25,
          "power_rep": 0,
          "folded": True,
          "rep": "all"}

######## Loading ########
with open(config.fitted_demography_file.format(**params)) as df:
    const_data = json.load(df)

params["model"] = "beta"
params["params"] = make_parameter_string(beta_params)
with open(config.fitted_demography_file.format(**params)) as df:
    beta_data = json.load(df)

params["model"] = "sel"
params["params"] = make_parameter_string(sel_params)
with open(config.fitted_demography_file.format(**params)) as df:
    sel_data = json.load(df)

params["model"] = "exp"
params["params"] = make_parameter_string(exp_params)
with open(config.fitted_demography_file.format(**params)) as df:
    exp_data = json.load(df)

######## Plotting ########

# Create the figure and define the colors for the four example models
fig, axs = plt.subplots(2, 1, figsize = (3.3, 4.2))
colors = ["tab:purple", "darkgreen", "tab:red", "tab:blue"]
# SFS's of some models are offset vertically
offsets = [1/1.7, 1, 1.4, 2]

#### Constant ####
axs[0].loglog(np.arange(1,20), np.array(const_data["sfs_obs"])[:-1] * offsets[0],
              "x", color = colors[0], label = "Const. Kingman")
axs[0].loglog(20, np.array(const_data["sfs_obs"])[-1] * offsets[0],
              "x", color = colors[0])
axs[0].loglog(np.arange(1,20), np.array(const_data["sfs_exp"])[:-1] * offsets[0],
              color = colors[0])
axs[0].loglog(20, np.array(const_data["sfs_exp"])[-1] * offsets[0],
              "_", color = colors[0])
# Demo #
y, t = demo_to_plot(const_data["sizes"], const_data["times"])
axs[1].loglog(t, y, color = colors[0], label = "Const. Kingman")

#### Beta ####
# SFS #
axs[0].loglog(np.arange(1,20), np.array(beta_data["sfs_obs"])[:-1] * offsets[1],
              "x", color = colors[1], label="Beta coal.")
axs[0].loglog(20, np.array(beta_data["sfs_obs"])[-1] * offsets[1],
              "x", color = colors[1], label="Beta coal.")
axs[0].loglog(np.arange(1,20), np.array(beta_data["sfs_exp"])[:-1] * offsets[1],
              color = colors[1])
axs[0].loglog(20, np.array(beta_data["sfs_exp"])[-1] * offsets[1],
              "_", color = colors[1])
# Demo #
y, t = demo_to_plot(beta_data["sizes"], beta_data["times"])
axs[1].loglog(t, y, color = colors[1], label = "Beta coal.")

#### Selective ####
# SFS #
axs[0].loglog(np.arange(1,20), np.array(sel_data["sfs_obs"])[:-1] * offsets[2],
              "x", color = colors[2], label="Selection")
axs[0].loglog(20, np.array(sel_data["sfs_obs"])[-1] * offsets[2],
              "x", color = colors[2], label="Selection")
axs[0].loglog(np.arange(1,20), np.array(sel_data["sfs_exp"])[:-1] * offsets[2],
              color = colors[2])
axs[0].loglog(20, np.array(sel_data["sfs_exp"])[-1] * offsets[2],
              "_", color = colors[2])
# Demo #
y, t = demo_to_plot(sel_data["sizes"], sel_data["times"])
axs[1].loglog(t, y, color = colors[2], label = "Selection")

#### Exponential ####
axs[0].loglog(np.arange(1,20), np.array(exp_data["sfs_obs"])[:-1] * offsets[3],
              "x", color = colors[3], label="Exp. growth")
axs[0].loglog(20, np.array(exp_data["sfs_obs"])[-1] * offsets[3],
              "x", color = colors[3])
axs[0].loglog(np.arange(1,20), np.array(exp_data["sfs_exp"])[:-1] * offsets[3],
              color = colors[3])
axs[0].loglog(20, np.array(exp_data["sfs_exp"])[-1] * offsets[3],
              "_", color = colors[3])
# Demo #
y, t = demo_to_plot(exp_data["sizes"], exp_data["times"])
axs[1].loglog(t, y, color = colors[3], label = "Exp. growth")


##### Formatting #####
# Add legend to upper subplot
legend_elements = [Line2D([0], [0], color="k", ls="none", marker="x", label="Simulated data"),
                   Line2D([0], [0], color="k", ls="-", label="Fit Kingman SFS")]
axs[0].legend(ncol=1, handles=legend_elements, fontsize=6, loc="lower left", title_fontsize=7)

# Format axis labels for upper subplot
axs[0].set_xticks([1, 10, 20], [r"$10^0$", r"$10^1$", "20+"], fontsize=6)
axs[0].tick_params(labelsize=6)
axs[0].set_xlabel("Site frequency", fontsize=7)
axs[0].set_ylabel("Relative fraction of sites", fontsize=7)
axs[0].set_title("Simulated and fit one-SFS", fontsize=7)

# Format axis labels for lower subplot
axs[1].legend(fontsize=6, title="Model", title_fontsize=7)
axs[1].tick_params(labelsize=6)
axs[1].set_xlabel("Time in the past (a.u.)", fontsize=7)
axs[1].set_ylabel("Population size (a.u.)", fontsize=7)
axs[1].set_title("Best-fit Kingman demographies", fontsize=7)

# Adjust subplot spacing
fig.tight_layout(pad=0.4, h_pad=0.4)

# Add labels to subplots and save
axs[0].text(-0.07, 1.05, "a.", fontweight = "bold", transform = axs[0].transAxes, fontsize=7)
axs[1].text(-0.07, 1.09, "b.", fontweight = "bold", transform = axs[1].transAxes, fontsize=7)

fig.savefig(save_path + "sfs.pdf")

