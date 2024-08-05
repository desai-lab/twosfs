import h5py
import numpy as np
import matplotlib.pyplot as plt
from twosfs.spectra import Spectra, spectra_from_hdf5, load_spectra
from twosfs.config import configuration_from_json, parse_parameter_string, make_parameter_string
from twosfs.analysis import demo_to_plot
import json
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# Define paths and chromosome arm names
chroms = ["2L", "2R", "3L", "3R"]
save_path = "figures/"

load_path = "../dros_data/Chr{}/"
demo_file = load_path + "Chr{}_4D_demo.folded=True.txt"
rec_file = load_path + "Chr{}_4D_rec_search.folded=True.hdf5"
initial_spectra = load_path + "Chr{}_4D_initial_spectra.hdf5"
ks_file = load_path + "Chr{}_4D_ks_distance.folded=True.hdf5"

######## Loading ########

# Colors used for plotting
colors = ["#44AF69", "#FCAB10", "#2B9EB3", "#DA3E52"]

# Load all data from the demography files
data = []
for c in chroms:
    with open(demo_file.format(c, c)) as df:
        data.append(json.load(df))

# np array that represents frequencies of mutations (will be the x-axis of SFS plots)
freqs = np.arange(1, len(data[0]["sfs_obs"]) + 1)

# Load KS distances from the data and the simulated Kingman fits and use them to find p-values
ks_sim = []
ks_data = []
p_vals = []
for ch in chroms:
    with h5py.File(rec_file.format(ch, ch)) as hf:
        ks_h = hf.get("spectra_high").attrs["ks_distance"]
        ks_l = hf.get("spectra_low").attrs["ks_distance"]
    with h5py.File(ks_file.format(ch, ch)) as hf:
        ks_sim.append(hf.get("ks_null")[:])
        if ks_h > ks_l:
            ks_data.append(ks_l)
            p_vals.append( sum( ks_l < np.array(hf.get("ks_null")[:])) / len(hf.get("ks_null")[:]) )
        else:
            ks_data.append(ks_h)
            p_vals.append( sum( ks_h < np.array(hf.get("ks_null")[:])) / len(hf.get("ks_null")[:]) )


##### Actual plotting #####
fig, axs = plt.subplots(2, 1, figsize=(3.3, 4.2))

# SFS's are shifted vertically to aid visualization
offsets = [1/1.5, 1, 1.5, 1.5**2]

# Plot demographies
for ch, d, co in zip(chroms, data, colors):
    y, t = demo_to_plot(d["sizes"], d["times"])
    axs[1].loglog(t, y, color=co)
# Plot SFS's
for ch, d, co, os in zip(chroms, data, colors, offsets):
    axs[0].loglog(freqs[:-1], os * np.array(d["sfs_obs"])[:-1], "x", color=co, label=ch)
    axs[0].loglog(freqs[-1], os * np.array(d["sfs_obs"])[-1], "x", color=co)
    axs[0].loglog(freqs[:-1], os * np.array(d["sfs_exp"])[:-1], color=co)
    axs[0].loglog(freqs[-1], os * np.array(d["sfs_exp"])[-1], "_", color=co)

# Add axis labels, axis ticks, legends, etc.
axs[0].set_xlabel("Site frequency", fontsize=7)
axs[0].set_ylabel("Relative fraction of sites", fontsize=7)
axs[0].set_title("Empirical and fit one-SFS", fontsize=7)

legend_elements = [Line2D([0], [0], color=colors[0], ls="-", marker="x", label=chroms[0]),
                   Line2D([0], [0], color=colors[1], ls="-", marker="x", label=chroms[1]),
                   Line2D([0], [0], color="k", ls="none", marker="x", label="Data"),
                   Line2D([0], [0], color=colors[2], ls="-", marker="x", label=chroms[2]),
                   Line2D([0], [0], color=colors[3], ls="-", marker="x", label=chroms[3]),
                   Line2D([0], [0], color="k", ls="-", label="Fit")]
axs[0].legend(ncol=2, handles=legend_elements, fontsize=6, loc="lower left")

axs[1].set_xlabel("Time in the past (a.u.)", fontsize=7)
axs[1].set_ylabel("Population size (a.u.)", fontsize=7)
axs[1].set_title("Best-fit Kingman demographies", fontsize=7)
axs[1].yaxis.set_minor_formatter(mticker.ScalarFormatter())
axs[1].set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
axs[1].set_yticklabels(["0.4", " ", "0.6", " ", " ", " ", "1.0"])

legend_elements = [Line2D([0], [0], color=colors[0], ls="-", label=chroms[0]),
                   Line2D([0], [0], color=colors[1], ls="-", label=chroms[1]),
                   Line2D([0], [0], color=colors[2], ls="-", label=chroms[2]),
                   Line2D([0], [0], color=colors[3], ls="-", label=chroms[3])]
axs[1].legend(title="Chromosome Arm", ncol=2, handles=legend_elements, fontsize=6, loc="upper right", title_fontsize=7)

axs[0].set_xticks([1, 10, 20], [r"$10^0$", r"$10^1$", "20+"], fontsize=6)
axs[0].tick_params(labelsize=6)
axs[1].tick_params(labelsize=6)

# Adjust spacing between subplots
fig.tight_layout(pad=0.4, h_pad=0.4)

# Add subplot labels
axs[0].text(-0.07, 1.05, "a.", fontweight = "bold", transform = axs[0].transAxes, fontsize=7)
axs[1].text(-0.07, 1.09, "b.", fontweight = "bold", transform = axs[1].transAxes, fontsize=7)

fig.savefig(save_path + "dros.pdf")


