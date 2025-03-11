import h5py
import numpy as np
import matplotlib.pyplot as plt
from twosfs.spectra import Spectra, spectra_from_hdf5, load_spectra
from twosfs.config import configuration_from_json, parse_parameter_string, make_parameter_string
import json
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# Define chromosome names and useful file paths
chroms = ["2L", "2R", "3L", "3R"]
save_path = "figures/"

load_path = "../dros_data/Chr{}/"
rec_file = load_path + "Chr{}_4D_rec_search.folded=True.hdf5"
initial_spectra = load_path + "Chr{}_4D_initial_spectra.hdf5"
ks_file = load_path + "Chr{}_4D_ks_distance.folded=True.json"

# Holds all genomic distances a multiple of three (3, 6, 9,...) to match the distance
# between fourfold degenerate sites
threefold = np.arange(3, 25, 3)

# The colors used to make these plots
colors = ["#44AF69", "#FCAB10", "#2B9EB3", "#DA3E52"]

######## Loading ########

# Lists hold the 2-SFS of the dros data and the Kingman fits to the dros data
twosfs_data = []
twosfs_fit = []
# Do the same for each chromosome
for ch in chroms:
    spec = load_spectra(initial_spectra.format(ch, ch))
    # We sum over several genomic distances here solely for visual purposes
    # The actual analysis does not do this
    # The data also already only contains sites at distances 3, 6, 9, etc.
    twosfs_data.append(np.sum(spec.normalized_twosfs(folded=True, k_max=20), axis=0)[1:,1:])
    # Load the simulated Kingman 2-SFS, picking out only sites at distances 3, 6, 9, etc.
    with h5py.File(rec_file.format(ch, ch)) as hf:
        p_h = hf.get("spectra_high").attrs["p_value"]
        p_l = hf.get("spectra_low").attrs["p_value"]
        if p_h > p_l:
            spec = spectra_from_hdf5(hf.get("spectra_high"))
            twosfs_fit.append(np.sum(spec.normalized_twosfs(folded=True, k_max=20)[threefold,1:,1:], axis=0))
        else:
            spec = spectra_from_hdf5(hf.get("spectra_low"))
            twosfs_fit.append(np.sum(spec.normalized_twosfs(folded=True, k_max=20)[threefold,1:,1:], axis=0))



# Now plot the violin plots of the KS distances
ks_sim = []
ks_data = []
p_vals = []

# Get the KS distances for each chromosome and use those to calculate p-values
for ch in chroms:
    with open(ks_file.format(ch, ch)) as f:
        results = json.load(f)
    ks_sim.append(results["ks_distribution"])
    ks_data.append(results["ks_distance"])
    p_vals.append(results["p_value"])

######## Actual plotting ######## 
fig = plt.figure(figsize=(3.0, 4.5))
# Creates a subplot with three rows, with the first two rows having two columns
# and the last row only having one
subfigs = fig.subfigures(2, 1, height_ratios=[2,0.8], hspace=0.5)
axs = subfigs[0].subplots(2, 2)

# Plot the 2-SFS
for ax, ch, two_data, two_fit in zip(axs.ravel(), chroms, twosfs_data, twosfs_fit):
    ax.pcolormesh( np.log2(two_data / two_fit), vmin = -0.5, vmax = 0.5, cmap = "PuOr_r")
    ax.set_title(ch, fontsize=7)

# Formant the axes for the 2-SFS
for ax in axs.ravel():
    ax.set_xticks([0.5, 9.5, 19.5], [])
    ax.set_yticks([0.5, 9.5, 19.5], [])
axs[0,0].set_yticklabels([1, 10, "20+"], fontsize=6)
axs[1,0].set_xticklabels([1, 10, "20+"], fontsize=6)
axs[1,0].set_yticklabels([1, 10, "20+"], fontsize=6)
axs[1,1].set_xticklabels([1, 10, "20+"], fontsize=6)

# Add axis labels and adjust the positions of the 2-SFS plots slightly
subfigs[0].supxlabel(r"Allele frequency site 1", fontsize=7)
subfigs[0].supylabel(r"Allele frequency site 2", fontsize=7)
subfigs[0].subplots_adjust(left=0.16, right=0.95, hspace=0.25)
subfigs[1].subplots_adjust(left=0.16, right=0.95, top=0.82, bottom=0.15)

# Plot the KS distance violin plots
ax = subfigs[1].subplots(1, 1)
vplot = ax.violinplot(ks_sim)

# Change the colors of all parts of the violin plots
for patch, co in zip(vplot["bodies"], colors):
    patch.set_color(co)
vplot["cmaxes"].set_colors(colors)
vplot["cmins"].set_colors(colors)
vplot["cbars"].set_colors(colors)
for i, co in enumerate(colors):
    ax.plot(i+1, ks_data[i], "*", color=co)

# Format axes for the KS distance plot
ax.set_xticks([1,2,3,4])
ax.set_xticklabels(chroms, fontsize=7)
ax.set_yticks([500, 1000, 1500], [500, 1000, 1500], fontsize=6)
ax.set_ylabel("KS statistic", fontsize=7)
ax.set_title("KS statistics from 2-SFS", fontsize=7)
ax.set_ylim(top=1750)

# Add a legend
legend_elements = [Line2D([0], [0], color="k", ls="none", marker="*", label="Data"),
                   Line2D([0], [0], color="gray", ls="-", lw="6", alpha=0.5, label="Null")]
ax.legend(ncol=2, handles=legend_elements, fontsize=6, loc="upper center")

# Add subpanel labels
axs[0,0].text(-0.12, 1.09, "a.", fontweight = "bold", transform = axs[0,0].transAxes, fontsize=7)
axs[0,1].text(-0.10, 1.09, "b.", fontweight = "bold", transform = axs[0,1].transAxes, fontsize=7)
axs[1,0].text(-0.12, 1.09, "c.", fontweight = "bold", transform = axs[1,0].transAxes, fontsize=7)
axs[1,1].text(-0.10, 1.09, "d.", fontweight = "bold", transform = axs[1,1].transAxes, fontsize=7)
ax.text(-0.10, 1.09, "e.", fontweight = "bold", transform = ax.transAxes, fontsize=7)

fig.savefig(save_path + "two_dros.pdf")

