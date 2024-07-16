import h5py
import numpy as np
import matplotlib.pyplot as plt
from twosfs.spectra import Spectra, spectra_from_hdf5, load_spectra
from twosfs.config import configuration_from_json, parse_parameter_string, make_parameter_string
import json
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

chroms = ["2L", "2R", "3L", "3R"]
save_path = "figures/"

load_path = "../dros_data/Chr{}/"
rec_file = load_path + "Chr{}_4D_rec_search.folded=True.hdf5"
initial_spectra = load_path + "Chr{}_4D_initial_spectra.hdf5"
ks_file = load_path + "Chr{}_4D_ks_distance.folded=True.hdf5"

threefold = 3*np.arange(8).astype("int") + 3

colors = ["#44AF69", "#FCAB10", "#2B9EB3", "#DA3E52"]

######## Loading ########

twosfs_data = []
twosfs_fit = []
for ch in chroms:
    spec = load_spectra(initial_spectra.format(ch, ch))
    print(ch)
    # print(np.sum(np.sum(spec.twosfs, axis=1), axis=1))
    print(np.sum(spec.twosfs) / 8)
    twosfs_data.append(np.sum(spec.normalized_twosfs(folded=True, k_max=20), axis=0)[1:,1:])
    with h5py.File(rec_file.format(ch, ch)) as hf:
        ks_h = hf.get("spectra_high").attrs["ks_distance"]
        ks_l = hf.get("spectra_low").attrs["ks_distance"]
        if ks_h < ks_l:
            spec = spectra_from_hdf5(hf.get("spectra_high"))
            twosfs_fit.append(np.sum(spec.normalized_twosfs(folded=True, k_max=20)[threefold,1:,1:], axis=0))
        else:
            spec = spectra_from_hdf5(hf.get("spectra_low"))
            twosfs_fit.append(np.sum(spec.normalized_twosfs(folded=True, k_max=20)[threefold,1:,1:], axis=0))

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


######## Actual plotting ######## 
fig, axs = plt.subplots(2, 2, figsize=(3.0, 3.0))

for ax, ch, two_data, two_fit in zip(axs.ravel(), chroms, twosfs_data, twosfs_fit):
    ax.pcolormesh( np.log2(two_data / two_fit), vmin = -0.5, vmax = 0.5, cmap = "PuOr_r")
    ax.set_title(ch, fontsize=7)

for ax in axs.ravel():
    ax.set_xticks([0.5, 9.5, 19.5], [])
    ax.set_yticks([0.5, 9.5, 19.5], [])
axs[0,0].set_yticklabels([1, 10, "20+"], fontsize=6)
axs[1,0].set_xticklabels([1, 10, "20+"], fontsize=6)
axs[1,0].set_yticklabels([1, 10, "20+"], fontsize=6)
axs[1,1].set_xticklabels([1, 10, "20+"], fontsize=6)

fig.supxlabel(r"Allele frequency site 1", fontsize=7)
fig.supylabel(r"Allele frequency site 2", fontsize=7)

# fig.tight_layout(pad=0.3, h_pad=0.5, w_pad=0.5)

axs[0,0].text(-0.12, 1.09, "a.", fontweight = "bold", transform = axs[0,0].transAxes, fontsize=7)
axs[0,1].text(-0.10, 1.09, "b.", fontweight = "bold", transform = axs[0,1].transAxes, fontsize=7)
axs[1,0].text(-0.12, 1.09, "c.", fontweight = "bold", transform = axs[1,0].transAxes, fontsize=7)
axs[1,1].text(-0.10, 1.09, "d.", fontweight = "bold", transform = axs[1,1].transAxes, fontsize=7)

fig.savefig(save_path + "two_dros.pdf")


# fig = plt.figure(figsize=(3.0, 4.5), vspace=0.5)
fig = plt.figure(figsize=(3.0, 4.5))
subfigs = fig.subfigures(2, 1, height_ratios=[2,0.8], hspace=0.5)

axs = subfigs[0].subplots(2, 2)

for ax, ch, two_data, two_fit in zip(axs.ravel(), chroms, twosfs_data, twosfs_fit):
    ax.pcolormesh( np.log2(two_data / two_fit), vmin = -0.5, vmax = 0.5, cmap = "PuOr_r")
    ax.set_title(ch, fontsize=7)

for ax in axs.ravel():
    ax.set_xticks([0.5, 9.5, 19.5], [])
    ax.set_yticks([0.5, 9.5, 19.5], [])
axs[0,0].set_yticklabels([1, 10, "20+"], fontsize=6)
axs[1,0].set_xticklabels([1, 10, "20+"], fontsize=6)
axs[1,0].set_yticklabels([1, 10, "20+"], fontsize=6)
axs[1,1].set_xticklabels([1, 10, "20+"], fontsize=6)

subfigs[0].supxlabel(r"Allele frequency site 1", fontsize=7)
subfigs[0].supylabel(r"Allele frequency site 2", fontsize=7)
# subfigs[0].subplots_adjust(wspace=0.15, hspace=0.15, left=0.16, right=0.95, top=0.92, bottom=0.15)
subfigs[0].subplots_adjust(left=0.16, right=0.95, hspace=0.25)
subfigs[1].subplots_adjust(left=0.16, right=0.95, top=0.82, bottom=0.15)

ax = subfigs[1].subplots(1, 1)

vplot = ax.violinplot(ks_sim)
for patch, co in zip(vplot["bodies"], colors):
    patch.set_color(co)
vplot["cmaxes"].set_colors(colors)
vplot["cmins"].set_colors(colors)
vplot["cbars"].set_colors(colors)

for i, co in enumerate(colors):
    ax.plot(i+1, ks_data[i], "*", color=co)

ax.set_xticks([1,2,3,4])
ax.set_xticklabels(chroms, fontsize=7)
ax.set_yticks([500, 1000, 1500], [500, 1000, 1500], fontsize=6)
# ax.set_xlabel("Chromosome arm", fontsize=7)
ax.set_ylabel("KS statistic", fontsize=7)
ax.set_title("KS statistics from 2-SFS", fontsize=7)
ax.set_ylim(top=1750)
legend_elements = [Line2D([0], [0], color="k", ls="none", marker="*", label="Data"),
                   Line2D([0], [0], color="gray", ls="-", lw="6", alpha=0.5, label="Null")]

ax.legend(ncol=2, handles=legend_elements, fontsize=6, loc="upper center")

# subfigs[0].tight_layout(pad=0.3, h_pad=0.5, w_pad=0.5)
# fig.tight_layout()

axs[0,0].text(-0.12, 1.09, "a.", fontweight = "bold", transform = axs[0,0].transAxes, fontsize=7)
axs[0,1].text(-0.10, 1.09, "b.", fontweight = "bold", transform = axs[0,1].transAxes, fontsize=7)
axs[1,0].text(-0.12, 1.09, "c.", fontweight = "bold", transform = axs[1,0].transAxes, fontsize=7)
axs[1,1].text(-0.10, 1.09, "d.", fontweight = "bold", transform = axs[1,1].transAxes, fontsize=7)
ax.text(-0.10, 1.09, "e.", fontweight = "bold", transform = ax.transAxes, fontsize=7)

fig.savefig(save_path + "two_dros_2.pdf")


