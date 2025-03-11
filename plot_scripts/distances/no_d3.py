import h5py
import numpy as np
import matplotlib.pyplot as plt
from twosfs.spectra import Spectra, spectra_from_hdf5, load_spectra
from twosfs.config import configuration_from_json, parse_parameter_string, make_parameter_string, insert_folder_name
from twosfs.analysis import get_power
import json
from matplotlib.lines import Line2D

fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.2))
save_path = "/n/home12/efenton/for_windows/newer_2sfs/paper/distances/"
beta_color = "darkgreen"

config = configuration_from_json("../../simulation_parameters.json", root = "../../")
ax = axs[0]

config.ks_distance_file = insert_folder_name(config.ks_distance_file, f"no_short", -1)

params = {"folded": True,
          "pair_density": config.power_pair_densities[0],
          "sequence_length": 25,
} 

power_const = []
tajd_const = []
for model in config.iter_models():
    if model[0] == "const":
        power_const.append(get_power(model, config=config, **params))
        tajd_const.append(
            load_spectra(config.format_initial_spectra_file(*model)).tajimas_d()
        )

power_beta = []
tajd_beta = []
alphas = []
for model in config.iter_models():
    if model[0] == "beta":
        power_beta.append(get_power(model, config=config, **params))
        alphas.append(model[1]["alpha"])
        tajd_beta.append(
            load_spectra(config.format_initial_spectra_file(*model)).tajimas_d()
        )

power_exp = []
tajd_exp = []
growth_rates = []
end_times = []
for model in config.iter_models():
    if model[0] == "exp":
        power_exp.append(get_power(model, config=config, **params))
        growth_rates.append(model[1]["growth_rate"])
        end_times.append(model[1]["end_time"])
        tajd_exp.append(
            load_spectra(config.format_initial_spectra_file(*model)).tajimas_d()
        )
power_exp = np.array(power_exp)
growth_rates = np.array(growth_rates)
end_times = np.array(end_times)

power_sel = []
tajd_sel = []
s_vals = []
mut_rates = []
for model in config.iter_forward_models():
    power_sel.append(get_power(model, config=config, **params))
    s_vals.append(model[1]["s"])
    mut_rates.append(model[1]["mut_rate"])
    tajd_sel.append(
        load_spectra(config.format_initial_spectra_file(*model)).tajimas_d()
    )
mut_rates = np.array(mut_rates)
s_vals = np.array(s_vals)
power_sel = np.array(power_sel)

ms = 4
ax.plot(tajd_beta[-1], power_beta[-1], ".", color = "k", label = "Constant-size Kingman")
ax.plot(tajd_beta, power_beta, ".", color = beta_color, label = "Beta coalescent")
ax.plot(tajd_sel, power_sel, ".", color = "tab:red", label = "Positive seleciton")
ax.plot(tajd_exp, power_exp, ".", color = "tab:blue", label = "Exponential growth")
ax.plot(tajd_const, power_const, ".", color = "k")

ax.set_ylabel("Power", fontsize=7)
ax.set_xlabel(r"Tajima's $D$", fontsize=7)
ax.set_title("Simulations", fontsize=7)
ax.set_ylim(-0.05, 1.05)
ax.set_xlim(-0.605, 0.02)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=6)

ax.tick_params(labelsize=6)

ax.legend(fontsize=6, title_fontsize=7, frameon=True, loc="center left")

chroms = ["2L", "2R", "3L", "3R"]
load_path = "../../dros_data/Chr{}/"
ks_file = load_path + "Chr{}_4D_ks_distance.folded=True.no_d=3.json"

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
# Creates a subplot with three rows, with the first two rows having two columns
# and the last row only having one
ax = axs[1]
vplot = ax.violinplot(ks_sim)
colors = ["#44AF69", "#FCAB10", "#2B9EB3", "#DA3E52"]

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
ax.set_title(r"$D.$ $melanogaster$ data", fontsize=7)
ax.set_ylim(top=1550)

# Add a legend
legend_elements = [Line2D([0], [0], color="k", ls="none", marker="*", label="Data"),
                   Line2D([0], [0], color="gray", ls="-", lw="6", alpha=0.5, label="Null")]
ax.legend(ncol=2, handles=legend_elements, fontsize=6, loc="upper center")

fig.suptitle(r"Analysis with sites at $d=3$ not included", fontsize=7)

fig.tight_layout(pad=0.2)

axs[0].text(-0.12, 1.07, "a.", fontweight = "bold", transform = axs[0].transAxes, fontsize=7)
axs[1].text(-0.12, 1.07, "b.", fontweight = "bold", transform = axs[1].transAxes, fontsize=7)

fig.savefig(save_path + f"no_d3.pdf")
