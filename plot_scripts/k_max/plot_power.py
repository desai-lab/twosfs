import h5py
import numpy as np
import matplotlib.pyplot as plt
from twosfs.spectra import Spectra, spectra_from_hdf5, load_spectra
from twosfs.config import configuration_from_json, parse_parameter_string, make_parameter_string, insert_folder_name
from twosfs.analysis import get_power
import json

k_max = 16

config = configuration_from_json("../../simulation_parameters.json", root = "../../")
save_path = "/n/home12/efenton/for_windows/newer_2sfs/paper/k_max/"
# save_path = "figures/"

config.initial_spectra_file = config.initial_spectra_file
config.ks_distance_file = insert_folder_name(config.ks_distance_file, f"k_max_{k_max}", -1)

params = {"folded": True,
          "pair_density": config.power_pair_densities[0],
          "sequence_length": config.power_sequence_lengths[0],
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


########## Power plots ##########

fig = plt.figure(figsize=(3.3, 3.6))
ax1 = plt.subplot2grid((4, 3), (0, 0), colspan = 1, rowspan = 2)
ax2 = plt.subplot2grid((4, 3), (0, 1), colspan = 1, rowspan = 2)
ax3 = plt.subplot2grid((4, 3), (0, 2), colspan = 1, rowspan = 2)
ax4 = plt.subplot2grid((4, 3), (2, 0), colspan = 3, rowspan = 2)

beta_color = "darkgreen"

ax1.plot(alphas, power_beta, ".", color=beta_color)
ax1.plot(2.0, power_const[-1], ".", color=beta_color)

colors_sel = ["indianred", "darkred"]
for i, mu in enumerate(config.positive_mut_rates):
    ax2.plot(s_vals[mut_rates==mu], power_sel[mut_rates==mu] , ".", color = colors_sel[i])
ax2.text(0.07, 0.68, r"$\mu=10^{-10}$", transform = ax2.transAxes, fontsize=6, color=colors_sel[0])
ax2.text(0.47, 0.035, r"$\mu=10^{-11}$", transform = ax2.transAxes, fontsize=6, color=colors_sel[1])

colors_exp = ["navy", "deepskyblue", "aquamarine"]
for i, t in enumerate(config.end_times):
    label = r"$t_0 = {}$".format(t)
    ax3.plot(growth_rates[end_times==t], power_exp[end_times==t], ".", color=colors_exp[i], label=label)
ax3.legend(fontsize=6, frameon=True)

ms = 4
ax4.plot(tajd_beta[-1], power_beta[-1], ".", color = "k", label = "Constant-size Kingman")
ax4.plot(tajd_beta, power_beta, ".", color = beta_color, label = "Beta coalescent")
ax4.plot(tajd_sel, power_sel, ".", color = "tab:red", label = "Positive seleciton")
ax4.plot(tajd_exp, power_exp, ".", color = "tab:blue", label = "Exponential growth")
ax4.plot(tajd_const, power_const, ".", color = "k")
ax4.legend(fontsize=6, title_fontsize=7, frameon=True, loc="center left")

ax1.set_ylabel("Power", fontsize=7)
ax1.set_xlabel(r"$\alpha$", fontsize=7)
ax2.set_xlabel("Sel. Coeff.", fontsize=7)
ax3.set_xlabel("Growth Rate", fontsize=7)
ax4.set_ylabel("Power", fontsize=7)
ax4.set_xlabel(r"Tajima's $D$", fontsize=7)

ax1.set_title("Beta coal.", fontsize=7)
ax2.set_title("Selection", fontsize=7)
ax3.set_title("Exp. growth", fontsize=7)
ax4.set_title(r"Power vs. Tajima's $D$", fontsize=7)

ax1.set_ylim(-0.05, 1.05)
ax1.set_xlim(2.05, 0.95)
ax2.set_ylim(-0.05, 1.05)
ax3.set_ylim(-0.05, 1.05)
ax4.set_ylim(-0.05, 1.05)
ax4.set_xlim(-0.605, 0.02)

ax1.set_xticks([2.0, 1.5, 1.0], fontsize=6)
ax2.set_xticks([0, 0.04, 0.08], fontsize=6)
ax3.set_xticks([0.0, 1.0, 2.0], ["0.0", "1.0", "2.0"], fontsize=6)
ax4.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=6)

for ax in [ax1, ax2, ax3, ax4]:
    ax.tick_params(labelsize=6)

for ax in [ax1, ax2, ax3]:
    ax.set_yticks([0, 0.5, 1], [])
ax1.set_yticks([0, 0.5, 1], [0, 0.5, 1])

fig.suptitle(r"$\mathbf{k_{max}=}$" + str(k_max), fontsize=7, fontweight = "bold")

fig.tight_layout(pad=0.2)

ax1.text(-0.12, 1.09, "a.", fontweight = "bold", transform = ax1.transAxes, fontsize=6)
ax2.text(-0.12, 1.09, "b.", fontweight = "bold", transform = ax2.transAxes, fontsize=6)
ax3.text(-0.12, 1.09, "c.", fontweight = "bold", transform = ax3.transAxes, fontsize=6)
ax4.text(-0.04, 1.09, "d.", fontweight = "bold", transform = ax4.transAxes, fontsize=6)

fig.savefig(save_path + f"power_{k_max}.pdf")
