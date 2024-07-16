import h5py
import numpy as np
import matplotlib.pyplot as plt
from twosfs.spectra import Spectra, spectra_from_hdf5, load_spectra
from twosfs.config import configuration_from_json, parse_parameter_string, make_parameter_string
import json

config = configuration_from_json("../simulation_parameters.json")
save_path = "figures/"
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


########## Power plots ##########
power_beta = np.load("power_files/beta_power.npy")
power_sel = np.load("power_files/sel_power.npy")
power_exp = np.load("power_files/exp_power.npy")

tajd_beta = np.load("power_files/beta_tajd.npy")
tajd_sel = np.load("power_files/sel_tajd.npy")
tajd_exp = np.load("power_files/exp_tajd.npy")
colors = ["tab:blue", "tab:red", "tab:green"]

"""
fig, axs = plt.subplots(2, 2, figsize=(3.3,3.3))
axs = axs.ravel()

# axs[0].plot(alphas, power_beta, ".", linestyle = "dashed")
axs[0].plot(alphas, power_beta, ".")

L = len(s_vals)
for i, mu in enumerate(mut_rates):
    label =  r"$\Theta_+ = {}$".format(round(mu*2000*1e5, 3))
    # axs[1].plot(s_vals, power_sel[i*L:(i+1)*L], ".", linestyle = "dashed", color = colors[i], label = label)
    axs[1].plot(s_vals, power_sel[i*L:(i+1)*L], ".", color = colors[i], label = label)
axs[1].legend(title="Positive mutation rate", fontsize=6, title_fontsize=7, bbox_to_anchor=(0.5, -0.05), ncol=2)

L = len(growth_rates)
for i, t in enumerate(end_times):
    label = r"$t_0 = {}$".format(t)
    # axs[2].plot(growth_rates, power_exp[i*L:(i+1)*L], ".", linestyle = "dashed", color = colors[i], label = label)
    axs[2].plot(growth_rates, power_exp[i*L:(i+1)*L], ".", color = colors[i], label = label)
axs[2].legend(title="Growth time", fontsize=6, title_fontsize=7)

# axs[3].plot(tajd_beta[-1], power_beta[-1], "o", color = "k", label = "Constant-size Kingman")
# axs[3].plot(tajd_beta, power_beta, "^", color = "tab:blue", label = "Beta coalescent")
# axs[3].plot(tajd_sel, power_sel, "P", color = "tab:red", label = "Positive seleciton")
# axs[3].plot(tajd_exp, power_exp, "X", color = "tab:green", label = "Exponential growth")
axs[3].plot(tajd_beta[-1], power_beta[-1], ".", color = "k", label = "Constant-size Kingman")
axs[3].plot(tajd_beta, power_beta, ".", color = "tab:blue", label = "Beta coalescent")
axs[3].plot(tajd_sel, power_sel, ".", color = "tab:red", label = "Positive seleciton")
axs[3].plot(tajd_exp, power_exp, ".", color = "tab:green", label = "Exponential growth")
axs[3].plot(tajd_beta[-1], power_beta[-1], ".", color = "k")
axs[3].legend(fontsize=6, title_fontsize=7)

axs[0].set_ylabel("Power", fontsize=7)
axs[0].set_xlabel(r"$\alpha$ Paramter", fontsize=7)
axs[1].set_xlabel("Selective Coefficient", fontsize=7)
axs[2].set_xlabel("Growth Rate", fontsize=7)
axs[2].set_ylabel("Power", fontsize=7)
axs[3].set_xlabel(r"Tajima's $D$", fontsize=7)

axs[0].set_title("Beta coalescent", fontsize=7)
axs[1].set_title("Positive selection", fontsize=7)
axs[2].set_title("Exponential growth", fontsize=7)
axs[3].set_title(r"Power vs. Tajima's $D$", fontsize=7)

axs[0].set_ylim(-0.05, 1.05)
axs[0].set_xlim(1.00, 2.05)
axs[1].set_ylim(-0.05, 1.05)
axs[2].set_ylim(-0.05, 1.05)
axs[3].set_ylim(-0.05, 1.05)
axs[3].set_xlim(-0.605, 0.02)

for ax in axs:
    ax.set_yticks([0, 0.5, 1], [])
axs[0].set_yticklabels(["0", "0.5", "1.0"], fontsize=6)
axs[2].set_yticklabels(["0", "0.5", "1.0"], fontsize=6)

axs[0].set_xticks([1.05, 1.5, 2.0], ["1.05", "1.5", "2.0"], fontsize=6)
axs[0].invert_xaxis()
axs[1].set_xticks([0, 0.04, 0.08], ["0.0", "0.04", "0.08"], fontsize=6)
axs[2].set_xticks([0, 1, 2], ["0.0", "1.0", "2.0"], fontsize=6)

fig.tight_layout()
plt.savefig(save_path + "power_2.pdf")
"""

fig = plt.figure(figsize=(3.3, 3.3))
ax1 = plt.subplot2grid((4, 3), (0, 0), colspan = 1, rowspan = 2)
ax2 = plt.subplot2grid((4, 3), (0, 1), colspan = 1, rowspan = 2)
ax3 = plt.subplot2grid((4, 3), (0, 2), colspan = 1, rowspan = 2)
ax4 = plt.subplot2grid((4, 3), (2, 0), colspan = 3, rowspan = 2)

beta_color = "darkgreen"

# ax1.plot(alphas, power_beta, ".", linestyle = "dashed", color="tab:green")
ax1.plot(alphas, power_beta, ".", color=beta_color)

colors_sel = ["indianred", "darkred"]
L = len(s_vals)
for i, mu in enumerate(mut_rates):
    label =  r"$\Theta_+ = {}$".format(round(mu*2000*1e5, 3))
    # ax2.plot(s_vals, power_sel[i*L:(i+1)*L], ".", linestyle = "dashed", color = colors_sel[i], label = label)
    ax2.plot(s_vals, power_sel[i*L:(i+1)*L], ".", color = colors_sel[i], label = label)
ax2.text(0.07, 0.68, r"$\mu=10^{-10}$", transform = ax2.transAxes, fontsize=6, color=colors_sel[0])
ax2.text(0.47, 0.035, r"$\mu=10^{-11}$", transform = ax2.transAxes, fontsize=6, color=colors_sel[1])
# ax2.legend(loc = "lower right", title="Positive mutation rate", fontsize=6, title_fontsize=7, frameon=False)

colors_exp = ["navy", "deepskyblue", "aquamarine"]
L = len(growth_rates)
for i, t in enumerate(end_times):
    label = r"$t_0 = {}$".format(t)
    # ax3.plot(growth_rates, power_exp[i*L:(i+1)*L], ".", linestyle = "dashed", color = colors_exp[i], label = label)
    ax3.plot(growth_rates, power_exp[i*L:(i+1)*L], ".", color = colors_exp[i], label = label)
# ax3.legend(title="Growth time", fontsize=6, title_fontsize=7, frameon=False)
ax3.legend(fontsize=6, title_fontsize=7, frameon=False)

ms = 4
ax4.plot(tajd_beta[-1], power_beta[-1], ".", color = "k", label = "Constant-size Kingman")
ax4.plot(tajd_beta, power_beta, ".", color = beta_color, label = "Beta coalescent")
ax4.plot(tajd_sel, power_sel, ".", color = "tab:red", label = "Positive seleciton")
ax4.plot(tajd_exp, power_exp, ".", color = "tab:blue", label = "Exponential growth")
"""
ax4.plot(tajd_beta[:-1], power_beta[:-1], "^", color = beta_color, label = "Beta coalescent", markersize=ms)
ax4.plot(tajd_sel, power_sel, "P", color = "tab:red", label = "Positive seleciton", markersize=ms)
ax4.plot(tajd_exp, power_exp, "X", color = "tab:blue", label = "Exponential growth", markersize=ms)
"""
ax4.plot(tajd_beta[-1], power_beta[-1], ".", color = "k")
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

fig.tight_layout(pad=0.2)

ax1.text(-0.12, 1.10, "a.", fontweight = "bold", transform = ax1.transAxes, fontsize=6)
ax2.text(-0.12, 1.10, "b.", fontweight = "bold", transform = ax2.transAxes, fontsize=6)
ax3.text(-0.12, 1.10, "c.", fontweight = "bold", transform = ax3.transAxes, fontsize=6)
ax4.text(-0.04, 1.10, "d.", fontweight = "bold", transform = ax4.transAxes, fontsize=6)

fig.savefig(save_path + "power.pdf")
