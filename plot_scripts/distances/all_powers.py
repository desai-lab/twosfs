import h5py
import numpy as np
import matplotlib.pyplot as plt
from twosfs.spectra import Spectra, spectra_from_hdf5, load_spectra
from twosfs.config import configuration_from_json, parse_parameter_string, make_parameter_string, insert_folder_name
from twosfs.analysis import get_power
import json

fig, axs = plt.subplots(2, 2, figsize=(6.6, 4.0))
save_path = "/n/home12/efenton/for_windows/newer_2sfs/paper/distances/"
beta_color = "darkgreen"

for d_max, ax in zip([19, 22, 25, 28], axs.ravel()):
    config = configuration_from_json("../../simulation_parameters.json", root = "../../")

    params = {"folded": True,
              "pair_density": config.power_pair_densities[0],
              "sequence_length": d_max,
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
    ax.set_title(r"$d_{max}=$" + str(d_max-1), fontsize=7)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.605, 0.02)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=6)

    ax.tick_params(labelsize=6)

axs[0,0].legend(fontsize=6, title_fontsize=7, frameon=True, loc="center left")


########## Power plots ##########

fig.suptitle(r"Power vs. Tajima's $D$ for different maximum distances", fontsize=7)

fig.tight_layout(pad=0.2)


axs[0,0].text(-0.12, 1.07, "a.", fontweight = "bold", transform = axs[0,0].transAxes, fontsize=7)
axs[0,1].text(-0.12, 1.07, "b.", fontweight = "bold", transform = axs[0,1].transAxes, fontsize=7)
axs[1,0].text(-0.12, 1.07, "c.", fontweight = "bold", transform = axs[1,0].transAxes, fontsize=7)
axs[1,1].text(-0.12, 1.07, "d.", fontweight = "bold", transform = axs[1,1].transAxes, fontsize=7)


fig.savefig(save_path + f"power_distances.pdf")

