import json
import tempfile

import h5py
import numpy as np
import matplotlib.pyplot as plt

from twosfs import spectra, statistics
from twosfs.config import configuration_from_json, parse_parameter_string, make_parameter_string
from twosfs.spectra import Spectra, spectra_from_hdf5
import time

class Slim_Data:
    def __init__(self, root = "../"):
        self.root = root
        self.params = []
        self.s_vals = []
        self.mut_rates = []
        self.spec_targ = {}
        self.spec_fit = {}
        self.demo_data = {}

    def add_data(self, params):
        targ_file = config.format_initial_spectra_file("sel", params)
        fit_file = config.format_fitted_spectra_file("sel", params, True, 
                                                     config.power_pair_densities[0],
                                                     config.power_sequence_lengths[0], 0)
        demo_file = config.format_fitted_demography_file("sel", params, True)
        param_string = make_parameter_string(params)

        with h5py.File(self.root + targ_file, "r") as hf:
            self.spec_targ[param_string] = spectra_from_hdf5(hf.get("spectra"))
        with h5py.File(self.root + fit_file, "r") as hf:
            self.spec_fit[param_string] = spectra_from_hdf5(hf.get("spectra"))
        with open(self.root + demo_file, "r") as df:
            self.demo_data[param_string] = json.load(df)

        self.params.append(param_string)
        self.s_vals.append(float(params["s"]))
        self.mut_rates.append(float(params["mut_rate"]))

    def key_from_params(self, s, mu):
        return np.array(self.params)[(np.array(self.s_vals) == s) * (np.array(self.mut_rates) == mu)][0]

    def two_sfs_from_params(self, s, mu, version, k_max = 20, d = 3):
        if version == "targ":
            return self.spec_targ[self.key_from_params(s, mu)].normalized_twosfs(k_max = k_max)[d, 1:, 1:]
        elif version == "fit":
            return self.spec_fit[self.key_from_params(s, mu)].normalized_twosfs(k_max = k_max)[d, 1:, 1:]

    def to_plot(self, s, mu, k_max = 20, d = 3):
        return self.two_sfs_from_params(s, mu, "fit", k_max, d) / self.two_sfs_from_params(s, mu, "targ", k_max, d)

    def demo_plot(self, s, mu):
        sizes = self.demo_data[self.key_from_params(s, mu)]["sizes"]
        times = self.demo_data[self.key_from_params(s, mu)]["times"]
        sizes_plot = []
        times_plot = [0]
        for t in times:
            times_plot.append(t)
            times_plot.append(t)
        times_plot.append(100)
        for x in sizes:
            sizes_plot.append(x)
            sizes_plot.append(x)
        return sizes_plot, times_plot

########## Load data ##########

config = configuration_from_json("../simulation_parameters.json")
save_path = "/n/home12/efenton/for_windows/newer_2sfs/"
sd = Slim_Data()

for model, params in config.iter_forward_models():
    sd.add_data(params)

########## Plot 2sfs ##########

fig, axs = plt.subplots(2, 4, figsize = (12, 6))
for s, ax in zip(config.positive_sel_coeffs, axs.ravel()[1:]):
    ax.pcolormesh( np.log2(sd.to_plot(s, 1e-09)), vmin = -0.5, vmax = 0.5, cmap = "PuOr_r")
    ax.set_title("s = {}".format(s))
axs[0,0].pcolormesh( np.log2(sd.to_plot(0, 1e-09)), vmin = -0.5, vmax = 0.5, cmap = "PuOr_r")
axs[0,0].set_title("s = 0")
fig.suptitle("Mutation rate 1e-09")
fig.savefig(save_path + "two_all_09.png")
plt.close(fig)

fig, axs = plt.subplots(2, 4, figsize = (12, 6))
for s, ax in zip(config.positive_sel_coeffs, axs.ravel()[1:]):
    ax.pcolormesh( np.log2(sd.to_plot(s, 1e-10)), vmin = -0.5, vmax = 0.5, cmap = "PuOr_r")
    ax.set_title("s = {}".format(s))
axs[0,0].pcolormesh( np.log2(sd.to_plot(0, 1e-09)), vmin = -0.5, vmax = 0.5, cmap = "PuOr_r")
axs[0,0].set_title("s = 0")
fig.suptitle("Mutation rate 1e-10")
fig.savefig(save_path + "two_all_10.png")
plt.close(fig)


plt.figure()
sizes, times = sd.demo_plot(0, 1e-09)
plt.loglog(times, sizes, label = "s = 0", color = "k")
for s in config.positive_sel_coeffs:
    sizes, times = sd.demo_plot(s, 1e-09)
    plt.loglog(times, sizes, label = "s = {}".format(s))
plt.legend()
plt.title("Mutation rate 1e-09")
plt.savefig(save_path + "demos_09.png")
plt.close()

plt.figure()
sizes, times = sd.demo_plot(0, 1e-09)
plt.loglog(times, sizes, label = "s = 0", color = "k")
for s in config.positive_sel_coeffs:
    sizes, times = sd.demo_plot(s, 1e-10)
    plt.loglog(times, sizes, label = "s = {}".format(s))
plt.legend()
plt.title("Mutation rate 1e-10")
plt.savefig(save_path + "demos_10.png")
plt.close()



"""
plt.figure()
plt.loglog(targ.normalized_onesfs(), label = "true")
plt.loglog(high.normalized_onesfs(), ".", label = "high")
plt.loglog(low.normalized_onesfs(), ".", label = "low")
plt.legend()
plt.savefig(save_path + "test.png")
plt.close()

plt.figure()
plt.pcolormesh( np.log2( high.normalized_twosfs(k_max = 20)[3] / targ.normalized_twosfs(k_max = 20)[3]), vmin = -0.5, vmax = 0.5, cmap="PuOr_r")
plt.colorbar()
plt.savefig(save_path + "test_two_high.png")
plt.close()

plt.figure()
plt.pcolormesh( np.log2( low.normalized_twosfs(k_max = 20)[3] / targ.normalized_twosfs(k_max = 20)[3]), vmin = -0.5, vmax = 0.5, cmap="PuOr_r")
plt.colorbar()
plt.savefig(save_path + "test_two_low.png")
plt.close()
"""


"""
# +
seed = 1000
output_file = tempfile.TemporaryFile()

model = "beta"
alpha = 1.6
params = {"alpha": alpha}
folded = False
configuration = config.configuration_from_json("../simulation_parameters.json")

root_dir = "../"
spectra_file = root_dir + configuration.format_initial_spectra_file(model, params)
demo_file = root_dir + configuration.format_fitted_demography_file(
    model=model, params=params, folded=folded
)

pair_density = configuration.power_pair_densities[0]
sequence_length = configuration.power_sequence_lengths[0]
# -

rng = np.random.default_rng(seed)
raw_spectra = spectra.load_spectra(spectra_file)
num_pairs = pair_density * statistics.degenerate_pairs(raw_spectra, sequence_length)
spectra_samp = statistics.sample_spectra(raw_spectra, num_pairs=num_pairs, rng=rng)

with open(demo_file) as f:
    model_parameters = json.load(f)
msprime_parameters = configuration.msprime_parameters
msprime_parameters["sequence_length"] = sequence_length
msprime_parameters["num_replicates"] = configuration.search_num_replicates
sim_kwargs = dict(
    model="pwc",
    model_parameters=model_parameters,
    msprime_parameters=msprime_parameters,
    random_seed=rng,
)


statistics.search_recombination_rates_save(
    output_file,
    spectra_samp,
    configuration.k_max,
    folded,
    sim_kwargs,
    configuration.search_r_low,
    configuration.search_r_high,
    configuration.search_iters,
)
with h5py.File(output_file, "r") as f:
    for x in f:
        print(x)
        print(dict(f[x].attrs))  # type: ignore
        # print(spectra.spectra_from_hdf5(f[x]))  # type: ignore

"""
