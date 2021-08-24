# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 1
# %aimport twosfs.spectra, twosfs.simulations, twosfs.demography, twosfs.config

import matplotlib.pyplot as plt
import msprime
import numpy as np
from scipy.special import rel_entr

from twosfs import config, demography, simulations, spectra

d_alpha = 0.05
ALPHAS = simulations.list_rounded_parameters(np.arange(1 + d_alpha, 2.0, d_alpha))
GS = [0.5, 1.0, 2.0]
TS = [0.5, 1.0, 2.0]

initial_spectra_file = (
    "../simulations/spectra/model={model}.params={parameter_string}.rep=all.npz"
)

spectra_const = spectra.load_spectra(
    initial_spectra_file.format(model="const", parameter_string="{}")
)
print(spectra_const.tajimas_pi())
print(spectra_const.recombination_rate)
print(spectra_const.scaled_recombination_rate())

spectra_beta = {
    a: spectra.load_spectra(
        initial_spectra_file.format(
            model="beta", parameter_string=simulations.make_parameter_string(alpha=a)
        )
    )
    for a in ALPHAS
}

for a, spec in spectra_beta.items():
    plt.scatter(a, spec.tajimas_pi(), color="k")

for a, spec in spectra_beta.items():
    print(a, spec.tajimas_pi(), spec.scaled_recombination_rate(), sep="\t")

spectra_exp = {
    (t, g): spectra.load_spectra(
        initial_spectra_file.format(
            model="exp",
            parameter_string=simulations.make_parameter_string(
                end_time=t, growth_rate=g
            ),
        )
    )
    for t in TS
    for g in GS
}

for (t, g), spec in spectra_exp.items():
    print(t, g, spec.tajimas_pi(), spec.scaled_recombination_rate(), sep="\t")

plt.loglog(np.arange(1, 100), spectra_const.normalized_onesfs()[1:-1])

for (t, g), spec in spectra_exp.items():
    plt.loglog(np.arange(1, 100), spec.normalized_onesfs()[1:-1])

for a, spec in spectra_beta.items():
    plt.loglog(np.arange(1, 100), spec.normalized_onesfs()[1:-1])

print("T", "G", "RE", sep="\t")
for (t, g), spec in spectra_exp.items():
    re = np.sum(
        rel_entr(
            spectra.lump_onesfs(spectra_const.normalized_onesfs(), kmax=20),
            spectra.lump_onesfs(spec.normalized_onesfs(), kmax=20),
        )
    )
    print(t, g, re, sep="\t")

print("A", "RE", sep="\t")
for a, spec in spectra_beta.items():
    re = np.sum(
        rel_entr(
            spectra.lump_onesfs(spectra_const.normalized_onesfs(), kmax=20),
            spectra.lump_onesfs(spec.normalized_onesfs(), kmax=20),
        )
    )
    print(a, re, sep="\t")

simulations.list_rounded_parameters(np.arange(1 + d_alpha, 2.0, d_alpha))

import json

# ?json.load

# ?open

config = twosfs.config.configuration_from_json("../simulation_parameters.json")

prefixes = simulations.model_prefixes(config)

demo_dict = {p: demography.make_pwc_demography(
    *demography.read_fastNeutrino_output(f"../simulations/fitted_demographies/{p}.demography=3Epoch.txt"))
             for p in filter(lambda x: x.startswith("model=exp") or x.startswith("model=const"), prefixes)}

times = np.linspace(0, 2, 100)
for p, demo in demo_dict.items():
    print(p)
    plt.plot(times, demo.debug().population_size_trajectory(times))
    # plt.ylim([0,3])
    plt.show()

for p in prefixes:
    sizes, start_times, initial_size = demography.read_fastNeutrino_output(
        f"../simulations/fitted_demographies/{p}.demography=3Epoch.txt")
    print(demography.make_pwc_demography(sizes, start_times, initial_size))

simulations.list_rounded_parameters(np.logspace(-1,1,5, base=2))

simulations.list_rounded_parameters(np.logspace(-0.5,0.5,7,base=2))

simulations.list_rounded_parameters(np.logspace(0,1,10,base=2))

0.71 * 1.41

"simulations/spectra/{prefix}.rep={rep}.npz".replace(".rep={rep}.", ".rep=all.")

json.dumps({})


# +
def f(x, **kwargs):
    return json.dumps(kwargs)

f(0)
# -

import re

regex = re.compile("^model\=\{(?P<model>\S+)\}\.params\=\{(?P<parameter_string>\S+)\}$")

m = regex.match("model={test}.params={test2}")
print(m.group("model"))
print(m.group("parameter_string"))

# ?regex.sub

configuration = config.configuration_from_json("../simulation_parameters.json")

for prefix in simulations.model_prefixes(configuration):
    print(simulations.parse_prefix(prefix))

from typing import Iterator


def iter_models(config) -> Iterator[tuple]:
    yield "const", dict()
    for


for model, params in configuration.iter_models():
    print(model, params, sep='\t')

print(configuration.initial_spectra_file)

type(configuration.iter_models())

{(1,{1:2}): "test"}
