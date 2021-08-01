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

from dataclasses import dataclass

import matplotlib.pyplot as plt

from twosfs.config import configuration_from_json
from twosfs.spectra import Spectra, load_spectra


@dataclass
class Data:
    model: str
    params: dict
    spectra: Spectra


configuration = configuration_from_json("../simulation_parameters.json")
root = "../"
data = [
    Data(
        model,
        params,
        load_spectra(root + configuration.format_initial_spectra_file(model, params)),
    )
    for model, params in configuration.iter_models()
]

for i, d in enumerate(data):
    print(i, d.model, d.params)


sp_test = data[28].spectra
num_epochs = 5
folded = True
pc = 2e-6
for i in range(100):
    demo = sp_test.fit_pwc_demography(
        folded=folded,
        k_max=configuration.k_max,
        num_epochs=num_epochs,
        num_restarts=1,
        penalty_coef=pc,
        interval_bounds=(1e-2, 10),
        # options={"ftol": 1e-11, "gtol": 1e-13},
    )
    if demo.kl_div < 1e-6:
        print(demo.kl_div, demo.sizes, demo.times, sep="\n")
        print()

sp_test = data[1].spectra
num_epochs = 5
folded = True
pc = 1e-6
for i in range(100):
    demo = sp_test.fit_pwc_demography(
        folded=folded,
        k_max=configuration.k_max,
        num_epochs=num_epochs,
        num_restarts=1,
        penalty_coef=pc,
        interval_bounds=(1e-2, 10),
        # options={"ftol": 1e-11, "gtol": 1e-13},
    )
    if demo.kl_div < 1e-5:
        print(demo.kl_div, demo.sizes, demo.times, sep="\n")
        print()

# +
num_epochs = 5
folded = False
pc = 1e-6

fitted_unfolded = [
    (
        d.model,
        d.params,
        d.spectra.fit_pwc_demography(
            folded=folded,
            k_max=configuration.k_max,
            num_epochs=num_epochs,
            num_restarts=50,
            penalty_coef=pc,
            interval_bounds=(1e-2, 10),
        ),
    )
    for d in data
]
# -

for model, params, pwc_demo in fitted_unfolded:
    print(model, params, pwc_demo.kl_div)

for _, params, pwc_demo in filter(lambda x: x[0] == "beta", fitted_unfolded):
    alpha = params["alpha"]
    plt.semilogy(alpha, pwc_demo.kl_div, "ok")

for _, params, pwc_demo in filter(lambda x: x[0] == "beta", fitted_unfolded):
    alpha = params["alpha"]
    plt.loglog([0.001] + list(pwc_demo.times), pwc_demo.sizes, drawstyle="steps-post")

for _, params, pwc_demo in filter(lambda x: x[0] == "exp", fitted_unfolded):
    g = params["growth_rate"]
    plt.semilogy(g, pwc_demo.kl_div, "ok")

for _, params, pwc_demo in filter(lambda x: x[0] == "exp", fitted_unfolded):
    g = params["growth_rate"]
    plt.loglog([0.001] + list(pwc_demo.times), pwc_demo.sizes, drawstyle="steps-post")


# +
num_epochs = 5
folded = True
pc = 1e-6

fitted_folded = [
    (
        d.model,
        d.params,
        d.spectra.fit_pwc_demography(
            folded=folded,
            k_max=configuration.k_max,
            num_epochs=num_epochs,
            num_restarts=50,
            penalty_coef=pc,
            interval_bounds=(1e-2, 10),
        ),
    )
    for d in data
]
# -

for model, params, pwc_demo in fitted_folded:
    print(model, params, pwc_demo.kl_div)

for _, params, pwc_demo in filter(lambda x: x[0] == "beta", fitted_folded):
    alpha = params["alpha"]
    plt.semilogy(alpha, pwc_demo.kl_div, "ok")

for _, params, pwc_demo in filter(lambda x: x[0] == "beta", fitted_folded):
    alpha = params["alpha"]
    plt.loglog([0.001] + list(pwc_demo.times), pwc_demo.sizes, drawstyle="steps-post")

for _, params, pwc_demo in filter(lambda x: x[0] == "exp", fitted_folded):
    g = params["growth_rate"]
    plt.semilogy(g, pwc_demo.kl_div, "ok")

for _, params, pwc_demo in filter(lambda x: x[0] == "exp", fitted_folded):
    g = params["growth_rate"]
    plt.loglog([0.001] + list(pwc_demo.times), pwc_demo.sizes, drawstyle="steps-post")
