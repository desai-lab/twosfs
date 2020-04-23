# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import dadi
import matplotlib.pyplot as plt
import numpy as np

# ## Creating and folding a spectrum

fs = dadi.Spectrum([0, 100, 20, 10, 1, 0])
fs.fold()

# Our data is already folded, so we can just just apply th folding method
# immediately

fs = dadi.Spectrum([0, 101, 30, 0, 0, 0]).fold()
fs

# ## Fitting a model
# Start with the expected standard Kingman SFS for $n = 25$.

n = 25
fs = dadi.Spectrum(100 / np.arange(n + 1))

# We will fit 3-epoch models using `Inference.optimize`.
# Later, we might need to consider different optimizers.

# ?dadi.Demographics1D.three_epoch

# ?dadi.Inference.optimize

# Start close to the true model: `(1, 1, t1, t2)`,
# where `t1` and `t2` are arbitrary.

p0 = (1.2, 0.8, 1.0, 1.0)
data = fs
model_function = dadi.Demographics1D.three_epoch
pts = 100

# %%time
params = dadi.Inference.optimize(p0, data, model_function, pts)
print(params)

# It works. And reasonably fast.

# ## Plotting

# ?dadi.Plotting.plot_1d_comp_multinom

# Get the expected SFS for the initial params and the fitted model

my_extrap_func = dadi.Numerics.make_extrap_func(model_function)
model0 = my_extrap_func(p0, [n], pts)
model = my_extrap_func(params, [n], pts)

# Make plots of the SFS and anscombe residuals in both cases.

dadi.Plotting.plot_1d_comp_multinom(model0, data)

dadi.Plotting.plot_1d_comp_multinom(model, data)

# ## Fitting a Xi-Beta SFS

alpha = 1.8
sim_file = f'../simulations/msprime/xibeta-alpha={alpha:.2f}.npz'
fs_xibeta_array = np.load(sim_file)['onesfs']
fs_xibeta_array

fs_xibeta = dadi.Spectrum(fs_xibeta_array).fold()
fs_xibeta.mask[10:] = True
fs_xibeta

plt.semilogy(fs_xibeta)

p0 = (2.0, 20.0, 1.0, 0.05)
data = fs_xibeta
model_function = dadi.Demographics1D.three_epoch
pts = 100

# %%time
params = dadi.Inference.optimize_log(p0, data, model_function, pts)
print(params)

n = 100
my_extrap_func = dadi.Numerics.make_extrap_func(model_function)
model = my_extrap_func(params, [n], pts)

dadi.Plotting.plot_1d_comp_multinom(model, data)
