# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 1
# %aimport twosfs.statistics


import numpy as np

from twosfs import statistics

# ## Test flip


def random_pdf(n):
    raw_pdf = np.random.uniform(size=(n, n))
    return raw_pdf / np.sum(raw_pdf)


pdf1 = random_pdf(3)
pdf2 = random_pdf(3)
statistics.max_ks_distance(pdf1, pdf2)
