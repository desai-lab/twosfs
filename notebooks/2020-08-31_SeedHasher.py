# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from hashlib import blake2b

import matplotlib.pyplot as plt
import numpy as np


def filename2seed(filename: str):
    """Generate a valid numpy seed by hashing a filename.

    Parameters
    ----------
    filename : str
        The simulation output filename.

    Returns
    -------
    int
        An integer between 0 and 2**32.

    Examples
    --------
    >>> filename2seed('path/to/my_simulation_output.npz')
    2974054299

    It is very unlikely to start two simulations with the same seed.

    >>> seed1 = filename2seed('path/to/my_simulation_output.rep1.npz')
    >>> seed2 = filename2seed('path/to/my_simulation_output.rep2.npz')
    >>> np.random.seed(seed1)
    >>> print(seed1, np.random.uniform())
    272825019 0.13286198770980562
    >>> np.random.seed(seed2)
    >>> print(seed2, np.random.uniform())
    2028164767 0.8321152367526514
    """
    h = blake2b(filename.encode(), digest_size=4)
    return int.from_bytes(h.digest(), 'big')


# +
seeds1 = np.zeros(100, dtype=int)
seeds2 = np.zeros(100, dtype=int)

for rep in range(100):
    fn1 = f"path/to/my_simulation_output.rep{rep}.txt"
    fn2 = f"path/to/my_other_simulation_output.rep{rep}.txt"
    seeds1[rep] = filename2seed(fn1)
    seeds2[rep] = filename2seed(fn2)
plt.plot(seeds1, '.')
plt.plot(seeds2, '.')
# -

plt.scatter(seeds1, seeds2)

nreps = 100000
seeds = np.zeros(nreps, dtype=int)
for rep in range(nreps):
    fn = f"path/to/my_simulation_output.rep{rep}.txt"
    seeds[rep] = int.from_bytes(
        blake2b(fn.encode(), digest_size=4).digest(), 'big')
print(min(seeds) / 2**32)
print(max(seeds) / 2**32)
plt.hist(seeds, bins=np.linspace(0, 2**32, 100))
