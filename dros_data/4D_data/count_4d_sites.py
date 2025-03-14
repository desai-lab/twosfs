import json
import numpy as np
from twosfs.spectra import load_spectra

f_name = "4d_sites.txt"
with open(f_name) as f:
    data = json.load(f)

bounds = [[1e6, 17e6], [6e6, 6e19], [1e6, 17e6], [10e6, 26e6]]

for chrom, bound in zip(["2L", "2R", "3L", "3R"], bounds):
    x = data[chrom]
    lower = np.searchsorted(x, bound[0])
    upper = np.searchsorted(x, bound[1])
    print(chrom)
    print((upper - lower) / 1e3)
    print()

spec = load_spectra("../Chr2L/Chr2L_4D_initial_spectra.hdf5")
print(sum(spec.onesfs))
print(spec.onesfs[1])


