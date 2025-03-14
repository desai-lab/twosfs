import matplotlib.pyplot as plt
import numpy as np
import h5py

chro = "Chr2R"

with h5py.File("{}/{}_ks_distance.folded=True.hdf5".format(chro, chro)) as hf:
    ks_null = np.array(hf.get("ks_null"))

with h5py.File("{}/{}_rec_search.folded=True.hdf5".format(chro, chro)) as hf:
    if dict(hf.get("spectra_high").attrs)["ks_distance"] < dict(hf.get("spectra_low").attrs)["ks_distance"]:
        ks_data = dict(hf.get("spectra_high").attrs)["ks_distance"]
    else:
        ks_data = dict(hf.get("spectra_low").attrs)["ks_distance"]

print(ks_null)
print(np.mean(ks_null))
print(ks_data)
