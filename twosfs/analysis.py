import numpy as np
import h5py
from twosfs.config import configuration_from_json, make_parameter_string

config = configuration_from_json("../simulation_parameters.json", root = "../")

def get_p_value(rec_search_file, ks_distance_file):
    """
    Calculates the empirical p-value from a KS distance file and a recombination search
    file

    Parameters
    ----------
        rec_search_file: path to a recombination search (.hdf5) file
        ks_distance_file: path to a ks (.hdf5) distance file
    """
    # Find the KS distance from the lower and upper recombination rates:
    with h5py.File(rec_search_file) as hf:
        ks_h = hf.get("spectra_high").attrs["ks_distance"]
        ks_l = hf.get("spectra_low").attrs["ks_distance"]
    # We use the recombination rate that is a better fit to the data - i.e. the one that has
    # a lower KS distance. This quantity represents the KS distance between the data and the
    # null. Then, p equals the fraction of resampled 2-SFS's with KS distance larger than
    # that.
    with h5py.File(ks_distance_file) as hf:
        if ks_h > ks_l:
            p = sum( ks_l < np.array(hf.get("ks_null"))) / len (hf.get("ks_null")[:])
        else:
            p = sum( ks_h < np.array(hf.get("ks_null"))) / len (hf.get("ks_null")[:])
    return p

def get_power(model, params, folded, pair_density, sequence_length, reps=config.power_reps):
    """
    Calculates the power to reject Kingman coalescence from a simulated model
    """
    p_vals = []
    # Convert the parameter dictionary to a string
    if type(params) == dict:
        params = make_parameter_string(params)
    for rep in range(reps):
        # Format the relevant files
        rec_search_file = config.recombination_search_file.format(model=model,
                                                                  params=params,
                                                                  folded=folded,
                                                                  pair_density=pair_density,
                                                                  sequence_length=sequence_length,
                                                                  power_rep=rep
                                                                 )
        ks_distance_file = config.ks_distance_file.format(model=model,
                                                          params=params,
                                                          folded=folded,
                                                          pair_density=pair_density,
                                                          sequence_length=sequence_length,
                                                          power_rep=rep
                                                         )
        p_vals.append(get_p_value(rec_search_file, ks_distance_file))

    # Power (at p=0.05) is the fraction of reps with p-value less than 0.05
    return sum(np.array(p_vals) < 0.05) / reps

def demo_to_plot(sizes, times):
    """
    Takes population sizes and change times as listed in demography files and reformats
    in a way that leads to stepwise plots. Returns two arrays:
        t = [0,  t1, t1, t2, t2, t3, t3, ...]
        y = [y1, y1, y2, y2, y3, y3, y4, ...]
    """
    t = [0]
    for time in times:
        t.append(time)
        t.append(time)
    t.append(t[-1] * 10)
    t = np.array(t) / t[-1]
    y = []
    for size in sizes:
        y.append(size)
        y.append(size)
    return y, t


