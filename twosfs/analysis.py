import numpy as np
import h5py
import json
from twosfs.config import configuration_from_json, make_parameter_string
import os

curr_dir = "/".join(os.path.realpath(__file__).split("/")[:-2]) + "/"
config = configuration_from_json(curr_dir + "simulation_parameters.json", root = curr_dir)


def get_power(model: tuple, folded: bool, pair_density: int,
    sequence_length: int, p_val_cutoff: float = 0.05, n_power_reps: int = 100, config=config
):
    """Returns the power from simulated models"""
    p_vals = []
    for power_rep in range(n_power_reps):
        f_name = config.format_ks_distance_file(*model,folded,pair_density,sequence_length,power_rep)
        with open(f_name) as f:
            p_vals.append(json.load(f)["p_value"])
    return sum(np.array(p_vals) < p_val_cutoff) / n_power_reps


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


