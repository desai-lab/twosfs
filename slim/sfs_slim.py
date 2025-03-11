import numpy as np
import pyslim
from twosfs.spectra import Spectra, add_spectra, spectra_from_TreeSequence
from dataclasses import dataclass, field
from os import PathLike
from typing import Any, Iterator, Union, Dict, List
import tskit

def iterate_tseqs(fname: PathLike, params: dict) -> Iterator:
    tseq = tskit.load(fname).simplify()
    tseq = tseq.simplify(params["samples"])
    tree_spacing = (1 - 2 * params["genome_cutoff"]) / params["num_trees"]

    for i in range(params["num_trees"]):
        left_bound = round( params["genome_length"] * (params["genome_cutoff"] + tree_spacing * i) )
        right_bound = round( params["genome_length"] * (params["genome_cutoff"] + tree_spacing * i) ) + params["num_bp"]

        yield tseq.keep_intervals( np.array([[left_bound, right_bound]]) ).trim()

def spectra_from_tree_file(fname: PathLike, params: dict) -> Spectra:
    return sum( spectra_from_TreeSequence(
                                           windows = np.arange(params["num_bp"] + 1),
                                           recombination_rate = params["recombination_rate"],
                                           tseq = tseq)
                for tseq in iterate_tseqs(fname, params) )

'''
def spectra_from_tree_file(fname, **params):
    tseq_array = get_tseq(fname, **params)
    spectra = sum( spectra_from_TreeSequence(windows, recombination_rate, tseq) for tseq in tseq_array )


def get_tseq(f_name, samples, length=1e8, num_bp=100, num_trees=6, cut=.2):

    tseq = pyslim.load(f_name).simplify()
    tseq = tseq.simplify(samples)

    spacing = (1-2*cut)/num_trees
    intervals = [[round(length*(cut+spacing*i)), round(length*(cut+spacing*i) + num_bp)] for i in range(num_trees)]

    tseq_array = []
    for i in range(num_trees):
        tseq_array.append(tseq.keep_intervals([intervals[i]]).trim())

    return tseq_array
'''
