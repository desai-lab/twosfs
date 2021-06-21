import numpy as np
import pyslim

def get_tseq(f_name, samples, length=1e8, num_bp=100, num_trees=6, cut=.2):

    tseq = pyslim.load(f_name).simplify()
    tseq = tseq.simplify(samples)

    spacing = (1-2*cut)/num_trees
    intervals = [[round(length*(cut+spacing*i)), round(length*(cut+spacing*i) + num_bp)] for i in range(num_trees)]

    tseq_array = []
    for i in range(num_trees):
        tseq_array.append(tseq.keep_intervals([intervals[i]]).trim())

    return tseq_array
