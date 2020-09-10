import numpy as np
import pyslim
import sys
import random

def get_sfs(f_name, sample_size,length):

    tseq = pyslim.load(f_name).simplify()

    windows = np.zeros(length + 3)
    windows[1:-1] = 5e7 - np.floor(length/2) + np.arange(101)
    windows[-1] = 1e8
    samples = [random.sample(range(2000), k=sample_size)]

    onesfs = np.zeros([sample_size + 1])
    twosfs = np.zeros([length, sample_size + 1, sample_size + 1])

    afs = tseq.allele_frequency_spectrum(sample_sets=samples, mode='branch', polarised=True, windows=windows)[1:-1,:]
    onesfs += np.mean(afs, axis=0)
    twosfs += afs[:,None,:]*afs[:,:,None]

    return onesfs, twosfs

