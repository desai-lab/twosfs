import numpy as np
import pyslim
import sys
import random

def get_sfs(f_name,sample_size):
    tseq = pyslim.load(f_name).simplify()
    
    windows = [0,5e7-5,5e7-4,5e7-3,5e7-2,5e7-1,5e7,5e7+1,5e7+2,5e7+3,5e7+4,1e8]

    onesfs = np.zeros([sample_size+1])
    samples = [random.sample(range(2000), k=sample_size)]

    afs = tseq.allele_frequency_spectrum(sample_sets=samples, mode='branch', polarised=True, windows=windows)
    print(afs.shape)
    onesfs += np.mean(afs[1:-1,:],axis=0)

    return onesfs

