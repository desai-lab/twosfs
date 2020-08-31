"""Helper functions for computing 2SFS and other statistics."""
import numpy as np


def sims2sfs(sims, sample_size, length):
    """Compute the SFS and 2SFS from msprime simulation output.

    Parameters
    ----------
    sims :
        msprime simulation output (a generator of tree sequences)
    sample_size :
        the sample size of the simulation
    length :
        the length of the simulated sequence (in basepairs)

    Returns
    -------
    onesfs : ndarray
        1D array containing average branch lengths.
        `onesfs[j]` is the length subtending `j` samples.
    twosfs : ndarray
        3D array containing the products of the branch lengths.
        `twosfs[i,j,k]` is the average of the product of the
        length subtending `j` samples at position `0` and the
        length subtending `k` samples at position `i`.
    """
    windows = np.arange(length + 1)
    onesfs = np.zeros((sample_size + 1))
    twosfs = np.zeros((length, sample_size + 1, sample_size + 1))
    n_sims = 0
    for tseq in sims:
        afs = tseq.allele_frequency_spectrum(mode='branch',
                                             windows=windows,
                                             polarised=True)
        onesfs += np.mean(afs, axis=0)
        twosfs += afs[0, :, None] * afs[:, None, :]
        n_sims += 1
    return onesfs / n_sims, twosfs / n_sims


def sims2pi(sims, num_replicates):
    """Compute pairwise diversity (T_2) for a generator of tree sequences."""
    pi = sum(tseq.diversity(mode='branch') for tseq in sims)
    pi /= num_replicates
    return (pi)


def sfs2pi(sfs):
    """Compute the average pairwise diversity from an SFS."""
    n = len(sfs) - 1
    k = np.arange(n + 1)
    weights = 2 * k * (n - k) / (n * (n - 1))
    return np.dot(sfs, weights)


def lump_sfs(sfs, kmax):
    """Lump all sfs bins for k <= kmax into one bin."""
    sfs_lumped = np.zeros(kmax + 1)
    sfs_lumped[:-1] = sfs[:kmax]
    sfs_lumped[-1] = np.sum(sfs[kmax:])
    return sfs_lumped
