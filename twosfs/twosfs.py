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
    """Lump all sfs bins for k>=kmax into one bin."""
    sfs_lumped = np.zeros(kmax + 1)
    sfs_lumped[:-1] = sfs[:kmax]
    sfs_lumped[-1] = np.sum(sfs[kmax:])
    return sfs_lumped


def lump_twosfs(twosfs, kmax):
    """Lump all 2-sfs bins for k>=kmax into one bin."""
    twosfs_lumped = np.zeros((twosfs.shape[0], kmax + 1, kmax + 1))
    twosfs_lumped[:, :-1, :-1] = twosfs[:, :kmax, :kmax]
    twosfs_lumped[:, -1, :-1] = np.sum(twosfs[:, kmax:, :kmax], axis=1)
    twosfs_lumped[:, :-1, -1] = np.sum(twosfs[:, :kmax, kmax:], axis=2)
    twosfs_lumped[:, -1, -1] = np.sum(twosfs[:, kmax:, kmax:], axis=(1, 2))
    return twosfs_lumped


def lump_spectra(sfs, twosfs, kmax):
    """Lump all bins for k>=kmax into one bin."""
    return lump_sfs(sfs, kmax), lump_twosfs(twosfs, kmax)


def save_spectra(filename, onesfs, twosfs):
    """Save SFS and 2-SFS to file."""
    np.savez_compressed(filename, onesfs=onesfs, twosfs=twosfs)


def load_spectra(filename):
    """Load SFS and 2-SFS from file."""
    data = np.load(filename)
    return data['onesfs'], data['twosfs']


def avg_spectra(spectra_list):
    """Average a list of tuples (onesfs, twosfs)."""
    return tuple(sum(x) / len(spectra_list) for x in zip(*spectra_list))


def export_to_fastNeutrino(filename: str, sfs, sfs_0=100):
    """Write SFS as a fastNeutrino input file.

    Parameters
    ----------
    filename : str
        The name of the fastNeutrino input file to write.
    sfs : array_like
        Length n+1 array containing the site frequency spectrum.
    sfs_0 : int
        The number in the zero-class. This does not effect fitting.
        (Default=100).
    """
    n = len(sfs) - 1
    # Normalize sfs to have T_2 = 4.
    sfs *= 4 / sfs2pi(sfs)
    sfs[0] = sfs_0
    with open(filename, 'w') as outfile:
        outfile.write(f'{n}\t1\n')
        outfile.write('\n'.join(map(str, sfs)) + '\n')
