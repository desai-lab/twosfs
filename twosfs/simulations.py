"""Helper functions for running msprime simulations."""
from hashlib import blake2b

import numpy as np
from scipy.special import betaln


def beta_timescale(alpha, pop_size=1.0):
    """Compute the timescale of the beta coalescent."""
    m = 2 + np.exp(alpha * np.log(2) + (1 - alpha) * np.log(3) - np.log(alpha - 1))
    N = pop_size / 2
    # The initial 2 is so that the rescaling by beta_timescale
    # gives T_2 = 4
    ret = 2 * np.exp(
        alpha * np.log(m)
        + (alpha - 1) * np.log(N)
        - np.log(alpha)
        - betaln(2 - alpha, alpha)
    )
    return ret


def filename2seed(filename: str):
    """Generate a valid numpy seed by hashing a filename.

    Parameters
    ----------
    filename : str
        The simulation output filename.

    Returns
    -------
    int
        An integer between 0 and 2**32.

    Examples
    --------
    >>> filename2seed('path/to/my_simulation_output.npz')
    2974054299

    It is very unlikely to start two simulations with the same seed.

    >>> seed1 = filename2seed('path/to/my_simulation_output.rep1.npz')
    >>> seed2 = filename2seed('path/to/my_simulation_output.rep2.npz')
    >>> np.random.seed(seed1)
    >>> print(seed1, np.random.uniform())
    272825019 0.13286198770980562
    >>> np.random.seed(seed2)
    >>> print(seed2, np.random.uniform())
    2028164767 0.8321152367526514
    """
    h = blake2b(filename.encode(), digest_size=4)
    return int.from_bytes(h.digest(), "big")


# TODO
# def sims2sfs(sims, sample_size, length):
#     """Compute the SFS and 2SFS from msprime simulation output.

#     Parameters
#     ----------
#     sims :
#         msprime simulation output (a generator of tree sequences)
#     sample_size :
#         the sample size of the simulation
#     length :
#         the length of the simulated sequence (in basepairs)

#     Returns
#     -------
#     onesfs : ndarray
#         1D array containing average branch lengths.
#         `onesfs[j]` is the length subtending `j` samples.
#     twosfs : ndarray
#         3D array containing the products of the branch lengths.
#         `twosfs[i,j,k]` is the average of the product of the
#         length subtending `j` samples at position `0` and the
#         length subtending `k` samples at position `i`.
#     """
#     windows = np.arange(length + 1)
#     onesfs = np.zeros((sample_size + 1))
#     twosfs = np.zeros((length, sample_size + 1, sample_size + 1))
#     n_sims = 0
#     for tseq in sims:
#         afs = tseq.allele_frequency_spectrum(
#             mode="branch", windows=windows, polarised=True
#         )
#         onesfs += np.mean(afs, axis=0)
#         twosfs += afs[0, :, None] * afs[:, None, :]
#         n_sims += 1
#     return onesfs / n_sims, twosfs / n_sims

# TODO
# def spectra_from_TreeSequences(
#     treeseqs, sample_size: int, length: int, recombination_rate: float
# ):
#     """Create a Spectra object from a list of tskit.TreeSequences.

#     Parameters
#     ----------
#     treeseqs :
#         The list of TreeSequences.
#     sample_size : int
#         The sample size.
#     length : int
#         The length of sequence to calculate 2SFS for.
#     recombination_rate : float
#         The recombination rate of the sequences.
#     """
#     sfs, twosfs = sims2sfs(treeseqs, sample_size, length)
#     return Spectra(sfs, twosfs, recombination_rate=recombination_rate, normalized=False)


def sims2pi(sims, num_replicates):
    """Compute pairwise diversity (T_2) for a generator of tree sequences."""
    pi = sum(tseq.diversity(mode="branch") for tseq in sims)
    pi /= num_replicates
    return pi
