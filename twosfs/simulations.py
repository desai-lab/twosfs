"""Helper functions for running msprime simulations."""
from hashlib import blake2b

import msprime
import numpy as np
from scipy.special import betaln

from twosfs.spectra import add_spectra, spectra_from_TreeSequence


def simulate_spectra(parameters: dict):
    """Run msprime simulations and return a Spectra object.

    Parameters
    ----------
    parameters : Dict
        Parameters to pass to msprime
    """
    sims = msprime.sim_ancestry(**parameters)
    windows = np.arange(parameters["length"] + 1)
    return add_spectra(
        spectra_from_TreeSequence(windows, parameters["recombination_rate"], tseq)
        for tseq in sims
    )


def expected_t2_beta(alpha, pop_size=1.0):
    """Compute the mean coalescent time of the diploid beta coalescent."""
    m = 2 + np.exp(alpha * np.log(2) - (alpha - 1) * np.log(3) - np.log(alpha - 1))
    return np.exp(
        np.log(4)
        + alpha * np.log(m)
        + (alpha - 1) * np.log(pop_size / 2)
        - np.log(alpha)
        - betaln(2 - alpha, alpha)
    )


def sims2pi(sims, num_replicates):
    """Compute pairwise diversity for a generator of tree sequences."""
    pi = sum(tseq.diversity(mode="branch") for tseq in sims)
    pi /= num_replicates
    return pi


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
