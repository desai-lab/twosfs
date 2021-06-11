"""Helper functions for running msprime simulations."""
from hashlib import blake2b

import msprime
import numpy as np
from scipy.special import betaln

from twosfs.demographicmodel import DemographicModel
from twosfs.spectra import add_spectra, spectra_from_TreeSequence


def parameter_map(prefix: str, default_parameters: dict):
    """Convert a file prefix to a set of msprime parameters."""
    if prefix == "kingman":
        parameters = {}
    elif prefix.startswith("xibeta"):
        alpha = float(prefix.split("alpha=")[1])
        # Rescale recombination to keep r*T_2 constant
        r = default_parameters["recombination_rate"] / beta_timescale(alpha)
        parameters = {
            "model": msprime.BetaCoalescent(alpha=alpha),
            "recombination_rate": r,
        }
    elif prefix.startswith("expgrowth"):
        p = {
            elem.split("=")[0]: float(elem.split("=")[1])
            for elem in prefix.split("-")[1:]
        }
        g = p["g"]
        t = p["t"]
        dm = DemographicModel()
        dm.add_epoch(0.0, 1.0, g)
        dm.add_epoch(t, np.exp(-g * t))
        dm.rescale()
        parameters = {"demographic_events": dm.get_demographic_events()}
    else:
        raise KeyError("Simulation prefix not found")
    return dict(default_parameters, **parameters)


def simulate_spectra(parameters: dict):
    """Run msprime simulations and return a Spectra object.

    Parameters
    ----------
    parameters : Dict
        Parameters to pass to msprime
    """
    sims = msprime.simulate(**parameters)
    windows = np.arange(parameters["length"] + 1)
    return add_spectra(
        spectra_from_TreeSequence(windows, parameters["recombination_rate"], tseq)
        for tseq in sims
    )


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


def sims2pi(sims, num_replicates):
    """Compute pairwise diversity (T_2) for a generator of tree sequences."""
    pi = sum(tseq.diversity(mode="branch") for tseq in sims)
    pi /= num_replicates
    return pi
