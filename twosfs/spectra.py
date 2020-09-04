"""Helper functions for computing 2SFS and other statistics."""
# TODO: rewrite docstring

import numpy as np


class Spectra:
    def __init__(
        self,
        sfs,
        twosfs,
        t2: float = None,
        normalized: bool = False,
        folded: bool = False,
        recombination_rate: float = None,
    ):
        r"""__init__.

        Parameters
        ----------
        sfs : ndarray
            1D array containing the site frequency spectrum for sample size $n$
            `sfs.shape == (n+1,)` and `sfs[i]` is expected number of i-ton mutations.
        twosfs :
            3D array containing the 2-SFS for distances $\{0,\ldots, l-1\}$
            `twosfs.shape == (l, n+1, n+1)`
        t2 : float
            The average branch length separating two individuals.
        normalized : bool
            If `normalized` is True, `sfs` and each entry of `twosfs` must sum to 1.
        folded : bool
            `folded` is True if `sfs` and `twosfs` represent minor allele frequencies.
        recombination_rate : float
            The per-site recombination rate.
        """
        self.sfs = sfs.copy()
        self.twosfs = twosfs.copy()
        self.sample_size = len(sfs) - 1
        self.length = twosfs.shape[0]
        self._check_dimensions()
        self.normalized = normalized
        self.t2 = self._compute_t2(t2, normalized)
        self.folded = folded
        if self.folded:
            self._check_folded()
        if recombination_rate is None:
            self.recombination_rate = 0.0
        else:
            self.recombination_rate = recombination_rate

    def _check_dimensions(self):
        if self.sfs.shape != (self.sample_size + 1,):
            raise ValueError("sfs must have shape (n + 1,)")
        if self.twosfs.shape != (
            self.length,
            self.sample_size + 1,
            self.sample_size + 1,
        ):
            raise ValueError("twosfs must have shape (length, n + 1, n + 1)")

    def _compute_t2(self, t2, normalized):
        if normalized:
            if t2 is None:
                raise ValueError("If normalized is True, you must specify t2.")
            if np.sum(self.sfs) != 1.0:
                raise ValueError("If normalized is True, sfs must sum to one.")
            if not np.allclose(np.sum(self.twosfs, axis=(1, 2)), 1.0):
                raise ValueError(
                    "If normalized is True, each entry of twosfs must sum to one."
                )
            return t2
        else:
            if t2 is not None:
                raise ValueError("If normalized is False, you may not specify t2.")
            return sfs2pi(self.sfs)

    def _check_folded(self):
        high_freq = self.sample_size // 2 + 1
        sfs_folded = np.allclose(self.sfs[high_freq:], 0.0)
        twosfs_folded = np.allclose(self.twosfs[:, high_freq:, high_freq:], 0.0)
        if not (sfs_folded and twosfs_folded):
            raise ValueError(
                "If folded is True, the spectra must have zero entries for k>=n//2+1."
            )

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return all(
                np.all(self.__dict__[k] == other.__dict__[k]) for k in self.__dict__
            )
        else:
            return NotImplemented

    def save(self, output_file) -> None:
        np.savez_compressed(output_file, **self.__dict__)

    def export_to_fastNeutrino(self, filename: str, sfs_0: float = 100.0) -> None:
        """Write SFS as a fastNeutrino input file.

        Parameters
        ----------
        filename : str
            The name of the fastNeutrino input file to write.
        sfs_0 : int
            The number in the zero-class. This does not effect fitting.
            (Default=100).
        """
        export_to_fastNeutrino(filename, self.sfs, sfs_0=sfs_0)

    def normalize(self) -> None:
        self.sfs /= np.sum(self.sfs)
        self.twosfs /= np.sum(self.twosfs, axis=(1, 2))[:, None, None]
        self.normalized = True

    def fold(self) -> None:
        n_fold = (self.sample_size + 1) // 2
        self.sfs[:n_fold] += self.sfs[: -(n_fold + 1) : -1]
        self.sfs[-n_fold:] = 0.0
        self.twosfs[:, :n_fold, :] += self.twosfs[:, : -(n_fold + 1) : -1, :]
        self.twosfs[:, :, :n_fold] += self.twosfs[:, :, : -(n_fold + 1) : -1]
        self.twosfs[:, -n_fold:, :] = 0.0
        self.twosfs[:, :, -n_fold:] = 0.0
        self.folded = True

    def lumped_sfs(self, kmax: int = None):
        return lump_sfs(self.sfs, kmax=kmax)

    def lumped_twosfs(self, kmax: int = None):
        return lump_twosfs(self.twosfs, kmax=kmax)


def spectra_from_TreeSequences(treeseqs, sample_size, length, recombination_rate):
    sfs, twosfs = sims2sfs(treeseqs, sample_size, length)
    return Spectra(sfs, twosfs, recombination_rate=recombination_rate, normalized=False)


# TODO: move to simulations
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
        afs = tseq.allele_frequency_spectrum(
            mode="branch", windows=windows, polarised=True
        )
        onesfs += np.mean(afs, axis=0)
        twosfs += afs[0, :, None] * afs[:, None, :]
        n_sims += 1
    return onesfs / n_sims, twosfs / n_sims


# TODO: move to simulations.py
def sims2pi(sims, num_replicates):
    """Compute pairwise diversity (T_2) for a generator of tree sequences."""
    pi = sum(tseq.diversity(mode="branch") for tseq in sims)
    pi /= num_replicates
    return pi


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


# TODO: remove once dependencies are gone
def lump_spectra(sfs, twosfs, kmax=None):
    """Lump all bins for k>=kmax into one bin."""
    if kmax is None:
        kmax = len(sfs - 1)
    return lump_sfs(sfs, kmax), lump_twosfs(twosfs, kmax)


def load_spectra(input_file):
    data = np.load(input_file)
    sfs = data["sfs"]
    twosfs = data["twosfs"]
    kwargs = {k: data[k] for k in ["t2", "normalized", "folded", "recombination_rate"]}
    if not kwargs["normalized"]:
        kwargs.pop("t2")
    spectra = Spectra(sfs, twosfs, **kwargs)
    return spectra


def avg_spectra(spectra_list):
    if any(s.normalized for s in spectra_list):
        raise ValueError("You may not average normalized spectra.")
    r = spectra_list[0].recombination_rate
    if not np.allclose([s.recombination_rate for s in spectra_list], r):
        raise ValueError(
            "You may not average spectra with different recombination rates."
        )
    sfs = sum(s.sfs for s in spectra_list) / len(spectra_list)
    twosfs = sum(s.twosfs for s in spectra_list) / len(spectra_list)
    return Spectra(sfs, twosfs, normalized=False, recombination_rate=r)


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
    with open(filename, "w") as outfile:
        outfile.write(f"{n}\t1\n")
        outfile.write("\n".join(map(str, sfs)) + "\n")
