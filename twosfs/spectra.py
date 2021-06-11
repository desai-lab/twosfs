"""Class and functions for manipulating SFS and 2SFS."""
from collections.abc import Iterable
from copy import deepcopy

import attr
import attr.validators as v
import numpy as np
import tskit

# Converters


def _float_array(value) -> np.ndarray:
    return np.array(value, dtype=float)


# Validators


def _nonnegative(instance, attribute, value):
    if np.any(value < 0):
        raise ValueError(f"{attribute.name} must be nonnegative.")


def _strictly_increasing(instance, attribute, value):
    if not np.all(np.diff(value) > 0):
        raise ValueError(f"{attribute.name} must be strictly increasing.")


def _1D(instance, attribute, value):
    if value.ndim != 1:
        raise ValueError(f"{attribute.name} must be a 1D array.")


def _3D(instance, attribute, value):
    if value.ndim != 3:
        raise ValueError(f"{attribute.name} must be a 3D array.")


def _matches_num_samples(instance, attribute, value):
    if value.shape[-1] != instance.num_samples + 1:
        raise ValueError(
            f"Last dimension of {attribute.name} must equal num_samples + 1."
        )


def _matches_windows(instance, attribute, value):
    if value.shape[0] != len(instance.windows) - 1:
        raise ValueError(
            f"First dimension of {attribute.name} must equal len(windows) - 1."
        )


def _last_dims_square(instance, attribute, value):
    if value.shape[-1] != value.shape[-2]:
        raise ValueError(f"Last two dimensions of {attribute.name} must be equal.")


def _zero_if_num_sites(instance, attribute, value):
    if instance.num_sites == 0 and np.any(value > 0):
        raise ValueError(
            f"If num_sites == 0, {attribute.name} must only contain zeros."
        )


def _zero_if_num_pairs(instance, attribute, value):
    if any((n == 0) & np.any(x > 0) for n, x in zip(instance.num_pairs, value)):
        raise ValueError(
            f"If num_pairs == 0, {attribute.name} must only contain zeros."
        )


@attr.s(eq=False)
class Spectra(object):
    """
    Stores SFS and 2SFS data.

    Attributes
    ----------
    num_samples : int
        The sample size (i.e. number of haploid genomes.)
    windows : ndarray
        The boundaries of the windows for computing the 2SFS
    recombination_rate : float
       The per-site recombination rate.
    num_sites : float
        The number of sites contributing to the SFS
    num_pairs : ndarray
        The number of pairs of sites contributing to the 2SFS
    sfs : ndarray
       1D array containing the site frequency spectrum for sample size n
       `sfs.shape == (n+1,)` and `sfs[i]` is expected number of i-ton mutations.
    twosfs : ndarray
       3D array containing the 2-SFS for each of l windows
       `twosfs.shape == (l, n+1, n+1)`
    """

    # attr constructor
    num_samples: int = attr.ib(validator=[v.instance_of(int), _nonnegative])
    windows: np.ndarray = attr.ib(
        converter=_float_array, validator=[_1D, _nonnegative, _strictly_increasing]
    )
    recombination_rate: float = attr.ib(
        converter=float, validator=[v.instance_of(float), _nonnegative]
    )
    num_sites: float = attr.ib(converter=float, validator=[_nonnegative])
    num_pairs: np.ndarray = attr.ib(
        converter=_float_array, validator=[_1D, _matches_windows, _nonnegative]
    )
    onesfs: np.ndarray = attr.ib(
        converter=_float_array,
        validator=[_1D, _matches_num_samples, _nonnegative, _zero_if_num_sites],
    )
    twosfs: np.ndarray = attr.ib(
        converter=_float_array,
        validator=[
            _3D,
            _matches_windows,
            _matches_num_samples,
            _last_dims_square,
            _nonnegative,
            _zero_if_num_pairs,
        ],
    )

    def __eq__(self, other) -> bool:
        """Equality is equality of elements."""
        if type(self) is not type(other):
            return NotImplemented
        return all(
            np.all(getattr(self, field.name) == getattr(other, field.name))
            for field in attr.fields(type(self))
        )

    def close(self, other) -> bool:
        """Equality is equality of elements."""
        if type(self) is not type(other):
            return NotImplemented
        return all(
            np.allclose(getattr(self, field.name), getattr(other, field.name))
            for field in attr.fields(type(self))
        )

    def compatible(self, other: "Spectra") -> bool:
        """
        Determine whether spectra are compatible for addition.

        Compatible spectra have the same:
        - num_samples
        - windows
        - recombination rate.
        """
        return (
            self.num_samples == other.num_samples
            and self.windows.shape == other.windows.shape
            and np.all(self.windows == other.windows)
            and self.recombination_rate == other.recombination_rate
        )

    def __add__(self, other) -> "Spectra":
        """Adding spectra adds extensive fields and preserves intensive ones."""
        if other == 0:
            # Lift numerical zero to zero_spectra. (For sum function to work.)
            return self.__add__(zero_spectra_like(self))
        elif type(self) is not type(other):
            return NotImplemented
        return add_spectra((self, other))

    def __radd__(self, other) -> "Spectra":
        """Addition of spectra is commutative."""
        return self.__add__(other)

    def normalized_onesfs(self, folded=False) -> np.ndarray:
        """Return the SFS normalized to one."""
        normed = self.onesfs / np.sum(self.onesfs)
        if folded:
            return foldonesfs(normed)
        else:
            return normed

    def normalized_twosfs(self, folded=False) -> np.ndarray:
        """Return the 2SFS normalized to one in each window."""
        normed = self.twosfs / np.sum(self.twosfs, axis=(1, 2))[:, None, None]
        if folded:
            return foldtwosfs(normed)
        else:
            return normed

    def t2(self) -> float:
        """Return the pairwise coalescence time (proportional to Tajima's pi)."""
        return t2(self.onesfs / self.num_sites)

    def export_to_fastNeutrino(
        self, filename: str, sfs_0: float = 100.0, folded: bool = False
    ) -> None:
        """Write SFS as a fastNeutrino input file.

        Parameters
        ----------
        filename : str
            The name of the fastNeutrino input file to write.
        sfs_0 : int
            The number in the zero-class. This does not effect fitting.
            (Default=100).
        folded : bool
            If True, fold the sfs before exporting.
        """
        export_to_fastNeutrino(
            filename, self.normalized_onesfs(folded=folded), sfs_0=sfs_0
        )

    def save(self, output_file) -> None:
        """Save Spectra to a .npz file.

        Parameters
        ----------
        output_file :
            May be a filename string or a file handle.
        """
        np.savez_compressed(output_file, **self.__dict__)


def add_spectra(specs: Iterable[Spectra]):
    """Add an iterable of compatible spectra."""
    it = iter(specs)
    ret = deepcopy(next(it))
    for s in it:
        if not ret.compatible(s):
            raise ValueError("Spectra are incompatible.")
        ret.num_sites += s.num_sites
        ret.num_pairs += s.num_pairs
        ret.onesfs += s.onesfs
        ret.twosfs += s.twosfs
    return ret


# Spectra constructors


def load_spectra(input_file):
    """Read a Spectra object from a .npz file created by Spectra.save()."""
    with np.load(input_file) as data:
        kws = dict(data)
    kws["num_samples"] = int(kws["num_samples"])
    return Spectra(**kws)


def zero_spectra(num_samples: int, windows, recombination_rate: float) -> Spectra:
    """Construct an empty Spectra object."""
    return Spectra(
        num_samples,
        windows,
        recombination_rate,
        0,
        np.zeros(len(windows) - 1),
        np.zeros(num_samples + 1),
        np.zeros((len(windows) - 1, num_samples + 1, num_samples + 1)),
    )


def zero_spectra_like(spectra: Spectra) -> Spectra:
    """Construct an empty Spectra object that is compatible with spectra."""
    return zero_spectra(
        spectra.num_samples, spectra.windows, spectra.recombination_rate
    )


def spectra_from_TreeSequence(
    windows, recombination_rate: float, tseq: tskit.TreeSequence
) -> Spectra:
    """Construct a Spectra object from a tskit.TreeSeqeunce."""
    num_samples = tseq.sample_size
    num_sites = windows[-1] - windows[0]
    num_pairs = np.diff(windows)
    afs = tseq.allele_frequency_spectrum(
        mode="branch", windows=windows, polarised=True, span_normalise=False
    )
    onesfs = np.sum(afs, axis=0)
    twosfs = afs[0, :, None] * afs[:, None, :]
    return Spectra(
        num_samples, windows, recombination_rate, num_sites, num_pairs, onesfs, twosfs
    )


def spectra_from_sites(
    num_samples: int,
    windows: np.ndarray,
    recombination_rate: float,
    allele_count_dict: dict[int, int],
) -> Spectra:
    """Create a Spectra from a dictionary of allele counts and positions.

    Parameters
    ----------
    num_samples : int
        The sample size (i.e. number of haploid genomes.)
    windows : ndarray
        The boundaries of the windows for computing the 2SFS
    recombination_rate : float
       The per-site recombination rate.
    allele_count_dict : Dict[int, int]
        A dictionary of `position: allele_count` pairs

    Returns
    -------
    Spectra

    """
    onesfs = np.zeros(num_samples + 1)
    twosfs = np.zeros((len(windows) - 1, num_samples + 1, num_samples + 1))
    num_sites = 0
    num_pairs = np.zeros(len(windows) - 1)
    for pos, ac1 in allele_count_dict.items():
        num_sites += 1
        onesfs[ac1] += 1
        for i, dist in enumerate(windows[:-1]):
            for d in range(dist, windows[i + 1]):
                try:
                    ac2 = allele_count_dict[pos + d]
                except KeyError:
                    continue
                num_pairs[i] += 1
                twosfs[i, ac1, ac2] += 1
                twosfs[i, ac2, ac1] += 1
    return Spectra(
        num_samples, windows, recombination_rate, num_sites, num_pairs, onesfs, twosfs
    )


# Functions of arrays.


def foldonesfs(onesfs: np.ndarray) -> np.ndarray:
    """Fold the SFS so that it represents minor allele frequencies."""
    n_fold = len(onesfs) // 2
    folded = np.zeros_like(onesfs)
    folded[:-n_fold] = onesfs[:-n_fold]
    folded[:n_fold] += onesfs[: -(n_fold + 1) : -1]
    return folded


def foldtwosfs(twosfs: np.ndarray) -> np.ndarray:
    """Fold the 2SFS so that it represents minor allele frequencies."""
    n_fold = twosfs.shape[-1] // 2
    folded = np.zeros_like(twosfs)
    folded[:, :-n_fold, :-n_fold] = twosfs[:, :-n_fold, :-n_fold]
    folded[:, :-n_fold, :n_fold] += twosfs[:, :-n_fold, : -(n_fold + 1) : -1]
    folded[:, :n_fold, :-n_fold] += twosfs[:, : -(n_fold + 1) : -1, :-n_fold]
    folded[:, :n_fold, :n_fold] += twosfs[:, : -(n_fold + 1) : -1, : -(n_fold + 1) : -1]
    return folded


def lump_onesfs(onesfs: np.ndarray, kmax: int) -> np.ndarray:
    """Lump all sfs bins for k>=kmax into one bin."""
    onesfs_lumped = np.zeros(kmax + 1)
    onesfs_lumped[:-1] = onesfs[:kmax]
    onesfs_lumped[-1] = np.sum(onesfs[kmax:])
    return onesfs_lumped


def lump_twosfs(twosfs: np.ndarray, kmax: int) -> np.ndarray:
    """Lump all 2-sfs bins for k>=kmax into one bin."""
    twosfs_lumped = np.zeros((twosfs.shape[0], kmax + 1, kmax + 1))
    twosfs_lumped[:, :-1, :-1] = twosfs[:, :kmax, :kmax]
    twosfs_lumped[:, -1, :-1] = np.sum(twosfs[:, kmax:, :kmax], axis=1)
    twosfs_lumped[:, :-1, -1] = np.sum(twosfs[:, :kmax, kmax:], axis=2)
    twosfs_lumped[:, -1, -1] = np.sum(twosfs[:, kmax:, kmax:], axis=(1, 2))
    return twosfs_lumped


def t2(onesfs: np.ndarray) -> float:
    """Compute the average pairwise diversity from an SFS."""
    n = len(onesfs) - 1
    k = np.arange(n + 1)
    weights = 2 * k * (n - k) / (n * (n - 1))
    return np.dot(onesfs, weights)


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
    sfs *= 4 / t2(sfs)
    sfs[0] = sfs_0
    with open(filename, "w") as outfile:
        outfile.write(f"{n}\t1\n")
        outfile.write("\n".join(map(str, sfs)) + "\n")
