"""Tests for the spectra module."""


from copy import deepcopy
from tempfile import NamedTemporaryFile, TemporaryFile

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
from hypothesis import assume, given

from twosfs.spectra import (
    Spectra,
    foldonesfs,
    foldtwosfs,
    load_spectra,
    lump_onesfs,
    lump_twosfs,
    zero_spectra_like,
)


@st.composite
def onesfss(
    draw,
    ints=st.integers(min_value=1, max_value=10),
    elements=st.floats(min_value=0.0, max_value=1e6),
    num_samples=None,
):
    if num_samples is None:
        num_samples = draw(ints)
    return draw(hnp.arrays(dtype=float, shape=(num_samples + 1,), elements=elements))


@st.composite
def twosfss(
    draw,
    ints=st.integers(min_value=1, max_value=10),
    elements=st.floats(min_value=0.0, max_value=1e6),
    num_samples=None,
    num_windows=None,
):
    if num_samples is None:
        num_samples = draw(ints)
    if num_windows is None:
        num_windows = draw(ints)
    return draw(
        hnp.arrays(
            dtype=float,
            shape=(num_windows, num_samples + 1, num_samples + 1),
            elements=elements,
        )
    )


@st.composite
def kinds(draw, elements=st.integers(min_value=1, max_value=10)):
    n_samples = draw(elements)
    windows = sorted(
        draw(
            st.lists(
                st.integers(min_value=0, max_value=int(1e6)),
                min_size=1,
                max_size=10,
                unique=True,
            )
        )
    )
    rec_rate = draw(st.floats(min_value=0.0))
    return n_samples, windows, rec_rate


@st.composite
def spectras_from_kind(
    draw,
    kind,
    elements=st.floats(min_value=0.0, max_value=1e6),
    integers=st.integers(min_value=0, max_value=int(1e8)),
):
    num_samples, windows, rec_rate = kind
    num_windows = len(windows)
    num_sites = draw(integers)
    num_pairs = draw(hnp.arrays(dtype=int, shape=num_windows, elements=integers))
    if num_sites == 0:
        onesfs = np.zeros(num_samples + 1)
    else:
        onesfs = draw(onesfss(num_samples=num_samples, elements=elements))
    twosfs = draw(
        twosfss(num_samples=num_samples, num_windows=num_windows, elements=elements)
    )
    for i in range(num_windows):
        if num_pairs[i] == 0:
            twosfs[i] = 0
    return Spectra(num_samples, windows, rec_rate, num_sites, num_pairs, onesfs, twosfs)


@st.composite
def spectras(draw, num=1):
    kind = draw(kinds())
    if num == 1:
        return draw(spectras_from_kind(kind))
    else:
        return [draw(spectras_from_kind(kind)) for i in range(num)]


@given(kinds())
def test_kinds(x):
    assert isinstance(x[0], int)
    assert isinstance(x[1], list)
    assert sorted(list(set(x[1]))) == x[1]
    assert isinstance(x[2], float)


@given(spectras())
def test_spectras(x):
    assert isinstance(x, Spectra)


@given(spectras())
def test_eq(x):
    assert deepcopy(x) == x


@given(spectras(num=3))
def test_sum_associates(xs):
    s1 = xs[0] + (xs[1] + xs[2])
    s2 = (xs[0] + xs[1]) + xs[2]
    assert s1.close(s2)


@given(spectras())
def test_sum_identity(x):
    assert x + zero_spectra_like(x) == x
    assert x + 0 == x
    assert 0 + x == x


@given(spectras(num=2))
def test_sum_commutes(xs):
    assert xs[0] + xs[1] == xs[1] + xs[0]


@given(spectras(num=2))
def test_sum_preserves_intensive(xs):
    s = xs[0] + xs[1]
    assert s.compatible(xs[0])


@given(spectras())
def test_sum_extensive(x):
    s = x + x
    assert np.all(s.onesfs == 2 * x.onesfs)
    assert np.all(s.twosfs == 2 * x.twosfs)
    assert s.num_sites == 2 * x.num_sites
    assert np.all(s.num_pairs == 2 * x.num_pairs)


@given(spectras(num=2))
def test_sum_function(xs):
    assert sum(xs) == xs[0] + xs[1]


@given(spectras())
def test_save_load(x):
    """Test that saving and loading are inverses."""
    with TemporaryFile() as tf:
        x.save(tf)
        tf.seek(0)
        loaded = load_spectra(tf)
    assert x == loaded


# TODO:
# - linear
# - nullspace
# - image (or cokernel)


@given(onesfss())
def test_foldonesfs_idempotent(x):
    assert np.all(foldonesfs(foldonesfs(x)) == foldonesfs(x))


@given(onesfss())
def test_foldonesfs_symmetric(x):
    sym = x + x[::-1]
    antisym = x - x[::-1]
    assert np.all(foldonesfs(antisym) == 0)
    assert np.all(sym == 0) or not np.all(foldonesfs(sym == 0))


@given(onesfss())
def test_foldonesfs_preserves_sum(x):
    assert np.isclose(np.sum(foldonesfs(x)), np.sum(x))


@given(twosfss())
def test_foldtwosfs_idempotent(x):
    assert np.all(foldtwosfs(foldtwosfs(x)) == foldtwosfs(x))


@given(twosfss())
def test_foldtwosfs_symmetric(x):
    assert np.allclose(foldtwosfs(x[:, ::-1, ::-1]), foldtwosfs(x))


@given(twosfss())
def test_foldtwosfs_preserves_sum(x):
    assert np.allclose(np.sum(foldtwosfs(x), axis=(1, 2)), np.sum(x, axis=(1, 2)))


@given(spectras())
def test_normalized_onesfs_normalized(x):
    assume(np.sum(x.onesfs) > 0)
    assert np.isclose(np.sum(x.normalized_onesfs()), 1)


@given(spectras())
def test_normalized_onesfs_preserves_ratios(x):
    assume(np.sum(x.onesfs) > 0)
    ratio = x.normalized_onesfs()[np.nonzero(x.onesfs)] / x.onesfs[np.nonzero(x.onesfs)]
    assert np.allclose(ratio, ratio[0])


@given(spectras())
def test_normalized_twosfs_normalized(x):
    assume(np.all(np.sum(x.twosfs, axis=(1, 2)) > 0))
    assert np.allclose(np.sum(x.normalized_twosfs(), axis=(1, 2)), 1)


@given(spectras())
def test_normalized_twosfs_preserves_ratios(x):
    assume(np.all(np.sum(x.twosfs, axis=(1, 2)) > 0))
    for normed, unnormed in zip(x.normalized_twosfs(), x.twosfs):
        ratio = normed[np.nonzero(normed)] / unnormed[np.nonzero(unnormed)]
        assert np.allclose(ratio, ratio[0])


@given(onesfss(), st.integers(min_value=1, max_value=10))
def test_lump_onesfs_preserves_sums(x, kmax):
    assume(kmax <= x.shape[-1])
    assert np.isclose(np.sum(lump_onesfs(x, kmax)), np.sum(x))


@given(twosfss(), st.integers(min_value=1, max_value=10))
def test_lump_twosfs_preserves_sums(x, kmax):
    assume(kmax <= x.shape[-1])
    assert np.allclose(
        np.sum(lump_twosfs(x, kmax), axis=(1, 2)), np.sum(x, axis=(1, 2))
    )


@given(onesfss(), st.integers(min_value=1, max_value=10))
def test_lumped_onesfs_preserves_initvals(x, kmax):
    assume(kmax <= x.shape[-1])
    lumped = lump_onesfs(x, kmax)
    assert lumped.shape == (kmax + 1,)
    assert np.all(lumped[:kmax] == x[:kmax])


@given(twosfss(), st.integers(min_value=1, max_value=10))
def test_lumped_twosfs_preserves_initvals(x, kmax):
    assume(kmax <= x.shape[-1])
    lumped = lump_twosfs(x, kmax)
    assert lumped.shape == (
        x.shape[0],
        kmax + 1,
        kmax + 1,
    )
    assert np.all(lumped[:, :kmax, :kmax] == x[:, :kmax, :kmax])


# def test_export_to_fastNeutrino(self):
#     """Test that exporting fastNeutrino output matches expectation."""
#     with NamedTemporaryFile() as tf:
#         self.spectra.export_to_fastNeutrino(tf.name)
#         tf.seek(0)
#         output = tf.read()
#     self.assertEqual(
#         output,
#         b"10\t1\n100.0\n1.0909090909090908\n1.0909090909090908\n1.0909090909090908"
#         b"\n1.0909090909090908\n1.0909090909090908\n1.0909090909090908\n"
#         b"1.0909090909090908\n1.0909090909090908\n1.0909090909090908\n"
#         b"1.0909090909090908\n",
#     )
