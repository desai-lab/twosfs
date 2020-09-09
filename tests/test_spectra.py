"""Tests for the spectra module."""


from tempfile import NamedTemporaryFile, TemporaryFile
from unittest import TestCase, main

import numpy as np

from twosfs import Spectra, avg_spectra, load_spectra, sfs2pi


class TestSpectra(TestCase):
    """Test the Spectra class."""

    def setUp(self):
        """Set up a sample spectrum and some variables for comparison."""
        self.sample_size = 10
        self.length = 5
        self.recombination_rate = 1.0
        self.sfs = np.ones(self.sample_size + 1)
        self.twosfs = np.ones((self.length, self.sample_size + 1, self.sample_size + 1))
        self.spectra = Spectra(
            self.sfs, self.twosfs, recombination_rate=self.recombination_rate
        )

    def test_dimensions(self):
        """Test that the provided spectra have the correct shape."""
        self.assertEqual(self.spectra.sample_size, self.sample_size)
        self.assertEqual(self.spectra.length, self.length)
        with self.assertRaises(ValueError):
            Spectra(
                np.empty((1, self.sample_size + 1)),
                np.empty((self.length, self.sample_size + 1, self.sample_size + 1)),
            )
        with self.assertRaises(ValueError):
            Spectra(
                np.empty(self.sample_size + 1),
                np.empty((self.sample_size + 1, self.sample_size + 1)),
            )
        with self.assertRaises(ValueError):
            Spectra(
                np.empty(self.sample_size + 1),
                np.empty((self.length, self.sample_size + 1, self.sample_size + 1, 1)),
            )
        with self.assertRaises(ValueError):
            Spectra(
                np.empty(self.sample_size + 1),
                np.empty((self.length, self.sample_size, self.sample_size + 1)),
            )
        with self.assertRaises(ValueError):
            Spectra(
                np.empty(self.sample_size + 1),
                np.empty((self.length, self.sample_size, self.sample_size)),
            )

    def test__init__recombination(self):
        """Check that the recombination rate defaults to zero and takes value."""
        self.assertEqual(Spectra(self.sfs, self.twosfs).recombination_rate, 0.0)
        self.assertEqual(self.spectra.recombination_rate, self.recombination_rate)

    def test___init__normalization(self):
        """
        Test the normalization flag of the initializer.

        - Sets t2 to provided value if normalized and t2 provided.
        - Computes t2 from sfs if not normalized and t2 not provided.
        - Raises exceptions in other cases.
        """
        sfs_normed = self.sfs / np.sum(self.sfs)
        twosfs_normed = self.twosfs / np.sum(self.twosfs, axis=(1, 2))[:, None, None]
        # If normalized, set t2
        self.assertEqual(
            Spectra(sfs_normed, twosfs_normed, t2=4.5, normalized=True).t2, 4.5
        )
        # If not normalized, set t2
        spectra = Spectra(self.sfs, self.twosfs, normalized=False)
        self.assertEqual(spectra.t2, sfs2pi(self.sfs))
        # t2 must be specified if normalized
        with self.assertRaises(ValueError):
            Spectra(sfs_normed, twosfs_normed, normalized=True)
        # t2 must not be specified if not normalized
        with self.assertRaises(ValueError):
            Spectra(self.sfs, self.twosfs, t2=1.0, normalized=False)
        # if normalized, check normalization
        with self.assertRaises(ValueError):
            spectra = Spectra(2 * sfs_normed, twosfs_normed, t2=1.0, normalized=True)
        with self.assertRaises(ValueError):
            spectra = Spectra(sfs_normed, 2 * twosfs_normed, t2=1.0, normalized=True)

    def test___init__folded(self):
        """
        Test that the folded kwarg to init is working properly.

        - Should set the folded attribute
        - Should raise an exception when given unfolded spectra
        """
        n = self.sample_size
        sfs_folded = np.ones(n + 1)
        sfs_folded[-(n + 1) // 2 :] = 0.0
        twosfs_folded = np.ones((self.length, n + 1, n + 1))
        twosfs_folded[:, -(n + 1) // 2 :, -(n + 1) // 2 :] = 0.0
        self.assertTrue(Spectra(sfs_folded, twosfs_folded, folded=True).folded)
        self.assertFalse(Spectra(self.sfs, self.twosfs, folded=False).folded)
        # Test default unfolded
        self.assertFalse(Spectra(self.sfs, self.twosfs).folded)
        with self.assertRaises(ValueError):
            Spectra(self.sfs, self.twosfs, folded=True)

    def test___eq__(self):
        """Test the equality relation."""
        a = Spectra(self.sfs, self.twosfs)
        b = Spectra(self.sfs.copy(), self.twosfs.copy())
        self.assertTrue(a == b)
        b.sfs[1] /= 2
        self.assertFalse(a == b)

    def test_save_load(self):
        """Test that saving and loading returns equivalent spectra."""
        with TemporaryFile() as tf:
            self.spectra.save(tf)
            tf.seek(0)
            loaded_spectra = load_spectra(tf)
        self.assertTrue(self.spectra == loaded_spectra)

    def test_export_to_fastNeutrino(self):
        """Test that exporting fastNeutrino output matches expectation."""
        with NamedTemporaryFile() as tf:
            self.spectra.export_to_fastNeutrino(tf.name)
            tf.seek(0)
            output = tf.read()
        self.assertEqual(
            output,
            b"10\t1\n100.0\n1.0909090909090908\n1.0909090909090908\n1.0909090909090908"
            b"\n1.0909090909090908\n1.0909090909090908\n1.0909090909090908\n"
            b"1.0909090909090908\n1.0909090909090908\n1.0909090909090908\n"
            b"1.0909090909090908\n",
        )

    def test_normalize(self):
        """Test that normalizing spectra makes them sum to one."""
        spectra = Spectra(self.sfs, self.twosfs)
        self.assertFalse(spectra.normalized)
        spectra.normalize()
        self.assertTrue(spectra.normalized)
        self.assertTrue(np.isclose(np.sum(spectra.sfs), 1))
        self.assertTrue(np.allclose(np.sum(spectra.twosfs, axis=(1, 2)), 1))

    def test_fold(self):
        """
        Testing that folding works properly.

        - Preserves sums
        - Zeros out high frequencies
        - Gives correctly folded spectra
        """
        self.spectra.fold()
        self.assertTrue(self.spectra.folded)
        # Folding preserves sums
        self.assertEqual(np.sum(self.spectra.sfs), np.sum(self.sfs))
        self.assertTrue(
            np.all(
                np.sum(self.spectra.twosfs, axis=(1, 2))
                == np.sum(self.twosfs, axis=(1, 2))
            )
        )
        # Should zero out high frequencies
        high_freq = self.sample_size // 2 + 1
        self.assertTrue(np.allclose(self.spectra.sfs[high_freq:], 0.0))
        self.assertTrue(
            np.allclose(self.spectra.twosfs[:, high_freq:, high_freq:], 0.0)
        )
        # Test exact values
        sfs = np.arange(4)
        twosfs = sfs[None, :, None] + sfs[None, None, :]
        spectra = Spectra(sfs, twosfs)
        spectra.fold()
        self.assertTrue(np.all(spectra.sfs == np.array([3, 3, 0, 0])))
        self.assertTrue(
            np.all(
                spectra.twosfs
                == np.array(
                    [[12, 12, 0, 0], [12, 12, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
                )
            )
        )

    def test_lumped_sfs(self):
        """Test that lumped sfs preserve values and give the right shape."""
        kmax = 5
        lumped_sfs = self.spectra.lumped_sfs(kmax)
        self.assertEqual(lumped_sfs.shape, (kmax + 1,))
        self.assertTrue(np.all(self.sfs[: kmax - 1] == lumped_sfs[: kmax - 1]))
        self.assertEqual(np.sum(self.sfs), np.sum(lumped_sfs))

    def test_lumped_twosfs(self):
        """Test that lumped twosfs preserve values and give the right shape."""
        kmax = 5
        lumped_twosfs = self.spectra.lumped_twosfs(kmax)
        self.assertEqual(lumped_twosfs.shape, (self.length, kmax + 1, kmax + 1))
        self.assertTrue(
            np.all(
                self.twosfs[:, : kmax - 1, : kmax - 1]
                == lumped_twosfs[:, : kmax - 1, : kmax - 1]
            )
        )
        self.assertEqual(np.sum(self.twosfs), np.sum(lumped_twosfs))


class TestAvgSpectra(TestCase):
    """Test the avg_spectra function."""

    def test_avg_spectra(self):
        """
        Test that avg_spectra is working properly.

        - Averages the branch length properties of two spectra.
        - Rejects mismatches in recombination rate.
        - Rejects normalized spectra.
        """
        n = 4
        length = 2
        sfs1 = np.ones(n)
        twosfs1 = np.ones((length, n, n))
        sfs2 = 2 * sfs1
        twosfs2 = 2 * twosfs1
        spectra1 = Spectra(sfs1, twosfs1, recombination_rate=1.0)
        spectra2 = Spectra(sfs2, twosfs2, recombination_rate=1.0)
        avg = avg_spectra([spectra1, spectra2])
        self.assertTrue(np.isclose(avg.t2, (spectra1.t2 + spectra2.t2) / 2))
        self.assertTrue(np.allclose(avg.sfs, (sfs1 + sfs2) / 2))
        self.assertTrue(np.allclose(avg.twosfs, (twosfs1 + twosfs2) / 2))
        # Wrong recombination rate
        spectra3 = Spectra(sfs2, twosfs2, recombination_rate=2.0)
        with self.assertRaises(ValueError):
            avg_spectra([spectra1, spectra3])
        # Can't average normalized spectra
        spectra4 = Spectra(sfs2, twosfs2, recombination_rate=1.0)
        spectra4.normalize()
        with self.assertRaises(ValueError):
            avg_spectra([spectra1, spectra4])


if __name__ == "__main__":
    main()
