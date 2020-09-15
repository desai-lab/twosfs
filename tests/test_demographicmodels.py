"""Test demographic models."""

from twosfs.demographicmodel import DemographicModel


def test_t2_constant():
    """Test that piecewise constant T2 agree between methods."""
    dm = DemographicModel()
    dm.add_epoch(0.0, 2.0)
    dm.add_epoch(2.0, 1.0)
    assert dm.t2() == dm._legacy_t2()
