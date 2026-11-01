import numpy as np
import pytest

from derivlab import VolModel


def test_flat_model_gives_one_vol_everywhere():
    v = VolModel.flat(0.3).vol(np.array([50.0, 100.0, 200.0]), 0.5, 100.0)
    assert np.allclose(v, 0.3)


def test_atm_vol_at_the_forward():
    for m in (VolModel.equity(0.2), VolModel.fx(0.1), VolModel(atm=0.3, skew=-0.05, smile=0.01)):
        assert float(m.vol(102.0, 0.5, 102.0)) == pytest.approx(m.atm)
        assert float(m.moneyness(102.0, 0.5, 102.0)) == pytest.approx(0.0)


def test_equity_skew_makes_low_strikes_expensive():
    m = VolModel.equity(0.2)
    v = m.vol(np.array([80.0, 90.0, 100.0, 110.0, 120.0]), 0.25, 100.0)
    assert np.all(np.diff(v) < 0)


def test_fx_smile_lifts_both_wings():
    m = VolModel.fx(0.09)
    lo, mid, hi = (float(m.vol(k, 0.5, 1.0)) for k in (0.9, 1.0, 1.1))
    assert lo > mid and hi > mid


def test_parallel_shift_is_parallel():
    m = VolModel.equity(0.2)
    K = np.linspace(70, 130, 13)
    diff = m.shifted(0.05).vol(K, 0.5, 100.0) - m.vol(K, 0.5, 100.0)
    assert np.allclose(diff, 0.05)
    assert m.shifted(0.05).level == pytest.approx(0.25)
    assert m.shifted(0.02).shifted(-0.02).vol(90.0, 0.5, 100.0) == pytest.approx(m.vol(90.0, 0.5, 100.0))


def test_floor_applies():
    m = VolModel(atm=0.10, skew=-0.2, floor=0.01)
    assert float(m.vol(300.0, 0.1, 100.0)) == pytest.approx(0.01)
    assert VolModel.flat(0.2).shifted(-0.5).level == pytest.approx(0.01)


def test_wings_are_flat_beyond_the_cutoff():
    m = VolModel.equity(0.2)                     # wing = 4 standard deviations
    T, F = 0.25, 100.0
    k_at_wing = F * np.exp(-4 * 0.2 * np.sqrt(T))
    assert float(m.vol(k_at_wing * 0.5, T, F)) == pytest.approx(float(m.vol(k_at_wing, T, F)))
    assert float(m.vol(k_at_wing * 1.01, T, F)) < float(m.vol(k_at_wing, T, F))
