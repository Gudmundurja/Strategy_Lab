import math

import numpy as np
import pytest

from derivlab.pricing import binomial_price, bs_greeks, bs_price, implied_vol


def test_textbook_values():
    # The standard S = K = 100, T = 1, r = 5%, vol = 20% example
    assert bs_price("call", 100, 100, 1, 0.05, 0, 0.2) == pytest.approx(10.4506, abs=1e-4)
    assert bs_price("put", 100, 100, 1, 0.05, 0, 0.2) == pytest.approx(5.5735, abs=1e-4)


@pytest.mark.parametrize("S,K,T,r,q,v", [(100, 90, 0.5, 0.03, 0.01, 0.3), (50, 60, 2, 0.0, 0.02, 0.15),
                                         (1.10, 1.05, 0.25, 0.04, 0.02, 0.09)])
def test_put_call_parity(S, K, T, r, q, v):
    c = bs_price("call", S, K, T, r, q, v)
    p = bs_price("put", S, K, T, r, q, v)
    assert c - p == pytest.approx(S * math.exp(-q * T) - K * math.exp(-r * T), abs=1e-10)


def test_vectorised_matches_scalar():
    S = np.linspace(50, 150, 11)
    vec = bs_price("call", S, 100, 0.5, 0.02, 0, 0.25)
    assert vec.shape == S.shape
    for s, v in zip(S, vec):
        assert v == pytest.approx(bs_price("call", float(s), 100, 0.5, 0.02, 0, 0.25))


def test_expired_and_zero_vol_limits():
    assert bs_price("call", 110, 100, 0, 0.05, 0, 0.2) == pytest.approx(10)
    assert bs_price("put", 90, 100, 0, 0.05, 0, 0.2) == pytest.approx(10)
    assert bs_price("call", 90, 100, 0, 0.05, 0, 0.2) == 0
    assert bs_price("call", 100, 90, 1, 0.05, 0, 0.0) == pytest.approx(100 - 90 * math.exp(-0.05))


def test_rejects_unknown_kind():
    with pytest.raises(ValueError):
        bs_price("straddle", 100, 100, 1, 0.0, 0.0, 0.2)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_match_finite_differences(kind):
    S, K, T, r, q, v = 100.0, 105.0, 0.4, 0.03, 0.01, 0.27
    f = lambda s=S, vol=v, t=T, rr=r: bs_price(kind, s, K, t, rr, q, vol)
    g = bs_greeks(kind, S, K, T, r, q, v)
    h = 1e-4
    assert g["delta"] == pytest.approx((f(s=S + h) - f(s=S - h)) / (2 * h), rel=1e-6)
    assert g["gamma"] == pytest.approx((f(s=S + 0.01) - 2 * f() + f(s=S - 0.01)) / 1e-4, rel=1e-4)
    assert g["vega"] == pytest.approx((f(vol=v + h) - f(vol=v - h)) / (2 * h), rel=1e-6)
    assert g["theta"] == pytest.approx(-(f(t=T + h) - f(t=T - h)) / (2 * h), rel=1e-5)   # theta = -dV/dT
    assert g["rho"] == pytest.approx((f(rr=r + h) - f(rr=r - h)) / (2 * h), rel=1e-6)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("vol", [0.05, 0.2, 0.8])
def test_implied_vol_round_trip(kind, vol):
    px = bs_price(kind, 100, 110, 0.75, 0.02, 0.0, vol)
    assert implied_vol(kind, px, 100, 110, 0.75, 0.02, 0.0) == pytest.approx(vol, abs=1e-8)


def test_implied_vol_outside_no_arbitrage_bounds_is_nan():
    assert math.isnan(implied_vol("call", 0.0, 100, 100, 1, 0.0))
    assert math.isnan(implied_vol("call", 150.0, 100, 100, 1, 0.0))     # more than the stock itself
    assert math.isnan(implied_vol("put", 1.0, 100, 150, 1, 0.0))        # below intrinsic


def test_binomial_converges_to_black_scholes():
    bs = bs_price("put", 100, 100, 1, 0.05, 0, 0.2)
    assert binomial_price("put", 100, 100, 1, 0.05, 0, 0.2, steps=1000, american=False) == pytest.approx(bs, abs=5e-3)


def test_american_put_is_worth_more_than_european():
    eu = binomial_price("put", 100, 100, 1, 0.05, 0, 0.2, american=False)
    am = binomial_price("put", 100, 100, 1, 0.05, 0, 0.2, american=True)
    assert am > eu + 0.3


def test_american_call_without_dividends_is_never_exercised_early():
    eu = binomial_price("call", 100, 95, 1, 0.05, 0, 0.25, american=False)
    am = binomial_price("call", 100, 95, 1, 0.05, 0, 0.25, american=True)
    assert am == pytest.approx(eu, abs=1e-10)
