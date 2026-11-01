import math

import numpy as np
import pytest

from derivlab import Leg, Market, Position, VolModel, presets
from derivlab.position import DAYS_PER_YEAR
from derivlab.pricing import bs_greeks, bs_price

R = 0.04
M = Market(spot=100.0, rate=R, vol=VolModel.flat(0.25))
T30 = 30 / DAYS_PER_YEAR


def price(kind, K, days=30, S=100.0, vol=0.25):
    return bs_price(kind, S, K, days / DAYS_PER_YEAR, R, 0.0, vol)


def test_long_call_limits():
    pos = Position([Leg("call", 1, 100, 30)], M, multiplier=100)
    a = pos.analyze()
    prem = price("call", 100)
    assert a.net_premium == pytest.approx(-100 * prem)
    assert a.max_loss == pytest.approx(-100 * prem)
    assert math.isinf(a.max_profit)
    assert a.breakevens == [pytest.approx(100 + prem, abs=1e-6)]


def test_bull_call_spread_limits():
    pos = Position([Leg("call", 1, 100, 30), Leg("call", -1, 110, 30)], M)
    debit = price("call", 100) - price("call", 110)
    a = pos.analyze()
    assert a.net_premium == pytest.approx(-debit)
    assert a.max_profit == pytest.approx(10 - debit)
    assert a.max_loss == pytest.approx(-debit)
    assert a.breakevens == [pytest.approx(100 + debit, abs=1e-6)]


def test_short_put_loss_is_large_but_bounded():
    a = Position([Leg("put", -1, 95, 30)], M).analyze()
    prem = price("put", 95)
    assert a.max_profit == pytest.approx(prem)
    assert a.max_loss == pytest.approx(-(95 - prem), abs=1e-3)       # the stock going to zero


def test_straddle_breakevens_are_strike_plus_minus_premium():
    pos = Position([Leg("call", 1, 100, 30), Leg("put", 1, 100, 30)], M)
    prem = price("call", 100) + price("put", 100)
    a = pos.analyze()
    assert a.breakevens == [pytest.approx(100 - prem, abs=1e-6), pytest.approx(100 + prem, abs=1e-6)]
    assert a.max_loss == pytest.approx(-prem)


def test_iron_condor_max_loss_is_width_minus_credit():
    legs = [Leg("put", 1, 90, 30), Leg("put", -1, 95, 30), Leg("call", -1, 105, 30), Leg("call", 1, 110, 30)]
    a = Position(legs, M).analyze()
    credit = a.net_premium
    assert credit > 0
    assert a.max_profit == pytest.approx(credit)
    assert a.max_loss == pytest.approx(-(5 - credit))


def test_box_spread_locks_in_the_risk_free_rate():
    legs = [Leg("call", 1, 95, 30), Leg("call", -1, 105, 30), Leg("put", 1, 105, 30), Leg("put", -1, 95, 30)]
    pos = Position(legs, M)
    S = np.linspace(50, 150, 101)
    pnl = pos.pnl(S, 30)
    assert np.allclose(pnl, 10 * (1 - math.exp(-R * T30)))
    a = pos.analyze()
    assert a.breakevens == [] and a.prob_profit == pytest.approx(1.0)


def test_conversion_locks_in_strike():
    legs = [Leg("underlying", 1), Leg("put", 1, 100, 30), Leg("call", -1, 100, 30)]
    pnl = Position(legs, M).pnl(np.linspace(60, 140, 9), 30)
    assert np.allclose(pnl, 100 * (1 - math.exp(-R * T30)))


def test_pnl_is_zero_at_inception_with_model_prices():
    pos = Position(presets.get("iron-condor").make_legs(100), M, multiplier=100)
    assert pos.pnl(100.0, 0.0) == pytest.approx(0.0, abs=1e-9)


def test_premium_override_changes_entry_cost():
    pos = Position([Leg("call", 1, 100, 30, premium=2.0)], M, multiplier=100)
    assert pos.entry_cost == pytest.approx(2.0)
    assert pos.pnl(100.0, 0.0) == pytest.approx((price("call", 100) - 2.0) * 100)


def test_multiplier_and_lots_scale_everything_linearly():
    legs = presets.get("call-ratio-spread").make_legs(100)
    base = Position(legs, M, multiplier=1)
    big = Position(legs, M, multiplier=100).scaled(3)
    S = np.linspace(80, 120, 9)
    assert np.allclose(big.pnl(S, 10), 300 * base.pnl(S, 10))
    gb, gs = big.greeks(), base.greeks()
    for k in ("delta", "gamma", "theta", "vega", "rho"):
        assert gb[k] == pytest.approx(300 * gs[k])
    ab, a1 = big.analyze(), base.analyze()
    assert ab.max_profit == pytest.approx(300 * a1.max_profit)
    assert ab.breakevens == pytest.approx(a1.breakevens)


def test_greeks_match_black_scholes_with_units():
    pos = Position([Leg("call", 2, 105, 60)], M, multiplier=100)
    g = pos.greeks()
    ref = bs_greeks("call", 100, 105, 60 / DAYS_PER_YEAR, R, 0.0, 0.25)
    assert g["delta"] == pytest.approx(200 * ref["delta"])
    assert g["theta"] == pytest.approx(200 * ref["theta"] / DAYS_PER_YEAR)
    assert g["vega"] == pytest.approx(200 * ref["vega"] / 100)
    assert g["cash_delta"] == pytest.approx(g["delta"] * 100)


def test_vega_predicts_a_one_point_vol_move():
    pos = Position(presets.get("long-straddle").make_legs(100), M, multiplier=100)
    assert pos.pnl(100.0, 0.0, 0.01) == pytest.approx(pos.greeks()["vega"], rel=1e-3)


def test_scenario_grid_centre_is_zero_and_shape():
    pos = Position(presets.get("short-strangle").make_legs(100), M, multiplier=100)
    grid = pos.scenario_grid()
    assert grid.shape == (8, 11)
    assert grid.loc[0.0, 0.0] == pytest.approx(0.0, abs=1e-9)
    assert grid.loc[20.0, 0.0] < 0 < grid.loc[-10.0, 0.0]          # short vol: loses when vol rises


def test_sticky_moneyness_changes_delta_only_with_a_smile():
    legs = [Leg("call", 1, 105, 60)]
    flat_s = Position(legs, Market(spot=100, rate=R, vol=VolModel.flat(0.2), sticky="strike")).greeks()["delta"]
    flat_m = Position(legs, Market(spot=100, rate=R, vol=VolModel.flat(0.2), sticky="moneyness")).greeks()["delta"]
    assert flat_s == pytest.approx(flat_m)
    sk = Market(spot=100, rate=R, vol=VolModel.equity(0.2), sticky="strike")
    mo = Market(spot=100, rate=R, vol=VolModel.equity(0.2), sticky="moneyness")
    d_strike = Position(legs, sk).greeks()["delta"]
    pos_m = Position(legs, mo)
    d_money = pos_m.greeks()["delta"]
    assert d_money > d_strike                                       # put skew: the smile slides up as spot rises
    fd = (pos_m.value(100.01) - pos_m.value(99.99)) / 0.02
    assert d_money == pytest.approx(float(fd), rel=1e-4)


def test_prob_profit_of_a_long_future_matches_the_lognormal():
    pos = Position([Leg("underlying", 1)], M, default_horizon=30)
    sig, T = 0.25, T30
    expected = 0.5 * math.erfc(-((R - 0.5 * sig * sig) * T / (sig * math.sqrt(T))) / math.sqrt(2))
    assert pos.prob_profit() == pytest.approx(expected, abs=1e-9)


def test_multiple_expiries_use_the_first():
    pos = Position(presets.get("calendar-spread").make_legs(100), M)
    assert pos.has_multiple_expiries and pos.horizon == 30
    a = pos.analyze()
    assert math.isfinite(a.max_profit) and math.isfinite(a.max_loss)
    assert len(a.breakevens) == 2


@pytest.mark.parametrize("strategy", presets.STRATEGIES, ids=lambda s: s.id)
def test_every_preset_analyses(strategy):
    pos = Position(strategy.make_legs(100), M, multiplier=100)
    a = pos.analyze()
    assert a.max_profit >= a.max_loss
    assert 0.0 <= a.prob_profit <= 1.0
    assert a.breakevens == sorted(a.breakevens)
    assert len(pos.leg_table()) == len(strategy.legs)


@pytest.mark.parametrize("strategy", presets.STRATEGIES, ids=lambda s: s.id)
def test_preset_greek_signs_match_the_model(strategy):
    """The '+'/'-'/'0' notes on each preset must agree with the Black-Scholes Greeks at inception."""
    g = Position(strategy.make_legs(100), M).greeks()
    values = dict(d=g["delta"], g=g["gamma"], t=g["theta"], v=g["vega"])
    near_zero = dict(d=0.1, g=1e-3, t=0.02, v=0.01)
    for key, sign in strategy.greeks.items():
        v = values[key]
        if sign == "+":
            assert v > 0, (key, v)
        elif sign == "-":
            assert v < 0, (key, v)
        elif sign == "0":
            assert abs(v) <= near_zero[key], (key, v)


def test_preset_strikes_snap_to_listed_increments():
    legs = presets.get("iron-condor").make_legs(4512.0)
    assert all(l.strike % 25 == 0 for l in legs)
    assert presets.strike_step(55.0) == 0.5


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        Leg("future", 1)
    with pytest.raises(ValueError):
        Leg("call", 1, -5, 30)
    with pytest.raises(ValueError):
        Leg("put", 1, 100, 0)
    with pytest.raises(ValueError):
        Market(spot=0)
    with pytest.raises(ValueError):
        Market(sticky="delta")
    with pytest.raises(ValueError):
        Position([], M)


@pytest.mark.parametrize("strategy", presets.STRATEGIES, ids=lambda s: s.id)
def test_the_smile_never_changes_whether_risk_is_unlimited(strategy):
    flat = Position(strategy.make_legs(100), M).analyze()
    skewed = Position(strategy.make_legs(100), Market(spot=100, rate=R, vol=VolModel.equity(0.25), sticky="moneyness")).analyze()
    assert math.isinf(flat.max_profit) == math.isinf(skewed.max_profit)
    assert math.isinf(flat.max_loss) == math.isinf(skewed.max_loss)
