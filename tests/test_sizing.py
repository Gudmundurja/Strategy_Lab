import math

import pytest

from derivlab import Leg, Market, Position, VolModel, presets, sizing

M = Market(spot=100.0, rate=0.04, vol=VolModel.flat(0.25))


def test_defined_risk_position_is_sized_by_max_loss():
    one = Position(presets.get("bull-call-spread").make_legs(100), M, multiplier=100)
    debit = -one.analyze().net_premium
    res = sizing.size_for_budget(one, 10_000)
    assert res.basis == "max loss"
    assert res.risk_per_lot == pytest.approx(debit)
    assert res.lots == math.floor(10_000 / debit)
    assert res.total_risk <= 10_000


def test_unlimited_loss_is_sized_by_stress():
    one = Position([Leg("call", -1, 105, 30)], M, multiplier=100)
    res = sizing.size_for_budget(one, 10_000)
    assert res.basis == "stress"
    worst = sizing.stress_loss(one, 0.20)
    assert res.risk_per_lot == pytest.approx(-worst)
    with pytest.raises(ValueError):
        sizing.size_for_budget(one, 10_000, basis="max loss")


def test_wider_stress_means_fewer_lots():
    one = Position(presets.get("short-strangle").make_legs(100), M, multiplier=100)
    narrow = sizing.size_for_budget(one, 50_000, basis="stress", spot_range=0.10)
    wide = sizing.size_for_budget(one, 50_000, basis="stress", spot_range=0.30)
    assert wide.risk_per_lot > narrow.risk_per_lot and wide.lots <= narrow.lots


def test_position_that_cannot_lose_has_no_lot_limit():
    one = Position(presets.get("box-spread").make_legs(100), M, multiplier=100)
    res = sizing.size_for_budget(one, 1_000)
    assert res.lots is None and res.risk_per_lot == 0


def test_delta_hedge_makes_the_book_delta_neutral():
    pos = Position(presets.get("risk-reversal").make_legs(100), M, multiplier=100).scaled(7)
    h = sizing.delta_hedge(pos)
    hedged = Position(pos.legs + [Leg("underlying", h["units"] / pos.multiplier)], M, multiplier=100)
    assert hedged.greeks()["delta"] == pytest.approx(0.0, abs=1e-9)
    fut = sizing.delta_hedge(pos, hedge_multiplier=50)
    assert fut["contracts"] == pytest.approx(h["units"] / 50)


def test_exposure_counts_contracts_and_notional():
    pos = Position(presets.get("collar").make_legs(100), M, multiplier=100).scaled(2)
    ex = sizing.exposure(pos)
    assert ex["option_contracts"] == 4
    assert ex["underlying_units"] == 200
    assert ex["option_notional"] == pytest.approx(4 * 100 * 100)
