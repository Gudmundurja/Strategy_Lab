"""Position sizing: how many lots fit a loss budget, delta hedges and exposure.

A *lot* is one unit of the strategy as written (for a 1x2 ratio spread: one long
and two short contracts). Quantities are in contracts and every money figure
already includes the contract multiplier.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .position import Position


def stress_loss(pos: Position, spot_range: float = 0.20, vol_shifts=(-0.05, 0.0, 0.10), n: int = 201) -> float:
    """Worst P&L (<= 0) for spot moves within +/- ``spot_range`` combined with each vol shift,
    checked both as an instant shock today and at the first expiry."""
    S = pos.market.spot * (1.0 + np.linspace(-spot_range, spot_range, n))
    worst = 0.0
    for days in (0.0, pos.horizon):
        for dv in vol_shifts:
            worst = min(worst, float(np.min(pos.pnl(S, days, dv))))
    return worst


@dataclass
class Sizing:
    basis: str                 # 'max loss' or 'stress'
    risk_per_lot: float        # money lost per lot in the sizing scenario (>= 0)
    lots: Optional[int]        # lots that fit the budget; None if the position cannot lose in that scenario
    total_risk: float          # risk_per_lot x lots
    premium_per_lot: float     # net option premium per lot, positive = credit received


def size_for_budget(one_lot: Position, budget: float, basis: str = "auto", spot_range: float = 0.20,
                    vol_shifts=(-0.05, 0.0, 0.10)) -> Sizing:
    """Largest whole number of lots whose loss stays within ``budget``.

    ``basis='max loss'`` uses the worst case at the first expiry, which only exists
    for defined-risk positions. ``basis='stress'`` uses :func:`stress_loss`.
    ``basis='auto'`` picks max loss when it is finite and stress otherwise.
    """
    if budget < 0:
        raise ValueError("budget must be non-negative")
    a = one_lot.analyze()
    defined = math.isfinite(a.max_loss)
    if basis == "auto":
        basis = "max loss" if defined else "stress"
    if basis == "max loss":
        if not defined:
            raise ValueError("the maximum loss is unlimited, size this position with a stress scenario")
        risk = max(-a.max_loss, 0.0)
    elif basis == "stress":
        risk = max(-stress_loss(one_lot, spot_range, vol_shifts), 0.0)
    else:
        raise ValueError("basis must be 'auto', 'max loss' or 'stress'")
    if risk <= 1e-12:
        return Sizing(basis, 0.0, None, 0.0, a.net_premium)
    lots = int(math.floor(budget / risk + 1e-9))
    return Sizing(basis, risk, lots, risk * lots, a.net_premium)


def delta_hedge(pos: Position, hedge_multiplier: float = 1.0) -> dict:
    """Trade that makes the position delta-neutral.

    ``units`` is the signed amount of underlying to trade (positive = buy);
    ``contracts`` expresses it in a hedge instrument with ``hedge_multiplier``
    (e.g. 50 for E-mini S&P 500 futures).
    """
    if hedge_multiplier <= 0:
        raise ValueError("hedge_multiplier must be positive")
    d = pos.greeks()["delta"]
    return dict(units=-d, contracts=-d / hedge_multiplier, cash=-d * pos.market.spot)


def exposure(pos: Position) -> dict:
    """Contracts, underlying units controlled and their notional value."""
    m, S = pos.multiplier, pos.market.spot
    contracts = sum(abs(l.qty) for l in pos.legs if l.is_option)
    opt_units = sum(abs(l.qty) * m for l in pos.legs if l.is_option)
    und_units = sum(abs(l.qty) * m for l in pos.legs if not l.is_option)
    g = pos.greeks()
    return dict(option_contracts=contracts, option_units=opt_units, option_notional=opt_units * S,
                underlying_units=und_units, underlying_notional=und_units * S,
                delta_units=g["delta"], delta_notional=g["cash_delta"])
