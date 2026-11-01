"""Structuring helpers: size tranches to target loss probabilities and price them.

The rating-agency style question "how much subordination does this tranche need?"
becomes, in the one-factor copula, a quantile of the pool loss distribution: a
tranche attaching at the (1 - p) quantile takes a loss with probability p.
"""
from __future__ import annotations

import math

import pandas as pd

from . import copula
from .structures import CapitalStructure, Tranche


def size_by_loss_probability(pd_annual: float, rho: float, lgd: float, maturity: float,
                             targets: list[tuple[str, float]], equity_name: str = "First loss",
                             rate: float = 0.03, round_to: float = 0.0025, holders: list[str] | None = None) -> CapitalStructure:
    """Build a structure bottom-up from target probabilities of loss over the deal life.

    ``targets`` lists ``(name, max probability of any loss)`` for each tranche above
    the first-loss piece, from junior to senior; probabilities must decrease. Each
    attachment point is rounded *up* to ``round_to`` (more subordination, never less).
    Coupons are set to the model's fair running spread; holders default to the
    classic SRT shape (first loss retained, mezzanine sold, senior retained).
    """
    if not targets:
        raise ValueError("give at least one tranche above the first loss")
    pd_h = copula.cumulative_pd(pd_annual, maturity)
    attaches = []
    for name, p in targets:
        if not (0 < p < 1):
            raise ValueError(f"{name}: target probability must be between 0 and 1")
        a = copula.attachment_for_target_hit_prob(p, pd_h, rho, lgd)
        if round_to:
            a = math.ceil(a / round_to - 1e-9) * round_to
        attaches.append(min(a, 0.99))
    for (n1, _), (n2, _), a1, a2 in zip(targets, targets[1:], attaches, attaches[1:]):
        if a2 <= a1:
            raise ValueError(f"{n2} must have a lower loss probability than {n1}")
    if attaches[0] <= 0:
        raise ValueError("the first loss piece came out empty; use a smaller target probability for the junior tranche")
    names = [equity_name] + [n for n, _ in targets]
    bounds = [0.0] + attaches + [1.0]
    if holders is None:
        holders = ["Retained"] + ["Sold"] * (len(names) - 2) + ["Retained"]
    tranches = []
    for i, name in enumerate(names):
        a, d = bounds[i], bounds[i + 1]
        s = copula.tranche_fair_spread(pd_annual, rho, lgd, a, d, maturity, rate)["spread"]
        tranches.append(Tranche(name, a, d, round(s, 6), holders[i]))
    return CapitalStructure(tranches)


def tranche_report(pd_annual: float, rho: float, lgd: float, maturity: float, structure: CapitalStructure,
                   rate: float = 0.03, losses=None) -> pd.DataFrame:
    """Risk and pricing for every tranche, senior first.

    Probabilities and expected losses (EL, % of the tranche) are over the whole life
    (cumulative PD). "Excess bp" is the coupon minus the fair spread.
    Pass simulated ``losses`` (fractions of the pool) to use a finite-pool Monte
    Carlo for them instead of the large-pool formula; the fair spread always uses
    the large-pool term structure.
    """
    pd_h = copula.cumulative_pd(pd_annual, maturity)
    rows = []
    for t in reversed(structure.tranches):
        if losses is None:
            m = copula.tranche_metrics(pd_h, rho, lgd, t.attach, t.detach)
        else:
            m = copula.tranche_metrics_from_losses(losses, t.attach, t.detach)
        fair = copula.tranche_fair_spread(pd_annual, rho, lgd, t.attach, t.detach, maturity, rate)["spread"]
        rows.append({"Tranche": t.name, "Attach %": t.attach * 100, "Detach %": t.detach * 100,
                     "P(loss) %": m["p_hit"] * 100, "P(wipeout) %": m["p_wipe"] * 100, "EL %": m["el"] * 100,
                     "Loss if hit %": m["lgd_given_hit"] * 100, "Fair spread bp": fair * 1e4,
                     "Coupon bp": t.coupon * 1e4, "Excess bp": (t.coupon - fair) * 1e4})
    return pd.DataFrame(rows)
