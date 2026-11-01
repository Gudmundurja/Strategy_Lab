"""A monthly cash-flow waterfall for an amortising securitisation.

Mechanics (deliberately simple and transparent):

* Collateral: level-pay amortising (auto/consumer/mortgage style) or bullet loans.
  Each month defaults hit the performing balance at the monthly default rate
  (from CDR), then scheduled principal, then prepayments (from CPR).
* Losses = defaults x severity, recognised at default; recoveries arrive after
  ``recovery_lag`` months.
* Interest waterfall: servicing fee -> note coupons senior to junior -> excess
  spread covers the month's losses (paid out as principal, "turbo") -> the rest
  is released to the equity (residual) tranche.
* Losses not covered by excess spread write down tranches from the bottom up.
* Principal (scheduled + prepayments + recoveries + loss cover) pays the notes
  sequentially, senior first; equity is repaid last.

The bottom tranche of the capital structure is treated as equity: it receives
no coupon, only residual cash and its remaining principal.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .structures import CapitalStructure


@dataclass
class PoolAssumptions:
    balance: float = 500_000_000.0
    wac: float = 0.085              # annual collateral coupon
    term_months: int = 60
    amortization: str = "level"     # 'level' or 'bullet'
    cpr: float = 0.10               # annual prepayment rate
    cdr: float = 0.02               # annual default rate
    severity: float = 0.50          # loss given default
    recovery_lag: int = 6           # months
    servicing_fee: float = 0.005    # annual, on performing balance


@dataclass
class WaterfallResult:
    periods: pd.DataFrame
    summary: pd.DataFrame


def _annual_to_monthly(rate: float) -> float:
    return 1.0 - (1.0 - rate) ** (1.0 / 12.0)


def _irr(cash_flows: np.ndarray, invest: float) -> float:
    """Annualised IRR of monthly cash flows for an investment made at t=0."""
    if invest <= 0 or cash_flows.sum() <= 0:
        return float("nan")
    t = np.arange(1, len(cash_flows) + 1)
    f = lambda r: -invest + np.sum(cash_flows / (1.0 + r) ** t)
    try:
        r = brentq(f, -0.9999, 10.0, maxiter=200)
    except ValueError:
        return float("nan")
    out = (1.0 + r) ** 12 - 1.0
    return 0.0 if abs(out) < 1e-9 else out


def run_waterfall(pool: PoolAssumptions, structure: CapitalStructure) -> WaterfallResult:
    tr = list(structure)                       # bottom (equity) first
    names = [t.name for t in tr]
    n_tr = len(tr)
    bal = np.array([t.thickness * pool.balance for t in tr], dtype=float)
    orig = bal.copy()
    cpn_m = np.array([0.0] + [t.coupon / 12.0 for t in tr[1:]])  # equity gets no coupon
    r_m = pool.wac / 12.0
    mdr = _annual_to_monthly(pool.cdr)
    smm = _annual_to_monthly(pool.cpr)
    fee_m = pool.servicing_fee / 12.0
    T = int(pool.term_months)
    horizon = T + max(int(pool.recovery_lag), 0)
    perf = float(pool.balance)
    recov = defaultdict(float)

    rows = []
    int_paid = np.zeros((horizon, n_tr))
    prin_paid = np.zeros((horizon, n_tr))
    resid = np.zeros(horizon)
    shortfall = np.zeros(n_tr)
    writedown = np.zeros(n_tr)

    for t in range(1, horizon + 1):
        beg_perf = perf
        if t <= T and perf > 1e-9:
            d = perf * mdr
            p_after = perf - d
            interest = p_after * r_m
            if pool.amortization == "bullet":
                sched = p_after if t == T else 0.0
            else:
                n_rem = T - t + 1
                sched = p_after * r_m / ((1 + r_m) ** n_rem - 1) if r_m > 0 else p_after / n_rem
            sched = min(sched, p_after)
            prepay = (p_after - sched) * smm if t < T else 0.0
            perf = p_after - sched - prepay
            loss = d * pool.severity
            recov[t + pool.recovery_lag] += d * (1.0 - pool.severity)
        else:
            d = interest = sched = prepay = loss = 0.0
        rec = recov.pop(t, 0.0)

        # interest waterfall
        avail = interest
        fee = min(beg_perf * fee_m, avail)
        avail -= fee
        for i in range(n_tr - 1, 0, -1):            # senior notes first, equity excluded
            due = bal[i] * cpn_m[i]
            paid = min(due, avail)
            avail -= paid
            int_paid[t - 1, i] = paid
            shortfall[i] += due - paid
        cover = min(avail, loss)
        avail -= cover
        wd = loss - cover

        # write down from the bottom up
        for i in range(n_tr):
            if wd <= 1e-12:
                break
            hit = min(bal[i], wd)
            bal[i] -= hit
            writedown[i] += hit
            wd -= hit

        # principal waterfall, senior first
        prin = sched + prepay + rec + cover
        for i in range(n_tr - 1, -1, -1):
            pay = min(bal[i], prin)
            bal[i] -= pay
            prin -= pay
            prin_paid[t - 1, i] = pay
        resid[t - 1] = avail + prin                  # leftover cash goes to the residual holder

        rows.append(dict(month=t, performing=perf, defaults=d, losses=loss, interest=interest, scheduled=sched,
                         prepayments=prepay, recoveries=rec, fees=fee, loss_cover=cover, residual=resid[t - 1],
                         **{f"bal: {names[i]}": bal[i] for i in range(n_tr)}))

    periods = pd.DataFrame(rows)
    years = periods["month"].to_numpy() / 12.0
    summ = []
    for i, t in enumerate(tr):
        cf = int_paid[:, i] + prin_paid[:, i] + (resid if i == 0 else 0.0)
        lost = writedown[i] + bal[i]                 # anything left unpaid at the end is also a loss
        pr = prin_paid[:, i]
        wal = float(np.sum(years * pr) / pr.sum()) if pr.sum() > 0 else float("nan")
        summ.append({"Tranche": t.name, "Attach %": t.attach * 100, "Detach %": t.detach * 100,
                     "Coupon %": 0.0 if i == 0 else t.coupon * 100, "Original": orig[i],
                     "Interest received": float(int_paid[:, i].sum()), "Principal received": float(pr.sum()),
                     "Residual cash": float(resid.sum()) if i == 0 else 0.0,
                     "Principal loss %": 100 * lost / orig[i] if orig[i] > 0 else 0.0,
                     "Interest shortfall": float(shortfall[i]), "WAL (yrs)": wal,
                     "IRR %": 100 * _irr(cf, orig[i])})
    summary = pd.DataFrame(summ).iloc[::-1].reset_index(drop=True)   # senior first for display
    return WaterfallResult(periods, summary)


def tranche_losses(pool: PoolAssumptions, structure: CapitalStructure) -> dict:
    res = run_waterfall(pool, structure)
    return dict(zip(res.summary["Tranche"], res.summary["Principal loss %"]))


def breakeven_cdr(pool: PoolAssumptions, structure: CapitalStructure, tranche_name: str,
                  tolerance_pct: float = 0.01, hi: float = 1.0) -> float:
    """Highest annual CDR at which the tranche loses less than ``tolerance_pct`` % of principal."""
    def loss_at(cdr):
        p = PoolAssumptions(**{**pool.__dict__, "cdr": cdr})
        return tranche_losses(p, structure)[tranche_name]
    if loss_at(0.0) > tolerance_pct:
        return 0.0
    if loss_at(hi) <= tolerance_pct:
        return hi
    lo = 0.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if loss_at(mid) > tolerance_pct:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-5:
            break
    return lo


def stress_table(pool: PoolAssumptions, structure: CapitalStructure, cdrs=(0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20)) -> pd.DataFrame:
    """Principal loss % by tranche across annual default-rate scenarios."""
    data = {}
    for c in cdrs:
        p = PoolAssumptions(**{**pool.__dict__, "cdr": c})
        data[f"CDR {c:.0%}"] = tranche_losses(p, structure)
    return pd.DataFrame(data)
