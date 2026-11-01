"""Monte Carlo for multi-leg positions built in the options lab: repricing under a model,
the model's implied-volatility smile, and P&L risk (value-at-risk and expected shortfall).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..position import DAYS_PER_YEAR, Position
from ..pricing import implied_vol
from .models import GBM, Heston
from .payoffs import European
from .pricer import model_vanilla_price, price_option
from .rng import NormalSource


def _iv(kind, price, S, K, T, r, q):
    try:
        return implied_vol(kind, price, S, K, T, r, q)
    except ValueError:
        return float("nan")


def reprice_position(pos: Position, model, n_paths: int = 20_000, sampling: str = "pseudo", seed: int = 7) -> pd.DataFrame:
    """Price every option leg by simulation under ``model`` and compare with Black-Scholes on the smile.

    With a GBM model each leg is simulated at its own smile vol, so the simulation must
    reproduce the options-lab prices (a check of the method). Heston and Merton price every
    leg in one consistent model; the implied vol of their prices is the smile that model
    produces.
    """
    m = pos.market
    rows = []
    for leg in pos.legs:
        if not leg.is_option:
            continue
        T = leg.days / DAYS_PER_YEAR
        smile_vol = float(pos.leg_vol(leg, m.spot))
        leg_model = GBM(smile_vol) if isinstance(model, GBM) else model
        res = price_option(European(leg.kind, leg.strike), leg_model, m.spot, m.rate, m.div_yield, T, 1, n_paths,
                           sampling, "auto", seed)
        formula = model_vanilla_price(leg_model, leg.kind, m.spot, leg.strike, T, m.rate, m.div_yield)
        rows.append({"Leg": f"{leg.side} {abs(leg.qty):g} {leg.kind} {leg.strike:g}", "Days": leg.days, "Qty": leg.qty,
                     "Smile vol %": smile_vol * 100, "Black-Scholes on smile": pos.model_price(leg),
                     "Model formula": formula, "Monte Carlo": res.price, "MC ±": 1.96 * res.stderr,
                     "Model implied vol %": _iv(leg.kind, formula, m.spot, leg.strike, T, m.rate, m.div_yield) * 100})
    return pd.DataFrame(rows)


def position_values(table: pd.DataFrame, multiplier: float) -> dict:
    """Value of the option legs (money, including multiplier) under each pricing column."""
    out = {}
    for col in ("Black-Scholes on smile", "Model formula", "Monte Carlo"):
        out[col] = float((table["Qty"] * table[col]).sum() * multiplier) if len(table) else 0.0
    return out


def model_smile(model, S: float, r: float, q: float, days_list, n_strikes: int = 25, width: float = 2.5) -> pd.DataFrame:
    """Implied vols the model produces across strikes (from its pricing formula)."""
    rows = []
    for days in days_list:
        T = days / DAYS_PER_YEAR
        F = S * math.exp((r - q) * T)
        level = model.total_vol()
        for x in np.linspace(-width, width, n_strikes):
            K = F * math.exp(x * level * math.sqrt(T))
            kind = "put" if K < F else "call"
            price = model_vanilla_price(model, kind, S, K, T, r, q)
            rows.append({"days": days, "strike": K, "iv": _iv(kind, price, S, K, T, r, q)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- P&L risk
def pnl_distribution(pos: Position, model, horizon_days: float, n_paths: int = 20_000, mu: float = 0.06,
                     seed: int = 7, link_vol: bool = True) -> dict:
    """Simulated change in the position's mark-to-model value over ``horizon_days``.

    Spot is simulated under the real-world drift ``mu``. The position is then fully
    revalued with Black-Scholes on the smile at the horizon (options that expired pay
    intrinsic value). Under Heston, with ``link_vol``, implied vols move in parallel by the
    change in simulated volatility, sqrt(v_h) - sqrt(v_0), so spot and vol risk interact.
    """
    m = pos.market
    h = horizon_days / DAYS_PER_YEAR
    times = np.array([h])
    st = model.simulate(m.spot, mu, m.div_yield, times, n_paths, NormalSource(model.dims(times), "pseudo", seed))
    S_h = st["S"][:, -1]
    base = float(pos.pnl(m.spot, 0.0))
    dvol = np.zeros(n_paths)
    if isinstance(model, Heston) and link_vol:
        dvol = np.sqrt(st["V"][:, -1]) - math.sqrt(model.v0)
        pnl = np.empty(n_paths)
        edges = np.unique(np.quantile(dvol, np.linspace(0, 1, 41)))
        bins = np.clip(np.searchsorted(edges, dvol, side="right") - 1, 0, max(len(edges) - 2, 0))
        for b in np.unique(bins):
            idx = bins == b
            pnl[idx] = pos.pnl(S_h[idx], horizon_days, float(np.median(dvol[idx])))
    else:
        pnl = pos.pnl(S_h, horizon_days, 0.0)
    return {"pnl": np.asarray(pnl, dtype=float) - base, "S": S_h, "dvol": dvol,
            "jumps": st.get("jumps")}


def var_es(pnl, alpha: float):
    """Value-at-risk and expected shortfall at confidence ``alpha`` as positive loss amounts."""
    losses = -np.asarray(pnl, dtype=float)
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    return var, float(tail.mean()) if len(tail) else var


def delta_normal_var(pos: Position, sigma: float, horizon_days: float, alpha: float) -> float:
    """Parametric VaR treating the position as its delta: z_alpha |cash delta| sigma sqrt(h)."""
    from scipy.special import ndtri
    z = float(ndtri(alpha))
    return z * abs(pos.greeks()["cash_delta"]) * sigma * math.sqrt(horizon_days / DAYS_PER_YEAR)


def risk_table(pnl, levels=(0.95, 0.975, 0.99)) -> pd.DataFrame:
    rows = []
    for a in levels:
        v, e = var_es(pnl, a)
        rows.append({"Confidence": f"{a:.1%}", "VaR": v, "Expected shortfall": e})
    return pd.DataFrame(rows)
