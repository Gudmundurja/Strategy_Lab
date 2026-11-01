"""Monte Carlo Greeks by three methods.

* **Bump and revalue** with common random numbers: reprice with spot (or vol) moved up
  and down using the *same* random draws, and take finite differences path by path.
  Works for any model and payoff, but needs extra simulations and is noisy for gamma of
  payoffs that jump (digitals, barriers).
* **Pathwise**: differentiate the payoff along each path, using dS_t/dS_0 = S_t / S_0 and
  dS_t/dsigma = S_t (ln(S_t/S_0) - (r - q + sigma^2/2) t) / sigma under GBM. Unbiased and
  low-variance, but fails for payoffs that jump (their derivative is zero almost everywhere).
* **Likelihood ratio**: differentiate the probability density instead of the payoff,
  multiplying the payoff by a score. Under GBM the score for S_0 depends only on the first
  shock Z_1: delta = E[payoff * Z_1 / (S_0 sigma sqrt(dt_1))]. Works for any payoff, but its
  variance grows as the first time step shrinks.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .models import GBM
from .payoffs import Asian, Barrier, Digital, European, Lookback
from .pricer import benchmark, observation_times
from .rng import NormalSource

VOL_BUMP = 0.01


def _mean_se(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))


def _discounted(product, model, S0, r, q, T, times, n, seed):
    st = model.simulate(S0, r, q, times, n, NormalSource(model.dims(times), "pseudo", seed))
    return math.exp(-r * T) * np.asarray(product.payoff(st["S"]), dtype=float), st["S"]


def bump_and_revalue(product, model, S0, r, q, T, times, n, seed, rel_bump=0.01):
    h = rel_bump * S0
    up, _ = _discounted(product, model, S0 + h, r, q, T, times, n, seed)
    mid, _ = _discounted(product, model, S0, r, q, T, times, n, seed)
    dn, _ = _discounted(product, model, S0 - h, r, q, T, times, n, seed)
    vu, _ = _discounted(product, model.bumped_vol(VOL_BUMP), S0, r, q, T, times, n, seed)
    vd, _ = _discounted(product, model.bumped_vol(-VOL_BUMP), S0, r, q, T, times, n, seed)
    return {"Delta": _mean_se((up - dn) / (2 * h)), "Gamma": _mean_se((up - 2 * mid + dn) / (h * h)),
            "Vega": _mean_se((vu - vd) / (2 * VOL_BUMP) / 100.0)}


def _pathwise_ok(product) -> bool:
    return isinstance(product, (European, Asian, Lookback))


def pathwise(product, model: GBM, S0, r, q, T, times, n, seed):
    """Pathwise delta and vega under GBM (None where the method does not apply)."""
    if not (isinstance(model, GBM) and _pathwise_ok(product)):
        return None
    disc = math.exp(-r * T)
    _, S = _discounted(product, model, S0, r, q, T, times, n, seed)
    sig = model.sigma
    t = np.concatenate([[0.0], np.asarray(times, dtype=float)])
    dS_dsig = S * (np.log(S / S0) - (r - q + 0.5 * sig * sig) * t) / sig      # zero in column 0
    phi = 1 if product.kind == "call" else -1
    if isinstance(product, European):
        itm = phi * (S[:, -1] - product.strike) > 0
        delta = disc * phi * itm * S[:, -1] / S0
        vega = disc * phi * itm * dS_dsig[:, -1]
    elif isinstance(product, Asian):
        fix = S[:, 1:]
        if product.average == "arithmetic":
            A = fix.mean(axis=1)
            dA = dS_dsig[:, 1:].mean(axis=1)
        else:
            A = np.exp(np.log(fix).mean(axis=1))
            dA = A * (dS_dsig[:, 1:] / fix).mean(axis=1)
        itm = phi * (A - product.strike) > 0
        delta = disc * phi * itm * A / S0
        vega = disc * phi * itm * dA
    else:                                                    # lookbacks
        rows = np.arange(len(S))
        lo, hi = S.argmin(axis=1), S.argmax(axis=1)
        if product.strike is None:
            pay = product.payoff(S)
            delta = disc * pay / S0                          # the payoff scales with S0
            vega = disc * ((dS_dsig[:, -1] - dS_dsig[rows, lo]) if phi == 1 else (dS_dsig[rows, hi] - dS_dsig[:, -1]))
        elif phi == 1:
            itm = S[rows, hi] > product.strike
            delta, vega = disc * itm * S[rows, hi] / S0, disc * itm * dS_dsig[rows, hi]
        else:
            itm = S[rows, lo] < product.strike
            delta, vega = -disc * itm * S[rows, lo] / S0, -disc * itm * dS_dsig[rows, lo]
    return {"Delta": _mean_se(delta), "Vega": _mean_se(vega / 100.0)}


def likelihood_ratio(product, model: GBM, S0, r, q, T, times, n, seed):
    """Likelihood-ratio delta, gamma and vega under GBM, for any payoff."""
    if not isinstance(model, GBM):
        return None
    Y, S = _discounted(product, model, S0, r, q, T, times, n, seed)
    sig = model.sigma
    dt = np.diff(np.concatenate([[0.0], np.asarray(times, dtype=float)]))
    Z = (np.log(S[:, 1:] / S[:, :-1]) - (r - q - 0.5 * sig * sig) * dt) / (sig * np.sqrt(dt))
    z1, dt1 = Z[:, 0], dt[0]
    delta = Y * z1 / (S0 * sig * math.sqrt(dt1))
    gamma = Y * (z1 * z1 - 1 - z1 * sig * math.sqrt(dt1)) / (S0 * S0 * sig * sig * dt1)
    vega = Y * ((Z * Z - 1) / sig - Z * np.sqrt(dt)).sum(axis=1)
    return {"Delta": _mean_se(delta), "Gamma": _mean_se(gamma), "Vega": _mean_se(vega / 100.0)}


def formula_greeks(product, model, S0, r, q, T, n_obs):
    """Greeks of the closed-form benchmark by central differences, when a benchmark exists."""
    f = lambda s, m: benchmark(product, m, s, r, q, T, n_obs)
    if f(S0, model) is None:
        return None
    h = 1e-3 * S0
    up, mid, dn = f(S0 + h, model)[1], f(S0, model)[1], f(S0 - h, model)[1]
    vu, vd = f(S0, model.bumped_vol(1e-4))[1], f(S0, model.bumped_vol(-1e-4))[1]
    return {"Delta": (up - dn) / (2 * h), "Gamma": (up - 2 * mid + dn) / (h * h), "Vega": (vu - vd) / 2e-4 / 100.0}


def greeks_table(product, model, S0, r, q, T, n_obs, n_paths=20_000, seed=7) -> pd.DataFrame:
    """Delta, gamma and vega (per vol point) by every applicable method, with standard errors."""
    times = observation_times(T, n_obs if product.path_dependent else 1)
    bump = bump_and_revalue(product, model, S0, r, q, T, times, n_paths, seed)
    pw = pathwise(product, model, S0, r, q, T, times, n_paths, seed)
    lr = likelihood_ratio(product, model, S0, r, q, T, times, n_paths, seed)
    exact = formula_greeks(product, model, S0, r, q, T, len(times))
    rows = []
    for g in ("Delta", "Gamma", "Vega"):
        row = {"Greek": g if g != "Vega" else "Vega (per vol pt)", "Formula": exact[g] if exact else np.nan}
        for name, res in (("Bump & revalue", bump), ("Pathwise", pw), ("Likelihood ratio", lr)):
            val = res.get(g) if res else None
            row[name], row[f"{name} ±"] = (val if val else (np.nan, np.nan))
        rows.append(row)
    return pd.DataFrame(rows)


def pathwise_note(product, model) -> str:
    if not isinstance(model, GBM):
        return "Pathwise and likelihood-ratio formulas here are derived for GBM; other models use bump and revalue."
    if isinstance(product, (Digital, Barrier)):
        return "Pathwise does not apply: the payoff jumps, and its derivative along a path is zero almost everywhere."
    return "Pathwise gamma does not exist for kinked payoffs; use bump and revalue or likelihood ratio for gamma."
