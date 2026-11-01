"""American options by Monte Carlo: the Longstaff-Schwartz (2001) least-squares method.

Simulation runs forward in time, but early exercise is decided backward: at each exercise
date, the holder compares the payoff from exercising now with the expected value of
continuing. Longstaff and Schwartz estimate that continuation value by regressing the
discounted future cash flows of in-the-money paths on a few functions of today's state
(here powers of S/K, plus variance terms under Heston).

Two passes keep the estimate honest: the exercise rule is fitted on one set of paths and
then applied to a fresh, independent set. A rule that is slightly sub-optimal can only
lower the value, so the result is a low-biased estimate of the true price.
"""
from __future__ import annotations

import math

import numpy as np

from .models import Heston
from .pricer import observation_times
from .rng import NormalSource


def _basis(x: np.ndarray, v, degree: int) -> np.ndarray:
    cols = [x ** p for p in range(degree + 1)]
    if v is not None:
        cols += [v, v * x, v * v]
    return np.column_stack(cols)


def _threshold(s: np.ndarray, exercise: np.ndarray, phi: int) -> float:
    """The price that best separates 'exercise' from 'wait' decisions on one date.

    For a put, exercise should happen below the boundary and waiting above it; the threshold
    that agrees with the most fitted decisions is robust to wiggles in the regression.
    """
    order = np.argsort(s)
    s, d = s[order], exercise[order].astype(float)
    if phi == 1:                                   # a call is exercised above the boundary
        s, d = s[::-1], d[::-1]
    score = np.concatenate([[0.0], np.cumsum(d)]) + np.concatenate([np.cumsum((1 - d)[::-1])[::-1], [0.0]])
    i = int(np.argmax(score))
    return float(s[max(i - 1, 0)])


def lsm_price(kind: str, S0: float, K: float, T: float, r: float, q: float, model, n_ex: int = 50,
              n_paths: int = 20_000, degree: int = 3, seed: int = 7) -> dict:
    """Longstaff-Schwartz price of an American (Bermudan with ``n_ex`` dates) call or put.

    Returns the price and standard error, the European value on the same pricing paths,
    the share of paths exercised early and the estimated exercise boundary.
    """
    phi = 1 if kind == "call" else -1
    pay = lambda s: np.maximum(phi * (s - K), 0.0)
    times = observation_times(T, n_ex)
    dims = model.dims(times)
    heston = isinstance(model, Heston)

    # pass 1: fit the exercise rule backward on training paths
    st = model.simulate(S0, r, q, times, n_paths, NormalSource(dims, "pseudo", seed))
    S, V = st["S"], st.get("V")
    step_disc = math.exp(-r * T / n_ex)
    cash = pay(S[:, -1])
    coefs, data_range = {}, {}                # data_range: boundary estimate per exercise date
    for m in range(n_ex - 1, 0, -1):
        cash *= step_disc
        ex = pay(S[:, m])
        itm = np.flatnonzero(ex > 0)
        if len(itm) <= degree + 4:
            continue
        X = _basis(S[itm, m] / K, V[itm, m] if heston else None, degree)
        beta = np.linalg.lstsq(X, cash[itm], rcond=None)[0]
        stop = ex[itm] > X @ beta
        cash[itm[stop]] = ex[itm[stop]]
        coefs[m] = beta
        if not heston and stop.sum() >= 50:     # under Heston the boundary also depends on variance
            data_range[m] = _threshold(S[itm, m], stop, phi)

    # pass 2: apply the rule to fresh paths
    st2 = model.simulate(S0, r, q, times, n_paths, NormalSource(dims, "pseudo", seed + 104729))
    S2, V2 = st2["S"], st2.get("V")
    value = np.zeros(n_paths)
    alive = np.ones(n_paths, dtype=bool)
    exercised_at = np.full(n_paths, np.nan)
    for m in range(1, n_ex):
        if m not in coefs:
            continue
        ex = pay(S2[:, m])
        cand = np.flatnonzero(alive & (ex > 0))
        if len(cand) == 0:
            continue
        cont = _basis(S2[cand, m] / K, V2[cand, m] if heston else None, degree) @ coefs[m]
        stop = cand[ex[cand] > cont]
        value[stop] = ex[stop] * math.exp(-r * times[m - 1])
        exercised_at[stop] = times[m - 1]
        alive[stop] = False
    value[alive] = pay(S2[alive, -1]) * math.exp(-r * T)
    european = pay(S2[:, -1]) * math.exp(-r * T)
    price = float(value.mean())
    se = float(value.std(ddof=1) / math.sqrt(n_paths))
    intrinsic = float(pay(np.array([S0]))[0])

    bt = np.array([times[m - 1] for m in sorted(data_range)])
    bs = np.array([data_range[m] for m in sorted(data_range)])
    if len(bs) >= 5:                                   # drop dates where the regression went astray
        med = np.array([np.median(bs[max(0, i - 3):i + 4]) for i in range(len(bs))])
        keep = np.abs(bs - med) <= 0.08 * K
        bt, bs = bt[keep], bs[keep]
    return dict(price=max(price, intrinsic), stderr=se, european=float(european.mean()),
                european_se=float(european.std(ddof=1) / math.sqrt(n_paths)),
                early_share=float(np.mean(~np.isnan(exercised_at))), boundary_t=bt, boundary_S=bs,
                exercise_times=exercised_at)


def binomial_with_boundary(kind: str, S0: float, K: float, T: float, r: float, q: float, sigma: float,
                           steps: int = 1000) -> dict:
    """Cox-Ross-Rubinstein American price and the critical exercise price at each step."""
    phi = 1 if kind == "call" else -1
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    disc = math.exp(-r * dt)
    j = np.arange(steps + 1)
    val = np.maximum(phi * (S0 * u ** (steps - j) * d ** j - K), 0.0)
    bt, bs = [], []
    for i in range(steps - 1, -1, -1):
        j = np.arange(i + 1)
        st = S0 * u ** (i - j) * d ** j
        cont = disc * (p * val[:-1] + (1 - p) * val[1:])
        ex = np.maximum(phi * (st - K), 0.0)
        early = (ex > cont + 1e-12) & (ex > 0)
        val = np.maximum(cont, ex)
        if early.any() and i > 0:
            bt.append(i * dt)
            bs.append(float(st[early].max() if phi == -1 else st[early].min()))
    return dict(price=float(val[0]), boundary_t=np.array(bt[::-1]), boundary_S=np.array(bs[::-1]))
