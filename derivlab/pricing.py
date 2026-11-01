"""Black-Scholes-Merton pricing and Greeks, vectorised with NumPy.

Conventions
-----------
* ``T`` is time to expiry in years, ``r`` the continuously compounded risk-free
  rate and ``q`` the continuous dividend (or foreign-interest) yield.
* ``vol`` is the annualised implied volatility as a decimal (0.25 = 25%).
* Greeks are returned per unit of underlying: vega and rho per 1.00 change in
  vol / rate (divide by 100 for "per vol point" / "per 1%"), theta per year.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

_SQRT_2PI = np.sqrt(2.0 * np.pi)
_TINY = 1e-12


def norm_pdf(x):
    return np.exp(-0.5 * np.square(x)) / _SQRT_2PI


def norm_cdf(x):
    return ndtr(x)


def _prep(S, K, T, vol):
    S, K, T, vol = (np.asarray(a, dtype=float) for a in (S, K, T, vol))
    return np.broadcast_arrays(S, K, T, vol)


def bs_price(kind: str, S, K, T, r: float, q: float, vol):
    """Price of a European call or put. Works on scalars or arrays.

    Expired options (T <= 0) return intrinsic value; zero vol returns the
    discounted intrinsic value of the forward.
    """
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    S, K, T, vol = _prep(S, K, T, vol)
    live = (T > _TINY) & (vol > _TINY)
    Ts = np.where(live, T, 1.0)
    vs = np.where(live, vol, 1.0)
    sq = np.sqrt(Ts)
    d1 = (np.log(S / K) + (r - q + 0.5 * vs * vs) * Ts) / (vs * sq)
    d2 = d1 - vs * sq
    dq, dr = np.exp(-q * np.maximum(T, 0.0)), np.exp(-r * np.maximum(T, 0.0))
    if kind == "call":
        px = S * dq * ndtr(d1) - K * dr * ndtr(d2)
        dead = np.maximum(S * dq - K * dr, 0.0)
    else:
        px = K * dr * ndtr(-d2) - S * dq * ndtr(-d1)
        dead = np.maximum(K * dr - S * dq, 0.0)
    out = np.where(live, px, dead)
    return out if out.ndim else float(out)


def bs_greeks(kind: str, S, K, T, r: float, q: float, vol) -> dict:
    """Delta, gamma, vega (per 1.00), theta (per year) and rho (per 1.00)."""
    S, K, T, vol = _prep(S, K, T, vol)
    live = (T > _TINY) & (vol > _TINY)
    Ts = np.where(live, T, 1.0)
    vs = np.where(live, vol, 1.0)
    sq = np.sqrt(Ts)
    d1 = (np.log(S / K) + (r - q + 0.5 * vs * vs) * Ts) / (vs * sq)
    d2 = d1 - vs * sq
    dq, dr = np.exp(-q * Ts), np.exp(-r * Ts)
    pdf = norm_pdf(d1)
    gamma = dq * pdf / (S * vs * sq)
    vega = S * dq * pdf * sq
    common = -S * dq * pdf * vs / (2 * sq)
    if kind == "call":
        delta = dq * ndtr(d1)
        theta = common - r * K * dr * ndtr(d2) + q * S * dq * ndtr(d1)
        rho = K * Ts * dr * ndtr(d2)
        dead_delta = np.where(S > K, 1.0, 0.0)
    else:
        delta = -dq * ndtr(-d1)
        theta = common + r * K * dr * ndtr(-d2) - q * S * dq * ndtr(-d1)
        rho = -K * Ts * dr * ndtr(-d2)
        dead_delta = np.where(S < K, -1.0, 0.0)
    zero = np.zeros_like(S)
    res = {
        "delta": np.where(live, delta, dead_delta),
        "gamma": np.where(live, gamma, zero),
        "vega": np.where(live, vega, zero),
        "theta": np.where(live, theta, zero),
        "rho": np.where(live, rho, zero),
        "d1": np.where(live, d1, np.nan),
        "d2": np.where(live, d2, np.nan),
    }
    return {k: (v if v.ndim else float(v)) for k, v in res.items()}


def implied_vol(kind: str, price: float, S: float, K: float, T: float, r: float, q: float = 0.0) -> float:
    """Implied volatility by Brent's method. Returns NaN outside no-arbitrage bounds."""
    if T <= 0 or price <= 0:
        return float("nan")
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    lower = max(S * dq - K * dr, 0.0) if kind == "call" else max(K * dr - S * dq, 0.0)
    upper = S * dq if kind == "call" else K * dr
    if not (lower < price < upper):
        return float("nan")
    return brentq(lambda v: bs_price(kind, S, K, T, r, q, v) - price, 1e-6, 10.0, xtol=1e-12)


def binomial_price(kind: str, S: float, K: float, T: float, r: float, q: float, vol: float,
                   steps: int = 500, american: bool = True) -> float:
    """Cox-Ross-Rubinstein tree, European or American exercise."""
    dt = T / steps
    u = np.exp(vol * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    j = np.arange(steps + 1)
    st = S * u ** (steps - j) * d ** j
    val = np.maximum(st - K, 0.0) if kind == "call" else np.maximum(K - st, 0.0)
    for i in range(steps - 1, -1, -1):
        j = np.arange(i + 1)
        val = disc * (p * val[:-1] + (1 - p) * val[1:])
        if american:
            st = S * u ** (i - j) * d ** j
            ex = np.maximum(st - K, 0.0) if kind == "call" else np.maximum(K - st, 0.0)
            val = np.maximum(val, ex)
    return float(val[0])
