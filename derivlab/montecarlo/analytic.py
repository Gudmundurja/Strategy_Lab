"""Closed-form and semi-analytic prices used as Monte Carlo benchmarks and control variates.

Every function prices with continuous rates ``r`` and yields ``q`` (cost of carry ``b = r - q``).

* Black-Scholes digital (cash-or-nothing) options.
* Geometric-average Asian options with discrete, equally spaced fixings (exact under GBM).
* Single barrier options with continuous monitoring (Reiner-Rubinstein), plus the
  Broadie-Glasserman-Kou shift that adapts them to discrete monitoring.
* Floating-strike lookback options with continuous monitoring (Goldman-Sosin-Gatto).
* Merton jump-diffusion options (Merton's Poisson-weighted Black-Scholes series).
* Heston stochastic-volatility options through the characteristic function and
  Lewis's single-integral formula (which also prices Black-Scholes and Merton).
* Two-asset formulas: Margrabe exchange option, Kirk's spread approximation,
  Stulz options on the maximum or minimum of two assets.
* Basket options: exact geometric basket and Levy's lognormal approximation.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.special import ndtr
from scipy.stats import multivariate_normal

from ..pricing import bs_price

BGK_BETA = 0.5825971579390106   # -zeta(1/2) / sqrt(2 pi), Broadie-Glasserman-Kou (1997)


def _phi(kind: str) -> int:
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    return 1 if kind == "call" else -1


# ---------------------------------------------------------------- digitals
def digital_price(kind: str, S: float, K: float, T: float, r: float, q: float, sigma: float, cash: float = 1.0) -> float:
    """Cash-or-nothing digital: pays ``cash`` if the option finishes in the money."""
    phi = _phi(kind)
    if T <= 0:
        return cash * float(phi * (S - K) > 0)
    d2 = (math.log(S / K) + (r - q - 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return cash * math.exp(-r * T) * float(ndtr(phi * d2))


# ---------------------------------------------------------------- Asians
def geometric_asian_moments(S: float, T: float, r: float, q: float, sigma: float, n_obs: int):
    """Mean and standard deviation of ln G, G = geometric average of S at t_i = i T / n, i = 1..n."""
    n = int(n_obs)
    mean = math.log(S) + (r - q - 0.5 * sigma * sigma) * T * (n + 1) / (2 * n)
    var = sigma * sigma * T * (n + 1) * (2 * n + 1) / (6 * n * n)
    return mean, math.sqrt(var)


def geometric_asian_price(kind: str, S: float, K: float, T: float, r: float, q: float, sigma: float, n_obs: int) -> float:
    """Fixed-strike geometric Asian option with n equally spaced fixings (exact: ln G is normal)."""
    phi = _phi(kind)
    m, s = geometric_asian_moments(S, T, r, q, sigma, n_obs)
    d1 = (m - math.log(K) + s * s) / s
    d2 = d1 - s
    return math.exp(-r * T) * phi * (math.exp(m + 0.5 * s * s) * float(ndtr(phi * d1)) - K * float(ndtr(phi * d2)))


# ---------------------------------------------------------------- barriers
def _barrier_terms(phi, eta, S, K, H, T, r, q, sigma):
    b = r - q
    sq = sigma * math.sqrt(T)
    mu = (b - 0.5 * sigma * sigma) / (sigma * sigma)
    x1 = math.log(S / K) / sq + (1 + mu) * sq
    x2 = math.log(S / H) / sq + (1 + mu) * sq
    y1 = math.log(H * H / (S * K)) / sq + (1 + mu) * sq
    y2 = math.log(H / S) / sq + (1 + mu) * sq
    dq, dr = math.exp((b - r) * T), math.exp(-r * T)
    N = lambda x: float(ndtr(x))
    A = phi * S * dq * N(phi * x1) - phi * K * dr * N(phi * x1 - phi * sq)
    B = phi * S * dq * N(phi * x2) - phi * K * dr * N(phi * x2 - phi * sq)
    C = (phi * S * dq * (H / S) ** (2 * (mu + 1)) * N(eta * y1)
         - phi * K * dr * (H / S) ** (2 * mu) * N(eta * y1 - eta * sq))
    D = (phi * S * dq * (H / S) ** (2 * (mu + 1)) * N(eta * y2)
         - phi * K * dr * (H / S) ** (2 * mu) * N(eta * y2 - eta * sq))
    return A, B, C, D


def barrier_price(kind: str, S: float, K: float, H: float, T: float, r: float, q: float, sigma: float,
                  direction: str, knock: str, monitoring_dt: float | None = None) -> float:
    """Single-barrier option (no rebate) under Black-Scholes.

    Continuous monitoring uses the Reiner-Rubinstein formulas. With ``monitoring_dt``
    (time between barrier checks) the barrier is shifted away from spot by
    ``exp(0.5826 sigma sqrt(dt))``, the Broadie-Glasserman-Kou correction for discrete
    monitoring. Knock-outs are priced as vanilla minus knock-in.
    """
    phi = _phi(kind)
    if direction not in ("up", "down") or knock not in ("in", "out"):
        raise ValueError("direction must be 'up'/'down' and knock 'in'/'out'")
    vanilla = bs_price(kind, S, K, T, r, q, sigma)
    if monitoring_dt:
        H = H * math.exp((BGK_BETA if direction == "up" else -BGK_BETA) * sigma * math.sqrt(monitoring_dt))
    touched = (S <= H) if direction == "down" else (S >= H)
    if touched:                                    # already at or through the barrier
        knock_in = vanilla
    else:
        eta = 1 if direction == "down" else -1
        A, B, C, D = _barrier_terms(phi, eta, S, K, H, T, r, q, sigma)
        if phi == 1 and direction == "down":
            knock_in = C if K > H else A - B + D
        elif phi == 1:                             # up-and-in call
            knock_in = A if K > H else B - C + D
        elif direction == "down":                  # down-and-in put
            knock_in = B - C + D if K > H else A
        else:                                      # up-and-in put
            knock_in = A - B + D if K > H else C
    knock_in = min(max(knock_in, 0.0), vanilla)
    return knock_in if knock == "in" else vanilla - knock_in


# ---------------------------------------------------------------- lookbacks
def lookback_floating_price(kind: str, S: float, T: float, r: float, q: float, sigma: float, extreme: float | None = None) -> float:
    """Floating-strike lookback with continuous monitoring (Goldman, Sosin and Gatto 1979).

    Call pays S_T - min(S); put pays max(S) - S_T. ``extreme`` is the running minimum (call)
    or maximum (put) so far; at inception it equals S. Discrete monitoring is worth less.
    """
    phi = _phi(kind)
    b = r - q
    if abs(b) < 1e-7:
        b = 1e-7 if b >= 0 else -1e-7                 # the formula has a removable singularity at b = 0
    m = S if extreme is None else extreme
    sq = sigma * math.sqrt(T)
    s2b = sigma * sigma / (2 * b)
    a1 = (math.log(S / m) + (b + 0.5 * sigma * sigma) * T) / sq
    a2 = a1 - sq
    dq, dr = math.exp((b - r) * T), math.exp(-r * T)
    N = lambda x: float(ndtr(x))
    if phi == 1:
        return (S * dq * N(a1) - m * dr * N(a2)
                + S * dr * s2b * ((S / m) ** (-2 * b / sigma ** 2) * N(-a1 + 2 * b * math.sqrt(T) / sigma) - math.exp(b * T) * N(-a1)))
    return (m * dr * N(-a2) - S * dq * N(-a1)
            + S * dr * s2b * (-(S / m) ** (-2 * b / sigma ** 2) * N(a1 - 2 * b * math.sqrt(T) / sigma) + math.exp(b * T) * N(a1)))


# ---------------------------------------------------------------- characteristic functions
def bs_cf(u, T: float, sigma: float):
    """Characteristic function of X = ln(S_T / F_T) under Black-Scholes."""
    return np.exp(-0.5 * sigma * sigma * T * (u * u + 1j * u))


def merton_cf(u, T: float, sigma: float, lam: float, mu_j: float, delta: float):
    """Characteristic function of X = ln(S_T / F_T) under Merton's jump-diffusion."""
    k = math.exp(mu_j + 0.5 * delta * delta) - 1
    jump = np.exp(1j * u * mu_j - 0.5 * delta * delta * u * u) - 1
    return np.exp(T * (-1j * u * (lam * k + 0.5 * sigma * sigma) - 0.5 * sigma * sigma * u * u + lam * jump))


def heston_cf(u, T: float, v0: float, kappa: float, theta: float, xi: float, rho: float):
    """Characteristic function of X = ln(S_T / F_T) under Heston, in the numerically stable
    'little Heston trap' form of Albrecher, Mayer, Schoutens and Tistaert (2007)."""
    u = np.asarray(u, dtype=complex)
    a = kappa - rho * xi * 1j * u
    d = np.sqrt(a * a + xi * xi * (1j * u + u * u))
    g = (a - d) / (a + d)
    e = np.exp(-d * T)
    C = kappa * theta / (xi * xi) * ((a - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
    D = (a - d) / (xi * xi) * (1 - e) / (1 - g * e)
    return np.exp(C + D * v0)


def cf_price(kind: str, S: float, K: float, T: float, r: float, q: float, cf) -> float:
    """European option price from the characteristic function of ln(S_T / F_T) (Lewis 2001):

        C = S e^{-qT} - sqrt(S K) e^{-(r+q)T/2} / pi * integral_0^inf Re[e^{-iuk} cf(u - i/2)] / (u^2 + 1/4) du

    with k = ln(K / F). Puts follow from put-call parity.
    """
    F = S * math.exp((r - q) * T)
    k = math.log(K / F)
    integrand = lambda u: (np.exp(-1j * u * k) * cf(u - 0.5j)).real / (u * u + 0.25)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)     # tolerance is far below what any chart needs
        integral = quad(integrand, 0.0, np.inf, limit=400, epsabs=1e-11, epsrel=1e-9)[0]
    call = S * math.exp(-q * T) - math.sqrt(S * K) * math.exp(-0.5 * (r + q) * T) * integral / math.pi
    call = max(call, max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0))
    if kind == "call":
        return call
    return max(call - S * math.exp(-q * T) + K * math.exp(-r * T), 0.0)


def heston_price(kind: str, S: float, K: float, T: float, r: float, q: float,
                 v0: float, kappa: float, theta: float, xi: float, rho: float) -> float:
    return cf_price(kind, S, K, T, r, q, lambda u: heston_cf(u, T, v0, kappa, theta, xi, rho))


def merton_price(kind: str, S: float, K: float, T: float, r: float, q: float,
                 sigma: float, lam: float, mu_j: float, delta: float, tol: float = 1e-14) -> float:
    """Merton (1976): a Poisson-weighted sum of Black-Scholes prices, one per number of jumps n,

        sigma_n^2 = sigma^2 + n delta^2 / T,   r_n = r - lam k + n ln(1 + k) / T,   lam' = lam (1 + k).
    """
    k = math.exp(mu_j + 0.5 * delta * delta) - 1
    lam_p = lam * (1 + k)
    total, weight_sum, n = 0.0, 0.0, 0
    while True:
        w = math.exp(-lam_p * T + n * math.log(lam_p * T) - math.lgamma(n + 1)) if lam_p * T > 0 else float(n == 0)
        sig_n = math.sqrt(sigma * sigma + n * delta * delta / T)
        r_n = r - lam * k + n * math.log1p(k) / T
        total += w * bs_price(kind, S, K, T, r_n, q, sig_n)
        weight_sum += w
        n += 1
        if 1 - weight_sum < tol or n > 400:
            return total


# ---------------------------------------------------------------- two assets
def margrabe_price(S1: float, S2: float, T: float, q1: float, q2: float, s1: float, s2: float, rho: float) -> float:
    """Option to exchange asset 2 for asset 1: pays max(S1 - S2, 0) (Margrabe 1978)."""
    s = math.sqrt(s1 * s1 + s2 * s2 - 2 * rho * s1 * s2)
    d1 = (math.log(S1 / S2) + (q2 - q1 + 0.5 * s * s) * T) / (s * math.sqrt(T))
    return S1 * math.exp(-q1 * T) * float(ndtr(d1)) - S2 * math.exp(-q2 * T) * float(ndtr(d1 - s * math.sqrt(T)))


def kirk_spread_price(kind: str, S1: float, S2: float, K: float, T: float, r: float, q1: float, q2: float,
                      s1: float, s2: float, rho: float) -> float:
    """Kirk's (1995) approximation for a spread option paying max(phi (S1 - S2 - K), 0)."""
    phi = _phi(kind)
    F1, F2 = S1 * math.exp((r - q1) * T), S2 * math.exp((r - q2) * T)
    w = F2 / (F2 + K)
    sk = math.sqrt(s1 * s1 - 2 * rho * s1 * s2 * w + (s2 * w) ** 2)
    d1 = (math.log(F1 / (F2 + K)) + 0.5 * sk * sk * T) / (sk * math.sqrt(T))
    d2 = d1 - sk * math.sqrt(T)
    call = math.exp(-r * T) * (F1 * float(ndtr(d1)) - (F2 + K) * float(ndtr(d2)))
    return call if phi == 1 else call - math.exp(-r * T) * (F1 - F2 - K)


def bivariate_normal_cdf(a: float, b: float, rho: float) -> float:
    if abs(rho) > 1 - 1e-12:
        return float(ndtr(min(a, b))) if rho > 0 else max(float(ndtr(a)) - float(ndtr(-b)), 0.0)
    return float(multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]]).cdf([a, b]))


def two_asset_extreme_price(kind: str, extreme: str, S1: float, S2: float, K: float, T: float, r: float,
                            q1: float, q2: float, s1: float, s2: float, rho: float) -> float:
    """Call or put on the maximum ('max', best-of) or minimum ('min', worst-of) of two assets
    (Stulz 1982; Johnson 1987). Puts come from put-call parity on the extreme."""
    phi = _phi(kind)
    if extreme not in ("max", "min"):
        raise ValueError("extreme must be 'max' or 'min'")
    sT = math.sqrt(T)
    s = math.sqrt(s1 * s1 + s2 * s2 - 2 * rho * s1 * s2)
    d = (math.log(S1 / S2) + (q2 - q1 + 0.5 * s * s) * T) / (s * sT)
    y1 = (math.log(S1 / K) + (r - q1 + 0.5 * s1 * s1) * T) / (s1 * sT)
    y2 = (math.log(S2 / K) + (r - q2 + 0.5 * s2 * s2) * T) / (s2 * sT)
    r1, r2 = (s1 - rho * s2) / s, (s2 - rho * s1) / s
    M = bivariate_normal_cdf
    e1, e2, er = S1 * math.exp(-q1 * T), S2 * math.exp(-q2 * T), K * math.exp(-r * T)
    exch = margrabe_price(S1, S2, T, q1, q2, s1, s2, rho)
    if extreme == "max":
        call = (e1 * M(y1, d, r1) + e2 * M(y2, -d + s * sT, r2)
                - er * (1 - M(-y1 + s1 * sT, -y2 + s2 * sT, rho)))
        pv_extreme = e2 + exch                         # max(S1, S2) = S2 + max(S1 - S2, 0)
    else:
        call = (e1 * M(y1, -d, -r1) + e2 * M(y2, d - s * sT, -r2)
                - er * M(y1 - s1 * sT, y2 - s2 * sT, rho))
        pv_extreme = e1 - exch                         # min(S1, S2) = S1 - max(S1 - S2, 0)
    call = max(call, 0.0)
    return call if phi == 1 else max(call - pv_extreme + er, 0.0)


# ---------------------------------------------------------------- baskets
def _basket_inputs(S, q, sigma, corr, weights):
    S, q, sigma, w = (np.asarray(a, dtype=float) for a in (S, q, sigma, weights))
    cov = np.asarray(corr, dtype=float) * np.outer(sigma, sigma)
    return S, q, sigma, w, cov


def geometric_basket_price(kind: str, S, K: float, T: float, r: float, q, sigma, corr, weights) -> float:
    """Option on W * prod(S_i^(w_i / W)), W = sum of weights: lognormal, so priced exactly.
    The natural control variate for an arithmetic basket."""
    phi = _phi(kind)
    S, q, sigma, w, cov = _basket_inputs(S, q, sigma, corr, weights)
    W = w.sum()
    a = w / W
    m = math.log(W) + float(a @ (np.log(S) + (r - q - 0.5 * sigma * sigma) * T))
    s = math.sqrt(float(a @ cov @ a) * T)
    d1 = (m - math.log(K) + s * s) / s
    return math.exp(-r * T) * phi * (math.exp(m + 0.5 * s * s) * float(ndtr(phi * d1)) - K * float(ndtr(phi * (d1 - s))))


def levy_basket_price(kind: str, S, K: float, T: float, r: float, q, sigma, corr, weights) -> float:
    """Levy (1992): match the first two moments of the arithmetic basket to a lognormal."""
    phi = _phi(kind)
    S, q, sigma, w, cov = _basket_inputs(S, q, sigma, corr, weights)
    F = S * np.exp((r - q) * T)
    m1 = float(w @ F)
    m2 = float((w * F) @ np.exp(cov * T) @ (w * F))
    s = math.sqrt(math.log(m2 / (m1 * m1)))
    d1 = (math.log(m1 / K) + 0.5 * s * s) / s
    return math.exp(-r * T) * phi * (m1 * float(ndtr(phi * d1)) - K * float(ndtr(phi * (d1 - s))))
