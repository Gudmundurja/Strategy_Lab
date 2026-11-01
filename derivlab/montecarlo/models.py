"""Price dynamics for Monte Carlo simulation.

Every model simulates paths observed at given times and exposes how many normal draws
it needs per path (``dims``) so the random source can be set up in advance.

* :class:`GBM` - geometric Brownian motion (the Black-Scholes world). Simulated exactly:
  ln S jumps by (mu - q - sigma^2/2) dt + sigma sqrt(dt) Z between observation dates.
* :class:`Heston` - stochastic variance v following a square-root process
  dv = kappa (theta - v) dt + xi sqrt(v) dW_v, correlated rho with the price shocks.
  Simulated with the full-truncation Euler scheme on ln S, on a sub-grid of at most
  ``max_dt`` years per step (Lord, Koekkoek and van Dijk 2010).
* :class:`Merton` - GBM plus Poisson jumps with normally distributed log sizes.
  Simulated exactly: the sum of N lognormal jumps over a step is N mu_J + sqrt(N) delta Z.

``mu`` is the drift rate: the risk-free rate r for pricing, an expected return for risk.

With Sobol' points the Brownian paths are built with a *Brownian bridge*: the first
(best-distributed) coordinate sets the end point of the path, the next ones the midpoints,
and so on. The paths have exactly the same distribution, but the coordinates that matter
most get the best quasi-random dimensions, which is what makes quasi-Monte Carlo pay off
for path-dependent options with many dates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from .rng import NormalSource


@lru_cache(maxsize=64)
def _bridge_order(k: int):
    """Construction order (index, left neighbour, right neighbour) for a Brownian bridge on k dates."""
    order, queue = [(k - 1, -1, None)], [(-1, k - 1)]
    while queue:
        left, right = queue.pop(0)
        if right - left <= 1:
            continue
        mid = (left + right) // 2
        order.append((mid, left, right))
        queue += [(left, mid), (mid, right)]
    return tuple(order)


def bridge_normals(Z: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Turn standard normals into the standardised increments of a Brownian bridge path.

    Column j of ``Z`` drives the j-th point in bridge order (end point first). The output
    has the same joint distribution as ``Z`` (independent standard normals), only assigned
    differently to the time steps.
    """
    t = np.cumsum(dt)
    W = np.empty_like(Z)
    for col, (i, left, right) in enumerate(_bridge_order(len(t))):
        if right is None:
            W[:, i] = math.sqrt(t[i]) * Z[:, col]
            continue
        tl = t[left] if left >= 0 else 0.0
        wl = W[:, left] if left >= 0 else 0.0
        frac = (t[i] - tl) / (t[right] - tl)
        sd = math.sqrt((t[i] - tl) * (t[right] - t[i]) / (t[right] - tl))
        W[:, i] = wl + frac * (W[:, right] - wl) + sd * Z[:, col]
    inc = np.diff(np.concatenate([np.zeros((len(Z), 1)), W], axis=1), axis=1)
    return inc / np.sqrt(dt)


def _steps(times) -> np.ndarray:
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or len(t) == 0 or np.any(np.diff(np.concatenate([[0.0], t])) <= 0):
        raise ValueError("observation times must be positive and strictly increasing")
    return np.diff(np.concatenate([[0.0], t]))


@dataclass(frozen=True)
class GBM:
    sigma: float = 0.25
    name = "Black-Scholes (GBM)"

    def dims(self, times) -> int:
        return len(_steps(times))

    def total_vol(self) -> float:
        return self.sigma

    def bumped_vol(self, dv: float) -> "GBM":
        return replace(self, sigma=self.sigma + dv)

    def simulate(self, S0: float, mu: float, q: float, times, n: int, src: NormalSource) -> dict:
        dt = _steps(times)
        Z = src.normals(n)
        if src.sampling == "sobol" and len(dt) > 1:
            Z = bridge_normals(Z, dt)
        inc = (mu - q - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * Z
        S = np.empty((n, len(dt) + 1))
        S[:, 0] = S0
        S[:, 1:] = S0 * np.exp(np.cumsum(inc, axis=1))
        return {"S": S}


@dataclass(frozen=True)
class Merton:
    sigma: float = 0.20      # diffusion volatility
    lam: float = 0.5         # expected jumps per year
    mu_j: float = -0.10      # mean log jump size
    delta: float = 0.15      # standard deviation of the log jump size
    name = "Merton jump-diffusion"

    @property
    def kbar(self) -> float:
        """Expected relative jump size E[e^J - 1]; the drift is reduced by lam * kbar to stay risk-neutral."""
        return math.exp(self.mu_j + 0.5 * self.delta ** 2) - 1

    def dims(self, times) -> int:
        return 2 * len(_steps(times))

    def total_vol(self) -> float:
        return math.sqrt(self.sigma ** 2 + self.lam * (self.mu_j ** 2 + self.delta ** 2))

    def bumped_vol(self, dv: float) -> "Merton":
        return replace(self, sigma=self.sigma + dv)

    def simulate(self, S0: float, mu: float, q: float, times, n: int, src: NormalSource) -> dict:
        dt = _steps(times)
        k = len(dt)
        Z = src.normals(n)
        Zd = bridge_normals(Z[:, :k], dt) if src.sampling == "sobol" and k > 1 else Z[:, :k]
        N = src.poisson(self.lam * dt, n, k)
        jumps = N * self.mu_j + np.sqrt(N) * self.delta * Z[:, k:]
        inc = (mu - q - self.lam * self.kbar - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * Zd + jumps
        S = np.empty((n, k + 1))
        S[:, 0] = S0
        S[:, 1:] = S0 * np.exp(np.cumsum(inc, axis=1))
        return {"S": S, "jumps": N.sum(axis=1)}


@dataclass(frozen=True)
class Heston:
    v0: float = 0.0625       # initial variance (vol^2)
    kappa: float = 2.0       # speed of mean reversion
    theta: float = 0.0625    # long-run variance
    xi: float = 0.45         # volatility of variance
    rho: float = -0.7        # correlation between price and variance shocks
    max_dt: float = 1 / 200  # largest Euler step in years
    name = "Heston stochastic vol"

    @property
    def feller(self) -> bool:
        """2 kappa theta >= xi^2: the variance never touches zero in continuous time."""
        return 2 * self.kappa * self.theta >= self.xi ** 2

    def substeps(self, times) -> np.ndarray:
        return np.maximum(1, np.ceil(_steps(times) / self.max_dt - 1e-9)).astype(int)

    def dims(self, times) -> int:
        return 2 * int(self.substeps(times).sum())

    def total_vol(self) -> float:
        return math.sqrt(self.v0)

    def bumped_vol(self, dv: float) -> "Heston":
        """Shift the whole vol level: initial and long-run vol both move by dv."""
        return replace(self, v0=(math.sqrt(self.v0) + dv) ** 2, theta=(math.sqrt(self.theta) + dv) ** 2)

    def simulate(self, S0: float, mu: float, q: float, times, n: int, src: NormalSource) -> dict:
        dt = _steps(times)
        m = self.substeps(times)
        Z = src.normals(n)
        Z1, Z2 = Z[:, 0::2], Z[:, 1::2]                      # price and variance shocks, interleaved
        if src.sampling == "sobol" and Z1.shape[1] > 1:
            fine = np.repeat(dt / m, m)
            Z1, Z2 = bridge_normals(Z1, fine), bridge_normals(Z2, fine)
        x = np.full(n, math.log(S0))
        v = np.full(n, float(self.v0))
        S = np.empty((n, len(dt) + 1))
        V = np.empty((n, len(dt) + 1))
        S[:, 0], V[:, 0] = S0, self.v0
        c = math.sqrt(1 - self.rho ** 2)
        col = 0
        for i, (step, sub) in enumerate(zip(dt, m)):
            h = step / sub
            sh = math.sqrt(h)
            for _ in range(sub):
                z1, z2 = Z1[:, col], Z2[:, col]
                col += 1
                vp = np.maximum(v, 0.0)
                sv = np.sqrt(vp) * sh
                x += (mu - q - 0.5 * vp) * h + sv * z1
                v += self.kappa * (self.theta - vp) * h + self.xi * sv * (self.rho * z1 + c * z2)
            S[:, i + 1] = np.exp(x)
            V[:, i + 1] = np.maximum(v, 0.0)
        return {"S": S, "V": V}


MODELS = (GBM, Heston, Merton)
