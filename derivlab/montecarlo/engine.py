"""The Monte Carlo estimator: sampling, control variates, standard errors and convergence.

``run`` is model- and product-agnostic: it takes a path simulator and a payoff, draws
paths in memory-friendly chunks and returns an :class:`MCResult`.

Error estimates
---------------
* Pseudo-random: each path is an independent sample, SE = s / sqrt(N).
* Antithetic: a path and its mirror image are not independent, so the pair average is
  the sample unit, SE = s_pairs / sqrt(N / 2).
* Sobol': quasi-random points are deliberately not independent, so the error comes from
  ``replications`` independently scrambled sequences: SE = s_replications / sqrt(R).

Control variates
----------------
With a quantity X whose expectation is known exactly, the estimator
``mean(Y) - b (mean(X) - E[X])`` with ``b = Cov(Y, X) / Var(X)`` has variance reduced by
the factor 1 / (1 - corr(Y, X)^2). ``b`` is estimated from the same sample (the bias this
introduces is of order 1/N and negligible here).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .rng import NormalSource, next_pow2

Z95 = 1.959963984540054


@dataclass
class Control:
    name: str
    values: Callable          # state -> discounted control values
    mean: float               # exact expectation of the discounted control


@dataclass
class MCResult:
    price: float
    stderr: float
    n_paths: int
    sampling: str
    seconds: float
    control: Optional[str] = None
    beta: Optional[float] = None
    units_y: np.ndarray = field(default=None, repr=False)     # sample units (Sobol: replications x points)
    units_x: Optional[np.ndarray] = field(default=None, repr=False)
    control_mean: Optional[float] = None

    @property
    def ci95(self):
        return self.price - Z95 * self.stderr, self.price + Z95 * self.stderr


def chunk_size(dims: int) -> int:
    """Largest power of two keeping a chunk of normals below ~32 MB (2^22 floats)."""
    return int(min(16384, max(256, 1 << int(math.floor(math.log2(max(1, (1 << 22) // max(dims, 1))))))))


def _estimate(y: np.ndarray, x: Optional[np.ndarray], ex: Optional[float]):
    """Point estimate, standard error and control coefficient from independent sample units."""
    n = len(y)
    if x is None:
        return float(y.mean()), float(y.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"), None
    vx = float(np.var(x, ddof=1))
    beta = float(np.cov(y, x, ddof=1)[0, 1] / vx) if vx > 0 else 0.0
    adj = y - beta * (x - ex)
    return float(adj.mean()), float(adj.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"), beta


def _estimate_replications(Y: np.ndarray, X: Optional[np.ndarray], ex: Optional[float]):
    """Randomised QMC: one estimate per scrambled replication, error from their spread."""
    beta = None
    if X is not None:
        vx = float(np.var(X, ddof=1))
        beta = float(np.cov(Y.ravel(), X.ravel(), ddof=1)[0, 1] / vx) if vx > 0 else 0.0
        Y = Y - beta * (X - ex)
    means = Y.mean(axis=1)
    return float(means.mean()), float(means.std(ddof=1) / math.sqrt(len(means))), beta


def run(simulate: Callable, payoff: Callable, dims: int, n_paths: int, discount: float,
        sampling: str = "pseudo", control: Optional[Control] = None, seed: Optional[int] = 7,
        replications: int = 16) -> MCResult:
    """Price E[discount * payoff] by simulation.

    ``simulate(n, src)`` returns a state dict (at least ``S``); ``payoff(state)`` returns
    undiscounted payoffs. With Sobol' sampling the path count is rounded to
    ``replications`` x a power of two.
    """
    t0 = time.perf_counter()
    chunk = chunk_size(dims)

    def draw(src, n):
        ys, xs = [], []
        done = 0
        while done < n:
            m = min(chunk, n - done)
            state = simulate(m, src)
            y = discount * np.asarray(payoff(state), dtype=float)
            x = np.asarray(control.values(state), dtype=float) if control else None
            if src.sampling == "antithetic":                      # pair average = one sample unit
                h = m // 2
                y = 0.5 * (y[:h] + y[h:])
                x = 0.5 * (x[:h] + x[h:]) if x is not None else None
            ys.append(y)
            if x is not None:
                xs.append(x)
            done += m
        return np.concatenate(ys), (np.concatenate(xs) if xs else None)

    ex = control.mean if control else None
    if sampling == "sobol":
        per = next_pow2(max(64, int(math.ceil(n_paths / replications))))
        seeds = np.random.SeedSequence(seed).spawn(replications)
        rows = [draw(NormalSource(dims, "sobol", s), per) for s in seeds]
        Y = np.vstack([r[0] for r in rows])
        X = np.vstack([r[1] for r in rows]) if control else None
        price, se, beta = _estimate_replications(Y, X, ex)
        total = replications * per
    else:
        total = int(n_paths) + (int(n_paths) % 2 if sampling == "antithetic" else 0)
        Y, X = draw(NormalSource(dims, sampling, seed), total)
        price, se, beta = _estimate(Y, X, ex)
    return MCResult(price, se, total, sampling, time.perf_counter() - t0, control.name if control else None,
                    beta, Y, X, ex)


def convergence(res: MCResult, n_points: int = 14) -> pd.DataFrame:
    """Estimate and 95% band as the number of paths grows (prefixes of the same run)."""
    rows = []
    if res.sampling == "sobol":
        per = res.units_y.shape[1]
        sizes = [1 << k for k in range(4, int(math.log2(per)) + 1)]
        for m in sizes:
            est, se, _ = _estimate_replications(res.units_y[:, :m], None if res.units_x is None else res.units_x[:, :m],
                                                res.control_mean)
            rows.append((m * res.units_y.shape[0], est, se))
    else:
        units = len(res.units_y)
        per_unit = 2 if res.sampling == "antithetic" else 1
        sizes = np.unique(np.geomspace(min(64, units), units, n_points).astype(int))
        for m in sizes:
            est, se, _ = _estimate(res.units_y[:m], None if res.units_x is None else res.units_x[:m], res.control_mean)
            rows.append((m * per_unit, est, se))
    df = pd.DataFrame(rows, columns=["paths", "estimate", "stderr"])
    df["lo"], df["hi"] = df["estimate"] - Z95 * df["stderr"], df["estimate"] + Z95 * df["stderr"]
    return df
