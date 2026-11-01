"""Options on several correlated assets under multi-dimensional GBM.

Terminal prices are simulated exactly: correlated normals come from the Cholesky factor
L of the correlation matrix (Z_corr = Z L^T), and

    S_i(T) = S_i exp((r - q_i - sigma_i^2 / 2) T + sigma_i sqrt(T) Z_corr_i).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from . import analytic
from .engine import Control, MCResult, run
from .payoffs import Basket, Rainbow, Spread


def correlation_matrix(n: int, pairs: dict) -> np.ndarray:
    """Build an n x n correlation matrix from {(i, j): rho} and check it is a valid one."""
    C = np.eye(n)
    for (i, j), rho in pairs.items():
        C[i, j] = C[j, i] = rho
    eig = np.linalg.eigvalsh(C)
    if eig.min() <= 1e-10:
        raise ValueError("these correlations are inconsistent (the matrix is not positive definite)")
    return C


def simulate_terminal(S0, r, q, sigma, corr, T, n, src) -> np.ndarray:
    S0, q, sigma = (np.asarray(a, dtype=float) for a in (S0, q, sigma))
    L = np.linalg.cholesky(np.asarray(corr, dtype=float))
    Z = src.normals(n) @ L.T
    return S0 * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * Z)


def benchmark(product, S0, r, q, sigma, corr, T):
    """Exact or approximate formula for the product, or None: (name, value)."""
    d = len(S0)
    if isinstance(product, Spread):
        rho = corr[0][1]
        if product.strike == 0 and product.kind == "call":
            return "Margrabe exchange formula (exact)", analytic.margrabe_price(S0[0], S0[1], T, q[0], q[1], sigma[0], sigma[1], rho)
        return "Kirk approximation", analytic.kirk_spread_price(product.kind, S0[0], S0[1], product.strike, T, r, q[0], q[1],
                                                                  sigma[0], sigma[1], rho)
    if isinstance(product, Rainbow) and d == 2:
        return "Stulz formula (exact)", analytic.two_asset_extreme_price(product.kind, product.extreme, S0[0], S0[1],
                                                                          product.strike, T, r, q[0], q[1], sigma[0], sigma[1], corr[0][1])
    if isinstance(product, Basket):
        return "Levy lognormal approximation", analytic.levy_basket_price(product.kind, S0, product.strike, T, r, q, sigma,
                                                                          corr, product.weights)
    return None


def control_variate(product, S0, r, q, sigma, corr, T) -> Control:
    disc = math.exp(-r * T)
    S0a, qa = np.asarray(S0, dtype=float), np.asarray(q, dtype=float)
    if isinstance(product, Basket):
        w = np.asarray(product.weights, dtype=float)
        a, W = w / w.sum(), w.sum()
        phi = 1 if product.kind == "call" else -1
        geo = lambda ST: disc * np.maximum(phi * (W * np.exp(np.log(ST) @ a) - product.strike), 0.0)
        return Control("Geometric basket (exact price known)", geo,
                       analytic.geometric_basket_price(product.kind, S0, product.strike, T, r, q, sigma, corr, w))
    if isinstance(product, Spread) and len(S0) == 2:
        exch = lambda ST: disc * np.maximum(ST[:, 0] - ST[:, 1], 0.0)
        return Control("Exchange option (Margrabe price known)", exch,
                       analytic.margrabe_price(S0[0], S0[1], T, q[0], q[1], sigma[0], sigma[1], corr[0][1]))
    return Control("Discounted terminal prices", lambda ST: disc * ST.sum(axis=1), float(np.sum(S0a * np.exp(-qa * T))))


def price_multi(product, S0, r, q, sigma, corr, T, n_paths=20_000, sampling="pseudo", control=None,
                seed: Optional[int] = 7) -> MCResult:
    cv = control_variate(product, S0, r, q, sigma, corr, T) if control == "auto" else None
    return run(lambda n, src: simulate_terminal(S0, r, q, sigma, corr, T, n, src), product.payoff, len(S0), n_paths,
               math.exp(-r * T), sampling, cv, seed)


def correlation_sweep(product, S0, r, q, sigma, T, rhos, n_paths=20_000, seed=7) -> pd.DataFrame:
    """Price as one common correlation between every pair of assets moves (same random numbers for each rho)."""
    d = len(S0)
    rows = []
    for rho in rhos:
        C = np.full((d, d), rho)
        np.fill_diagonal(C, 1.0)
        if np.linalg.eigvalsh(C).min() <= 1e-10:
            continue
        res = price_multi(product, S0, r, q, sigma, C, T, n_paths, "pseudo", "auto", seed)
        bm = benchmark(product, S0, r, q, sigma, C, T)
        rows.append({"rho": rho, "price": res.price, "stderr": res.stderr, "benchmark": bm[1] if bm else np.nan})
    return pd.DataFrame(rows)
