"""Portfolio credit loss models: the one-factor Gaussian copula.

Each borrower defaults when  sqrt(rho) * M + sqrt(1 - rho) * eps_i  <  N^-1(PD).
Given the common factor M, defaults are independent with probability

    p(M) = N( (N^-1(PD) - sqrt(rho) * M) / sqrt(1 - rho) ).

For a large homogeneous pool (LHP) the loss fraction is exactly LGD * p(M), which
gives Vasicek's closed-form loss distribution, the same model behind the Basel
IRB formula. For small or lumpy pools use :func:`simulate_losses`.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri


def _check(pd, rho, lgd):
    if not (0 < pd < 1):
        raise ValueError("pd must be between 0 and 1")
    if not (0 < rho < 1):
        raise ValueError("rho must be between 0 and 1")
    if not (0 < lgd <= 1):
        raise ValueError("lgd must be in (0, 1]")


def vasicek_cdf(x, pd: float, rho: float, lgd: float = 1.0):
    """P(pool loss <= x) for a large homogeneous pool. x is a loss fraction of pool notional."""
    _check(pd, rho, lgd)
    x = np.asarray(x, dtype=float)
    u = np.clip(x / lgd, 1e-300, 1 - 1e-16)
    inner = (np.sqrt(1 - rho) * ndtri(u) - ndtri(pd)) / np.sqrt(rho)
    out = np.where(x <= 0, 0.0, np.where(x >= lgd, 1.0, ndtr(inner)))
    return out if out.ndim else float(out)


def vasicek_pdf(x, pd: float, rho: float, lgd: float = 1.0):
    """Density of the pool loss fraction (per unit of loss)."""
    _check(pd, rho, lgd)
    x = np.asarray(x, dtype=float)
    u = np.clip(x / lgd, 1e-300, 1 - 1e-16)
    y = ndtri(u)
    z = (np.sqrt(1 - rho) * y - ndtri(pd)) / np.sqrt(rho)
    dens = np.sqrt((1 - rho) / rho) * np.exp(-0.5 * z * z + 0.5 * y * y) / lgd
    out = np.where((x <= 0) | (x >= lgd), 0.0, dens)
    return out if out.ndim else float(out)


def vasicek_quantile(alpha: float, pd: float, rho: float, lgd: float = 1.0) -> float:
    """Loss fraction not exceeded with probability alpha (e.g. alpha=0.999 for IRB)."""
    _check(pd, rho, lgd)
    return float(lgd * ndtr((ndtri(pd) + np.sqrt(rho) * ndtri(alpha)) / np.sqrt(1 - rho)))


_GL_X, _GL_W = np.polynomial.legendre.leggauss(32)


def _gauss_legendre(f, a: float, b: float, pieces: int = 8) -> float:
    """Composite 32-point Gauss-Legendre quadrature of a smooth function on [a, b]."""
    if b <= a:
        return 0.0
    edges = np.linspace(a, b, pieces + 1)
    mid, half = 0.5 * (edges[1:] + edges[:-1]), 0.5 * (edges[1:] - edges[:-1])
    x = mid[:, None] + half[:, None] * _GL_X[None, :]
    return float(np.sum(half[:, None] * _GL_W[None, :] * f(x)))


def expected_min_loss(k: float, pd: float, rho: float, lgd: float = 1.0) -> float:
    """E[min(L, k)] for the large-pool loss fraction L.

    L <= k exactly when the common factor M is above m* = (N^-1(PD) - sqrt(1-rho) N^-1(k/LGD)) / sqrt(rho), so

        E[min(L, k)] = k * P(M < m*) + LGD * integral from m* to inf of p(m) phi(m) dm,

    and the remaining integrand is smooth, so Gauss-Legendre quadrature is accurate to ~1e-12.
    """
    _check(pd, rho, lgd)
    if k <= 0:
        return 0.0
    if k >= lgd:
        return pd * lgd
    c, s, t = ndtri(pd), np.sqrt(rho), np.sqrt(1 - rho)
    m_star = float((c - t * ndtri(k / lgd)) / s)
    integrand = lambda m: ndtr((c - s * m) / t) * np.exp(-0.5 * m * m) / np.sqrt(2 * np.pi)
    return float(k * ndtr(m_star) + lgd * _gauss_legendre(integrand, max(m_star, -10.0), 10.0))


def tranche_el(pd: float, rho: float, lgd: float, attach: float, detach: float) -> float:
    """Expected loss of the [attach, detach] tranche as a fraction of tranche notional (large pool).

    Tranche loss = min(L, D) - min(L, A), so EL = (E[min(L, D)] - E[min(L, A)]) / (D - A).
    """
    if detach <= attach:
        raise ValueError("detach must exceed attach")
    return (expected_min_loss(detach, pd, rho, lgd) - expected_min_loss(attach, pd, rho, lgd)) / (detach - attach)


def tranche_hit_prob(pd: float, rho: float, lgd: float, attach: float) -> float:
    """Probability the tranche takes any loss: P(L > attach)."""
    return 1.0 - vasicek_cdf(attach, pd, rho, lgd) if attach < lgd else 0.0


def tranche_metrics(pd: float, rho: float, lgd: float, attach: float, detach: float) -> dict:
    el = tranche_el(pd, rho, lgd, attach, detach)
    p_hit = tranche_hit_prob(pd, rho, lgd, attach)
    p_wipe = 1.0 - vasicek_cdf(detach, pd, rho, lgd) if detach < lgd else 0.0
    return dict(el=el, p_hit=p_hit, p_wipe=p_wipe, lgd_given_hit=el / p_hit if p_hit > 1e-15 else 0.0)


def cumulative_pd(pd_annual: float, years):
    """Cumulative default probability with a constant annual PD (flat hazard)."""
    out = 1.0 - (1.0 - pd_annual) ** np.asarray(years, dtype=float)
    return out if out.ndim else float(out)


def tranche_fair_spread(pd_annual: float, rho: float, lgd: float, attach: float, detach: float,
                        maturity: float, rate: float = 0.03, freq: int = 4) -> dict:
    """Fair running spread of a tranche under the LHP model with a flat hazard term structure.

    Premium leg: spread x accrual x DF x expected outstanding tranche notional.
    Protection leg: discounted increments of expected tranche loss.
    """
    n = max(1, int(round(maturity * freq)))
    t = np.arange(1, n + 1) / freq
    el = np.array([tranche_el(cumulative_pd(pd_annual, ti), rho, lgd, attach, detach) for ti in t])
    el_prev = np.concatenate([[0.0], el[:-1]])
    tau = 1.0 / freq
    df = np.exp(-rate * t)
    df_mid = np.exp(-rate * (t - tau / 2))
    rpv01 = float(np.sum(tau * df * (1.0 - 0.5 * (el + el_prev))))
    prot = float(np.sum(df_mid * (el - el_prev)))
    return dict(spread=prot / rpv01 if rpv01 > 0 else float("nan"), rpv01=rpv01, protection=prot,
                el_path=el, times=t, el_total=float(el[-1]))


def attachment_for_target_hit_prob(target: float, pd: float, rho: float, lgd: float) -> float:
    """Lowest attachment point whose probability of taking any loss is at most ``target``.

    This is just the (1 - target) quantile of the pool loss distribution.
    """
    return vasicek_quantile(1.0 - target, pd, rho, lgd)


def attachment_for_target_el(target_el: float, pd: float, rho: float, lgd: float, detach: float = 1.0) -> float:
    """Attachment point that gives the [A, detach] tranche an expected loss of ``target_el``."""
    f = lambda a: tranche_el(pd, rho, lgd, a, detach) - target_el
    hi = min(detach, lgd) - 1e-9
    if f(0.0) <= 0:
        return 0.0
    if f(hi) >= 0:
        return hi
    return float(brentq(f, 0.0, hi, xtol=1e-7))


def simulate_losses(n_loans: int, pd: float, rho: float, lgd: float, n_sims: int = 50_000,
                    exposures=None, seed: int | None = 7) -> np.ndarray:
    """Monte Carlo pool loss fractions for a finite pool.

    Equal exposures use the exact conditional-binomial shortcut; pass ``exposures``
    (array of sizes) for a lumpy pool.
    """
    _check(pd, rho, lgd)
    rng = np.random.default_rng(seed)
    M = rng.standard_normal(n_sims)
    if exposures is None:
        p = ndtr((ndtri(pd) - np.sqrt(rho) * M) / np.sqrt(1 - rho))
        return lgd * rng.binomial(int(n_loans), p) / float(n_loans)
    w = np.asarray(exposures, dtype=float)
    w = w / w.sum()
    thr = ndtri(pd)
    out = np.empty(n_sims)
    chunk = max(1, int(4_000_000 // max(len(w), 1)))
    for s in range(0, n_sims, chunk):
        m = M[s:s + chunk, None]
        eps = rng.standard_normal((m.shape[0], len(w)))
        defaulted = (np.sqrt(rho) * m + np.sqrt(1 - rho) * eps) < thr
        out[s:s + chunk] = lgd * (defaulted @ w)
    return out


def tranche_metrics_from_losses(losses, attach: float, detach: float) -> dict:
    L = np.asarray(losses, dtype=float)
    tl = np.clip(L - attach, 0.0, detach - attach) / (detach - attach)
    p_hit = float(np.mean(L > attach + 1e-12))
    el = float(tl.mean())
    return dict(el=el, p_hit=p_hit, p_wipe=float(np.mean(L >= detach - 1e-12)), lgd_given_hit=el / p_hit if p_hit > 0 else 0.0)
