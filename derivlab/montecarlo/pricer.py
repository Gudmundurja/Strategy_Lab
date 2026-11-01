"""Single-asset Monte Carlo pricing: the high-level API.

>>> from derivlab.montecarlo import GBM, Asian, price_option
>>> res = price_option(Asian("call", 100), GBM(0.25), S0=100, r=0.04, q=0, T=0.5, n_obs=126,
...                    sampling="antithetic", control="auto")
>>> res.price, res.stderr
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from ..pricing import bs_price
from . import analytic
from .engine import Control, MCResult, run
from .models import GBM, Heston, Merton
from .payoffs import Asian, Barrier, Digital, European, Lookback

METHODS = {                       # UI label -> (sampling, use control variate)
    "Plain": ("pseudo", False),
    "Antithetic": ("antithetic", False),
    "Control variate": ("pseudo", True),
    "Antithetic + control variate": ("antithetic", True),
    "Sobol (quasi-random)": ("sobol", False),
    "Sobol + control variate": ("sobol", True),
}


def observation_times(T: float, n_obs: int) -> np.ndarray:
    n = max(int(n_obs), 1)
    return T * np.arange(1, n + 1) / n


def model_vanilla_price(model, kind: str, S: float, K: float, T: float, r: float, q: float) -> float:
    """European option price under the model: Black-Scholes, Merton's series or Heston's formula."""
    if isinstance(model, GBM):
        return float(bs_price(kind, S, K, T, r, q, model.sigma))
    if isinstance(model, Merton):
        return analytic.merton_price(kind, S, K, T, r, q, model.sigma, model.lam, model.mu_j, model.delta)
    if isinstance(model, Heston):
        return analytic.heston_price(kind, S, K, T, r, q, model.v0, model.kappa, model.theta, model.xi, model.rho)
    raise TypeError(f"unknown model {model!r}")


def benchmark(product, model, S0: float, r: float, q: float, T: float, n_obs: int):
    """A closed-form or semi-analytic price to check the simulation against: (name, value) or None."""
    gbm = isinstance(model, GBM)
    if isinstance(product, European):
        name = {GBM: "Black-Scholes formula", Merton: "Merton series formula", Heston: "Heston formula"}[type(model)]
        return name, model_vanilla_price(model, product.kind, S0, product.strike, T, r, q)
    if not gbm:
        return None
    if isinstance(product, Digital):
        return "Black-Scholes digital formula", product.cash * analytic.digital_price(product.kind, S0, product.strike, T, r, q, model.sigma)
    if isinstance(product, Asian) and product.average == "geometric":
        return "Exact discrete geometric-average formula", analytic.geometric_asian_price(
            product.kind, S0, product.strike, T, r, q, model.sigma, n_obs)
    if isinstance(product, Barrier):
        return "Barrier formula with discrete-monitoring correction (approx.)", analytic.barrier_price(
            product.kind, S0, product.strike, product.barrier, T, r, q, model.sigma, product.direction, product.knock,
            monitoring_dt=T / n_obs)
    if isinstance(product, Lookback) and product.strike is None:
        return "Continuous-monitoring formula (an upper bound)", analytic.lookback_floating_price(
            product.kind, S0, T, r, q, model.sigma)
    return None


def control_variate(product, model, S0: float, r: float, q: float, T: float, n_obs: int) -> Control:
    """The most useful control variate available for this product and model.

    * Arithmetic Asian under GBM: the geometric Asian (exact price, correlation ~0.99).
    * Digitals, barriers, fixed-strike lookbacks and Asians under other models: the
      vanilla option with the same strike, priced by the model's formula.
    * Everything else: the discounted terminal price, whose expectation is S0 e^{-qT}.
    """
    disc = math.exp(-r * T)
    if isinstance(product, Asian) and product.average == "arithmetic" and isinstance(model, GBM):
        geo = Asian(product.kind, product.strike, "geometric")
        return Control("Geometric Asian (exact price known)", lambda st: disc * geo.payoff(st["S"]),
                       analytic.geometric_asian_price(product.kind, S0, product.strike, T, r, q, model.sigma, n_obs))
    vanilla_like = isinstance(product, (Digital, Barrier, Asian)) or (isinstance(product, Lookback) and product.strike is not None)
    if vanilla_like:
        v = European(product.kind, product.strike)
        return Control("Vanilla option, same strike", lambda st: disc * v.payoff(st["S"]),
                       model_vanilla_price(model, product.kind, S0, product.strike, T, r, q))
    return Control("Discounted terminal price", lambda st: disc * st["S"][:, -1], S0 * math.exp(-q * T))


def price_option(product, model, S0: float, r: float, q: float, T: float, n_obs: int = 1, n_paths: int = 20_000,
                 sampling: str = "pseudo", control: Optional[str] = None, seed: Optional[int] = 7) -> MCResult:
    """Monte Carlo price of a single-asset product. ``control='auto'`` picks :func:`control_variate`."""
    times = observation_times(T, n_obs if product.path_dependent else 1)
    cv = control_variate(product, model, S0, r, q, T, len(times)) if control == "auto" else None
    return run(lambda n, src: model.simulate(S0, r, q, times, n, src), lambda st: product.payoff(st["S"]),
               model.dims(times), n_paths, math.exp(-r * T), sampling, cv, seed)


def compare_methods(product, model, S0: float, r: float, q: float, T: float, n_obs: int, n_paths: int,
                    seed: Optional[int] = 7, methods=None) -> pd.DataFrame:
    """Price with every variance-reduction method at the same path budget."""
    rows, base = [], None
    for label in methods or METHODS:
        sampling, cv = METHODS[label]
        res = price_option(product, model, S0, r, q, T, n_obs, n_paths, sampling, "auto" if cv else None, seed)
        if base is None:
            base = res
        vrf = (base.stderr / res.stderr) ** 2 if res.stderr > 0 else float("inf")
        rows.append({"Method": label, "Price": res.price, "Std error": res.stderr, "95% CI ±": 1.96 * res.stderr,
                     "Paths": res.n_paths, "Variance reduction": vrf, "Time (ms)": res.seconds * 1000,
                     "Efficiency": vrf * base.seconds / res.seconds if res.seconds > 0 else float("nan"),
                     "Control": res.control or "-"})
    return pd.DataFrame(rows)


def sample_paths(model, S0: float, r: float, q: float, T: float, n_obs: int, n: int = 30, seed: int = 1):
    """A few paths for plotting: times (years) and prices (n, steps + 1)."""
    from .rng import NormalSource
    times = observation_times(T, n_obs)
    st = model.simulate(S0, r, q, times, n, NormalSource(model.dims(times), "pseudo", seed))
    return np.concatenate([[0.0], times]), st["S"]


def discounted_payoffs(product, model, S0, r, q, T, n_obs, n_paths=20_000, seed=7) -> np.ndarray:
    """Per-path discounted payoffs (plain sampling) for histograms."""
    return price_option(product, model, S0, r, q, T, n_obs, n_paths, "pseudo", None, seed).units_y
