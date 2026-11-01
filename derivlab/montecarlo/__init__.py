"""Monte Carlo pricing: models, payoffs, variance reduction, American options, Greeks and risk.

Quick start::

    from derivlab.montecarlo import GBM, Heston, Asian, price_option
    res = price_option(Asian("call", 100), Heston(), S0=100, r=0.04, q=0.0, T=0.5, n_obs=126,
                       n_paths=50_000, sampling="antithetic", control="auto")
    print(res.price, res.stderr, res.ci95)
"""
from . import american, analytic, greeks, multiasset, portfolio
from .engine import MCResult, convergence, run
from .models import GBM, Heston, Merton
from .payoffs import Asian, Barrier, Basket, Digital, European, Lookback, Rainbow, Spread
from .pricer import (METHODS, benchmark, compare_methods, control_variate, discounted_payoffs, model_vanilla_price,
                     observation_times, price_option, sample_paths)

__all__ = ["american", "analytic", "greeks", "multiasset", "portfolio", "MCResult", "convergence", "run", "GBM",
           "Heston", "Merton", "Asian", "Barrier", "Basket", "Digital", "European", "Lookback", "Rainbow", "Spread",
           "METHODS", "benchmark", "compare_methods", "control_variate", "discounted_payoffs", "model_vanilla_price",
           "observation_times", "price_option", "sample_paths"]
