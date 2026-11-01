"""Payoffs evaluated on simulated paths.

Single-asset payoffs receive ``S`` with shape (paths, observations + 1); column 0 is
today's spot and the last column is the price at maturity. Path-dependent features
(averages, barriers, extremes) use the observation dates, so they are *discretely
monitored*, as real contracts are.

Multi-asset payoffs receive the terminal prices with shape (paths, assets).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _phi(kind: str) -> int:
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    return 1 if kind == "call" else -1


# ---------------------------------------------------------------- single asset
@dataclass(frozen=True)
class European:
    kind: str
    strike: float
    path_dependent = False

    def payoff(self, S):
        return np.maximum(_phi(self.kind) * (S[:, -1] - self.strike), 0.0)

    def label(self):
        return f"European {self.kind} K={self.strike:g}"


@dataclass(frozen=True)
class Digital:
    kind: str
    strike: float
    cash: float = 1.0
    path_dependent = False

    def payoff(self, S):
        return self.cash * (_phi(self.kind) * (S[:, -1] - self.strike) > 0)

    def label(self):
        return f"Digital {self.kind} K={self.strike:g} pays {self.cash:g}"


@dataclass(frozen=True)
class Asian:
    """Fixed-strike Asian option on the average of the fixings (today's price excluded)."""
    kind: str
    strike: float
    average: str = "arithmetic"   # or 'geometric'
    path_dependent = True

    def average_price(self, S):
        fix = S[:, 1:]
        return fix.mean(axis=1) if self.average == "arithmetic" else np.exp(np.log(fix).mean(axis=1))

    def payoff(self, S):
        return np.maximum(_phi(self.kind) * (self.average_price(S) - self.strike), 0.0)

    def label(self):
        return f"{self.average.title()} Asian {self.kind} K={self.strike:g}"


@dataclass(frozen=True)
class Barrier:
    """Knock-in or knock-out option. The barrier is checked today and on every observation date."""
    kind: str
    strike: float
    barrier: float
    direction: str = "down"       # 'down' or 'up'
    knock: str = "out"            # 'in' or 'out'
    path_dependent = True

    def touched(self, S):
        return S.min(axis=1) <= self.barrier if self.direction == "down" else S.max(axis=1) >= self.barrier

    def payoff(self, S):
        vanilla = np.maximum(_phi(self.kind) * (S[:, -1] - self.strike), 0.0)
        hit = self.touched(S)
        return vanilla * (hit if self.knock == "in" else ~hit)

    def label(self):
        return f"{self.direction.title()}-and-{self.knock} {self.kind} K={self.strike:g} H={self.barrier:g}"


@dataclass(frozen=True)
class Lookback:
    """Lookback option. Floating strike (strike=None): call pays S_T - min S, put pays max S - S_T.
    Fixed strike: call pays max(max S - K, 0), put pays max(K - min S, 0)."""
    kind: str
    strike: Optional[float] = None
    path_dependent = True

    def payoff(self, S):
        lo, hi, ST = S.min(axis=1), S.max(axis=1), S[:, -1]
        if self.strike is None:
            return ST - lo if self.kind == "call" else hi - ST
        return np.maximum(hi - self.strike, 0.0) if self.kind == "call" else np.maximum(self.strike - lo, 0.0)

    def label(self):
        return f"Lookback {self.kind} ({'floating strike' if self.strike is None else f'K={self.strike:g}'})"


# ---------------------------------------------------------------- multi asset (terminal prices)
@dataclass(frozen=True)
class Basket:
    kind: str
    strike: float
    weights: tuple

    def payoff(self, ST):
        return np.maximum(_phi(self.kind) * (ST @ np.asarray(self.weights, dtype=float) - self.strike), 0.0)

    def label(self):
        return f"Basket {self.kind} K={self.strike:g}"


@dataclass(frozen=True)
class Spread:
    """Pays max(phi (S1 - S2 - K), 0)."""
    kind: str
    strike: float

    def payoff(self, ST):
        return np.maximum(_phi(self.kind) * (ST[:, 0] - ST[:, 1] - self.strike), 0.0)

    def label(self):
        return f"Spread {self.kind} K={self.strike:g}"


@dataclass(frozen=True)
class Rainbow:
    """Best-of ('max') or worst-of ('min') option on the extreme of several assets."""
    kind: str
    strike: float
    extreme: str = "max"

    def payoff(self, ST):
        x = ST.max(axis=1) if self.extreme == "max" else ST.min(axis=1)
        return np.maximum(_phi(self.kind) * (x - self.strike), 0.0)

    def label(self):
        return f"{'Best' if self.extreme == 'max' else 'Worst'}-of {self.kind} K={self.strike:g}"
