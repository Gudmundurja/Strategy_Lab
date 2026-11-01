"""A simple, transparent implied-volatility smile.

The smile is quadratic in *standardised moneyness*

    x = ln(K / F) / (atm * sqrt(T))

i.e. the number of ATM standard deviations the strike sits away from the
forward. Measuring moneyness this way keeps the skew comparable across
expiries:

    vol(K, T) = atm + shift + skew * x + smile * x**2      (floored at ``floor``)

``skew`` < 0 gives an equity-style put skew (low strikes trade at higher vol),
``smile`` > 0 lifts both wings (FX-style). With both at zero you get the flat
volatility assumed by textbook Black-Scholes. ``shift`` is a parallel move of
the whole smile, used for "what if implied vol rises 5 points" scenarios.

Beyond ``wing`` standard deviations the smile is held flat. A quadratic keeps
rising forever, which would give absurd vols (and arbitrage) for far-away strikes,
e.g. when spot is stressed 50% under a sticky-moneyness smile.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class VolModel:
    atm: float = 0.25
    skew: float = 0.0     # vol change (decimal) per 1 standardised move; e.g. -0.03
    smile: float = 0.0    # curvature (decimal) per standardised move squared; e.g. 0.005
    floor: float = 0.01
    shift: float = 0.0    # parallel shift (decimal) applied to every strike
    wing: float = 4.0     # the smile is flat beyond +/- this many standard deviations

    @classmethod
    def flat(cls, vol: float) -> "VolModel":
        return cls(atm=vol)

    @classmethod
    def equity(cls, atm: float = 0.20) -> "VolModel":
        """Typical index shape: steep put skew, mild curvature."""
        return cls(atm=atm, skew=-0.035, smile=0.006)

    @classmethod
    def fx(cls, atm: float = 0.09) -> "VolModel":
        """Roughly symmetric smile with a slight tilt."""
        return cls(atm=atm, skew=-0.003, smile=0.008)

    @property
    def is_flat(self) -> bool:
        return self.skew == 0 and self.smile == 0

    @property
    def level(self) -> float:
        """At-the-money vol including any parallel shift."""
        return max(self.atm + self.shift, self.floor)

    def shifted(self, dv: float) -> "VolModel":
        """The same smile moved up or down by ``dv`` (decimal) at every strike."""
        return replace(self, shift=self.shift + dv)

    def moneyness(self, K, T, F):
        T = np.maximum(np.asarray(T, dtype=float), 1.0 / 365.0)
        return np.log(np.asarray(K, dtype=float) / np.asarray(F, dtype=float)) / (self.atm * np.sqrt(T))

    def vol(self, K, T, F):
        """Implied vol for strike(s) K at expiry T (years) given forward F."""
        if self.is_flat:
            shape = np.broadcast(np.asarray(K), np.asarray(T), np.asarray(F)).shape
            return np.full(shape, self.level)
        x = np.clip(self.moneyness(K, T, F), -self.wing, self.wing)
        return np.maximum(self.atm + self.shift + self.skew * x + self.smile * x * x, self.floor)
