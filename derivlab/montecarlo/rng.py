"""Random inputs for Monte Carlo: pseudo-random, antithetic and scrambled Sobol' normals.

A :class:`NormalSource` hands out standard normal draws chunk by chunk with a fixed
number of dimensions per path (one per time step and risk factor).

* ``pseudo``: independent draws from NumPy's PCG64 generator.
* ``antithetic``: every chunk is ``[Z, -Z]``, so paths come in mirror-image pairs. The
  pair average has lower variance for payoffs that are monotone in the shocks.
* ``sobol``: a scrambled Sobol' low-discrepancy sequence mapped through the inverse
  normal CDF (randomised quasi-Monte Carlo). Points fill the space more evenly than
  random draws, so errors can shrink faster than 1/sqrt(N).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

SAMPLING = ("pseudo", "antithetic", "sobol")


def next_pow2(n: int) -> int:
    return 1 << max(0, math.ceil(math.log2(max(int(n), 1))))


class NormalSource:
    def __init__(self, dim: int, sampling: str = "pseudo", seed=None):
        if sampling not in SAMPLING:
            raise ValueError(f"sampling must be one of {SAMPLING}")
        self.dim = max(int(dim), 1)
        self.sampling = sampling
        self.rng = np.random.default_rng(seed)
        self._engine = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng) if sampling == "sobol" else None

    def normals(self, n: int) -> np.ndarray:
        """(n, dim) standard normals. For antithetic sampling n must be even: rows j and j + n/2 mirror each other."""
        if self.sampling == "pseudo":
            return self.rng.standard_normal((n, self.dim))
        if self.sampling == "antithetic":
            if n % 2:
                raise ValueError("antithetic chunks need an even number of paths")
            half = self.rng.standard_normal((n // 2, self.dim))
            return np.concatenate([half, -half])
        u = self._engine.random(n)
        return ndtri(np.clip(u, 1e-12, 1 - 1e-12))

    def poisson(self, lam, n: int, k: int) -> np.ndarray:
        """(n, k) Poisson counts (jump numbers). Antithetic pairs share their counts."""
        lam = np.broadcast_to(np.asarray(lam, dtype=float), (k,))
        if self.sampling == "antithetic":
            half = self.rng.poisson(lam, size=(n // 2, k))
            return np.concatenate([half, half])
        return self.rng.poisson(lam, size=(n, k))
