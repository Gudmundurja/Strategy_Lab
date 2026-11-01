"""Option positions: legs, market, P&L curves, risk statistics and Greeks."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .pricing import bs_greeks, bs_price, norm_cdf
from .volatility import VolModel

KINDS = ("call", "put", "underlying")
DAYS_PER_YEAR = 365.0


@dataclass
class Leg:
    """One leg of a position.

    ``qty`` is signed (+ long / - short) and measured in contracts. ``strike`` and
    ``days`` (calendar days to expiry) are ignored for the underlying. ``premium``
    overrides the model price paid or received per unit of underlying.
    """

    kind: str
    qty: float
    strike: float = float("nan")
    days: float = 30.0
    premium: Optional[float] = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if self.is_option:
            if not (self.strike > 0):
                raise ValueError("option legs need a positive strike")
            if not (self.days > 0):
                raise ValueError("option legs need days to expiry > 0")

    @property
    def is_option(self) -> bool:
        return self.kind != "underlying"

    @property
    def side(self) -> str:
        return "Buy" if self.qty >= 0 else "Sell"


@dataclass
class Market:
    spot: float = 100.0
    rate: float = 0.04
    div_yield: float = 0.0
    vol: VolModel = field(default_factory=VolModel)
    sticky: str = "strike"  # 'strike': each strike keeps its vol; 'moneyness': smile moves with spot

    def __post_init__(self):
        if not self.spot > 0:
            raise ValueError("spot must be positive")
        if self.sticky not in ("strike", "moneyness"):
            raise ValueError("sticky must be 'strike' or 'moneyness'")

    def forward(self, T, spot=None):
        s = self.spot if spot is None else spot
        return np.asarray(s, dtype=float) * np.exp((self.rate - self.div_yield) * np.asarray(T, dtype=float))


@dataclass
class Analysis:
    horizon_days: float
    net_premium: float          # > 0 credit received, < 0 debit paid (options only, in money)
    underlying_cost: float      # cost of underlying legs (in money)
    max_profit: float           # math.inf if unlimited
    max_loss: float             # -math.inf if unlimited (negative number otherwise)
    breakevens: list
    prob_profit: float          # risk-neutral probability that P&L > 0 at the horizon
    greeks: dict

    @property
    def is_credit(self) -> bool:
        return self.net_premium > 0


class Position:
    """A multi-leg position priced with Black-Scholes on a (possibly skewed) vol model.

    All money figures are ``per unit price x qty x multiplier``: with
    ``multiplier=100`` one contract covers 100 shares, including underlying legs.
    """

    def __init__(self, legs: Iterable[Leg], market: Market, multiplier: float = 1.0, default_horizon: float = 30.0):
        self.legs = list(legs)
        if not self.legs:
            raise ValueError("a position needs at least one leg")
        self.market = market
        self.multiplier = float(multiplier)
        self.default_horizon = default_horizon

    # ---------- pricing ----------
    def leg_vol(self, leg: Leg, S, days_elapsed: float = 0.0, vol_shift: float = 0.0):
        if not leg.is_option:
            return np.zeros_like(np.asarray(S, dtype=float))
        m = self.market
        T = max(leg.days - days_elapsed, 0.0) / DAYS_PER_YEAR
        ref = m.spot if m.sticky == "strike" else S
        model = m.vol.shifted(vol_shift) if vol_shift else m.vol
        return model.vol(leg.strike, max(T, 1e-6), m.forward(T, ref))

    def leg_value(self, leg: Leg, S, days_elapsed: float = 0.0, vol_shift: float = 0.0):
        S = np.asarray(S, dtype=float)
        if not leg.is_option:
            return S.copy()
        m = self.market
        T = max(leg.days - days_elapsed, 0.0) / DAYS_PER_YEAR
        return bs_price(leg.kind, S, leg.strike, T, m.rate, m.div_yield, self.leg_vol(leg, S, days_elapsed, vol_shift))

    def model_price(self, leg: Leg) -> float:
        return float(self.leg_value(leg, self.market.spot))

    def entry_price(self, leg: Leg) -> float:
        if not leg.is_option:
            return self.market.spot
        return float(leg.premium) if leg.premium is not None and np.isfinite(leg.premium) else self.model_price(leg)

    @property
    def entry_cost(self) -> float:
        """Signed cost of setting up the position per unit multiplier (positive = paid)."""
        return sum(l.qty * self.entry_price(l) for l in self.legs)

    @property
    def horizon(self) -> float:
        """Days to the first option expiry (the date expiry P&L is evaluated)."""
        opts = [l.days for l in self.legs if l.is_option]
        return float(min(opts)) if opts else float(self.default_horizon)

    @property
    def has_multiple_expiries(self) -> bool:
        return len({l.days for l in self.legs if l.is_option}) > 1

    def value(self, S, days_elapsed: float = 0.0, vol_shift: float = 0.0):
        S = np.asarray(S, dtype=float)
        total = np.zeros_like(S)
        for l in self.legs:
            total = total + l.qty * self.leg_value(l, S, days_elapsed, vol_shift)
        return total

    def pnl(self, S, days_elapsed: float = 0.0, vol_shift: float = 0.0):
        """P&L in money after ``days_elapsed`` days if spot is S and vols shift by ``vol_shift``."""
        return (self.value(S, days_elapsed, vol_shift) - self.entry_cost) * self.multiplier

    def pnl_at_horizon(self, S, vol_shift: float = 0.0):
        return self.pnl(S, self.horizon, vol_shift)

    # ---------- risk ----------
    def greeks(self, S: Optional[float] = None, days_elapsed: float = 0.0, vol_shift: float = 0.0) -> dict:
        """Position Greeks in money: delta (units of underlying), gamma (delta change per 1.00 move),
        theta per calendar day, vega per vol point, rho per 1% rate move, plus cash delta.

        With a sticky-moneyness smile, delta and gamma are taken by bumping spot and
        letting the smile move with it, so they include the skew effect ("smile delta").
        """
        m = self.market
        S = m.spot if S is None else float(S)
        g = dict(delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)
        live = [l for l in self.legs if l.is_option and l.days - days_elapsed > 0]
        for l in self.legs:
            if not l.is_option:
                g["delta"] += l.qty
        for l in live:
            T = (l.days - days_elapsed) / DAYS_PER_YEAR
            gr = bs_greeks(l.kind, S, l.strike, T, m.rate, m.div_yield, float(self.leg_vol(l, S, days_elapsed, vol_shift)))
            g["delta"] += l.qty * gr["delta"]
            g["gamma"] += l.qty * gr["gamma"]
            g["theta"] += l.qty * gr["theta"] / DAYS_PER_YEAR
            g["vega"] += l.qty * gr["vega"] / 100.0
            g["rho"] += l.qty * gr["rho"] / 100.0
        if live and m.sticky == "moneyness" and not m.vol.is_flat:
            h = S * 1e-3
            v = lambda s: sum(l.qty * float(self.leg_value(l, s, days_elapsed, vol_shift)) for l in live)
            up, mid, dn = v(S + h), v(S), v(S - h)
            g["delta"] = sum(l.qty for l in self.legs if not l.is_option) + (up - dn) / (2 * h)
            g["gamma"] = (up - 2 * mid + dn) / (h * h)
        out = {k: v * self.multiplier for k, v in g.items()}
        out["cash_delta"] = out["delta"] * S
        out["dollar_gamma_1pct"] = out["gamma"] * S * S * 0.01
        return out

    def scaled(self, lots: float) -> "Position":
        """The same position with every quantity multiplied by ``lots``."""
        legs = [Leg(l.kind, l.qty * lots, l.strike, l.days, l.premium) for l in self.legs]
        return Position(legs, self.market, self.multiplier, self.default_horizon)

    def _grid(self, n: int = 6000):
        """Spot grid for max profit/loss and breakevens: dense, out to 5x spot, and every strike
        included exactly so the kinks of the expiry payoff are never missed."""
        S0 = self.market.spot
        ks = [l.strike for l in self.legs if l.is_option]
        hi = max(5.0 * S0, 2.0 * max(ks) if ks else 0.0)
        return np.unique(np.concatenate([np.linspace(S0 * 1e-6, hi, n), ks, [S0]]))

    def _breakevens(self, x: np.ndarray, y: np.ndarray) -> list:
        """Spot levels where the P&L curve ``y`` over grid ``x`` changes sign."""
        bes = []
        eps = 1e-10 * max(1.0, abs(self.multiplier)) * self.market.spot
        nz = np.flatnonzero(np.abs(y) > eps)            # ignore exact zeros, find sign changes between them
        for a, b in zip(nz[:-1], nz[1:]):
            if np.sign(y[a]) == np.sign(y[b]):
                continue
            if b == a + 1:
                xb = x[a] - y[a] * (x[b] - x[a]) / (y[b] - y[a])
            else:                                       # P&L touches zero on grid points in between
                xb = 0.5 * (x[a + 1] + x[b - 1])
            if not bes or abs(xb - bes[-1]) > self.market.spot * 1e-3:
                bes.append(float(xb))
        return bes

    def breakevens(self, vol_shift: float = 0.0) -> list:
        """Underlying prices at which the P&L at the first expiry is zero."""
        x = self._grid()
        return self._breakevens(x, self.pnl(x, self.horizon, vol_shift))

    def analyze(self, vol_shift: float = 0.0) -> Analysis:
        """Premium, max profit and loss, breakevens, probability of profit and Greeks.

        Max profit/loss are measured at the first expiry over all spot levels and
        reported as +/- infinity when the payoff keeps rising/falling in the far tail.
        """
        H = self.horizon
        x = self._grid()
        y = self.pnl(x, H, vol_shift)
        tail = slice(-60, None)
        slope = (y[tail][-1] - y[tail][0]) / (x[tail][-1] - x[tail][0])
        tol = 1e-3 * self.multiplier
        max_p = math.inf if slope > tol else float(np.max(y))
        max_l = -math.inf if slope < -tol else float(np.min(y))
        bes = self._breakevens(x, y)
        net = -sum(l.qty * self.entry_price(l) for l in self.legs if l.is_option) * self.multiplier
        und = sum(l.qty * self.entry_price(l) for l in self.legs if not l.is_option) * self.multiplier
        return Analysis(H, net, und, max_p, max_l, bes, self.prob_profit(vol_shift, bes), self.greeks(vol_shift=vol_shift))

    def prob_profit(self, vol_shift: float = 0.0, breakevens: Optional[list] = None) -> float:
        """Risk-neutral probability that the P&L at the first expiry is positive.

        Spot at the horizon is lognormal at the (shifted) ATM vol. The breakevens
        split the price axis into intervals of constant sign, and each profitable
        interval contributes its exact lognormal probability.
        """
        m = self.market
        T = self.horizon / DAYS_PER_YEAR
        sig = m.vol.shifted(vol_shift).level
        bes = self.breakevens(vol_shift) if breakevens is None else sorted(breakevens)
        mu = math.log(m.spot) + (m.rate - m.div_yield - 0.5 * sig * sig) * T
        sd = sig * math.sqrt(T)

        def cdf(level: float) -> float:
            if level <= 0:
                return 0.0
            if math.isinf(level):
                return 1.0
            return float(norm_cdf((math.log(level) - mu) / sd))

        edges = [0.0] + bes + [math.inf]
        tol = 1e-9 * max(1.0, abs(self.multiplier))
        total = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            if math.isinf(hi):
                probe = lo * 1.5 if lo > 0 else m.spot
            elif lo == 0.0:
                probe = hi * 0.5
            else:
                probe = 0.5 * (lo + hi)
            if float(self.pnl(probe, self.horizon, vol_shift)) > tol:
                total += cdf(hi) - cdf(lo)
        return min(max(total, 0.0), 1.0)

    def scenario_grid(self, spot_moves=None, vol_shifts=None, days_elapsed: float = 0.0) -> pd.DataFrame:
        """P&L for spot moves (fractions) x parallel vol shifts (decimals) after ``days_elapsed`` days."""
        spot_moves = np.array([-0.2, -0.15, -0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1, 0.15, 0.2]) if spot_moves is None else np.asarray(spot_moves, dtype=float)
        vol_shifts = np.array([-0.10, -0.05, -0.02, 0, 0.02, 0.05, 0.10, 0.20]) if vol_shifts is None else np.asarray(vol_shifts, dtype=float)
        S = self.market.spot * (1 + spot_moves)
        rows = [self.pnl(S, days_elapsed, dv) for dv in vol_shifts]
        return pd.DataFrame(rows, index=pd.Index(np.round(vol_shifts * 100, 2), name="vol shift (pts)"),
                            columns=pd.Index(np.round(spot_moves * 100, 2), name="spot move (%)"))

    def leg_table(self) -> pd.DataFrame:
        m = self.market
        rows = []
        for l in self.legs:
            if l.is_option:
                T = l.days / DAYS_PER_YEAR
                v = float(self.leg_vol(l, m.spot))
                gr = bs_greeks(l.kind, m.spot, l.strike, T, m.rate, m.div_yield, v)
                rows.append(dict(Side=l.side, Qty=abs(l.qty), Type=l.kind.title(), Strike=l.strike, Days=l.days,
                                 **{"Vol %": v * 100, "Model price": self.model_price(l), "Entry price": self.entry_price(l),
                                    "Delta": l.qty * gr["delta"] * self.multiplier, "Gamma": l.qty * gr["gamma"] * self.multiplier,
                                    "Vega / pt": l.qty * gr["vega"] / 100 * self.multiplier, "Theta / day": l.qty * gr["theta"] / DAYS_PER_YEAR * self.multiplier}))
            else:
                rows.append(dict(Side=l.side, Qty=abs(l.qty), Type="Underlying", Strike=np.nan, Days=np.nan,
                                 **{"Vol %": np.nan, "Model price": m.spot, "Entry price": m.spot, "Delta": l.qty * self.multiplier,
                                    "Gamma": 0.0, "Vega / pt": 0.0, "Theta / day": 0.0}))
        return pd.DataFrame(rows)

    def curves(self, n: int = 400, days_elapsed: float = 0.0, vol_shift: float = 0.0, lo: float = None, hi: float = None) -> pd.DataFrame:
        """Spot grid with P&L at the horizon and after ``days_elapsed`` days (for plotting)."""
        S0 = self.market.spot
        ks = [l.strike for l in self.legs if l.is_option]
        lo = lo if lo is not None else max(S0 * 0.02, min([S0] + ks) * 0.78)
        hi = hi if hi is not None else max([S0] + ks) * 1.22
        x = np.linspace(lo, hi, n)
        return pd.DataFrame({"spot": x, "expiry": self.pnl(x, self.horizon, vol_shift),
                             "now": self.pnl(x, min(days_elapsed, self.horizon), vol_shift)})
