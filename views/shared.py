"""State shared between pages: the options-lab market and position.

The options lab owns these inputs; the Monte Carlo page reads the same session-state keys
so it can reprice and risk-manage the position you built.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from derivlab import Leg, Market, Position, VolModel, presets

SMILES = {
    "Flat (textbook Black-Scholes)": (0.0, 0.0),
    "Equity index put skew": (-0.035, 0.006),
    "FX-style smile": (-0.003, 0.008),
    "Custom": None,
}
MULTIPLIERS = {
    "1 · per unit of underlying": 1,
    "10 · e.g. Euro Stoxx 50 index options": 10,
    "50 · e.g. options on E-mini S&P 500 futures": 50,
    "100 · US equity and SPX options": 100,
    "1,000 · e.g. crude oil futures options": 1000,
    "Custom": None,
}
COLUMNS = ["Side", "Qty", "Type", "Strike", "Days", "Premium"]
MARKET_DEFAULTS = dict(spot=100.0, rate=4.0, div=0.0, atm=25.0, smile_shape="Flat (textbook Black-Scholes)", skew=-3.5,
                       smile=0.6, sticky="Sticky strike", mult_choice="100 · US equity and SPX options", mult_custom=100.0,
                       lots=1, strategy="bull-call-spread", days=30)


def init_state(extra: dict | None = None):
    ss = st.session_state
    for k, v in {**MARKET_DEFAULTS, **(extra or {})}.items():
        ss.setdefault(k, v)
    if "legs_df" not in ss:
        reset_legs()


def preset_frame(strategy_id: str, spot: float, days: int) -> pd.DataFrame:
    rows = []
    for l in presets.get(strategy_id).make_legs(spot, days):
        rows.append({"Side": l.side, "Qty": abs(l.qty), "Type": l.kind.title() if l.is_option else "Underlying",
                     "Strike": l.strike if l.is_option else np.nan, "Days": l.days if l.is_option else np.nan,
                     "Premium": np.nan})
    return pd.DataFrame(rows, columns=COLUMNS).astype({"Qty": "float64", "Strike": "float64", "Days": "float64", "Premium": "float64"})


def reset_legs():
    ss = st.session_state
    ss.legs_df = preset_frame(ss.strategy, float(ss.spot), int(ss.days))
    ss.legs_saved = ss.legs_df
    ss.legs_ver = ss.get("legs_ver", 0) + 1


def parse_legs(df: pd.DataFrame):
    legs, problems = [], []
    for i, r in enumerate(df.to_dict("records"), start=1):
        side, qty, typ = r.get("Side"), r.get("Qty"), r.get("Type")
        if side not in ("Buy", "Sell") or typ not in ("Call", "Put", "Underlying") or qty is None or pd.isna(qty):
            continue                                           # incomplete row, e.g. just added
        if qty <= 0:
            continue
        q = float(qty) if side == "Buy" else -float(qty)
        if typ == "Underlying":
            legs.append(Leg("underlying", q))
            continue
        K, D, prem = r.get("Strike"), r.get("Days"), r.get("Premium")
        if K is None or pd.isna(K) or K <= 0:
            problems.append(f"Row {i}: enter a positive strike.")
            continue
        if D is None or pd.isna(D) or D < 1:
            problems.append(f"Row {i}: enter days to expiry (1 or more).")
            continue
        premium = None if prem is None or pd.isna(prem) else float(prem)
        legs.append(Leg(typ.lower(), q, float(K), float(D), premium))
    return legs, problems


def current_market() -> Market:
    ss = st.session_state
    skew, smile = SMILES[ss.smile_shape] or (ss.skew / 100.0, ss.smile / 100.0)
    vol = VolModel(atm=ss.atm / 100.0, skew=skew, smile=smile)
    return Market(spot=float(ss.spot), rate=ss.rate / 100.0, div_yield=ss.div / 100.0, vol=vol,
                  sticky="strike" if ss.sticky == "Sticky strike" else "moneyness")


def current_multiplier() -> float:
    ss = st.session_state
    return float(MULTIPLIERS[ss.mult_choice] or ss.mult_custom)


def current_position():
    """The options-lab position (all lots) and any problems found in the legs table, or (None, problems)."""
    ss = st.session_state
    legs, problems = parse_legs(ss.get("legs_saved", ss.legs_df))
    if not legs:
        return None, problems
    return Position(legs, current_market(), multiplier=current_multiplier()).scaled(int(ss.lots)), problems
