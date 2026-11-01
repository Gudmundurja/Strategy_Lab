"""Small number-formatting helpers shared by the app and the examples."""
from __future__ import annotations

import math


def money(x: float, dp: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if math.isinf(x):
        return "Unlimited"
    return f"{x:,.{dp}f}"


def signed(x: float, dp: int = 2) -> str:
    if math.isinf(x):
        return "Unlimited" if x > 0 else "-Unlimited"
    return f"{x:+,.{dp}f}"


def pct(x: float, dp: int = 1) -> str:
    """0.0525 -> '5.3%'."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100 * x:,.{dp}f}%"


def compact(x: float, dp: int = 1) -> str:
    """1_750_000_000 -> '1.75bn', 610_000_000 -> '610.0m'."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    a = abs(x)
    if a >= 1e9:
        return f"{x / 1e9:,.2f}bn"
    if a >= 1e6:
        return f"{x / 1e6:,.{dp}f}m"
    if a >= 1e4:
        return f"{x / 1e3:,.{dp}f}k"
    return f"{x:,.0f}"
