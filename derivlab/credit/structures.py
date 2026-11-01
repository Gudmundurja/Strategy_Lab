"""Capital structures: tranches defined by attachment and detachment points."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Tranche:
    name: str
    attach: float           # fraction of pool notional, e.g. 0.03
    detach: float           # e.g. 0.07
    coupon: float = 0.0       # annual coupon or protection premium (decimal), used by waterfall/SRT tools
    holder: str = "Retained"  # 'Retained' or 'Sold' (for SRT economics)

    @property
    def thickness(self) -> float:
        return self.detach - self.attach


class CapitalStructure:
    """An ordered, gap-free stack of tranches from 0% to 100% of the pool."""

    def __init__(self, tranches: list[Tranche]):
        ts = sorted(tranches, key=lambda t: t.attach)
        if not ts:
            raise ValueError("need at least one tranche")
        problems = []
        if abs(ts[0].attach) > 1e-9:
            problems.append("the first tranche must attach at 0%")
        if abs(ts[-1].detach - 1.0) > 1e-9:
            problems.append("the last tranche must detach at 100%")
        for a, b in zip(ts, ts[1:]):
            if abs(a.detach - b.attach) > 1e-9:
                problems.append(f"gap or overlap between {a.name} ({a.detach:.2%}) and {b.name} ({b.attach:.2%})")
        for t in ts:
            if t.detach <= t.attach:
                problems.append(f"{t.name}: detachment must exceed attachment")
            if t.attach < -1e-12 or t.detach > 1.0 + 1e-12:
                problems.append(f"{t.name}: attachment and detachment must lie between 0% and 100%")
        names = [t.name for t in ts]
        for n in sorted({n for n in names if names.count(n) > 1}):
            problems.append(f"two tranches are both called '{n}'")
        if problems:
            raise ValueError("; ".join(problems))
        self.tranches = ts

    def __iter__(self):
        return iter(self.tranches)

    def __len__(self):
        return len(self.tranches)

    @property
    def senior(self) -> Tranche:
        return self.tranches[-1]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([dict(Tranche=t.name, **{"Attach %": t.attach * 100, "Detach %": t.detach * 100,
                                                       "Coupon %": t.coupon * 100, "Holder": t.holder}) for t in self.tranches])

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "CapitalStructure":
        """Build from a table with columns Tranche, Attach %, Detach %, Coupon % and Holder.

        Rows without both attachment and detachment are ignored (e.g. a blank row
        just added in an editor); missing names, coupons and holders get defaults.
        """
        def val(r, col, default):
            v = r.get(col, default)
            return default if v is None or (not isinstance(v, str) and pd.isna(v)) or v == "" else v

        ts = []
        for i, (_, r) in enumerate(df.iterrows()):
            a, d = val(r, "Attach %", None), val(r, "Detach %", None)
            if a is None or d is None:
                continue
            ts.append(Tranche(str(val(r, "Tranche", f"Tranche {i + 1}")).strip(), float(a) / 100, float(d) / 100,
                              float(val(r, "Coupon %", 0.0)) / 100, str(val(r, "Holder", "Retained"))))
        return cls(ts)

    @classmethod
    def from_sizes(cls, sizes: list[tuple[str, float, float]]) -> "CapitalStructure":
        """Build bottom-up from (name, thickness, coupon) with thickness as a fraction of the pool.
        The last (most senior) tranche is stretched or trimmed to detach at 100%."""
        ts, a = [], 0.0
        for name, size, cpn in sizes:
            ts.append(Tranche(name, a, a + size, cpn))
            a += size
        ts[-1].detach = 1.0
        return cls(ts)

    def to_tuples(self) -> tuple:
        """Hashable representation (handy for caching)."""
        return tuple((t.name, t.attach, t.detach, t.coupon, t.holder) for t in self.tranches)


# Illustrative structures. Coupons are indicative spreads/coupons for the waterfall tool, not market quotes.
PRESETS: dict[str, list[Tranche]] = {
    "CDX IG standard tranches": [
        Tranche("Equity 0-3%", 0.00, 0.03, 0.05), Tranche("Junior mezz 3-7%", 0.03, 0.07, 0.0125),
        Tranche("Senior mezz 7-15%", 0.07, 0.15, 0.0035), Tranche("Super senior 15-100%", 0.15, 1.00, 0.0010)],
    "iTraxx Main standard tranches": [
        Tranche("Equity 0-3%", 0.00, 0.03, 0.05), Tranche("Mezz 3-6%", 0.03, 0.06, 0.012),
        Tranche("Senior mezz 6-12%", 0.06, 0.12, 0.004), Tranche("Senior 12-100%", 0.12, 1.00, 0.001)],
    "Bank SRT (synthetic)": [
        Tranche("First loss 0-1%", 0.000, 0.010, 0.00, "Retained"), Tranche("Mezzanine 1-8.5%", 0.010, 0.085, 0.10, "Sold"),
        Tranche("Senior 8.5-100%", 0.085, 1.000, 0.00, "Retained")],
    "Auto ABS": [
        Tranche("Equity / residual", 0.00, 0.04, 0.00), Tranche("Class C", 0.04, 0.08, 0.065),
        Tranche("Class B", 0.08, 0.15, 0.050), Tranche("Class A", 0.15, 1.00, 0.042)],
    "CLO (broadly syndicated loans)": [
        Tranche("Equity", 0.00, 0.09, 0.00), Tranche("Class E", 0.09, 0.14, 0.095), Tranche("Class D", 0.14, 0.20, 0.075),
        Tranche("Class C", 0.20, 0.27, 0.062), Tranche("Class B", 0.27, 0.38, 0.055), Tranche("Class A", 0.38, 1.00, 0.050)],
}


def preset(name: str) -> CapitalStructure:
    return CapitalStructure([Tranche(t.name, t.attach, t.detach, t.coupon, t.holder) for t in PRESETS[name]])
