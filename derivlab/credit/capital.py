"""Regulatory capital for loan pools and securitisation tranches (simplified).

* :func:`irb_k` - Basel IRB capital requirement K (unexpected loss) per unit of EAD
  for corporate and retail exposures.
* :func:`sec_irba_risk_weight` - the SEC-IRBA risk weight of a securitisation
  position (Basel CRE44 / EU CRR Article 259), with the STS treatment of Article 260.
* :func:`srt_economics` - capital relief and its cost for a synthetic risk transfer.

For learning and first-pass structuring only. A real SRT needs the full rulebook:
maturity rules, the significant-risk-transfer tests, the maximum capital cap,
eligibility of the protection, supervisory approval and so on.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
from scipy.special import ndtr, ndtri

from .structures import CapitalStructure, Tranche

# (A, B, C, D, E) supervisory parameters for p, CRR Article 259(3)
_SEC_IRBA_COEFS = {
    ("wholesale", True, True): (0.00, 3.56, -1.85, 0.55, 0.07),   # senior, granular (N >= 25)
    ("wholesale", True, False): (0.11, 2.61, -2.91, 0.68, 0.07),  # senior, non-granular
    ("wholesale", False, True): (0.16, 2.87, -1.03, 0.21, 0.07),  # non-senior, granular
    ("wholesale", False, False): (0.22, 2.35, -2.46, 0.48, 0.07), # non-senior, non-granular
    ("retail", True, None): (0.00, 0.00, -7.48, 0.71, 0.24),
    ("retail", False, None): (0.00, 0.00, -5.78, 0.55, 0.27),
}

ASSET_CLASSES = ("corporate", "retail_mortgage", "retail_qrre", "retail_other")


def irb_correlation(pd_: float, asset_class: str = "corporate") -> float:
    if asset_class == "corporate":
        w = (1 - math.exp(-50 * pd_)) / (1 - math.exp(-50))
        return 0.12 * w + 0.24 * (1 - w)
    if asset_class == "retail_mortgage":
        return 0.15
    if asset_class == "retail_qrre":
        return 0.04
    if asset_class == "retail_other":
        w = (1 - math.exp(-35 * pd_)) / (1 - math.exp(-35))
        return 0.03 * w + 0.16 * (1 - w)
    raise ValueError(f"asset_class must be one of {ASSET_CLASSES}")


def irb_k(pd_: float, lgd: float, maturity: float = 2.5, asset_class: str = "corporate",
          pd_floor: float = 0.0005, scaling: float = 1.0) -> float:
    """IRB capital requirement K per unit EAD (unexpected loss only, excludes EL)."""
    p = max(pd_, pd_floor)
    if p >= 1.0:
        return 0.0
    R = irb_correlation(p, asset_class)
    cond = ndtr((ndtri(p) + math.sqrt(R) * ndtri(0.999)) / math.sqrt(1 - R))
    k = lgd * cond - p * lgd
    if asset_class == "corporate":
        b = (0.11852 - 0.05478 * math.log(p)) ** 2
        m = min(max(maturity, 1.0), 5.0)
        k *= (1 + (m - 2.5) * b) / (1 - 1.5 * b)
    return max(k, 0.0) * scaling


def k_irb_pool(pd_: float, lgd: float, maturity: float = 2.5, asset_class: str = "corporate",
               pd_floor: float = 0.0005, scaling: float = 1.0) -> float:
    """KIRB for SEC-IRBA: capital including the expected-loss portion, per unit of pool EAD."""
    return irb_k(pd_, lgd, maturity, asset_class, pd_floor, scaling) + max(pd_, pd_floor) * lgd


def sec_irba_p(kirb: float, lgd: float, n_eff: float, mt: float, retail: bool, senior: bool, sts: bool = False) -> float:
    mt = min(max(mt, 1.0), 5.0)
    if retail:
        A, B, C, D, E = _SEC_IRBA_COEFS[("retail", senior, None)]
    else:
        A, B, C, D, E = _SEC_IRBA_COEFS[("wholesale", senior, n_eff >= 25)]
    raw = A + B / max(n_eff, 1.0) + C * kirb + D * lgd + E * mt
    return max(0.3, 0.5 * raw if sts else raw)


def k_ssfa(attach: float, detach: float, kirb: float, p: float) -> float:
    a = -1.0 / (p * kirb)
    u = detach - kirb
    l = max(attach - kirb, 0.0)
    if u <= l:
        return 1.0
    return (math.exp(a * u) - math.exp(a * l)) / (a * (u - l))


def sec_irba_risk_weight(attach: float, detach: float, kirb: float, lgd: float, n_eff: float, mt: float,
                         retail: bool = False, senior: bool | None = None, sts: bool = False) -> dict:
    """Risk weight (decimal, 0.15 = 15%) of the [attach, detach] position under SEC-IRBA."""
    senior = (detach >= 1.0 - 1e-12) if senior is None else senior
    floor = 0.10 if (sts and senior) else 0.15
    if detach <= kirb:
        return dict(rw=12.5, p=float("nan"), kssfa=1.0, floor=floor, senior=senior, case="D <= KIRB: 1,250%")
    p = sec_irba_p(kirb, lgd, n_eff, mt, retail, senior, sts)
    ks = k_ssfa(attach, detach, kirb, p)
    if attach >= kirb:
        rw = 12.5 * ks
        case = "A >= KIRB"
    else:
        rw = ((kirb - attach) / (detach - attach)) * 12.5 + ((detach - kirb) / (detach - attach)) * 12.5 * ks
        case = "A < KIRB < D"
    return dict(rw=min(max(rw, floor), 12.5), p=p, kssfa=ks, floor=floor, senior=senior, case=case)


@dataclass
class SRTInputs:
    ead: float = 2_000_000_000.0
    pd: float = 0.012
    lgd: float = 0.40
    maturity: float = 2.5
    asset_class: str = "corporate"
    n_eff: float = 300.0
    tranche_maturity: float = 3.0
    sts: bool = False
    capital_ratio: float = 0.12        # capital held per unit RWA (CET1 target incl. buffers)
    cost_of_equity: float = 0.12
    pd_floor: float = 0.0005
    scaling: float = 1.0


def srt_economics(inp: SRTInputs, structure: CapitalStructure) -> dict:
    """Capital before/after a synthetic risk transfer and the cost of the relief.

    Retained tranches are risk-weighted with SEC-IRBA; tranches marked 'Sold' are
    assumed fully protected by eligible (e.g. cash-collateralised) protection at 0%.
    The post-transfer RWA is capped at the pre-transfer RWA.
    """
    retail = inp.asset_class != "corporate"
    k = irb_k(inp.pd, inp.lgd, inp.maturity, inp.asset_class, inp.pd_floor, inp.scaling)
    kirb = k_irb_pool(inp.pd, inp.lgd, inp.maturity, inp.asset_class, inp.pd_floor, inp.scaling)
    rwa_before = k * 12.5 * inp.ead
    rows, rwa_after, cost = [], 0.0, 0.0
    for t in structure:
        notional = t.thickness * inp.ead
        res = sec_irba_risk_weight(t.attach, t.detach, kirb, inp.lgd, inp.n_eff, inp.tranche_maturity, retail,
                                   senior=t is structure.senior, sts=inp.sts)
        sold = t.holder.lower().startswith("sold")
        rwa = 0.0 if sold else res["rw"] * notional
        rwa_after += rwa
        if sold:
            cost += t.coupon * notional
        rows.append({"Tranche": t.name, "Attach %": t.attach * 100, "Detach %": t.detach * 100, "Holder": "Sold" if sold else "Retained",
                     "Notional": notional, "SEC-IRBA RW %": res["rw"] * 100, "p": res["p"], "Case": res["case"],
                     "RWA": rwa, "Annual premium": t.coupon * notional if sold else 0.0})
    capped = min(rwa_after, rwa_before)
    released = (rwa_before - capped) * inp.capital_ratio
    cost_of_relief = cost / released if released > 0 else float("nan")
    return dict(k=k, kirb=kirb, rw_pool=k * 12.5, rwa_before=rwa_before, rwa_after_uncapped=rwa_after,
                rwa_after=capped, rwa_reduction=rwa_before - capped, capital_released=released,
                annual_cost=cost, cost_of_relief=cost_of_relief,
                economic=bool(cost_of_relief < inp.cost_of_equity) if released > 0 else False,
                table=pd.DataFrame(rows))


def senior_boundary_sweep(inp: SRTInputs, structure: CapitalStructure, points) -> pd.DataFrame:
    """Re-run :func:`srt_economics` while moving the boundary between the senior tranche
    and the tranche just below it (typically the sold mezzanine).

    Shows the core SRT trade-off: a thicker sold tranche releases more capital but
    costs more premium, and the senior risk-weight floor caps the benefit.
    """
    ts = structure.tranches
    if len(ts) < 2:
        raise ValueError("need at least two tranches")
    lo = ts[-2].attach
    rows = []
    for d in points:
        if not (lo + 1e-6 < d < 1.0 - 1e-6):
            continue
        new = [Tranche(t.name, t.attach, t.detach, t.coupon, t.holder) for t in ts]
        new[-2].detach = d
        new[-1].attach = d
        r = srt_economics(inp, CapitalStructure(new))
        senior_rw = float(r["table"]["SEC-IRBA RW %"].iloc[-1])
        rows.append({"Boundary %": d * 100, "RWA after": r["rwa_after"], "Capital released": r["capital_released"],
                     "Annual cost": r["annual_cost"], "Cost of relief %": r["cost_of_relief"] * 100,
                     "Senior RW %": senior_rw})
    return pd.DataFrame(rows)
