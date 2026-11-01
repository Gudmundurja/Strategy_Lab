import math

import numpy as np
import pytest
from scipy.stats import norm

from derivlab.credit import capital
from derivlab.credit.structures import CapitalStructure, Tranche, preset


def test_irb_matches_published_basel_reference_points():
    # Corporate, LGD 45%, M = 2.5: risk weight 92.32% at PD 1% and 14.44% at PD 0.03%
    assert capital.irb_k(0.01, 0.45, 2.5) * 12.5 == pytest.approx(0.9232, abs=5e-5)
    assert capital.irb_k(0.0003, 0.45, 2.5, pd_floor=0.0003) * 12.5 == pytest.approx(0.1444, abs=5e-5)


def test_irb_retail_against_an_independent_formula():
    pd_, lgd, R = 0.01, 0.20, 0.15
    k = lgd * norm.cdf((norm.ppf(pd_) + math.sqrt(R) * norm.ppf(0.999)) / math.sqrt(1 - R)) - pd_ * lgd
    assert capital.irb_k(pd_, lgd, asset_class="retail_mortgage") == pytest.approx(k, rel=1e-10)
    # no maturity adjustment for retail
    assert capital.irb_k(pd_, lgd, 5.0, "retail_mortgage") == capital.irb_k(pd_, lgd, 1.0, "retail_mortgage")


def test_irb_correlations():
    assert capital.irb_correlation(1e-6) == pytest.approx(0.24, abs=1e-4)
    assert capital.irb_correlation(0.5) == pytest.approx(0.12, abs=1e-6)
    assert capital.irb_correlation(0.02, "retail_qrre") == 0.04
    with pytest.raises(ValueError):
        capital.irb_correlation(0.01, "sovereign")


def test_kirb_adds_expected_loss():
    k = capital.irb_k(0.012, 0.4)
    assert capital.k_irb_pool(0.012, 0.4) == pytest.approx(k + 0.012 * 0.4)


def test_sec_irba_p_parameter():
    # wholesale senior granular: p = 0 + 3.56/N - 1.85*KIRB + 0.55*LGD + 0.07*MT
    p = capital.sec_irba_p(0.0747, 0.40, 300, 3.0, retail=False, senior=True)
    assert p == pytest.approx(3.56 / 300 - 1.85 * 0.0747 + 0.55 * 0.40 + 0.07 * 3.0)
    assert capital.sec_irba_p(0.0747, 0.40, 300, 3.0, retail=False, senior=True, sts=True) == 0.3   # halved, floored
    assert capital.sec_irba_p(0.2, 0.1, 300, 1.0, retail=True, senior=True) == 0.3                 # floor
    assert capital.sec_irba_p(0.0747, 0.40, 300, 9.0, False, True) == capital.sec_irba_p(0.0747, 0.40, 300, 5.0, False, True)


def test_ssfa_function_matches_the_formula():
    A, D, K, p = 0.10, 0.20, 0.07, 0.5
    a = -1 / (p * K)
    u, l = D - K, A - K
    expected = (math.exp(a * u) - math.exp(a * l)) / (a * (u - l))
    assert capital.k_ssfa(A, D, K, p) == pytest.approx(expected)


def test_risk_weight_regions_and_floors():
    K = 0.0747
    below = capital.sec_irba_risk_weight(0.0, 0.05, K, 0.4, 300, 3)
    assert below["rw"] == 12.5 and below["case"].startswith("D <= KIRB")
    senior = capital.sec_irba_risk_weight(0.30, 1.0, K, 0.4, 300, 3)
    assert senior["rw"] == pytest.approx(0.15)
    sts = capital.sec_irba_risk_weight(0.30, 1.0, K, 0.4, 300, 3, sts=True)
    assert sts["rw"] == pytest.approx(0.10)
    straddle = capital.sec_irba_risk_weight(0.05, 0.10, K, 0.4, 300, 3)
    assert straddle["case"] == "A < KIRB < D" and 0.15 < straddle["rw"] < 12.5


def test_risk_weight_falls_as_attachment_rises():
    rws = [capital.sec_irba_risk_weight(a, a + 0.02, 0.0747, 0.4, 300, 3, senior=False)["rw"] for a in np.arange(0.06, 0.30, 0.01)]
    assert all(x >= y - 1e-12 for x, y in zip(rws, rws[1:]))


def test_bank_srt_example():
    inp = capital.SRTInputs()
    res = capital.srt_economics(inp, preset("Bank SRT (synthetic)"))
    assert res["rwa_before"] == pytest.approx(res["k"] * 12.5 * inp.ead)
    tbl = res["table"].set_index("Tranche")
    assert tbl.loc["First loss 0-1%", "SEC-IRBA RW %"] == pytest.approx(1250)
    assert tbl.loc["Mezzanine 1-8.5%", "RWA"] == 0                              # sold
    assert 15 <= tbl.loc["Senior 8.5-100%", "SEC-IRBA RW %"] < 25
    assert res["rwa_after"] == pytest.approx(tbl["RWA"].sum())
    assert res["capital_released"] == pytest.approx((res["rwa_before"] - res["rwa_after"]) * inp.capital_ratio)
    assert res["annual_cost"] == pytest.approx(0.10 * 0.075 * inp.ead)
    assert res["cost_of_relief"] == pytest.approx(res["annual_cost"] / res["capital_released"])


def test_selling_everything_releases_everything_and_keeping_everything_nothing():
    inp = capital.SRTInputs()
    sold = CapitalStructure([Tranche("A", 0, 0.1, 0.05, "Sold"), Tranche("B", 0.1, 1.0, 0.01, "Sold")])
    r = capital.srt_economics(inp, sold)
    assert r["rwa_after"] == 0 and r["capital_released"] == pytest.approx(r["rwa_before"] * inp.capital_ratio)
    kept = CapitalStructure([Tranche("A", 0, 0.1), Tranche("B", 0.1, 1.0)])
    r2 = capital.srt_economics(inp, kept)
    assert r2["annual_cost"] == 0 and r2["rwa_after"] <= r2["rwa_before"] and not r2["economic"]


def test_thicker_sold_tranche_releases_more_capital():
    inp = capital.SRTInputs()
    sweep = capital.senior_boundary_sweep(inp, preset("Bank SRT (synthetic)"), np.linspace(0.02, 0.25, 24))
    released = sweep["Capital released"].to_numpy()
    assert np.all(np.diff(released) >= -1e-6)
    assert sweep["Senior RW %"].iloc[-1] == pytest.approx(15.0)
