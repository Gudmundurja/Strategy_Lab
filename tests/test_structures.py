import numpy as np
import pandas as pd
import pytest

from derivlab.credit import design
from derivlab.credit.structures import PRESETS, CapitalStructure, Tranche, preset


@pytest.mark.parametrize("name", list(PRESETS))
def test_presets_are_valid_and_round_trip(name):
    cs = preset(name)
    assert cs.tranches[0].attach == 0 and cs.senior.detach == 1
    again = CapitalStructure.from_frame(cs.to_frame())
    assert [(t.name, t.attach, t.detach, t.holder) for t in again] == [(t.name, t.attach, t.detach, t.holder) for t in cs]
    assert np.allclose([t.coupon for t in again], [t.coupon for t in cs])


@pytest.mark.parametrize("tranches,message", [
    ([Tranche("A", 0.01, 0.1), Tranche("B", 0.1, 1)], "attach at 0%"),
    ([Tranche("A", 0, 0.1), Tranche("B", 0.1, 0.9)], "detach at 100%"),
    ([Tranche("A", 0, 0.1), Tranche("B", 0.12, 1)], "gap or overlap"),
    ([Tranche("A", 0, 0.1), Tranche("B", 0.08, 1)], "gap or overlap"),
    ([Tranche("A", 0, 0.1), Tranche("A", 0.1, 1)], "both called"),
])
def test_invalid_structures_explain_the_problem(tranches, message):
    with pytest.raises(ValueError, match=message):
        CapitalStructure(tranches)


def test_from_frame_tolerates_blank_rows_and_missing_values():
    df = pd.DataFrame([
        {"Tranche": "Equity", "Attach %": 0, "Detach %": 5, "Coupon %": np.nan, "Holder": None},
        {"Tranche": None, "Attach %": 5, "Detach %": 100, "Coupon %": 1.5, "Holder": "Sold"},
        {"Tranche": None, "Attach %": np.nan, "Detach %": np.nan, "Coupon %": np.nan, "Holder": None},
    ])
    cs = CapitalStructure.from_frame(df)
    assert len(cs) == 2
    assert cs.tranches[0].coupon == 0 and cs.tranches[0].holder == "Retained"
    assert cs.senior.name == "Tranche 2" and cs.senior.coupon == pytest.approx(0.015)


def test_from_sizes_stacks_bottom_up():
    cs = CapitalStructure.from_sizes([("Equity", 0.05, 0.0), ("Mezz", 0.10, 0.06), ("Senior", 0.80, 0.03)])
    assert [(t.attach, t.detach) for t in cs] == [(0, 0.05), (0.05, pytest.approx(0.15)), (pytest.approx(0.15), 1.0)]


def test_structuring_assistant_hits_its_targets():
    pd_a, rho, lgd, T = 0.012, 0.2, 0.4, 3.0
    cs = design.size_by_loss_probability(pd_a, rho, lgd, T, [("Mezzanine", 0.30), ("Senior", 0.005)])
    assert [t.holder for t in cs] == ["Retained", "Sold", "Retained"]
    rep = design.tranche_report(pd_a, rho, lgd, T, cs).set_index("Tranche")
    assert rep.loc["Senior", "P(loss) %"] <= 0.5 and rep.loc["Senior", "P(loss) %"] > 0.3
    assert rep.loc["Mezzanine", "P(loss) %"] <= 30
    for t in cs:
        assert round(t.attach / 0.0025, 9) == int(round(t.attach / 0.0025))     # on the 0.25% grid
    assert rep.loc["Mezzanine", "Excess bp"] == pytest.approx(0, abs=0.01)       # coupons set to fair spread


def test_structuring_assistant_rejects_inconsistent_targets():
    with pytest.raises(ValueError):
        design.size_by_loss_probability(0.012, 0.2, 0.4, 3.0, [("Mezzanine", 0.01), ("Senior", 0.05)])


def test_report_is_senior_first_with_finite_pool_option():
    from derivlab.credit import copula
    cs = preset("CDX IG standard tranches")
    losses = copula.simulate_losses(125, 0.02, 0.2, 0.6, 20_000, seed=3)
    rep = design.tranche_report(0.01, 0.2, 0.6, 2.0, cs, losses=losses)
    assert rep["Tranche"].iloc[0] == cs.senior.name
    assert set(["P(loss) %", "P(wipeout) %", "EL %", "Fair spread bp", "Excess bp"]).issubset(rep.columns)
