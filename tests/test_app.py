"""Smoke tests for the Streamlit app: every page, template and strategy renders without errors."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from derivlab import presets  # noqa: E402
from derivlab.credit.structures import PRESETS  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


@pytest.fixture
def lab():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    return at


def test_every_strategy_renders(lab):
    for s in presets.STRATEGIES:
        lab.selectbox(key="strategy").set_value(s.id).run()
        assert not lab.exception, s.id
        assert len(lab.metric) >= 12


def test_contract_size_scales_the_numbers(lab):
    base = float(metric(lab, "Net debit").replace(",", ""))
    lab.selectbox(key="mult_choice").set_value("Custom").run()
    lab.number_input(key="mult_custom").set_value(25.0).run()
    lab.number_input(key="lots").set_value(3).run()
    assert not lab.exception
    assert float(metric(lab, "Net debit").replace(",", "")) == pytest.approx(base / 100 * 25 * 3, rel=1e-4)


def test_smiles_and_sticky_rules_render(lab):
    for shape in ["Equity index put skew", "FX-style smile", "Custom", "Flat (textbook Black-Scholes)"]:
        lab.selectbox(key="smile_shape").set_value(shape).run()
        assert not lab.exception, shape
    lab.radio(key="sticky").set_value("Sticky moneyness").run()
    lab.radio(key="basis").set_value("Stress scenario").run()
    lab.selectbox(key="strategy").set_value("short-straddle").run()
    assert not lab.exception


def test_credit_lab_templates_and_options(lab):
    lab.switch_page("views/credit_lab.py").run()
    assert not lab.exception
    for name in PRESETS:
        lab.selectbox(key="cs_preset").set_value(name).run()
        assert not lab.exception, name
    lab.radio(key="cr_mode").set_value("Finite pool (Monte Carlo)").run()
    lab.toggle(key="cr_lumpy").set_value(True).run()
    lab.toggle(key="srt_sts").set_value(True).run()
    lab.selectbox(key="srt_ac").set_value("Retail: residential mortgages").run()
    lab.selectbox(key="wf_amort").set_value("Bullet").run()
    assert not lab.exception


def test_structuring_assistant_replaces_the_structure(lab):
    lab.switch_page("views/credit_lab.py").run()
    before = metric(lab, "KIRB")
    next(b for b in lab.button if b.label == "Use this structure").click().run()
    assert not lab.exception
    assert metric(lab, "KIRB") == before                        # same pool...
    assert metric(lab, "RWA after") != "610.6m"                 # ...but a different structure
