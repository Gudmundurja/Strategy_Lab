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


def _apptest_reruns_segmented_controls() -> bool:
    """Older Streamlit releases' AppTest cannot rerun a page that has a segmented control (the app itself works)."""
    try:
        at = AppTest.from_string("import streamlit as st\nst.segmented_control('x', ['A', 'B'], key='k')\n").run()
        at.run()
        return not at.exception
    except Exception:
        return False


needs_button_groups = pytest.mark.skipif(not _apptest_reruns_segmented_controls(),
                                         reason="this Streamlit version's AppTest cannot rerun segmented controls")
MC_SECTIONS = ["Exotics", "Variance reduction", "Greeks", "American", "Multi-asset", "Your position", "VaR & ES"]
MC_MODELS = ["Black-Scholes (GBM)", "Heston stochastic vol", "Merton jump-diffusion"]


@pytest.fixture
def mc(lab):
    lab.switch_page("views/montecarlo_lab.py").run()
    assert not lab.exception
    lab.session_state["mc_paths"] = 5_000          # fewer paths keeps the test fast
    lab.run()
    return lab


@needs_button_groups
def test_monte_carlo_page_renders_every_section_under_every_model(mc):
    for model in MC_MODELS:
        mc.selectbox(key="mc_model").set_value(model).run()
        for sec in MC_SECTIONS:
            mc.session_state["mc_section"] = sec
            mc.run()
            assert not mc.exception, (model, sec)


@needs_button_groups
def test_monte_carlo_products_and_options(mc):
    for method in ["Plain", "Control variate", "Sobol + control variate"]:
        mc.selectbox(key="mc_method").set_value(method).run()
        for product in ["European", "Digital (cash-or-nothing)", "Asian, geometric average", "Barrier",
                        "Lookback, floating strike", "Lookback, fixed strike"]:
            mc.selectbox(key="ex_product").set_value(product).run()
            assert not mc.exception, (method, product)
    assert metric(mc, "Formula") == "none"                       # no closed form for a fixed-strike lookback
    mc.selectbox(key="ex_product").set_value("Barrier").run()
    mc.selectbox(key="ex_dir").set_value("up").run()
    mc.number_input(key="ex_barrier").set_value(120.0).run()
    mc.selectbox(key="ex_knock").set_value("in").run()
    assert not mc.exception and metric(mc, "Formula") != "none"
    mc.session_state["mc_section"] = "Multi-asset"
    mc.run()
    mc.radio(key="ma_n").set_value(3).run()
    for product in ["Basket", "Best-of", "Worst-of"]:
        mc.selectbox(key="ma_product").set_value(product).run()
        assert not mc.exception, product
    mc.session_state["mc_section"] = "American"
    mc.run()
    mc.selectbox(key="am_kind").set_value("call").run()
    assert not mc.exception
