import numpy as np
import pandas as pd
import pytest

from derivlab.credit import waterfall
from derivlab.credit.structures import preset

ABS = preset("Auto ABS")


def pool(**kw):
    return waterfall.PoolAssumptions(**{**waterfall.PoolAssumptions().__dict__, **kw})


def balances(res, cs):
    return res.periods[[f"bal: {t.name}" for t in cs]].sum(axis=1).to_numpy()


def test_no_defaults_means_everyone_is_repaid():
    res = waterfall.run_waterfall(pool(cdr=0.0), ABS)
    s = res.summary
    assert np.allclose(s["Principal loss %"], 0)
    assert np.allclose(s["Principal received"], s["Original"])
    assert np.allclose(s["Interest shortfall"], 0)


def test_cash_in_equals_cash_out():
    res = waterfall.run_waterfall(pool(cdr=0.06), ABS)
    p, s = res.periods, res.summary
    cash_in = p[["interest", "scheduled", "prepayments", "recoveries"]].to_numpy().sum()
    cash_out = p["fees"].sum() + s["Interest received"].sum() + s["Principal received"].sum() + s["Residual cash"].sum()
    assert cash_out == pytest.approx(cash_in, rel=1e-10)


@pytest.mark.parametrize("cdr", [0.0, 0.05, 0.25])
def test_notes_are_backed_by_collateral_every_month(cdr):
    # outstanding notes = performing loans + recoveries still to come
    res = waterfall.run_waterfall(pool(cdr=cdr, severity=0.6), ABS)
    p = res.periods
    pending = (p["defaults"] * (1 - 0.6)).cumsum() - p["recoveries"].cumsum()
    assert np.allclose(balances(res, ABS), p["performing"] + pending, atol=1e-4)


def test_losses_hit_the_bottom_of_the_stack_first():
    loss = waterfall.run_waterfall(pool(cdr=0.15), ABS).summary["Principal loss %"].to_numpy()   # senior first
    assert np.all(np.diff(loss) >= -1e-9)
    assert loss[-1] == pytest.approx(100.0)


def test_sequential_pay_retires_the_senior_first():
    wal = waterfall.run_waterfall(pool(), ABS).summary["WAL (yrs)"].to_numpy()                    # senior first
    assert np.all(np.diff(wal) > 0)


def test_faster_prepayments_shorten_the_senior():
    slow = waterfall.run_waterfall(pool(cpr=0.0), ABS).summary["WAL (yrs)"].iloc[0]
    fast = waterfall.run_waterfall(pool(cpr=0.3), ABS).summary["WAL (yrs)"].iloc[0]
    assert fast < slow


def test_excess_spread_is_the_first_line_of_defence():
    rich = waterfall.run_waterfall(pool(cdr=0.02, wac=0.085), ABS).summary.set_index("Tranche")
    thin = waterfall.run_waterfall(pool(cdr=0.02, wac=0.04), ABS).summary.set_index("Tranche")
    assert rich.loc["Equity / residual", "Principal loss %"] == pytest.approx(0)
    assert thin.loc["Equity / residual", "Principal loss %"] > 0


def test_bullet_pool_repays_at_maturity():
    res = waterfall.run_waterfall(pool(amortization="bullet", cpr=0.0, cdr=0.0, term_months=36), ABS)
    p = res.periods
    assert p.loc[p["month"] < 36, "scheduled"].sum() == 0
    assert p.loc[p["month"] == 36, "scheduled"].iloc[0] == pytest.approx(500_000_000)


def test_breakeven_cdr_is_the_edge_of_loss():
    base = pool()
    be = waterfall.breakeven_cdr(base, ABS, "Class B")
    assert 0 < be < 1
    at = waterfall.tranche_losses(pool(cdr=be), ABS)["Class B"]
    above = waterfall.tranche_losses(pool(cdr=be + 0.01), ABS)["Class B"]
    assert at <= 0.01 < above
    assert waterfall.breakeven_cdr(base, ABS, "Class A") > be


def test_stress_table_losses_rise_with_defaults():
    table = waterfall.stress_table(pool(), ABS)
    assert isinstance(table, pd.DataFrame) and list(table.index)[0] == "Class A"
    assert np.all(np.diff(table.to_numpy(), axis=1) >= -1e-9)
