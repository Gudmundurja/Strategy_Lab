import math

import numpy as np
import pytest

from derivlab.credit import copula
from derivlab.credit.structures import preset

PD, RHO, LGD = 0.03, 0.20, 0.60


def test_quantile_inverts_cdf():
    for alpha in (0.5, 0.9, 0.99, 0.999):
        x = copula.vasicek_quantile(alpha, PD, RHO, LGD)
        assert copula.vasicek_cdf(x, PD, RHO, LGD) == pytest.approx(alpha, abs=1e-10)


def test_cdf_is_a_distribution_and_pdf_integrates_to_one():
    x = np.linspace(1e-7, LGD - 1e-7, 200_001)
    cdf = copula.vasicek_cdf(x, PD, RHO, LGD)
    assert np.all(np.diff(cdf) >= -1e-15)
    assert copula.vasicek_cdf(0.0, PD, RHO, LGD) == 0.0 and copula.vasicek_cdf(LGD, PD, RHO, LGD) == 1.0
    pdf = copula.vasicek_pdf(x, PD, RHO, LGD)
    assert float(np.sum((pdf[1:] + pdf[:-1]) * np.diff(x)) / 2) == pytest.approx(1.0, abs=1e-5)


def test_mean_loss_is_pd_times_lgd():
    # E[L] = integral of P(L > x) over [0, LGD]
    assert copula.tranche_el(PD, RHO, LGD, 0.0, LGD) * LGD == pytest.approx(PD * LGD, rel=1e-8)


@pytest.mark.parametrize("name", ["CDX IG standard tranches", "Bank SRT (synthetic)", "CLO (broadly syndicated loans)"])
def test_tranche_losses_add_up_to_the_pool(name):
    cs = preset(name)
    total = sum(copula.tranche_el(PD, RHO, LGD, t.attach, t.detach) * t.thickness for t in cs)
    assert total == pytest.approx(PD * LGD, rel=1e-6)


def test_correlation_moves_risk_from_equity_to_senior():
    eq = [copula.tranche_el(PD, r, LGD, 0.0, 0.03) for r in (0.05, 0.2, 0.5)]
    sen = [copula.tranche_el(PD, r, LGD, 0.15, 1.0) for r in (0.05, 0.2, 0.5)]
    assert eq[0] > eq[1] > eq[2]
    assert sen[0] < sen[1] < sen[2]


def test_tranche_beyond_max_loss_never_loses():
    assert copula.tranche_el(PD, RHO, 0.4, 0.5, 1.0) == 0.0
    assert copula.tranche_hit_prob(PD, RHO, 0.4, 0.5) == 0.0
    with pytest.raises(ValueError):
        copula.tranche_el(PD, RHO, LGD, 0.1, 0.1)
    with pytest.raises(ValueError):
        copula.vasicek_cdf(0.1, PD, 1.0, LGD)


def test_monte_carlo_converges_to_the_large_pool():
    L = copula.simulate_losses(5000, PD, RHO, LGD, 200_000, seed=1)
    assert L.mean() == pytest.approx(PD * LGD, rel=0.02)
    mc = copula.tranche_metrics_from_losses(L, 0.03, 0.07)
    lhp = copula.tranche_metrics(PD, RHO, LGD, 0.03, 0.07)
    assert mc["el"] == pytest.approx(lhp["el"], abs=0.004)
    assert mc["p_hit"] == pytest.approx(lhp["p_hit"], abs=0.004)


def test_lumpy_pool_has_the_same_mean_but_fatter_tail():
    rng = np.random.default_rng(3)
    lumpy = copula.simulate_losses(50, PD, RHO, LGD, 40_000, exposures=rng.lognormal(0, 1.2, 50), seed=2)
    even = copula.simulate_losses(50, PD, RHO, LGD, 40_000, seed=2)
    assert lumpy.mean() == pytest.approx(PD * LGD, rel=0.05)
    assert np.quantile(lumpy, 0.999) > np.quantile(even, 0.999) * 0.95


def test_attachment_solvers_hit_their_targets():
    a = copula.attachment_for_target_hit_prob(0.01, PD, RHO, LGD)
    assert copula.tranche_hit_prob(PD, RHO, LGD, a) == pytest.approx(0.01, abs=1e-9)
    a2 = copula.attachment_for_target_el(0.001, PD, RHO, LGD)
    assert copula.tranche_el(PD, RHO, LGD, a2, 1.0) == pytest.approx(0.001, rel=1e-4)


def test_fair_spreads_fall_with_seniority():
    spreads = [copula.tranche_fair_spread(0.01, 0.25, 0.6, t.attach, t.detach, 5.0)["spread"]
               for t in preset("CDX IG standard tranches")]
    assert all(s > 0 for s in spreads)
    assert spreads == sorted(spreads, reverse=True)


def test_whole_pool_spread_obeys_the_credit_triangle():
    # spread ~= hazard rate x LGD when the "tranche" is the whole pool
    pd_a, lgd = 0.02, 0.6
    s = copula.tranche_fair_spread(pd_a, 0.3, lgd, 0.0, 1.0, 5.0, 0.03)["spread"]
    assert s == pytest.approx(-math.log(1 - pd_a) * lgd, rel=0.03)


def test_cumulative_pd():
    assert copula.cumulative_pd(0.02, 1.0) == pytest.approx(0.02)
    assert copula.cumulative_pd(0.02, 5.0) == pytest.approx(1 - 0.98 ** 5)
    assert isinstance(copula.cumulative_pd(0.02, 3.0), float)
