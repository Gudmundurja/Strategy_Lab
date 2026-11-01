import math

import numpy as np
import pytest

from derivlab import Leg, Market, Position, VolModel, presets
from derivlab.montecarlo import (GBM, METHODS, Asian, Barrier, Basket, Digital, European, Heston, Lookback, Merton,
                                 Rainbow, Spread, american, benchmark, compare_methods, convergence, greeks, multiasset,
                                 portfolio, price_option)
from derivlab.montecarlo import analytic as A
from derivlab.montecarlo.models import bridge_normals
from derivlab.montecarlo.rng import NormalSource, next_pow2
from derivlab.pricing import binomial_price, bs_greeks, bs_price, implied_vol

S0, R, Q, T = 100.0, 0.04, 0.01, 0.5
G = GBM(0.25)


def within(res, ref, k=4.0, abs_tol=0.0):
    assert abs(res.price - ref) <= k * res.stderr + abs_tol, (res.price, res.stderr, ref)


# ---------------------------------------------------------------- random numbers
def test_antithetic_draws_come_in_mirror_pairs():
    z = NormalSource(3, "antithetic", 1).normals(8)
    assert np.allclose(z[:4], -z[4:])
    with pytest.raises(ValueError):
        NormalSource(3, "antithetic", 1).normals(7)


def test_sobol_points_are_standard_normal():
    z = NormalSource(4, "sobol", 1).normals(4096)
    assert np.abs(z.mean(axis=0)).max() < 0.01
    assert np.abs(z.std(axis=0) - 1).max() < 0.02
    assert next_pow2(1000) == 1024 and next_pow2(1024) == 1024


def test_brownian_bridge_keeps_the_distribution_and_sets_the_end_point_first():
    rng = np.random.default_rng(0)
    dt = np.array([0.1, 0.2, 0.1, 0.3, 0.1, 0.2])
    Z = rng.standard_normal((200_000, 6))
    X = bridge_normals(Z, dt)
    assert np.abs(np.cov(X.T) - np.eye(6)).max() < 0.012
    assert np.allclose((X * np.sqrt(dt)).sum(axis=1), math.sqrt(dt.sum()) * Z[:, 0])


# ---------------------------------------------------------------- formulas
@pytest.mark.parametrize("K", [70.0, 100.0, 135.0])
def test_characteristic_function_pricer_reproduces_black_scholes(K):
    for kind in ("call", "put"):
        cf = A.cf_price(kind, S0, K, T, R, Q, lambda u: A.bs_cf(u, T, 0.25))
        assert cf == pytest.approx(bs_price(kind, S0, K, T, R, Q, 0.25), abs=1e-8)


@pytest.mark.parametrize("K", [75.0, 100.0, 125.0])
def test_merton_series_matches_its_characteristic_function(K):
    m = Merton(0.2, 0.8, -0.15, 0.2)
    series = A.merton_price("put", S0, K, T, R, Q, m.sigma, m.lam, m.mu_j, m.delta)
    integral = A.cf_price("put", S0, K, T, R, Q, lambda u: A.merton_cf(u, T, m.sigma, m.lam, m.mu_j, m.delta))
    assert series == pytest.approx(integral, abs=1e-8)
    assert A.merton_price("call", S0, K, T, R, Q, 0.2, 0.0, -0.1, 0.1) == pytest.approx(bs_price("call", S0, K, T, R, Q, 0.2))


def test_heston_formula_limits_and_parity():
    assert A.heston_cf(-1j, 1.0, 0.04, 2.0, 0.06, 0.5, -0.7) == pytest.approx(1.0)
    almost_bs = A.heston_price("call", S0, 105, T, R, Q, 0.0625, 2.0, 0.0625, 1e-4, -0.5)
    assert almost_bs == pytest.approx(bs_price("call", S0, 105, T, R, Q, 0.25), abs=1e-4)
    h = (0.04, 1.5, 0.09, 0.6, -0.5)
    c, p = A.heston_price("call", S0, 95, 1.0, R, Q, *h), A.heston_price("put", S0, 95, 1.0, R, Q, *h)
    assert c - p == pytest.approx(S0 * math.exp(-Q) - 95 * math.exp(-R), abs=1e-8)


def test_negative_spot_vol_correlation_gives_a_put_skew():
    ivs = [implied_vol("put" if K < S0 else "call", A.heston_price("put" if K < S0 else "call", S0, K, T, R, Q,
                                                                   0.0625, 2.0, 0.0625, 0.45, -0.7), S0, K, T, R, Q)
           for K in (80.0, 100.0, 120.0)]
    assert ivs[0] > ivs[1] > ivs[2]


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("direction,H", [("down", 90.0), ("up", 112.0)])
@pytest.mark.parametrize("K", [95.0, 105.0])
def test_barrier_formulas_knock_in_plus_knock_out_is_vanilla(kind, direction, H, K):
    vanilla = bs_price(kind, S0, K, T, R, Q, 0.25)
    for dt in (None, T / 126):
        knock_in = A.barrier_price(kind, S0, K, H, T, R, Q, 0.25, direction, "in", dt)
        knock_out = A.barrier_price(kind, S0, K, H, T, R, Q, 0.25, direction, "out", dt)
        assert knock_in >= 0 and knock_out >= 0
        assert knock_in + knock_out == pytest.approx(vanilla)


def test_barrier_already_breached_today():
    assert A.barrier_price("call", S0, 100, 101, T, R, Q, 0.25, "down", "out") == 0.0
    assert A.barrier_price("call", S0, 100, 101, T, R, Q, 0.25, "down", "in") == pytest.approx(bs_price("call", S0, 100, T, R, Q, 0.25))


def test_closed_form_special_cases():
    assert A.geometric_asian_price("call", S0, 100, T, R, Q, 0.25, 1) == pytest.approx(bs_price("call", S0, 100, T, R, Q, 0.25))
    assert A.kirk_spread_price("call", 100, 95, 0.0, T, R, 0.01, 0.02, 0.25, 0.3, 0.4) == pytest.approx(
        A.margrabe_price(100, 95, T, 0.01, 0.02, 0.25, 0.3, 0.4), rel=1e-10)
    assert A.geometric_basket_price("call", [100.0], 100, T, R, [Q], [0.25], [[1.0]], [1.0]) == pytest.approx(
        bs_price("call", S0, 100, T, R, Q, 0.25))
    cmax = A.two_asset_extreme_price("call", "max", 100, 95, 100, T, R, 0.01, 0.02, 0.25, 0.3, 0.4)
    cmin = A.two_asset_extreme_price("call", "min", 100, 95, 100, T, R, 0.01, 0.02, 0.25, 0.3, 0.4)
    assert cmax + cmin == pytest.approx(bs_price("call", 100, 100, T, R, 0.01, 0.25) + bs_price("call", 95, 100, T, R, 0.02, 0.3))
    assert A.lookback_floating_price("call", S0, T, R, Q, 0.25) > bs_price("call", S0, S0, T, R, Q, 0.25)


# ---------------------------------------------------------------- simulation against formulas
def test_simulated_prices_are_martingales():
    for model in (G, Heston(), Merton()):
        times = np.array([0.25, 0.5, 1.0])
        st = model.simulate(S0, R, Q, times, 40_000, NormalSource(model.dims(times), "antithetic", 9))
        disc = st["S"][:, -1] * math.exp(-(R - Q) * 1.0)
        assert disc.mean() == pytest.approx(S0, rel=0.01)


@pytest.mark.parametrize("method", list(METHODS))
def test_every_method_prices_a_european_call(method):
    sampling, cv = METHODS[method]
    res = price_option(European("call", 105), G, S0, R, Q, T, 1, 20_000, sampling, "auto" if cv else None, 3)
    within(res, bs_price("call", S0, 105, T, R, Q, 0.25))


def test_digital_and_geometric_asian_match_their_formulas():
    within(price_option(Digital("put", 95), G, S0, R, Q, T, 1, 50_000, "antithetic", None, 4),
           A.digital_price("put", S0, 95, T, R, Q, 0.25))
    within(price_option(Asian("call", 100, "geometric"), G, S0, R, Q, T, 52, 50_000, "antithetic", None, 4),
           A.geometric_asian_price("call", S0, 100, T, R, Q, 0.25, 52))


@pytest.mark.parametrize("product", [Barrier("call", 100, 90, "down", "out"), Barrier("put", 100, 112, "up", "out"),
                                     Barrier("put", 105, 90, "down", "in")], ids=lambda p: p.label())
def test_discrete_barriers_match_the_corrected_formula(product):
    res = price_option(product, G, S0, R, Q, T, 126, 50_000, "pseudo", "auto", 5)
    ref = benchmark(product, G, S0, R, Q, T, 126)[1]
    within(res, ref, abs_tol=0.01 * ref)               # the shift is an approximation


def test_knock_in_plus_knock_out_is_vanilla_path_by_path():
    args = (G, S0, R, Q, T, 52, 10_000, "pseudo", None, 8)
    ins = price_option(Barrier("call", 100, 95, "down", "in"), *args)
    outs = price_option(Barrier("call", 100, 95, "down", "out"), *args)
    vanilla = price_option(Barrier("call", 100, 1e-9, "down", "out"), *args)    # a barrier that is never touched
    assert np.allclose(ins.units_y + outs.units_y, vanilla.units_y)
    assert vanilla.price == pytest.approx(bs_price("call", S0, 100, T, R, Q, 0.25), abs=4 * vanilla.stderr)


def test_arithmetic_asian_sits_between_geometric_and_vanilla():
    args = (G, S0, R, Q, T, 52, 20_000, "pseudo", None, 2)
    arith = price_option(Asian("call", 100), *args)
    geo = price_option(Asian("call", 100, "geometric"), *args)
    assert np.all(arith.units_y >= geo.units_y - 1e-12)   # average >= geometric average on every path
    assert arith.price < bs_price("call", S0, 100, T, R, Q, 0.25)


def test_discrete_lookback_creeps_up_to_the_continuous_formula():
    cont = A.lookback_floating_price("put", S0, T, R, Q, 0.25)
    coarse = price_option(Lookback("put"), G, S0, R, Q, T, 26, 20_000, "sobol", None, 1).price
    fine = price_option(Lookback("put"), G, S0, R, Q, T, 1000, 20_000, "sobol", None, 1).price
    assert coarse < fine < cont
    assert fine == pytest.approx(cont, rel=0.03)


@pytest.mark.parametrize("model", [Heston(), Heston(0.04, 1.5, 0.09, 0.6, -0.5), Merton(), Merton(0.15, 1.0, 0.05, 0.25)],
                         ids=["heston", "heston-b", "merton", "merton-b"])
@pytest.mark.parametrize("K", [85.0, 100.0, 115.0])
def test_heston_and_merton_simulations_match_their_formulas(model, K):
    kind = "put" if K < S0 else "call"
    res = price_option(European(kind, K), model, S0, R, Q, T, 1, 40_000, "sobol", "auto", 6)
    ref = benchmark(European(kind, K), model, S0, R, Q, T, 1)[1]
    within(res, ref, abs_tol=0.005 * ref + 0.005)      # Euler steps add a small bias under Heston


def test_heston_variance_reverts_to_its_long_run_level():
    h = Heston(0.04, 1.5, 0.09, 0.6, -0.5)
    st = h.simulate(S0, R, Q, np.array([1.0]), 60_000, NormalSource(h.dims([1.0]), "pseudo", 3))
    assert st["V"][:, -1].mean() == pytest.approx(h.theta + (h.v0 - h.theta) * math.exp(-h.kappa), rel=0.02)
    assert np.all(st["V"] >= 0)


def test_variance_reduction_helps_where_it_should():
    table = compare_methods(Asian("call", 100), G, S0, R, Q, T, 52, 8_000, 3).set_index("Method")
    assert table.loc["Control variate", "Variance reduction"] > 100          # geometric Asian control
    assert table.loc["Sobol (quasi-random)", "Variance reduction"] > 20      # with the Brownian bridge
    assert table.loc["Antithetic", "Variance reduction"] > 1.2


def test_convergence_table():
    res = price_option(European("call", 100), G, S0, R, Q, T, 1, 20_000, "pseudo", None, 1)
    conv = convergence(res)
    assert conv["paths"].iloc[-1] == 20_000
    assert conv["estimate"].iloc[-1] == pytest.approx(res.price)
    assert conv["stderr"].iloc[-1] < conv["stderr"].iloc[0] / 5
    sob = convergence(price_option(European("call", 100), G, S0, R, Q, T, 1, 20_000, "sobol", None, 1))
    assert sob["paths"].iloc[-1] == 16 * 2048


# ---------------------------------------------------------------- multi-asset
def test_correlation_matrix_must_be_valid():
    with pytest.raises(ValueError):
        multiasset.correlation_matrix(3, {(0, 1): 0.9, (0, 2): 0.9, (1, 2): -0.9})
    assert multiasset.correlation_matrix(2, {(0, 1): 0.3})[0, 1] == 0.3


@pytest.mark.parametrize("product", [Spread("call", 0.0), Rainbow("call", 100, "max"), Rainbow("call", 100, "min"),
                                     Rainbow("put", 100, "max"), Rainbow("put", 100, "min")], ids=lambda p: p.label())
def test_two_asset_options_match_exact_formulas(product):
    S, q, sig = [100.0, 95.0], [0.01, 0.02], [0.25, 0.3]
    C = multiasset.correlation_matrix(2, {(0, 1): 0.4})
    res = multiasset.price_multi(product, S, R, q, sig, C, T, 100_000, "antithetic", None, 11)
    within(res, multiasset.benchmark(product, S, R, q, sig, C, T)[1])


def test_basket_control_variate_and_levy_approximation():
    S, q, sig = [100.0, 100.0, 100.0], [0.0, 0.01, 0.02], [0.2, 0.3, 0.25]
    C = multiasset.correlation_matrix(3, {(0, 1): 0.5, (0, 2): 0.3, (1, 2): 0.4})
    prod = Basket("call", 100, (1 / 3, 1 / 3, 1 / 3))
    cv = multiasset.price_multi(prod, S, R, q, sig, C, T, 20_000, "pseudo", "auto", 2)
    plain = multiasset.price_multi(prod, S, R, q, sig, C, T, 20_000, "pseudo", None, 2)
    assert cv.stderr < plain.stderr / 5
    assert abs(cv.price - plain.price) < 4 * plain.stderr
    assert multiasset.benchmark(prod, S, R, q, sig, C, T)[1] == pytest.approx(cv.price, rel=0.01)


def test_correlation_moves_basket_and_spread_in_opposite_directions():
    S, q, sig = [100.0, 100.0], [0.0, 0.0], [0.25, 0.25]
    basket = multiasset.correlation_sweep(Basket("call", 100, (0.5, 0.5)), S, R, q, sig, T, [-0.5, 0.0, 0.5], 20_000, 1)
    spread = multiasset.correlation_sweep(Spread("call", 0.0), S, R, q, sig, T, [-0.5, 0.0, 0.5], 20_000, 1)
    assert np.all(np.diff(basket["price"]) > 0) and np.all(np.diff(spread["price"]) < 0)


# ---------------------------------------------------------------- American
def test_longstaff_schwartz_matches_the_binomial_tree():
    res = american.lsm_price("put", 100, 100, 1.0, 0.05, 0.0, GBM(0.2), 50, 50_000, 3, 3)
    tree = binomial_price("put", 100, 100, 1.0, 0.05, 0.0, 0.2, 1000, True)
    assert res["price"] == pytest.approx(tree, rel=0.02)
    assert res["price"] > res["european"] + 3 * res["european_se"]
    assert 0 < res["early_share"] < 1


def test_american_call_without_dividends_is_european():
    res = american.lsm_price("call", 100, 100, 1.0, 0.05, 0.0, GBM(0.2), 25, 20_000, 3, 4)
    assert res["price"] == pytest.approx(bs_price("call", 100, 100, 1.0, 0.05, 0.0, 0.2), abs=4 * res["stderr"] + 0.02)


def test_exercise_boundaries_rise_towards_the_strike():
    tree = american.binomial_with_boundary("put", 100, 100, 1.0, 0.05, 0.0, 0.2, 500)
    b = tree["boundary_S"]
    assert np.all(b < 100) and b[-1] > b[0]
    lsm = american.lsm_price("put", 100, 100, 1.0, 0.05, 0.0, GBM(0.2), 50, 20_000, 3, 3)
    assert len(lsm["boundary_S"]) > 10 and np.all(lsm["boundary_S"] < 100)


# ---------------------------------------------------------------- Greeks
def test_three_greek_methods_agree_for_a_european_call():
    tbl = greeks.greeks_table(European("call", 100), G, S0, R, Q, T, 1, 40_000, 5).set_index("Greek")
    ref = bs_greeks("call", S0, 100, T, R, Q, 0.25)
    exact = {"Delta": ref["delta"], "Gamma": ref["gamma"], "Vega (per vol pt)": ref["vega"] / 100}
    for g, v in exact.items():
        assert tbl.loc[g, "Formula"] == pytest.approx(v, rel=1e-3)
        for m in ("Bump & revalue", "Likelihood ratio") + (("Pathwise",) if g != "Gamma" else ()):
            assert abs(tbl.loc[g, m] - v) <= 4 * tbl.loc[g, f"{m} ±"] + 1e-3 * abs(v), (g, m)


def test_pathwise_does_not_apply_to_digitals_but_likelihood_ratio_does():
    tbl = greeks.greeks_table(Digital("call", 105), G, S0, R, Q, T, 1, 40_000, 5).set_index("Greek")
    assert np.isnan(tbl.loc["Delta", "Pathwise"])
    assert abs(tbl.loc["Delta", "Likelihood ratio"] - tbl.loc["Delta", "Formula"]) < 4 * tbl.loc["Delta", "Likelihood ratio ±"]


def test_bump_and_revalue_works_under_every_model():
    for model in (Heston(), Merton()):
        tbl = greeks.greeks_table(European("call", 100), model, S0, R, Q, T, 1, 20_000, 5).set_index("Greek")
        assert abs(tbl.loc["Delta", "Bump & revalue"] - tbl.loc["Delta", "Formula"]) < 0.02
        assert np.isnan(tbl.loc["Delta", "Pathwise"])


# ---------------------------------------------------------------- positions and risk
MKT = Market(spot=100, rate=R, div_yield=Q, vol=VolModel.equity(0.25))


def test_gbm_repricing_reproduces_black_scholes_on_the_smile():
    pos = Position(presets.get("iron-condor").make_legs(100, 45), MKT, multiplier=100)
    tbl = portfolio.reprice_position(pos, GBM(0.25), 40_000, "antithetic", 3)
    assert np.allclose(tbl["Model formula"], tbl["Black-Scholes on smile"])
    assert np.all(np.abs(tbl["Monte Carlo"] - tbl["Black-Scholes on smile"]) <= 2.5 * tbl["MC ±"])
    vals = portfolio.position_values(tbl, 100)
    assert vals["Black-Scholes on smile"] == pytest.approx(-pos.analyze().net_premium)


def test_model_smile_of_heston_is_skewed():
    sm = portfolio.model_smile(Heston(), 100, R, Q, [60])
    assert sm["iv"].iloc[0] > sm["iv"].iloc[len(sm) // 2] > sm["iv"].iloc[-1]


def test_var_and_es_of_a_stock_match_the_lognormal():
    from scipy.special import ndtri
    pos = Position([Leg("underlying", 1)], Market(spot=100, rate=R, vol=VolModel.flat(0.25)))
    out = portfolio.pnl_distribution(pos, GBM(0.25), 10, 200_000, 0.06, 5)
    h = 10 / 365
    exact = 100 - 100 * math.exp((0.06 - 0.5 * 0.0625) * h + 0.25 * math.sqrt(h) * float(ndtri(0.01)))
    var, es = portfolio.var_es(out["pnl"], 0.99)
    assert var == pytest.approx(exact, rel=0.02)
    assert es > var
    assert portfolio.var_es(out["pnl"], 0.95)[0] < var
    assert portfolio.delta_normal_var(pos, 0.25, 10, 0.99) == pytest.approx(float(ndtri(0.99)) * 100 * 0.25 * math.sqrt(h))


def test_short_gamma_risk_is_understated_by_delta_normal_var():
    pos = Position(presets.get("short-straddle").make_legs(100, 30), Market(spot=100, rate=R, vol=VolModel.flat(0.25)), 100)
    out = portfolio.pnl_distribution(pos, GBM(0.25), 10, 50_000, 0.04, 3)
    assert portfolio.var_es(out["pnl"], 0.99)[0] > 3 * portfolio.delta_normal_var(pos, 0.25, 10, 0.99)


def test_heston_risk_links_vol_to_spot():
    pos = Position(presets.get("long-straddle").make_legs(100, 30), MKT, 100)
    out = portfolio.pnl_distribution(pos, Heston(), 10, 20_000, 0.04, 3, link_vol=True)
    assert np.corrcoef(np.log(out["S"]), out["dvol"])[0, 1] < -0.3        # rho = -0.7: vol up when spot down
    flat = portfolio.pnl_distribution(pos, Heston(), 10, 20_000, 0.04, 3, link_vol=False)
    assert np.all(flat["dvol"] == 0)


def test_path_chart_only_shows_a_running_average_for_asians():
    from derivlab.montecarlo import charts as mcc, sample_paths
    t, S = sample_paths(G, S0, R, Q, T, 20, 5, 1)
    plain = [tr.name for tr in mcc.paths_figure(t, S, 100, 90, np.zeros(5, bool), average=False).data]
    asian = [tr.name for tr in mcc.paths_figure(t, S, 100, average=True).data]
    assert "Its running average" not in plain and "Touched the barrier" in plain
    assert "Its running average" in asian
