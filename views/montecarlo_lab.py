"""Monte Carlo pricer: exotic, American and multi-asset options, variance reduction, Greeks and P&L risk."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st

from derivlab.fmt import money, pct
from derivlab.montecarlo import (GBM, METHODS, Asian, Barrier, Basket, Digital, European, Heston, Lookback, Merton,
                                 Rainbow, Spread, american, benchmark, compare_methods, control_variate, convergence,
                                 greeks, multiasset, portfolio, price_option, sample_paths)
from derivlab.montecarlo import charts as mcc
from derivlab.position import DAYS_PER_YEAR, Leg, Market, Position
from derivlab.pricing import bs_price
from derivlab.volatility import VolModel
from views.shared import current_position, init_state

SECTIONS = ["Exotics", "Variance reduction", "Greeks", "American", "Multi-asset", "Your position", "VaR & ES"]
MODELS = ["Black-Scholes (GBM)", "Heston stochastic vol", "Merton jump-diffusion"]
PRODUCTS = ["European", "Digital (cash-or-nothing)", "Asian, arithmetic average", "Asian, geometric average", "Barrier",
            "Lookback, floating strike", "Lookback, fixed strike"]
PATH_DEPENDENT = PRODUCTS[2:]
PATHS = [5_000, 10_000, 20_000, 50_000, 100_000, 200_000]
MULTI_PRODUCTS = ["Basket", "Spread (asset 1 − asset 2)", "Best-of", "Worst-of"]
MC_DEFAULTS = dict(
    mc_section=SECTIONS[0], mc_last=SECTIONS[0], mc_model=MODELS[0], mc_paths=20_000, mc_method="Antithetic", mc_seed=7,
    mc_h_v0=25.0, mc_h_theta=25.0, mc_h_kappa=2.0, mc_h_xi=0.45, mc_h_rho=-0.7,
    mc_m_sigma=20.0, mc_m_lam=0.5, mc_m_mu=-10.0, mc_m_delta=15.0,
    ex_product="Asian, arithmetic average", ex_kind="call", ex_strike=100.0, ex_days=182, ex_obs=126,
    ex_barrier=90.0, ex_dir="down", ex_knock="out",
    am_kind="put", am_strike=100.0, am_days=365, am_ex=50, am_degree=3,
    ma_n=2, ma_product="Basket", ma_kind="call", ma_strike=100.0, ma_days=182, ma_rho12=0.5, ma_rho13=0.3, ma_rho23=0.3,
    ma_s1=100.0, ma_s2=100.0, ma_s3=100.0, ma_v1=25.0, ma_v2=30.0, ma_v3=20.0, ma_q1=0.0, ma_q2=0.0, ma_q3=0.0,
    ma_w1=0.5, ma_w2=0.5, ma_w3=0.0, rk_days=10, rk_mu=6.0, rk_conf="99%", rk_link=True,
)
CONF = {"95%": 0.95, "97.5%": 0.975, "99%": 0.99}

ss = st.session_state
init_state(MC_DEFAULTS)


# ---------------------------------------------------------------- builders (plain tuples so results can be cached)
def model_key() -> tuple:
    if ss.mc_model == MODELS[1]:
        return ("heston", (ss.mc_h_v0 / 100) ** 2, ss.mc_h_kappa, (ss.mc_h_theta / 100) ** 2, ss.mc_h_xi, ss.mc_h_rho)
    if ss.mc_model == MODELS[2]:
        return ("merton", ss.mc_m_sigma / 100, ss.mc_m_lam, ss.mc_m_mu / 100, ss.mc_m_delta / 100)
    return ("gbm", ss.atm / 100)


def build_model(key: tuple):
    kind, *p = key
    return {"gbm": GBM, "heston": Heston, "merton": Merton}[kind](*p)


def product_key() -> tuple:
    S = float(ss.spot)
    return (ss.ex_product, ss.ex_kind, S * ss.ex_strike / 100, S * ss.ex_barrier / 100, ss.ex_dir, ss.ex_knock)


def build_product(key: tuple):
    name, kind, K, H, direction, knock = key
    if name == "European":
        return European(kind, K)
    if name.startswith("Digital"):
        return Digital(kind, K)
    if name.startswith("Asian"):
        return Asian(kind, K, "arithmetic" if "arithmetic" in name else "geometric")
    if name == "Barrier":
        return Barrier(kind, K, H, direction, knock)
    return Lookback(kind, None if "floating" in name else K)


def market_args():
    return float(ss.spot), ss.rate / 100, ss.div / 100


def position_key(pos: Position) -> tuple:
    m, v = pos.market, pos.market.vol
    legs = tuple((l.kind, l.qty, l.strike, l.days, l.premium) for l in pos.legs)
    return legs, (m.spot, m.rate, m.div_yield, v.atm, v.skew, v.smile, m.sticky), pos.multiplier


def build_position(key: tuple) -> Position:
    legs, (spot, rate, div, atm, skew, smile, sticky), mult = key
    market = Market(spot=spot, rate=rate, div_yield=div, vol=VolModel(atm=atm, skew=skew, smile=smile), sticky=sticky)
    return Position([Leg(*l) for l in legs], market, multiplier=mult)


# ---------------------------------------------------------------- cached computations
@st.cache_data(show_spinner=False, max_entries=64)
def run_price(pk, mk, S, r, q, T, n_obs, n, method, seed):
    sampling, cv = METHODS[method]
    return price_option(build_product(pk), build_model(mk), S, r, q, T, n_obs, n, sampling, "auto" if cv else None, seed)


@st.cache_data(show_spinner=False, max_entries=32)
def run_compare(pk, mk, S, r, q, T, n_obs, n, seed):
    return compare_methods(build_product(pk), build_model(mk), S, r, q, T, n_obs, n, seed)


@st.cache_data(show_spinner=False, max_entries=32)
def run_convergence(pk, mk, S, r, q, T, n_obs, n, method, seed):
    base = convergence(run_price(pk, mk, S, r, q, T, n_obs, n, "Plain", seed))
    other = convergence(run_price(pk, mk, S, r, q, T, n_obs, n, method, seed)) if method != "Plain" else None
    return base, other


@st.cache_data(show_spinner=False, max_entries=32)
def run_greeks(pk, mk, S, r, q, T, n_obs, n, seed):
    return greeks.greeks_table(build_product(pk), build_model(mk), S, r, q, T, n_obs, n, seed)


@st.cache_data(show_spinner=False, max_entries=32)
def run_paths(mk, S, r, q, T, n_obs, seed):
    return sample_paths(build_model(mk), S, r, q, T, max(n_obs, 2), 30, seed)


@st.cache_data(show_spinner=False, max_entries=32)
def run_lsm(kind, S, K, T, r, q, mk, n_ex, n, degree, seed):
    return american.lsm_price(kind, S, K, T, r, q, build_model(mk), n_ex, n, degree, seed)


@st.cache_data(show_spinner=False, max_entries=32)
def run_binomial(kind, S, K, T, r, q, sigma):
    return american.binomial_with_boundary(kind, S, K, T, r, q, sigma, 1000)


@st.cache_data(show_spinner=False, max_entries=32)
def run_multi(product, S, r, q, sig, corr, T, n, method, seed):
    sampling, cv = METHODS[method]
    prod = build_multi_product(product)
    return multiasset.price_multi(prod, list(S), r, list(q), list(sig), np.array(corr), T, n, sampling, "auto" if cv else None, seed)


@st.cache_data(show_spinner=False, max_entries=32)
def run_sweep(product, S, r, q, sig, T, n, seed):
    return multiasset.correlation_sweep(build_multi_product(product), list(S), r, list(q), list(sig), T,
                                        np.round(np.linspace(-0.9, 0.95, 38), 3), n, seed)


@st.cache_data(show_spinner=False, max_entries=16)
def run_reprice(pkey, mk, n, method, seed):
    sampling = METHODS[method][0]
    return portfolio.reprice_position(build_position(pkey), build_model(mk), n, sampling, seed)


@st.cache_data(show_spinner=False, max_entries=16)
def run_model_smile(mk, S, r, q, days):
    return portfolio.model_smile(build_model(mk), S, r, q, days)


@st.cache_data(show_spinner=False, max_entries=16)
def run_pnl(pkey, mk, horizon, n, mu, seed, link):
    return portfolio.pnl_distribution(build_position(pkey), build_model(mk), horizon, n, mu, seed, link)


def equal_weights():
    """Reset basket weights to an equally weighted average when the number of assets changes."""
    d = int(ss.ma_n)
    for i in range(3):
        ss[f"ma_w{i + 1}"] = round(1 / d, 4) if i < d else 0.0


def build_multi_product(key):
    name, kind, K, weights, d = key
    if name == "Basket":
        return Basket(kind, K, tuple(weights[:d]))
    if name.startswith("Spread"):
        return Spread(kind, K)
    return Rainbow(kind, K, "max" if name == "Best-of" else "min")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.subheader("Market")
    st.number_input("Spot price", min_value=0.01, step=1.0, format="%.2f", key="spot")
    c1, c2 = st.columns(2)
    c1.number_input("Rate %", step=0.25, format="%.2f", key="rate")
    c2.number_input("Div. yield %", step=0.25, format="%.2f", key="div")
    st.caption("Shared with the options lab.")
    st.subheader("Model")
    st.selectbox("Price dynamics", MODELS, key="mc_model",
                 help="GBM is the Black-Scholes world. Heston lets volatility move randomly; Merton adds sudden jumps.")
    if ss.mc_model == MODELS[0]:
        st.slider("Volatility %", 1.0, 150.0, step=0.5, key="atm", help="Shared with the options lab's ATM vol.")
    elif ss.mc_model == MODELS[1]:
        st.slider("Initial vol %", 1.0, 100.0, step=0.5, key="mc_h_v0")
        st.slider("Long-run vol %", 1.0, 100.0, step=0.5, key="mc_h_theta")
        st.slider("Mean reversion κ (per year)", 0.1, 10.0, step=0.1, key="mc_h_kappa",
                  help="How fast variance is pulled back to its long-run level.")
        st.slider("Vol of vol ξ", 0.05, 1.5, step=0.05, key="mc_h_xi", help="How much the variance itself fluctuates.")
        st.slider("Spot-vol correlation ρ", -0.99, 0.99, step=0.01, key="mc_h_rho",
                  help="Negative: vol rises when the price falls, which creates an equity-style skew.")
        hm = build_model(model_key())
        st.caption(("Feller condition holds: " if hm.feller else "Feller condition fails: variance can hit zero; ")
                   + f"2κθ = {2 * hm.kappa * hm.theta:.3f} vs ξ² = {hm.xi ** 2:.3f}.")
    else:
        st.slider("Diffusion vol %", 1.0, 100.0, step=0.5, key="mc_m_sigma")
        st.slider("Jumps per year λ", 0.0, 5.0, step=0.1, key="mc_m_lam")
        st.slider("Average jump % (log)", -50.0, 50.0, step=1.0, key="mc_m_mu")
        st.slider("Jump size vol %", 0.0, 60.0, step=1.0, key="mc_m_delta")
        st.caption(f"Total volatility ≈ {build_model(model_key()).total_vol():.1%} a year, diffusion plus jumps.")
    st.subheader("Simulation")
    st.select_slider("Paths", PATHS, key="mc_paths", format_func=lambda v: f"{v:,}")
    st.selectbox("Method", list(METHODS), key="mc_method",
                 help="How the random numbers are drawn and whether a control variate is used. "
                      "Compare them in the Variance reduction section.")
    st.number_input("Random seed", min_value=0, step=1, key="mc_seed", help="Same seed, same paths: results are reproducible.")

S, r, q = market_args()
mk = model_key()
model = build_model(mk)
N, METHOD, SEED = int(ss.mc_paths), ss.mc_method, int(ss.mc_seed)

st.title("Monte Carlo pricer")
st.caption("Price what has no formula by simulating thousands of price paths and averaging the discounted payoffs. "
           "Every result comes with a 95% confidence interval and, where one exists, a formula to check it against.")
sec = st.segmented_control("Section", SECTIONS, key="mc_section", label_visibility="collapsed")
if sec is None:
    sec = ss.mc_last
else:
    ss.mc_last = sec


def product_inputs():
    c = st.columns([2.2, 0.9, 1.1, 1, 1.2])
    c[0].selectbox("Product", PRODUCTS, key="ex_product")
    c[1].selectbox("Type", ["call", "put"], key="ex_kind", format_func=str.title)
    floating = ss.ex_product == "Lookback, floating strike"
    c[2].number_input("Strike, % of spot", min_value=1.0, max_value=500.0, step=1.0, format="%.1f", key="ex_strike",
                      disabled=floating, help="Not used by a floating-strike lookback." if floating else None)
    c[3].number_input("Days to expiry", min_value=1, max_value=3650, step=1, key="ex_days")
    c[4].number_input("Observation dates", min_value=1, max_value=1000, step=1, key="ex_obs",
                      disabled=ss.ex_product not in PATH_DEPENDENT,
                      help="Fixings of the average, checks of the barrier or of the max/min. 126 is about every trading "
                           "day over six months.")
    if ss.ex_product == "Barrier":
        c = st.columns([1.2, 1, 1, 2.6])
        c[0].number_input("Barrier, % of spot", min_value=1.0, max_value=500.0, step=1.0, format="%.1f", key="ex_barrier")
        c[1].selectbox("Barrier is", ["down", "up"], key="ex_dir", format_func=lambda v: f"{v} (below spot)" if v == "down" else f"{v} (above spot)")
        c[2].selectbox("On touching it", ["out", "in"], key="ex_knock",
                       format_func=lambda v: "knock out" if v == "out" else "knock in")
        breached = (ss.ex_dir == "down" and ss.ex_barrier >= 100) or (ss.ex_dir == "up" and ss.ex_barrier <= 100)
        if breached:
            c[3].warning("The barrier is already breached today: a down barrier must sit below spot, an up barrier above.")
    pk = product_key()
    T = ss.ex_days / DAYS_PER_YEAR
    n_obs = int(ss.ex_obs) if ss.ex_product in PATH_DEPENDENT else 1
    return pk, build_product(pk), T, n_obs


def fmt_pm(v, se, dp=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:,.{dp}f}" + ("" if se is None or math.isnan(se) else f" ± {1.96 * se:,.{dp}f}")


# ================================================================= Exotics
if sec == "Exotics":
    pk, prod, T, n_obs = product_inputs()
    with st.spinner(f"Simulating {N:,} paths…"):
        res = run_price(pk, mk, S, r, q, T, n_obs, N, METHOD, SEED)
    bm = benchmark(prod, model, S, r, q, T, n_obs)
    lo, hi = res.ci95
    k = st.columns(4)
    k[0].metric("Monte Carlo price", f"{res.price:,.4f}", help=f"{res.n_paths:,} paths, {METHOD.lower()}.")
    k[1].metric("95% confidence interval", f"± {1.96 * res.stderr:,.4f}", help=f"{lo:,.4f} to {hi:,.4f}")
    if bm:
        z = (res.price - bm[1]) / res.stderr if res.stderr > 0 else 0.0
        k[2].metric("Formula", f"{bm[1]:,.4f}", help=bm[0])
        k[3].metric("Difference", f"{z:+.1f} std. errors",
                    help="Within about ±2 standard errors the simulation agrees with the formula.")
    else:
        vanilla = bs_price(prod.kind, S, getattr(prod, "strike", None) or S, T, r, q, model.total_vol())
        k[2].metric("Formula", "none", help="No closed form exists for this product: this is where simulation earns its keep.")
        k[3].metric("Vanilla, for reference", f"{vanilla:,.4f}", help="Black-Scholes price of the plain option with the same strike.")
    st.caption((f"Formula: {bm[0]}. " if bm else "") + f"{res.n_paths:,} paths in {res.seconds * 1000:,.0f} ms"
               + (f", control variate: {res.control.lower()}" if res.control else "") + ".")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Sample paths**")
        times, paths = run_paths(mk, S, r, q, T, n_obs, SEED + 1)
        touched = prod.touched(paths) if isinstance(prod, Barrier) else None
        st.plotly_chart(mcc.paths_figure(times, paths, getattr(prod, "strike", None),
                                         prod.barrier if isinstance(prod, Barrier) else None, touched,
                                         average=isinstance(prod, Asian)), key="mc_paths_chart")
    with c2:
        st.markdown("**What each path pays**")
        plain = run_price(pk, mk, S, r, q, T, n_obs, N, "Plain", SEED)
        st.plotly_chart(mcc.payoff_histogram(plain.units_y, plain.price), key="mc_hist")
        st.caption("One plain run with the same number of paths. Its average differs from the headline price only by "
                   "sampling error: that spread is exactly what the confidence interval measures.")
    with st.expander("How the price is computed"):
        st.markdown(
            "1. Simulate the underlying along many paths under the **risk-neutral** measure (drift = rate − dividend yield).\n"
            "2. Work out each path's payoff, using only the observation dates, as a real contract would.\n"
            "3. Discount the payoffs at the risk-free rate and average them. The law of large numbers makes the average "
            "converge to the price; the central limit theorem gives the confidence interval: ±1.96 × std. deviation / √N.\n\n"
            "Halving the error takes **four times** the paths. That is why the variance-reduction tricks in the next "
            "section matter.")

# ================================================================= Variance reduction
elif sec == "Variance reduction":
    pk, prod, T, n_obs = product_inputs()
    with st.spinner("Pricing with every method…"):
        table = run_compare(pk, mk, S, r, q, T, n_obs, N, SEED)
        base_conv, method_conv = run_convergence(pk, mk, S, r, q, T, n_obs, N, METHOD, SEED)
    bm = benchmark(prod, model, S, r, q, T, n_obs)
    st.dataframe(table.drop(columns=["Control"]), hide_index=True, column_config={
        "Price": st.column_config.NumberColumn(format="%.4f"),
        "Std error": st.column_config.ProgressColumn("Std error", format="%.5f", min_value=0.0,
                                                     max_value=float(table["Std error"].max()) or 1.0),
        "95% CI ±": st.column_config.NumberColumn(format="%.4f"), "Paths": st.column_config.NumberColumn(format="%d"),
        "Variance reduction": st.column_config.NumberColumn(format="%.1f×"),
        "Time (ms)": st.column_config.NumberColumn(format="%.0f"), "Efficiency": st.column_config.NumberColumn(format="%.1f×")})
    st.caption((f"{bm[0]}: {bm[1]:,.4f}. " if bm else "No formula exists for this product. ")
               + "Variance reduction = (plain std error ÷ method std error)²: how many times more plain paths you would "
                 "need for the same accuracy. Efficiency also accounts for run time. Sobol' runs use 16 scrambles of a "
                 "power-of-two number of points.")
    st.markdown(f"**Convergence** · plain sampling vs {METHOD.lower()}")
    st.plotly_chart(mcc.convergence_figure(base_conv, method_conv, bm[1] if bm else None, METHOD), key="mc_conv")
    st.markdown(
        "* **Antithetic paths** pair every path with its mirror image (shocks −Z). Errors partly cancel for payoffs that rise "
        "or fall steadily with the price; for a straddle-like payoff they barely help.\n"
        "* **Control variates** use a related quantity with a known price: the simulation's error on it is used to correct "
        f"the estimate. Here: *{control_variate(prod, model, S, r, q, T, n_obs).name}*. The closer the "
        "correlation, the bigger the gain: the geometric Asian controls an arithmetic Asian almost perfectly.\n"
        "* **Sobol' quasi-random numbers** fill the space evenly instead of randomly, with paths built by a Brownian bridge "
        "so the best dimensions decide the overall shape. The error can fall nearly as fast as 1/N instead of 1/√N "
        "(see the right-hand chart). Sixteen independent scrambles give the error estimate.")

# ================================================================= Greeks
elif sec == "Greeks":
    pk, prod, T, n_obs = product_inputs()
    with st.spinner("Estimating Greeks…"):
        tbl = run_greeks(pk, mk, S, r, q, T, n_obs, N, SEED)
    show = pd.DataFrame({"Greek": tbl["Greek"],
                         "Formula": [fmt_pm(v, None) for v in tbl["Formula"]],
                         "Bump & revalue": [fmt_pm(v, s) for v, s in zip(tbl["Bump & revalue"], tbl["Bump & revalue ±"])],
                         "Pathwise": [fmt_pm(v, s) for v, s in zip(tbl["Pathwise"], tbl["Pathwise ±"])],
                         "Likelihood ratio": [fmt_pm(v, s) for v, s in zip(tbl["Likelihood ratio"], tbl["Likelihood ratio ±"])]})
    st.dataframe(show, hide_index=True)
    st.caption("Values ± 95% confidence. Delta per 1.00 move in spot, gamma per 1.00 move, vega per vol point; for one "
               "option on one unit. " + greeks.pathwise_note(prod, model))
    st.markdown(
        "* **Bump and revalue** reprices with spot moved ±1% (and vol ±1 point) using the **same random numbers**. Without "
        "them the noise in two separate simulations would swamp the small difference you want.\n"
        "* **Pathwise** differentiates each path's payoff: for a call, delta = discounted 1{S_T > K} × S_T / S₀. Cheap and "
        "precise, but blind to jumps in the payoff, so it cannot do digitals, barriers or gamma.\n"
        "* **Likelihood ratio** differentiates the probability of each path instead, so it works for any payoff, at the "
        "price of more noise, especially with many observation dates (the score uses only the first, short, step).")

# ================================================================= American
elif sec == "American":
    c = st.columns(5)
    c[0].selectbox("Type", ["put", "call"], key="am_kind", format_func=str.title)
    c[1].number_input("Strike, % of spot", min_value=1.0, max_value=500.0, step=1.0, format="%.1f", key="am_strike")
    c[2].number_input("Days to expiry", min_value=5, max_value=3650, step=5, key="am_days")
    c[3].number_input("Exercise dates", min_value=2, max_value=250, step=1, key="am_ex",
                      help="Dates on which the holder may exercise. More dates bring a Bermudan closer to an American option.")
    c[4].selectbox("Regression degree", [2, 3, 4, 5], key="am_degree", help="Powers of S/K used to estimate the continuation value.")
    K, T = S * ss.am_strike / 100, ss.am_days / DAYS_PER_YEAR
    with st.spinner("Fitting the exercise rule and pricing…"):
        lsm = run_lsm(ss.am_kind, S, K, T, r, q, mk, int(ss.am_ex), N, int(ss.am_degree), SEED)
        binom = run_binomial(ss.am_kind, S, K, T, r, q, model.sigma) if isinstance(model, GBM) else None
    euro_formula = benchmark(European(ss.am_kind, K), model, S, r, q, T, 1)[1]
    k = st.columns(4)
    k[0].metric("Longstaff-Schwartz price", f"{lsm['price']:,.4f}", help=f"± {1.96 * lsm['stderr']:,.4f} (95%)")
    k[1].metric("European price", f"{euro_formula:,.4f}", help="Same option without early exercise, from the model's formula.")
    k[2].metric("Early-exercise premium", f"{lsm['price'] - euro_formula:,.4f}")
    if binom:
        k[3].metric("Binomial tree (1,000 steps)", f"{binom['price']:,.4f}", help="An independent check: Cox-Ross-Rubinstein with early exercise.")
    else:
        k[3].metric("Paths exercised early", pct(lsm["early_share"]))
    st.plotly_chart(mcc.boundary_figure(lsm, binom, K), key="mc_boundary")
    st.caption(f"{lsm['early_share']:.0%} of paths are exercised early. "
               + ("Under Heston the exercise boundary depends on the variance as well as the price, so it is not drawn. "
                  if isinstance(model, Heston) else "Boundary points are shown where at least 50 training paths were exercised; "
                  "dates whose fitted rule jumps far from its neighbours are left out. ")
               + "The simulation can only exercise on its dates (a Bermudan option), and a fitted rule is never perfectly "
                 "optimal, so the estimate sits slightly below the true American value.")
    st.markdown(
        "**How Longstaff-Schwartz works.** Simulate paths forward, then walk backward through the exercise dates. At each "
        "date, regress the discounted future cash flows of in-the-money paths on powers of S/K: the fitted value is the "
        "expected value of waiting. Exercise wherever the payoff now beats it. The rule is fitted on one set of paths and "
        "applied to a fresh set, so the price is not flattered by hindsight. American calls on assets without dividends are "
        "never exercised early, which the method reproduces.")

# ================================================================= Multi-asset
elif sec == "Multi-asset":
    c = st.columns([1, 1.6, 0.9, 1, 1])
    c[0].radio("Assets", [2, 3], key="ma_n", horizontal=True, on_change=equal_weights)
    d = int(ss.ma_n)
    options = MULTI_PRODUCTS if d == 2 else [p for p in MULTI_PRODUCTS if not p.startswith("Spread")]
    if ss.ma_product not in options:
        ss.ma_product = options[0]
    c[1].selectbox("Product", options, key="ma_product")
    c[2].selectbox("Type", ["call", "put"], key="ma_kind", format_func=str.title)
    c[3].number_input("Strike", min_value=0.0, step=1.0, format="%.2f", key="ma_strike")
    c[4].number_input("Days to expiry", min_value=1, max_value=3650, step=1, key="ma_days")
    cols = st.columns(d)
    for i in range(d):
        with cols[i]:
            st.markdown(f"**Asset {i + 1}**")
            a, b = st.columns(2)
            a.number_input("Spot", min_value=0.01, step=1.0, format="%.2f", key=f"ma_s{i + 1}")
            b.number_input("Vol %", min_value=1.0, max_value=150.0, step=1.0, format="%.1f", key=f"ma_v{i + 1}")
            a.number_input("Div. yield %", step=0.25, format="%.2f", key=f"ma_q{i + 1}")
            if ss.ma_product == "Basket":
                b.number_input("Weight", min_value=0.0, step=0.1, format="%.2f", key=f"ma_w{i + 1}")
    pairs = [(0, 1)] if d == 2 else [(0, 1), (0, 2), (1, 2)]
    c = st.columns(len(pairs))
    rho = {}
    for (i, j), col in zip(pairs, c):
        rho[(i, j)] = col.slider(f"Correlation {i + 1}-{j + 1}", -0.95, 0.95, step=0.05, key=f"ma_rho{i + 1}{j + 1}")
    try:
        corr = multiasset.correlation_matrix(d, rho)
    except ValueError as e:
        st.error(f"{e}.")
        st.stop()
    Sv = tuple(float(ss[f"ma_s{i + 1}"]) for i in range(d))
    qv = tuple(ss[f"ma_q{i + 1}"] / 100 for i in range(d))
    sv = tuple(ss[f"ma_v{i + 1}"] / 100 for i in range(d))
    wv = tuple(float(ss[f"ma_w{i + 1}"]) for i in range(3))
    if ss.ma_product == "Basket" and sum(wv[:d]) <= 0:
        st.warning("Give at least one asset a positive weight.")
        st.stop()
    T = ss.ma_days / DAYS_PER_YEAR
    pkey = (ss.ma_product, ss.ma_kind, float(ss.ma_strike), wv, d)
    with st.spinner("Simulating correlated paths…"):
        res = run_multi(pkey, Sv, r, qv, sv, tuple(map(tuple, corr)), T, N, METHOD, SEED)
    bm = multiasset.benchmark(build_multi_product(pkey), list(Sv), r, list(qv), list(sv), corr, T)
    k = st.columns(4)
    k[0].metric("Monte Carlo price", f"{res.price:,.4f}")
    k[1].metric("95% confidence interval", f"± {1.96 * res.stderr:,.4f}")
    if bm:
        k[2].metric("Formula", f"{bm[1]:,.4f}", help=bm[0])
        k[3].metric("Difference", f"{(res.price - bm[1]) / res.stderr:+.1f} std. errors" if res.stderr > 0 else "0")
        st.caption(f"{bm[0]}." + (" An approximation, so small gaps are expected." if "approx" in bm[0].lower() else ""))
    else:
        k[2].metric("Formula", "none")
    st.markdown("**The correlation trade** · price as one common correlation between every pair moves")
    with st.spinner("Repricing across correlations…"):
        sweep = run_sweep(pkey, Sv, r, qv, sv, T, min(N, 20_000), SEED)
    st.plotly_chart(mcc.correlation_figure(sweep, float(np.mean(list(rho.values())))), key="mc_corr")
    st.caption("Baskets get cheaper as correlation falls (diversification lowers the basket's volatility). Spreads and "
               "best-of options get dearer (the assets drift apart). Worst-of options get cheaper. Correlated assets are "
               "simulated with the Cholesky factor of the correlation matrix.")

# ================================================================= Your position
elif sec == "Your position":
    pos, problems = current_position()
    if pos is None or not any(l.is_option for l in pos.legs):
        st.info("Build a position with at least one option leg in the options lab first.")
        st.stop()
    st.markdown(f"Repricing the position from the options lab under **{model.name}**. "
                + ("With GBM each leg is simulated at its own smile vol, so the simulation should reproduce the options-lab "
                   "prices: a check of the method." if isinstance(model, GBM) else
                   "This model has no smile input: it produces its own skew, so its prices differ from Black-Scholes on your smile."))
    pkey = position_key(pos)
    with st.spinner("Simulating every leg…"):
        tbl = run_reprice(pkey, mk, N, METHOD, SEED)
    vals = portfolio.position_values(tbl, pos.multiplier)
    k = st.columns(3)
    k[0].metric("Options value, Black-Scholes on your smile", money(vals["Black-Scholes on smile"]))
    k[1].metric(f"{'Black-Scholes at leg vols' if isinstance(model, GBM) else model.name} formula", money(vals["Model formula"]))
    k[2].metric("Monte Carlo", money(vals["Monte Carlo"]),
                delta=f"{vals['Monte Carlo'] - vals['Black-Scholes on smile']:+,.2f} vs your smile", delta_color="off")
    st.dataframe(tbl, hide_index=True, column_config={
        c: st.column_config.NumberColumn(format="%.4f") for c in ["Black-Scholes on smile", "Model formula", "Monte Carlo", "MC ±"]}
        | {c: st.column_config.NumberColumn(format="%.2f") for c in ["Smile vol %", "Model implied vol %"]}
        | {"Days": st.column_config.NumberColumn(format="%d"), "Qty": st.column_config.NumberColumn(format="%g")})
    st.caption("Values are long-option prices per unit; the position totals above include quantities, lots and the multiplier. "
               "MC ± is the 95% confidence half-width. Model implied vol is the Black-Scholes vol that matches the model's price.")
    days = tuple(sorted({l.days for l in pos.legs if l.is_option}))[:3]
    sm = run_model_smile(mk, S, r, q, days)
    user = sm[["days", "strike"]].copy()
    user["iv"] = [float(pos.market.vol.vol(k_, d_ / DAYS_PER_YEAR, pos.market.forward(d_ / DAYS_PER_YEAR)))
                  for k_, d_ in zip(user["strike"], user["days"])]
    st.markdown("**The smile each model produces** · solid: the model; dashed: the smile set in the options lab")
    st.plotly_chart(mcc.model_smile_figure(sm, user, model.name), key="mc_model_smile")
    st.caption("Heston with negative spot-vol correlation produces an equity-style skew; Merton's jumps steepen the "
               "short-dated wings. Matching a model to the market's smile is called calibration.")

# ================================================================= VaR & ES
elif sec == "VaR & ES":
    pos, problems = current_position()
    if pos is None:
        st.info("Build a position in the options lab first.")
        st.stop()
    c = st.columns(4)
    c[0].number_input("Horizon (days)", min_value=1, max_value=365, step=1, key="rk_days")
    c[1].number_input("Expected return % a year", step=1.0, format="%.1f", key="rk_mu",
                      help="Risk uses the real-world drift, not the risk-free rate. Over short horizons it matters little.")
    c[2].selectbox("Confidence", list(CONF), key="rk_conf")
    c[3].toggle("Link vol (Heston)", key="rk_link", disabled=not isinstance(model, Heston),
                help="Shifts every implied vol by the change in simulated volatility, so spot and vol move together.")
    alpha = CONF[ss.rk_conf]
    with st.spinner("Simulating and revaluing…"):
        out = run_pnl(position_key(pos), mk, int(ss.rk_days), N, ss.rk_mu / 100, SEED, bool(ss.rk_link))
    pnl = out["pnl"]
    var, es = portfolio.var_es(pnl, alpha)
    dn = portfolio.delta_normal_var(pos, model.total_vol(), int(ss.rk_days), alpha)
    k = st.columns(5)
    k[0].metric(f"VaR {ss.rk_conf}", money(var, 0), help="The loss exceeded on only (1 − confidence) of paths.")
    k[1].metric(f"Expected shortfall {ss.rk_conf}", money(es, 0), help="The average loss on the paths beyond the VaR.")
    k[2].metric("Delta-normal VaR", money(dn, 0), help="Treats the position as its delta: z × |cash delta| × vol × √horizon.")
    k[3].metric("Mean P&L", money(float(pnl.mean()), 0))
    k[4].metric("Chance of a loss", pct(float(np.mean(pnl < 0))))
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(mcc.pnl_histogram(pnl, var, es, alpha), key="mc_pnl_hist")
    with c2:
        st.plotly_chart(mcc.pnl_scatter(out["S"], pnl, S), key="mc_pnl_scatter")
    st.dataframe(portfolio.risk_table(pnl), hide_index=True, column_config={
        "VaR": st.column_config.NumberColumn(format="%.0f"), "Expected shortfall": st.column_config.NumberColumn(format="%.0f")})
    st.caption(f"Full revaluation: {N:,} spot scenarios under {model.name} over {int(ss.rk_days)} days, each priced with "
               "Black-Scholes on the options-lab smile. Delta-normal VaR treats the position as a straight line, so it "
               "misses curvature (gamma), capped losses and fat tails: it understates the risk of short options and can "
               "overstate it for spreads whose loss is capped. Expected shortfall measures how bad the tail is, which is "
               "why Basel's market-risk rules (FRTB) moved from VaR to ES.")
