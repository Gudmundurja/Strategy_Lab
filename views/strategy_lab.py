"""Options strategy lab: build a position, then stress it."""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

from derivlab import Leg, Market, Position, VolModel, charts, presets, sizing
from derivlab.fmt import money, pct
from derivlab.position import DAYS_PER_YEAR
from derivlab.pricing import bs_price

SMILES = {
    "Flat (textbook Black-Scholes)": (0.0, 0.0),
    "Equity index put skew": (-0.035, 0.006),
    "FX-style smile": (-0.003, 0.008),
    "Custom": None,
}
MULTIPLIERS = {
    "1 · per unit of underlying": 1,
    "10 · e.g. Euro Stoxx 50 index options": 10,
    "50 · e.g. options on E-mini S&P 500 futures": 50,
    "100 · US equity and SPX options": 100,
    "1,000 · e.g. crude oil futures options": 1000,
    "Custom": None,
}
VOL_SHIFTS = [-20, -15, -10, -5, -2, 0, 2, 5, 10, 15, 20, 30]
GREEK_SIGN = {"+": "+", "-": "−", "0": "0", "±": "±"}
COLUMNS = ["Side", "Qty", "Type", "Strike", "Days", "Premium"]
LEG_COLUMNS = {
    "Side": st.column_config.SelectboxColumn("Side", options=["Buy", "Sell"], required=True),
    "Qty": st.column_config.NumberColumn("Qty", help="Contracts per lot", min_value=0, step=1, format="%d", required=True),
    "Type": st.column_config.SelectboxColumn("Type", options=["Call", "Put", "Underlying"], required=True),
    "Strike": st.column_config.NumberColumn("Strike", min_value=0.0, format="%.2f"),
    "Days": st.column_config.NumberColumn("Days to expiry", min_value=1, step=1, format="%d"),
    "Premium": st.column_config.NumberColumn("Premium override", min_value=0.0, format="%.4f",
                                             help="The price per unit you actually traded at. Leave blank to use the model price."),
}
DEFAULTS = dict(spot=100.0, rate=4.0, div=0.0, atm=25.0, smile_shape="Flat (textbook Black-Scholes)", skew=-3.5, smile=0.6,
                sticky="Sticky strike", mult_choice="100 · US equity and SPX options", mult_custom=100.0, lots=1,
                strategy="bull-call-spread", days=30, budget=10_000.0, basis="Max loss at expiry", stress_rng=20, hedge_mult=1.0,
                grid_rng=20, grid_vols=[-10, -5, 0, 5, 10, 20], pay_vol=0)

ss = st.session_state
for _k, _v in DEFAULTS.items():
    ss.setdefault(_k, _v)


# ---------------------------------------------------------------- legs state
def preset_frame(strategy_id: str, spot: float, days: int) -> pd.DataFrame:
    rows = []
    for l in presets.get(strategy_id).make_legs(spot, days):
        rows.append({"Side": l.side, "Qty": abs(l.qty), "Type": l.kind.title() if l.is_option else "Underlying",
                     "Strike": l.strike if l.is_option else np.nan, "Days": l.days if l.is_option else np.nan,
                     "Premium": np.nan})
    return pd.DataFrame(rows, columns=COLUMNS).astype({"Qty": "float64", "Strike": "float64", "Days": "float64", "Premium": "float64"})


def reset_legs():
    ss.legs_df = preset_frame(ss.strategy, float(ss.spot), int(ss.days))
    ss.legs_saved = ss.legs_df
    ss.legs_ver = ss.get("legs_ver", 0) + 1


if "legs_df" not in ss:
    reset_legs()
elif f"legs_{ss.legs_ver}" not in ss and "legs_saved" in ss:
    ss.legs_df = ss.legs_saved          # coming back from another page: restore the last edits


def parse_legs(df: pd.DataFrame):
    legs, problems = [], []
    for i, r in enumerate(df.to_dict("records"), start=1):
        side, qty, typ = r.get("Side"), r.get("Qty"), r.get("Type")
        if side not in ("Buy", "Sell") or typ not in ("Call", "Put", "Underlying") or qty is None or pd.isna(qty):
            continue                                           # incomplete row, e.g. just added
        if qty <= 0:
            continue
        q = float(qty) if side == "Buy" else -float(qty)
        if typ == "Underlying":
            legs.append(Leg("underlying", q))
            continue
        K, D, prem = r.get("Strike"), r.get("Days"), r.get("Premium")
        if K is None or pd.isna(K) or K <= 0:
            problems.append(f"Row {i}: enter a positive strike.")
            continue
        if D is None or pd.isna(D) or D < 1:
            problems.append(f"Row {i}: enter days to expiry (1 or more).")
            continue
        premium = None if prem is None or pd.isna(prem) else float(prem)
        legs.append(Leg(typ.lower(), q, float(K), float(D), premium))
    return legs, problems


def position_code(legs, market: Market, mult: float, lots: int) -> str:
    v = market.vol
    out = ["from derivlab import Leg, Market, Position, VolModel", "",
           f"market = Market(spot={market.spot:.6g}, rate={market.rate:.6g}, div_yield={market.div_yield:.6g},",
           f"                vol=VolModel(atm={v.atm:.6g}, skew={v.skew:.6g}, smile={v.smile:.6g}), sticky={market.sticky!r})",
           "legs = ["]
    for l in legs:
        if l.is_option:
            extra = f", premium={l.premium:.6g}" if l.premium is not None else ""
            out.append(f"    Leg({l.kind!r}, {l.qty:g}, strike={l.strike:.6g}, days={l.days:g}{extra}),")
        else:
            out.append(f"    Leg('underlying', {l.qty:g}),")
    out += ["]", f"pos = Position(legs, market, multiplier={mult:g}).scaled({lots})", "",
            "a = pos.analyze()",
            "print('max profit', a.max_profit, 'max loss', a.max_loss, 'breakevens', a.breakevens)",
            "print('probability of profit', round(a.prob_profit, 3))",
            "print(pos.greeks())",
            "print(pos.scenario_grid().round(0))   # P&L: vol shifts (rows) x spot moves (columns)"]
    return "\n".join(out)


def max_loss_text(x: float) -> str:
    if math.isinf(x):
        return "Unlimited"
    return "None" if x >= 0 else money(x)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.subheader("Market")
    st.number_input("Spot price", min_value=0.01, step=1.0, format="%.2f", key="spot")
    c1, c2 = st.columns(2)
    c1.number_input("Rate %", step=0.25, format="%.2f", key="rate", help="Continuously compounded risk-free rate.")
    c2.number_input("Div. yield %", step=0.25, format="%.2f", key="div",
                    help="Continuous dividend yield. For FX use the foreign interest rate; for commodities the convenience yield.")
    st.subheader("Volatility")
    st.slider("ATM implied vol %", 1.0, 150.0, step=0.5, key="atm")
    st.selectbox("Smile", list(SMILES), key="smile_shape",
                 help="How implied vol varies with strike. See the Volatility smile tab.")
    if SMILES[ss.smile_shape] is None:
        st.slider("Skew (vol pts per std. dev.)", -10.0, 10.0, step=0.1, key="skew",
                  help="Negative: low strikes (puts) trade at higher vol, as in equity indices.")
        st.slider("Curvature (vol pts per std. dev.²)", 0.0, 3.0, step=0.05, key="smile",
                  help="Positive: both wings trade at higher vol, as in FX.")
    st.radio("When spot moves", ["Sticky strike", "Sticky moneyness"], key="sticky",
             help="Sticky strike: each strike keeps its implied vol. Sticky moneyness: the smile slides with spot, "
                  "so a fixed strike's vol changes as spot moves.")
    st.subheader("Contract size")
    st.selectbox("Contract multiplier", list(MULTIPLIERS), key="mult_choice",
                 help="Units of the underlying that one contract covers. All P&L and Greeks below include it.")
    if MULTIPLIERS[ss.mult_choice] is None:
        st.number_input("Custom multiplier", min_value=0.0001, step=1.0, key="mult_custom")
    st.number_input("Lots", min_value=1, step=1, key="lots",
                    help="How many units of the strategy you trade. Every leg's quantity is multiplied by this.")

skew, smile = SMILES[ss.smile_shape] or (ss.skew / 100.0, ss.smile / 100.0)
vol = VolModel(atm=ss.atm / 100.0, skew=skew, smile=smile)
market = Market(spot=float(ss.spot), rate=ss.rate / 100.0, div_yield=ss.div / 100.0, vol=vol,
                sticky="strike" if ss.sticky == "Sticky strike" else "moneyness")
mult = float(MULTIPLIERS[ss.mult_choice] or ss.mult_custom)
lots = int(ss.lots)

# ---------------------------------------------------------------- strategy + legs
st.title("Options strategy lab")
st.caption("Pick a strategy or build your own, then stress it: payoff, Greeks, a spot × vol scenario grid, the volatility smile "
           "and position sizing. Prices are Black-Scholes on the smile set in the sidebar.")

ids = [s.id for s in presets.STRATEGIES]
c1, c2, c3 = st.columns([3, 1.1, 1.3], vertical_alignment="bottom")
c1.selectbox("Strategy", ids, key="strategy", on_change=reset_legs,
             format_func=lambda i: f"{presets.get(i).category} · {presets.get(i).name}")
c2.number_input("Days to expiry", min_value=1, max_value=1825, step=1, key="days", on_change=reset_legs)
c3.button("Re-centre on spot", on_click=reset_legs, icon=":material/restart_alt:", width="stretch",
          help="Reload the strategy's legs with strikes placed around the current spot.")

s = presets.get(ss.strategy)
with st.container(border=True):
    left, right = st.columns([3, 1.4])
    left.markdown(f"**When to use:** {s.when}  \n**How to build:** {s.build}  \n**Avoid when:** {s.avoid}  \n**Tip:** {s.tip}")
    signs = " · ".join(f"{g} {GREEK_SIGN.get(s.greeks.get(k, '±'), '±')}" for g, k in zip("ΔΓΘν", "dgtv"))
    right.markdown(f":blue-badge[{s.category}] :violet-badge[{s.vol_view}]  \n:gray-badge[Cost: {s.cost}] "
                   f":orange-badge[Risk: {s.risk}]  \n**Greeks at inception:** {signs}")

st.markdown("**Legs** · quantities are per lot; edit any cell, add or delete rows")
edited = st.data_editor(ss.legs_df, key=f"legs_{ss.legs_ver}", num_rows="dynamic", hide_index=True,
                        column_config=LEG_COLUMNS, column_order=COLUMNS)
ss.legs_saved = edited
legs, problems = parse_legs(edited)
for p in problems:
    st.warning(p)
if not legs:
    st.info("Add at least one complete leg (side, quantity and type, plus strike and days for options).")
    st.stop()

one_lot = Position(legs, market, multiplier=mult)
pos = one_lot.scaled(lots)
a = pos.analyze()
g = a.greeks
H = int(math.ceil(pos.horizon))
has_options = any(l.is_option for l in legs)

# ---------------------------------------------------------------- headline numbers
with st.container(border=True):
    m = st.columns(4)
    if has_options:
        m[0].metric("Net credit" if a.is_credit else "Net debit", money(abs(a.net_premium)),
                    help="Option premium received (credit) or paid (debit), including the multiplier and lots.")
    else:
        m[0].metric("Net premium", "None")
    m[1].metric("Max profit", money(a.max_profit), help="At the first expiry, over all possible prices of the underlying.")
    m[2].metric("Max loss", max_loss_text(a.max_loss),
                help="At the first expiry." + ("" if a.max_loss < 0 else f" The worst case is still a gain of {money(a.max_loss)}."))
    be = ", ".join(f"{b:,.2f}" for b in a.breakevens[:2]) + (" …" if len(a.breakevens) > 2 else "")
    m[3].metric("Breakeven" + ("s" if len(a.breakevens) > 1 else ""), be or "None",
                help="Underlying prices where the P&L at the first expiry is zero: "
                     + (", ".join(f"{b:,.2f}" for b in a.breakevens) or "none") + ".")
    n = st.columns(4)
    n[0].metric("Prob. of profit", pct(a.prob_profit),
                help="Risk-neutral probability that the P&L at the first expiry is positive, using a lognormal distribution "
                     "at the ATM vol. A guide, not a forecast: real-world odds differ.")
    if math.isfinite(a.max_profit) and math.isfinite(a.max_loss) and a.max_loss < 0:
        rr = f"{a.max_profit / -a.max_loss:.2f} : 1"
    else:
        rr = "Unlimited" if math.isinf(a.max_profit) and math.isfinite(a.max_loss) else "n/a"
    n[1].metric("Reward : risk", rr, help="Max profit divided by max loss.")
    n[2].metric("Delta", f"{g['delta']:,.2f}", help="Equivalent position in units of the underlying.")
    n[3].metric("Cash delta", money(g["cash_delta"], 0), help="Delta × spot: the money exposure to the underlying.")
    o = st.columns(4)
    o[0].metric("Gamma", f"{g['gamma']:,.4f}", help="Change in delta for a 1.00 rise in the underlying.")
    o[1].metric("Theta / day", f"{g['theta']:,.2f}", help="P&L from one calendar day passing, all else equal.")
    o[2].metric("Vega / vol pt", f"{g['vega']:,.2f}", help="P&L if every implied vol rises by one point.")
    o[3].metric("Rho / 1%", f"{g['rho']:,.2f}", help="P&L if interest rates rise by one percentage point.")
if pos.has_multiple_expiries:
    st.caption(f"Legs expire on different dates. Expiry figures are measured at the first expiry (day {pos.horizon:g}), "
               "with the longer-dated legs valued at their remaining Black-Scholes value.")

t_pay, t_grid, t_greeks, t_smile, t_size, t_legs = st.tabs(
    ["Payoff", "Spot × vol grid", "Greeks", "Volatility smile", "Sizing & hedge", "Legs & Python"])

# ---------------------------------------------------------------- payoff
with t_pay:
    c1, c2 = st.columns(2)
    de = c1.slider("Dashed line: days from now", 0, H, 0, key=f"pay_days_{H}")
    dv = c2.slider("Implied vol shift (vol points)", -20, 40, step=1, key="pay_vol")
    a_dv = pos.analyze(vol_shift=dv / 100.0) if dv else a
    st.plotly_chart(charts.payoff_figure(pos, de, dv / 100.0, a_dv.breakevens), key="payoff_chart")
    st.caption("Solid line: P&L at expiry (green above zero, red below). Dashed line: the position marked to the model on the "
               "chosen day with the chosen vol shift. The gap between them is the time value still in the options.")

# ---------------------------------------------------------------- scenario grid
with t_grid:
    c1, c2, c3 = st.columns([1.3, 0.9, 2.2])
    when = c1.slider("Valuation date: days from now", 0, H, 0, key=f"grid_days_{H}",
                     help="0 = an instant shock today. The last value is the first expiry.")
    rng = c2.selectbox("Spot range", [5, 10, 15, 20, 30, 50], key="grid_rng", format_func=lambda v: f"±{v}%")
    vols = c3.multiselect("Implied vol shifts (points)", VOL_SHIFTS, key="grid_vols")
    shifts = np.array(sorted(vols or [0]), dtype=float) / 100.0
    moves = np.round(np.linspace(-rng, rng, 11), 6) / 100.0
    grid = pos.scenario_grid(moves, shifts, when)
    st.plotly_chart(charts.scenario_heatmap(grid), key="grid_chart")
    st.caption("Each cell is the P&L if spot moves by the column's percentage and every implied vol moves by the row's points, "
               "valued on the chosen day. Green = profit, red = loss.")
    st.download_button("Download grid (CSV)", grid.to_csv().encode(), file_name="scenario_grid.csv", mime="text/csv",
                       icon=":material/download:")

# ---------------------------------------------------------------- greeks
with t_greeks:
    gd = st.slider("Days from now", 0, H, 0, key=f"greeks_days_{H}")
    st.plotly_chart(charts.greeks_figure(pos, gd), key="greeks_chart")
    d_txt = (f"A +1.00 move in the underlying {'makes' if g['delta'] >= 0 else 'loses'} about {money(abs(g['delta']))}.")
    if abs(g["gamma"]) < 1e-12:
        g_txt = "No convexity: delta stays the same as spot moves."
    elif g["gamma"] > 0:
        g_txt = "Positive: delta rises as spot rises, so you get longer into rallies and shorter into sell-offs."
    else:
        g_txt = "Negative: delta falls as spot rises, so you get shorter into rallies and longer into sell-offs."
    t_txt = (f"One day passing {'earns' if g['theta'] >= 0 else 'costs'} about {money(abs(g['theta']))}, all else equal.")
    v_txt = (f"If implied vol rises one point (say 25% to 26%) you {'make' if g['vega'] >= 0 else 'lose'} about "
             f"{money(abs(g['vega']))}.")
    rows = [
        ("Delta", f"{g['delta']:,.3f}", "units of underlying", d_txt),
        ("Gamma", f"{g['gamma']:,.5f}", "delta per 1.00 move", g_txt),
        ("Theta", f"{g['theta']:,.2f}", "per calendar day", t_txt),
        ("Vega", f"{g['vega']:,.2f}", "per vol point", v_txt),
        ("Rho", f"{g['rho']:,.2f}", "per 1% rate rise", "Interest-rate sensitivity; matters for long-dated options."),
        ("Cash delta", money(g["cash_delta"], 0), "money", "Delta × spot: the size of the equivalent stock position."),
        ("Dollar gamma", money(g["dollar_gamma_1pct"], 0), "per 1% move", "How much cash delta changes when spot moves 1%."),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Greek", "Value", "Unit", "What it tells you"]), hide_index=True)
    if market.sticky == "moneyness" and not vol.is_flat:
        st.caption("With a sticky-moneyness smile, delta and gamma include the smile's move with spot (the 'smile delta').")

# ---------------------------------------------------------------- smile
with t_smile:
    opts = [l for l in pos.legs if l.is_option]
    if not opts:
        st.info("This position has no option legs, so the smile does not affect it.")
    else:
        if vol.is_flat:
            st.info("The smile is flat: every strike has the same implied vol, as textbook Black-Scholes assumes. "
                    "Choose *Equity index put skew* or *FX-style smile* in the sidebar to see how skew changes prices.")
        expiries = sorted({l.days for l in opts})
        fig = charts.smile_figure(vol, [(d, float(market.forward(d / DAYS_PER_YEAR))) for d in expiries],
                                  [(l.strike, l.days, float(market.forward(l.days / DAYS_PER_YEAR))) for l in opts])
        st.plotly_chart(fig, key="smile_chart")
        rows, effect = [], 0.0
        for l in opts:
            T = l.days / DAYS_PER_YEAR
            F = float(market.forward(T))
            v_smile = float(pos.leg_vol(l, market.spot))
            p_smile = pos.model_price(l)
            p_flat = float(bs_price(l.kind, market.spot, l.strike, T, market.rate, market.div_yield, vol.level))
            eff = l.qty * (p_smile - p_flat) * mult
            effect += eff
            rows.append({"Leg": f"{l.side} {abs(l.qty):g} {l.kind} {l.strike:g}", "Days": l.days,
                         "Std. devs from forward": float(vol.moneyness(l.strike, T, F)), "Smile vol %": v_smile * 100,
                         "ATM vol %": vol.level * 100, "Price on smile": p_smile, "Price at flat ATM vol": p_flat,
                         "Effect on cost": eff})
        st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
            "Std. devs from forward": st.column_config.NumberColumn(format="%.2f"),
            "Smile vol %": st.column_config.NumberColumn(format="%.2f"), "ATM vol %": st.column_config.NumberColumn(format="%.2f"),
            "Price on smile": st.column_config.NumberColumn(format="%.4f"),
            "Price at flat ATM vol": st.column_config.NumberColumn(format="%.4f"),
            "Effect on cost": st.column_config.NumberColumn(format="%.2f")})
        d_strike = Position(pos.legs, replace(market, sticky="strike"), pos.multiplier).greeks()["delta"]
        d_money = Position(pos.legs, replace(market, sticky="moneyness"), pos.multiplier).greeks()["delta"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Smile effect on entry cost", money(effect),
                  help="Cost of the position on the smile minus its cost if every strike traded at the ATM vol. "
                       "Positive: the smile makes the position more expensive to put on.")
        c2.metric("Delta, sticky strike", f"{d_strike:,.2f}")
        c3.metric("Delta, sticky moneyness", f"{d_money:,.2f}")
        st.markdown(
            "**Reading the smile.** Moneyness is measured in standard deviations: ln(K/F) / (ATM vol × √T), so the same "
            "shape applies to every expiry. Equity indices show a *put skew*: crash insurance is in demand, so low strikes "
            "trade at higher vol. That makes put spreads cheaper (you sell the rich lower strike) and risk reversals "
            "costly to buy downside. **Sticky strike vs sticky moneyness** is a hedging question: if the smile slides with "
            "spot, a fixed strike's vol changes as spot moves, which changes the delta you should hedge.")

# ---------------------------------------------------------------- sizing
with t_size:
    st.markdown("**How many lots fit your risk budget?**")
    c1, c2, c3 = st.columns(3)
    budget = c1.number_input("Risk budget (largest loss you accept)", min_value=0.0, step=1000.0, format="%.0f", key="budget")
    basis_label = c2.radio("Size by", ["Max loss at expiry", "Stress scenario"], key="basis",
                           help="Max loss only exists for defined-risk positions. The stress scenario moves spot by up to "
                                "± the range below and shocks implied vol by −5 and +10 points, today and at expiry.")
    stress = c3.slider("Stress: spot move ±%", 5, 50, step=5, key="stress_rng")
    basis = "max loss" if basis_label.startswith("Max") else "stress"
    one_a = one_lot.analyze()
    if basis == "max loss" and not math.isfinite(one_a.max_loss):
        st.warning("This position's maximum loss is unlimited, so it is sized with the stress scenario instead.")
        basis = "stress"
    res = sizing.size_for_budget(one_lot, budget, basis, stress / 100.0)
    k = st.columns(4)
    k[0].metric("Risk per lot", money(res.risk_per_lot), help=f"Basis: {res.basis}.")
    if res.lots is None:
        k[1].metric("Lots that fit", "No limit")
        st.success("This position does not lose money in the sizing scenario, so the budget does not limit its size.")
    else:
        k[1].metric("Lots that fit", f"{res.lots:,}")
        k[2].metric("Risk at that size", money(res.total_risk))
        used = res.risk_per_lot * lots
        share = f" · {used / budget:.0%} of budget" if budget > 0 else ""
        k[3].metric(f"Your {lots} lot{'s' if lots > 1 else ''}{share}", money(used),
                    help="Risk of the size set in the sidebar, on the same basis.")
        if res.lots == 0:
            st.warning("Even one lot exceeds the budget. Use a smaller multiplier, a defined-risk structure or a bigger budget.")
    st.divider()
    st.markdown("**Exposure at your current size**")
    ex = sizing.exposure(pos)
    k = st.columns(4)
    k[0].metric("Option contracts", f"{ex['option_contracts']:,.0f}")
    k[1].metric("Underlying controlled", f"{ex['option_units'] + ex['underlying_units']:,.0f} units",
                help="Contracts × multiplier, summed over all legs.")
    k[2].metric("Notional controlled", money(ex["option_notional"] + ex["underlying_notional"], 0), help="Units controlled × spot.")
    k[3].metric("Delta-equivalent", f"{ex['delta_units']:,.1f} units", help=f"Cash delta {money(ex['delta_notional'], 0)}.")
    st.divider()
    st.markdown("**Delta hedge**")
    c1, c2 = st.columns([1, 2])
    hm = c1.number_input("Hedge instrument multiplier", min_value=0.0001, step=1.0, key="hedge_mult",
                         help="1 = trade the underlying itself. 50 = E-mini S&P 500 futures, 5 = Micro E-mini.")
    h = sizing.delta_hedge(pos, hm)
    verb = "Buy" if h["units"] > 0 else "Sell"
    if abs(h["units"]) < 1e-9:
        c2.success("The position is already delta-neutral.")
    else:
        c2.info(f"**{verb} {abs(h['units']):,.2f} units** of the underlying (cash {money(abs(h['cash']), 0)})"
                + (f", i.e. **{abs(h['contracts']):,.2f} contracts** of the hedge instrument" if hm != 1 else "")
                + " to make the position delta-neutral.")
    st.caption("Delta-neutral only holds for small moves right now: gamma changes delta as spot moves and time passes, "
               "so a hedged option book has to be rebalanced.")

# ---------------------------------------------------------------- legs + code
with t_legs:
    st.dataframe(pos.leg_table(), hide_index=True, column_config={
        "Strike": st.column_config.NumberColumn(format="%.2f"), "Days": st.column_config.NumberColumn(format="%d"),
        "Vol %": st.column_config.NumberColumn(format="%.2f"), "Model price": st.column_config.NumberColumn(format="%.4f"),
        "Entry price": st.column_config.NumberColumn(format="%.4f"), "Delta": st.column_config.NumberColumn(format="%.2f"),
        "Gamma": st.column_config.NumberColumn(format="%.4f"), "Vega / pt": st.column_config.NumberColumn(format="%.2f"),
        "Theta / day": st.column_config.NumberColumn(format="%.2f")})
    st.caption("Quantities include lots; Greeks include the multiplier.")
    st.markdown("**The same position in Python**")
    st.code(position_code(legs, market, mult, lots), language="python")
