"""Credit structuring lab: design tranches, then test their risk, cash flows and capital relief."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st

from derivlab import charts
from derivlab.credit import capital, copula, design, structures, waterfall
from derivlab.credit.structures import CapitalStructure, Tranche
from derivlab.fmt import compact, pct

ASSET_CLASSES = {"Corporate": "corporate", "Retail: residential mortgages": "retail_mortgage",
                 "Retail: qualifying revolving (cards)": "retail_qrre", "Retail: other": "retail_other"}
AMORT = ["Level pay", "Bullet"]
POOL_MODELS = ["Large pool (closed form)", "Finite pool (Monte Carlo)"]
CS_COLUMNS = {
    "Tranche": st.column_config.TextColumn("Tranche", required=True),
    "Attach %": st.column_config.NumberColumn("Attach %", min_value=0.0, max_value=100.0, format="%.2f",
                                              help="Pool loss at which this tranche starts losing money."),
    "Detach %": st.column_config.NumberColumn("Detach %", min_value=0.0, max_value=100.0, format="%.2f",
                                              help="Pool loss at which this tranche is wiped out."),
    "Coupon %": st.column_config.NumberColumn("Coupon %", min_value=0.0, format="%.3f",
                                              help="Annual note coupon (waterfall) or protection premium (SRT)."),
    "Holder": st.column_config.SelectboxColumn("Holder", options=["Retained", "Sold"],
                                               help="For SRT: Sold = the bank buys protection on this tranche from investors."),
}
DEFAULTS = dict(cs_preset="Bank SRT (synthetic)", cr_pd=1.2, cr_rho=0.20, cr_lgd=40.0, cr_T=3.0, cr_r=3.0, cr_mode=POOL_MODELS[0],
                cr_n=200, cr_sims=50_000, cr_lumpy=False, as_mezz=30.0, as_senior=0.5,
                wf_bal=500.0, wf_wac=8.5, wf_term=60, wf_amort=AMORT[0], wf_cpr=10.0, wf_cdr=2.0, wf_sev=50.0, wf_lag=6,
                wf_fee=0.5, srt_ead=2000.0, srt_pd=1.2, srt_lgd=40.0, srt_m=2.5, srt_ac="Corporate", srt_n=300,
                srt_mt=3.0, srt_sts=False, srt_cap=12.0, srt_coe=12.0)

ss = st.session_state
for _k, _v in DEFAULTS.items():
    ss.setdefault(_k, _v)


def load_structure(df: pd.DataFrame):
    ss.cs_df = df
    ss.cs_saved = df
    ss.cs_ver = ss.get("cs_ver", 0) + 1


def load_preset():
    load_structure(structures.preset(ss.cs_preset).to_frame())


def load_named(name: str):
    ss.cs_preset = name
    load_preset()


if "cs_df" not in ss:
    load_preset()
elif f"cs_{ss.cs_ver}" not in ss and "cs_saved" in ss:
    ss.cs_df = ss.cs_saved          # coming back from another page: restore the last edits


@st.cache_data(show_spinner=False, max_entries=20)
def mc_losses(n: int, pd_h: float, rho: float, lgd: float, sims: int, lumpy: bool) -> np.ndarray:
    exposures = np.random.default_rng(11).lognormal(0.0, 1.0, int(n)) if lumpy else None
    return copula.simulate_losses(int(n), pd_h, rho, lgd, int(sims), exposures=exposures, seed=7)


@st.cache_data(show_spinner=False, max_entries=50)
def wf_breakevens(pool_key: tuple, cs_key: tuple) -> dict:
    pool = waterfall.PoolAssumptions(**dict(pool_key))
    struct = CapitalStructure([Tranche(*t) for t in cs_key])
    return {t.name: waterfall.breakeven_cdr(pool, struct, t.name) for t in struct}


def num(fmt: str):
    return st.column_config.NumberColumn(format=fmt)


# ---------------------------------------------------------------- structure editor
st.title("Credit structuring lab")
st.caption("Design a capital structure, then test it three ways: how risky each tranche is (one-factor Gaussian copula), "
           "how cash flows through the deal (a monthly waterfall) and how much bank capital a synthetic risk transfer "
           "releases (SEC-IRBA).")

c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
c1.selectbox("Template", list(structures.PRESETS), key="cs_preset", on_change=load_preset)
c2.button("Reload template", on_click=load_preset, icon=":material/restart_alt:", width="stretch")
st.markdown("**Capital structure** · tranches stack from 0% to 100% of the pool, and the bottom one takes the first losses")
edited = st.data_editor(ss.cs_df, key=f"cs_{ss.cs_ver}", num_rows="dynamic", hide_index=True,
                        column_config=CS_COLUMNS, column_order=list(CS_COLUMNS))
ss.cs_saved = edited
try:
    cs = CapitalStructure.from_frame(edited)
except ValueError as e:
    st.error(f"The capital structure needs fixing: {e}.")
    st.stop()
st.caption("Coupon % is the note coupon in the waterfall and the annual premium paid on sold tranches in the SRT tab.")

t_risk, t_wf, t_srt = st.tabs(["Tranche risk & pricing", "Cash-flow waterfall", "SRT capital relief"])

# ---------------------------------------------------------------- tranche risk
with t_risk:
    st.markdown("Each borrower defaults when a mix of one common factor (the economy) and its own luck falls below a "
                "threshold set by its default probability. **Correlation** decides how much defaults cluster, and "
                "clustering moves risk between tranches: it makes the equity safer and the senior tranches riskier.")
    c = st.columns(5)
    c[0].number_input("Annual default prob. %", min_value=0.01, max_value=50.0, step=0.1, format="%.2f", key="cr_pd")
    c[1].slider("Asset correlation ρ", 0.01, 0.95, step=0.01, key="cr_rho")
    c[2].number_input("Loss given default %", min_value=1.0, max_value=100.0, step=5.0, format="%.1f", key="cr_lgd")
    c[3].number_input("Maturity (years)", min_value=0.25, max_value=30.0, step=0.25, format="%.2f", key="cr_T")
    c[4].number_input("Discount rate %", min_value=0.0, max_value=20.0, step=0.25, format="%.2f", key="cr_r")
    pd_a, rho, lgd, T, r = ss.cr_pd / 100, float(ss.cr_rho), ss.cr_lgd / 100, float(ss.cr_T), ss.cr_r / 100
    pd_h = copula.cumulative_pd(pd_a, T)
    st.radio("Pool model", POOL_MODELS, key="cr_mode", horizontal=True,
             help="The large-pool formula (Vasicek) assumes infinitely many small loans. Monte Carlo simulates a "
                  "finite pool, where a single default is a visible jump.")
    losses = None
    if ss.cr_mode == POOL_MODELS[1]:
        c = st.columns(3, vertical_alignment="bottom")
        c[0].number_input("Number of loans", min_value=5, max_value=5000, step=5, key="cr_n")
        c[1].select_slider("Simulations", options=[10_000, 25_000, 50_000, 100_000], key="cr_sims", format_func=lambda v: f"{v:,}")
        c[2].toggle("Unequal loan sizes", key="cr_lumpy", help="Draw loan sizes from a lognormal distribution so a few big loans dominate.")
        losses = mc_losses(int(ss.cr_n), pd_h, rho, lgd, int(ss.cr_sims), bool(ss.cr_lumpy))
    k = st.columns(4)
    k[0].metric("Default prob. over the life", pct(pd_h, 2), help=f"1 - (1 - annual PD)^{T:g}")
    k[1].metric("Pool expected loss", pct(pd_h * lgd, 2))
    q99 = float(np.quantile(losses, 0.99)) if losses is not None else copula.vasicek_quantile(0.99, pd_h, rho, lgd)
    q999 = float(np.quantile(losses, 0.999)) if losses is not None else copula.vasicek_quantile(0.999, pd_h, rho, lgd)
    k[2].metric("Pool loss, 1-in-100", pct(q99, 2), help="Exceeded with 1% probability over the life of the deal.")
    k[3].metric("Pool loss, 1-in-1,000", pct(q999, 2), help="Exceeded with 0.1% probability over the life of the deal.")

    rep = design.tranche_report(pd_a, rho, lgd, T, cs, r, losses)
    report_cfg = {"Attach %": num("%.2f"), "Detach %": num("%.2f"), "P(loss) %": num("%.3f"), "P(wipeout) %": num("%.3f"),
                  "EL %": num("%.3f"), "Loss if hit %": num("%.1f"), "Fair spread bp": num("%.1f"), "Coupon bp": num("%.1f"),
                  "Excess bp": num("%+.1f")}
    st.dataframe(rep, hide_index=True, column_config=report_cfg)
    st.caption("P(loss): probability the tranche loses anything. P(wipeout): probability it loses everything. EL: expected "
               "loss as % of the tranche. All over the whole life. Fair spread: the running premium that pays for the "
               "expected loss (quarterly, large-pool model). Excess = coupon - fair spread: what investors earn for "
               "bearing the risk beyond its expected cost.")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.loss_distribution_figure(pd_h, rho, lgd, cs, losses), key="loss_dist")
    with c2:
        st.plotly_chart(charts.el_vs_correlation_figure(pd_h, lgd, cs, rho), key="el_rho")
    st.caption("Left: how likely each level of pool loss is, with the tranche bands shaded. Right: the correlation trade. "
               "Higher correlation fattens the tail, so senior expected loss rises while equity expected loss falls.")

    st.subheader("Structuring assistant")
    st.markdown("How much subordination does each tranche need? In this model a tranche that attaches at the 99.5th "
                "percentile of pool losses takes a loss with 0.5% probability, so you can size tranches by their "
                "**probability of any loss over the deal's life**. Rating models ask a similar question, with many more "
                "assumptions.")
    c = st.columns([1, 1, 1.4], vertical_alignment="bottom")
    c[0].number_input("Mezzanine: max P(loss) %", min_value=0.1, max_value=99.0, step=1.0, format="%.1f", key="as_mezz")
    c[1].number_input("Senior: max P(loss) %", min_value=0.001, max_value=50.0, step=0.1, format="%.3f", key="as_senior")
    try:
        proposal = design.size_by_loss_probability(pd_a, rho, lgd, T, [("Mezzanine", ss.as_mezz / 100), ("Senior", ss.as_senior / 100)], rate=r)
    except ValueError as e:
        proposal = None
        st.warning(f"{e}.")
    if proposal is not None:
        c[2].button("Use this structure", on_click=load_structure, args=(proposal.to_frame(),), icon=":material/check:",
                    type="primary", width="stretch")
        st.dataframe(design.tranche_report(pd_a, rho, lgd, T, proposal, r), hide_index=True, column_config=report_cfg)
        st.caption("Attachment points are rounded up to the next 0.25% (more protection, never less). Coupons are set to "
                   "each tranche's fair spread and holders follow the usual SRT shape: first loss retained, mezzanine "
                   "sold, senior retained. Using it replaces the structure at the top of the page.")

# ---------------------------------------------------------------- waterfall
with t_wf:
    st.markdown("A cash securitisation passes the pool's cash to investors in strict order. Each month, interest pays the "
                "servicing fee, then note coupons from the top down. What is left, the **excess spread**, first covers the "
                "month's losses and then goes to the equity holder. Principal (scheduled payments, prepayments and "
                "recoveries) repays the notes **sequentially**, senior first. Losses that excess spread cannot cover write "
                "the notes down **from the bottom up**. The bottom tranche is the equity: no coupon, just what is left.")
    c = st.columns(5)
    c[0].number_input("Pool balance (millions)", min_value=0.1, step=50.0, format="%.1f", key="wf_bal")
    c[1].number_input("Pool coupon (WAC) %", min_value=0.0, max_value=40.0, step=0.25, format="%.2f", key="wf_wac")
    c[2].number_input("Term (months)", min_value=6, max_value=480, step=6, key="wf_term")
    c[3].selectbox("Amortisation", AMORT, key="wf_amort",
                   help="Level pay: equal monthly payments, like auto loans and mortgages. Bullet: all principal at maturity.")
    c[4].number_input("Servicing fee % / yr", min_value=0.0, max_value=5.0, step=0.05, format="%.2f", key="wf_fee")
    c = st.columns(4)
    c[0].number_input("Default rate (CDR) % / yr", min_value=0.0, max_value=100.0, step=0.5, format="%.2f", key="wf_cdr")
    c[1].number_input("Prepayment rate (CPR) % / yr", min_value=0.0, max_value=100.0, step=1.0, format="%.1f", key="wf_cpr")
    c[2].number_input("Loss severity %", min_value=0.0, max_value=100.0, step=5.0, format="%.1f", key="wf_sev")
    c[3].number_input("Recovery lag (months)", min_value=0, max_value=60, step=1, key="wf_lag")
    pool = waterfall.PoolAssumptions(balance=float(ss.wf_bal) * 1e6, wac=ss.wf_wac / 100, term_months=int(ss.wf_term),
                                     amortization="level" if ss.wf_amort == AMORT[0] else "bullet", cpr=ss.wf_cpr / 100,
                                     cdr=ss.wf_cdr / 100, severity=ss.wf_sev / 100, recovery_lag=int(ss.wf_lag),
                                     servicing_fee=ss.wf_fee / 100)
    notes = cs.tranches[1:]
    if any(t.coupon == 0 for t in notes):
        c1, c2 = st.columns([4, 1], vertical_alignment="center")
        c1.info("Some notes above the equity pay no coupon, as in the synthetic SRT template where the bank keeps the "
                "senior risk itself, so the equity collects almost all the interest. For a cash deal, use a cash template.")
        c2.button("Load Auto ABS", on_click=load_named, args=("Auto ABS",), icon=":material/swap_horiz:", width="stretch")
    wf = waterfall.run_waterfall(pool, cs)
    per = wf.periods
    excess = pool.wac - pool.servicing_fee - sum(t.thickness * t.coupon for t in notes)
    total_loss = float(per["losses"].sum())
    eq_irr = float(wf.summary["IRR %"].iloc[-1])
    k = st.columns(4)
    k[0].metric("Day-one excess spread", pct(excess, 2) + " a year",
                help="Pool coupon minus servicing fee minus the note coupons, all as % of the pool: the first line of defence.")
    k[1].metric("Lifetime pool losses", pct(total_loss / pool.balance, 2), help="Defaults × severity, as % of the original pool.")
    k[2].metric("Covered by excess spread", pct(float(per["loss_cover"].sum()) / total_loss) if total_loss > 0 else "n/a",
                help="Share of losses paid out of excess spread rather than written off against the tranches.")
    k[3].metric("Equity IRR", ("n/a" if not math.isfinite(eq_irr) else "> 1,000%" if eq_irr > 1000 else pct(eq_irr / 100)),
                help="Annualised return on the equity tranche bought at par.")
    be = wf_breakevens(tuple(sorted(pool.__dict__.items())), cs.to_tuples())
    summ = wf.summary.copy()
    summ["Breakeven CDR %"] = [be[n] * 100 for n in summ["Tranche"]]
    view = summ.drop(columns=["Attach %", "Detach %"]).rename(columns={
        "Interest received": "Interest", "Principal received": "Principal", "Residual cash": "Residual",
        "Principal loss %": "Loss %", "Interest shortfall": "Shortfall", "WAL (yrs)": "WAL yrs", "Breakeven CDR %": "BE CDR %"})
    st.dataframe(view, hide_index=True, column_config={
        "Coupon %": num("%.2f"), "Original": num("compact"), "Interest": num("compact"), "Principal": num("compact"),
        "Residual": num("compact"), "Loss %": num("%.2f"), "Shortfall": num("compact"), "WAL yrs": num("%.2f"),
        "IRR %": num("%.2f"), "BE CDR %": num("%.2f")})
    st.caption("Residual: excess spread and leftover cash paid to the equity. Loss %: principal not repaid. Shortfall: coupon "
               "not paid. WAL: weighted average life of principal repayments. BE CDR: breakeven default rate, the highest "
               "constant annual default rate at which the tranche still gets all its principal back.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Tranche balances** · sequential pay retires the senior first")
        st.plotly_chart(charts.tranche_balances_figure(per, [t.name for t in cs]), key="wf_bal_chart")
    with c2:
        st.markdown("**Collateral cash flows**")
        st.plotly_chart(charts.collateral_flows_figure(per), key="wf_flow_chart")
    st.markdown("**Stress test** · principal loss by tranche as the default rate rises (other assumptions unchanged)")
    st.plotly_chart(charts.stress_heatmap(waterfall.stress_table(pool, cs)), key="wf_stress")
    with st.expander("Monthly cash flows"):
        st.dataframe(per, hide_index=True, column_config={c: num("localized") for c in per.columns if c != "month"})
        st.download_button("Download (CSV)", per.to_csv(index=False).encode(), file_name="waterfall_cash_flows.csv",
                           mime="text/csv", icon=":material/download:")

# ---------------------------------------------------------------- SRT
with t_srt:
    st.markdown("In a **synthetic risk transfer** a bank keeps a loan portfolio on its balance sheet but buys credit "
                "protection on one slice, usually the mezzanine, from investors who post cash collateral. If the transfer "
                "is recognised, the capital charge falls from the IRB requirement on the whole pool to securitisation risk "
                "weights (SEC-IRBA) on the tranches the bank still holds. Mark the tranches you sell as **Sold** in the "
                "structure table.")
    c = st.columns(5)
    c[0].number_input("Portfolio EAD (millions)", min_value=1.0, step=100.0, format="%.0f", key="srt_ead")
    c[1].number_input("1-year PD %", min_value=0.03, max_value=50.0, step=0.1, format="%.2f", key="srt_pd")
    c[2].number_input("LGD %", min_value=1.0, max_value=100.0, step=5.0, format="%.1f", key="srt_lgd")
    c[3].selectbox("Asset class", list(ASSET_CLASSES), key="srt_ac")
    corp = ASSET_CLASSES[ss.srt_ac] == "corporate"
    c[4].number_input("Loan maturity M (years)", min_value=1.0, max_value=5.0, step=0.5, format="%.1f", key="srt_m",
                      disabled=not corp, help="Effective maturity of the loans; only used for corporate exposures.")
    c = st.columns(5, vertical_alignment="bottom")
    c[0].number_input("Effective number of loans N", min_value=1, max_value=1_000_000, step=10, key="srt_n",
                      help="Wholesale pools with N of 25 or more count as granular.")
    c[1].number_input("Tranche maturity MT (years)", min_value=1.0, max_value=5.0, step=0.5, format="%.1f", key="srt_mt",
                      help="The rules floor it at 1 year and cap it at 5.")
    c[2].number_input("Capital ratio %", min_value=1.0, max_value=30.0, step=0.5, format="%.1f", key="srt_cap",
                      help="Capital held per unit of risk-weighted assets, e.g. the CET1 target including buffers.")
    c[3].number_input("Cost of equity %", min_value=0.0, max_value=40.0, step=0.5, format="%.1f", key="srt_coe")
    c[4].toggle("STS securitisation", key="srt_sts", help="Simple, transparent and standardised (EU): p is halved "
                "(floor 0.3) and the senior floor falls from 15% to 10%.")
    inp = capital.SRTInputs(ead=float(ss.srt_ead) * 1e6, pd=ss.srt_pd / 100, lgd=ss.srt_lgd / 100, maturity=float(ss.srt_m),
                            asset_class=ASSET_CLASSES[ss.srt_ac], n_eff=float(ss.srt_n), tranche_maturity=float(ss.srt_mt),
                            sts=bool(ss.srt_sts), capital_ratio=ss.srt_cap / 100, cost_of_equity=ss.srt_coe / 100)
    res = capital.srt_economics(inp, cs)
    if not any(t.holder.lower().startswith("sold") for t in cs):
        c1, c2 = st.columns([4, 1], vertical_alignment="center")
        c1.info("No tranche is marked Sold, so no risk is transferred. Set Holder to Sold on the tranche(s) you buy "
                "protection on, or load the Bank SRT template.")
        c2.button("Load Bank SRT", on_click=load_named, args=("Bank SRT (synthetic)",), icon=":material/swap_horiz:",
                  width="stretch")
    k = st.columns(4)
    k[0].metric("IRB capital K", pct(res["k"], 2), help="Unexpected-loss capital per unit of exposure (Basel IRB formula).")
    k[1].metric("KIRB", pct(res["kirb"], 2), help="K plus expected loss (PD × LGD). Any tranche detaching below KIRB gets 1,250%.")
    k[2].metric("RWA before", compact(res["rwa_before"]))
    k[3].metric("RWA after", compact(res["rwa_after"]),
                delta=f"{-res['rwa_reduction'] / res['rwa_before']:.0%}" if res["rwa_before"] > 0 else None, delta_color="inverse")
    k = st.columns(4)
    k[0].metric("Capital released", compact(res["capital_released"]), help="(RWA before - RWA after) × capital ratio.")
    k[1].metric("Annual protection cost", compact(res["annual_cost"]), help="Premium on the sold tranches.")
    cor = res["cost_of_relief"]
    k[2].metric("Cost of capital relief", pct(cor, 1) if math.isfinite(cor) else "n/a",
                help="Annual premium ÷ capital released: what each unit of freed capital costs per year.")
    k[3].metric("Cost of equity", pct(inp.cost_of_equity, 1))
    if res["capital_released"] > 0 and math.isfinite(cor):
        if res["economic"]:
            st.success(f"Economic: freeing capital this way costs {pct(cor, 1)} a year, less than the {pct(inp.cost_of_equity, 1)} "
                       "cost of equity it replaces.")
        else:
            st.warning(f"Not economic: the relief costs {pct(cor, 1)} a year, more than the {pct(inp.cost_of_equity, 1)} cost of equity.")
    srt_view = res["table"].copy()
    srt_view["p"] = [f"{v:.3f}" if math.isfinite(v) else "-" for v in srt_view["p"]]
    st.dataframe(srt_view, hide_index=True, column_config={
        "Attach %": num("%.2f"), "Detach %": num("%.2f"), "Notional": num("compact"), "SEC-IRBA RW %": num("%.1f"),
        "RWA": num("compact"), "Annual premium": num("compact")})
    st.caption("p is the SEC-IRBA supervisory parameter: the higher it is, the more capital a tranche above KIRB attracts. "
               "Tranches detaching below KIRB get 1,250%, i.e. are effectively deducted from capital.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Risk weight by tranche**")
        st.plotly_chart(charts.rwa_figure(res["table"]), key="srt_rw")
    with c2:
        if len(cs) >= 2:
            below = cs.tranches[-2]
            st.markdown(f"**How thick should the sold tranche be?** · moving the top of *{below.name}* ({below.holder.lower()})")
            hi = max(below.attach + 0.05, min(0.30, 3 * cs.senior.attach))
            sweep = capital.senior_boundary_sweep(inp, cs, np.linspace(below.attach + 0.0025, hi, 36))
            if len(sweep):
                st.plotly_chart(charts.srt_sweep_figure(sweep, inp.cost_of_equity, cs.senior.attach), key="srt_sweep")
    st.caption("A thicker sold tranche releases more capital, but once the senior tranche reaches its risk-weight floor the "
               "extra premium buys little extra relief, so the cost of relief rises.")
    with st.expander("What this calculator leaves out"):
        st.markdown("A learning tool, not a regulatory calculator. It ignores the significant-risk-transfer tests, maturity "
                    "mismatch between protection and loans, synthetic excess spread, the output floor, counterparty "
                    "treatment of non-collateralised protection, and supervisory approval. Sold tranches are assumed "
                    "fully cash-collateralised (0% risk weight).")
