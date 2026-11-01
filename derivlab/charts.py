"""Plotly figures for the Strategy Lab and the Credit Structuring Lab.

Every function returns a ``plotly.graph_objects.Figure`` so the same charts work
in Streamlit, Jupyter (``fig.show()``) or as HTML (``fig.write_html(...)``).
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .credit import copula
from .position import DAYS_PER_YEAR, Position
from .volatility import VolModel

GAIN, LOSS, ACCENT, AMBER, INK, FAINT = "#0B8A5E", "#C8432F", "#3555E0", "#B26A00", "#7D8794", "#9AA3B0"
GAIN_FILL, LOSS_FILL = "rgba(11,138,94,0.16)", "rgba(200,67,47,0.15)"
SERIES = [ACCENT, AMBER, "#7A4FD0", GAIN, LOSS, "#1A8FA3", INK, "#C2509A"]


def _base(fig: go.Figure, height: int = 420, **kw) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", **kw)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


def _with_crossings(x: np.ndarray, y: np.ndarray):
    """Insert exact zero crossings so green and red segments meet at zero."""
    xs, ys = [x[0]], [y[0]]
    for i in range(1, len(x)):
        if (y[i - 1] < 0 < y[i]) or (y[i - 1] > 0 > y[i]):
            xc = x[i - 1] - y[i - 1] * (x[i] - x[i - 1]) / (y[i] - y[i - 1])
            xs.append(xc)
            ys.append(0.0)
        xs.append(x[i])
        ys.append(y[i])
    return np.array(xs), np.array(ys)


# ---------------- options ----------------
def payoff_figure(pos: Position, days_elapsed: float = 0.0, vol_shift: float = 0.0, breakevens=(), height: int = 440) -> go.Figure:
    """P&L at the first expiry (green/red) and on an earlier date (dashed)."""
    c = pos.curves(days_elapsed=days_elapsed, vol_shift=vol_shift)
    x, y = _with_crossings(c["spot"].to_numpy(), c["expiry"].to_numpy())
    pos_y = np.where(y >= 0, y, np.nan)
    neg_y = np.where(y <= 0, y, np.nan)
    H = pos.horizon
    exp_name = f"At first expiry (day {H:g})" if pos.has_multiple_expiries else "At expiry"
    now_name = "Today" if days_elapsed <= 0 else f"Day {days_elapsed:g}"
    if vol_shift:
        now_name += f", vol {vol_shift * 100:+g} pts"
        if pos.has_multiple_expiries:
            exp_name += f", vol {vol_shift * 100:+g} pts"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=pos_y, name=exp_name, mode="lines", line=dict(color=GAIN, width=2.6),
                             fill="tozeroy", fillcolor=GAIN_FILL, hovertemplate="%{y:,.2f}<extra>" + exp_name + "</extra>"))
    fig.add_trace(go.Scatter(x=x, y=neg_y, name=exp_name, mode="lines", line=dict(color=LOSS, width=2.6), showlegend=False,
                             fill="tozeroy", fillcolor=LOSS_FILL, hoverinfo="skip"))
    if days_elapsed < H:
        fig.add_trace(go.Scatter(x=c["spot"], y=c["now"], name=now_name, mode="lines", line=dict(color=ACCENT, width=2, dash="dash"),
                                 hovertemplate="%{y:,.2f}<extra>" + now_name + "</extra>"))
    fig.add_hline(y=0, line=dict(color=INK, width=1), opacity=0.6)
    fig.add_vline(x=pos.market.spot, line=dict(color=AMBER, width=1.4, dash="dot"),
                  annotation=dict(text="spot", font=dict(color=AMBER, size=11)), annotation_position="top")
    for k in sorted({l.strike for l in pos.legs if l.is_option}):
        if c["spot"].iloc[0] < k < c["spot"].iloc[-1]:
            fig.add_vline(x=k, line=dict(color=FAINT, width=1, dash="dot"))
    bes = [b for b in breakevens if c["spot"].iloc[0] < b < c["spot"].iloc[-1]]
    if bes:
        fig.add_trace(go.Scatter(x=bes, y=[0] * len(bes), mode="markers", name="Breakeven",
                                 marker=dict(size=9, color="white", line=dict(color=INK, width=2)),
                                 hovertemplate="breakeven %{x:,.2f}<extra></extra>"))
    _base(fig, height)
    fig.update_xaxes(title_text="underlying price")
    fig.update_yaxes(title_text="P&L")
    return fig


def scenario_heatmap(grid, title: str = "") -> go.Figure:
    """Heatmap of a Position.scenario_grid table (rows: vol shifts, columns: spot moves)."""
    z = grid.to_numpy()
    lim = float(np.nanmax(np.abs(z))) or 1.0
    fig = go.Figure(go.Heatmap(z=z, x=[f"{c:+g}%" for c in grid.columns], y=[f"{r:+g} pts" for r in grid.index],
                               zmid=0, zmin=-lim, zmax=lim, colorscale=[[0, LOSS], [0.5, "rgba(140,140,140,0.12)"], [1, GAIN]],
                               texttemplate="%{z:,.0f}", textfont=dict(size=11), colorbar=dict(title="P&L"),
                               hovertemplate="spot %{x}, vol %{y}: %{z:,.2f}<extra></extra>"))
    _base(fig, 60 + 38 * len(grid.index) + 80)
    fig.update_layout(hovermode="closest", title=title or None)
    fig.update_xaxes(title_text="spot move", showgrid=False, type="category")
    fig.update_yaxes(title_text="implied vol shift", showgrid=False, autorange="reversed", type="category")
    return fig


def smile_figure(vol: VolModel, expiries, strikes=()) -> go.Figure:
    """Implied vol against strike.

    ``expiries``: list of ``(days, forward)``, one curve each.
    ``strikes``: list of ``(strike, days, forward)`` to mark the legs on their curve.
    """
    fig = go.Figure()
    F0 = expiries[0][1]
    lo = min([F0 * 0.6] + [k * 0.95 for k, _, _ in strikes])
    hi = max([F0 * 1.4] + [k * 1.05 for k, _, _ in strikes])
    K = np.linspace(lo, hi, 241)
    for i, (days, F) in enumerate(expiries):
        fig.add_trace(go.Scatter(x=K, y=vol.vol(K, days / DAYS_PER_YEAR, F) * 100, mode="lines",
                                 line=dict(color=SERIES[i % len(SERIES)], width=2.4), name=f"{days:g}-day expiry",
                                 hovertemplate="K %{x:,.2f}: %{y:.2f}%<extra>" + f"{days:g}d" + "</extra>"))
    if strikes:
        ks = np.array([k for k, _, _ in strikes])
        vs = np.array([float(vol.vol(k, d / DAYS_PER_YEAR, F)) * 100 for k, d, F in strikes])
        fig.add_trace(go.Scatter(x=ks, y=vs, mode="markers", name="Your strikes",
                                 marker=dict(size=10, color=AMBER, line=dict(color="white", width=1)),
                                 hovertemplate="K %{x:,.2f}: %{y:.2f}%<extra></extra>"))
    fig.add_vline(x=F0, line=dict(color=FAINT, dash="dot"), annotation_text="forward")
    _base(fig, 360)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title_text="strike")
    fig.update_yaxes(title_text="implied vol %")
    return fig


def greeks_figure(pos: Position, days_elapsed: float = 0.0, vol_shift: float = 0.0) -> go.Figure:
    """Delta, gamma, theta and vega of the position across spot."""
    S0 = pos.market.spot
    xs = np.linspace(S0 * 0.7, S0 * 1.3, 121)
    g = [pos.greeks(S=s, days_elapsed=days_elapsed, vol_shift=vol_shift) for s in xs]
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Delta", "Gamma", "Theta per day", "Vega per vol point"),
                        shared_xaxes=True, vertical_spacing=0.14, horizontal_spacing=0.08)
    for (r, cc, key, col) in [(1, 1, "delta", ACCENT), (1, 2, "gamma", AMBER), (2, 1, "theta", LOSS), (2, 2, "vega", GAIN)]:
        fig.add_trace(go.Scatter(x=xs, y=[d[key] for d in g], mode="lines", line=dict(color=col, width=2.2), name=key,
                                 showlegend=False, hovertemplate="%{y:,.3f}<extra>" + key + "</extra>"), row=r, col=cc)
        fig.add_vline(x=S0, line=dict(color=FAINT, dash="dot"), row=r, col=cc)
        fig.add_hline(y=0, line=dict(color=INK, width=1), opacity=0.4, row=r, col=cc)
    _base(fig, 460)
    fig.update_layout(hovermode="x")
    return fig


# ---------------- credit ----------------
def loss_distribution_figure(pd_h: float, rho: float, lgd: float, structure, mc_losses=None) -> go.Figure:
    """Pool loss density (large-pool formula, optional Monte Carlo histogram) with tranche bands."""
    top = min(lgd, max(0.25 * lgd, copula.vasicek_quantile(0.995, pd_h, rho, lgd) * 1.15))
    x = np.linspace(top / 600, top, 600)
    dens = copula.vasicek_pdf(x, pd_h, rho, lgd) / 100.0          # per 1% of pool loss
    fig = go.Figure()
    for i, t in enumerate(structure):
        lo, hi = t.attach * 100, min(t.detach * 100, top * 100)
        if lo < top * 100:
            label = {}
            if hi - lo >= 0.12 * top * 100:                              # only label bands wide enough to read
                label = dict(annotation_text=t.name, annotation_position="top left", annotation=dict(font_size=10))
            fig.add_vrect(x0=lo, x1=hi, fillcolor=SERIES[i % len(SERIES)], opacity=0.08, line_width=0, **label)
    if mc_losses is not None:
        fig.add_trace(go.Histogram(x=np.asarray(mc_losses) * 100, histnorm="probability density", name="Finite pool (Monte Carlo)",
                                   marker_color="rgba(178,106,0,0.45)", xbins=dict(start=0, end=top * 100, size=top * 100 / 80)))
    ymax = float(np.nanmax(dens[5:])) * 1.15 if len(dens) > 5 else None
    fig.add_trace(go.Scatter(x=x * 100, y=dens, mode="lines", line=dict(color=ACCENT, width=2.4), name="Large pool (Vasicek)",
                             hovertemplate="loss %{x:.2f}%: %{y:.4f}<extra></extra>"))
    fig.add_vline(x=pd_h * lgd * 100, line=dict(color=INK, dash="dash"), annotation_text="expected loss",
                  annotation_position="bottom right")
    _base(fig, 400)
    fig.update_xaxes(title_text="pool loss over the life, % of notional", range=[0, top * 100])
    fig.update_yaxes(title_text="probability density (per 1%)", range=[0, ymax] if ymax else None)
    return fig


def el_vs_correlation_figure(pd_h: float, lgd: float, structure, current_rho: float) -> go.Figure:
    """Tranche expected loss as correlation changes: equity falls, senior rises."""
    rhos = np.linspace(0.01, 0.9, 45)
    fig = go.Figure()
    lowest = 100.0
    for i, t in enumerate(structure):
        el = np.array([copula.tranche_el(pd_h, r, lgd, t.attach, t.detach) * 100 for r in rhos])
        el = np.where(el > 1e-6, el, np.nan)                         # log axis: hide numerically zero values
        if np.isfinite(el).any():
            lowest = min(lowest, float(np.nanmin(el)))
        fig.add_trace(go.Scatter(x=rhos, y=el, mode="lines", name=t.name, line=dict(color=SERIES[i % len(SERIES)], width=2.2),
                                 hovertemplate="rho %{x:.2f}: %{y:.4g}%<extra>" + t.name + "</extra>"))
    fig.add_vline(x=current_rho, line=dict(color=FAINT, dash="dot"), annotation_text="your rho")
    _base(fig, 400)
    lo = int(np.floor(np.log10(lowest)))
    step = 1 if 2 - lo <= 5 else 2
    decades = list(range(2, lo - 1, -step))[::-1]
    fig.update_xaxes(title_text="asset correlation rho")
    fig.update_yaxes(title_text="expected loss, % of tranche (log scale)", type="log", range=[lo, 2.05],
                     tickvals=[10.0 ** k for k in decades], ticktext=[_pct_label(k) for k in decades])
    return fig


def _pct_label(k: int) -> str:
    """Label for 10**k percent without scientific notation: 2 -> '100%', -3 -> '0.001%'."""
    return f"{10 ** k:,.0f}%" if k >= 0 else f"{10.0 ** k:.{-k}f}%"


def tranche_balances_figure(periods, names) -> go.Figure:
    fig = go.Figure()
    for i, n in enumerate(names):
        fig.add_trace(go.Scatter(x=periods["month"], y=periods[f"bal: {n}"], name=n, mode="lines", stackgroup="one",
                                 line=dict(width=0.5, color=SERIES[i % len(SERIES)]), hovertemplate="%{y:,.0f}<extra>" + n + "</extra>"))
    _base(fig, 380)
    fig.update_xaxes(title_text="month")
    fig.update_yaxes(title_text="outstanding balance")
    return fig


def collateral_flows_figure(periods) -> go.Figure:
    fig = go.Figure()
    labels = {"scheduled": "Scheduled principal", "prepayments": "Prepayments", "defaults": "Defaults",
              "recoveries": "Recoveries", "residual": "Excess spread to equity"}
    for col, color in [("scheduled", ACCENT), ("prepayments", "#1A8FA3"), ("defaults", LOSS), ("recoveries", GAIN), ("residual", AMBER)]:
        fig.add_trace(go.Scatter(x=periods["month"], y=periods[col], name=labels[col], mode="lines", line=dict(color=color, width=2),
                                 hovertemplate="%{y:,.0f}<extra>" + labels[col] + "</extra>"))
    _base(fig, 380)
    fig.update_xaxes(title_text="month")
    fig.update_yaxes(title_text="cash flow per month")
    return fig


def stress_heatmap(table) -> go.Figure:
    """Principal loss % by tranche (rows, senior first) and default-rate scenario (columns)."""
    z = table.to_numpy(dtype=float)
    fig = go.Figure(go.Heatmap(z=z, x=list(table.columns), y=list(table.index), zmin=0, zmax=100,
                               colorscale=[[0, "rgba(140,140,140,0.10)"], [0.0001, "rgba(200,67,47,0.22)"], [1, LOSS]],
                               texttemplate="%{z:.1f}%", textfont=dict(size=11), showscale=False,
                               hovertemplate="%{y}, %{x}: %{z:.2f}% principal loss<extra></extra>"))
    _base(fig, 60 + 40 * len(table.index) + 60)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title_text="annual default rate (CDR) scenario", showgrid=False, type="category", side="top")
    fig.update_yaxes(showgrid=False, autorange="reversed", type="category")
    return fig


def rwa_figure(table) -> go.Figure:
    """SEC-IRBA risk weight per tranche, sold tranches highlighted."""
    colors = [AMBER if h == "Sold" else ACCENT for h in table["Holder"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=table["Tranche"], y=table["SEC-IRBA RW %"], marker_color=colors, name="Risk weight %",
                         text=[f"{v:,.0f}%" for v in table["SEC-IRBA RW %"]], textposition="outside", cliponaxis=False,
                         hovertemplate="%{x}: %{y:,.1f}%<extra></extra>"))
    _base(fig, 360)
    fig.update_layout(hovermode="closest", showlegend=False)
    ticks = [10, 15, 20, 50, 100, 250, 500, 1250]
    fig.update_yaxes(title_text="risk weight % (log scale; amber = sold)", type="log", range=[np.log10(7), np.log10(3000)],
                     tickvals=ticks, ticktext=[f"{t:,}%" for t in ticks])
    return fig


def srt_sweep_figure(sweep, cost_of_equity: float, current: float | None = None) -> go.Figure:
    """Capital released and cost of relief as the sold tranche gets thicker."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=sweep["Boundary %"], y=sweep["Capital released"], name="Capital released",
                         marker_color="rgba(53,85,224,0.35)", hovertemplate="%{y:,.0f}<extra>capital released</extra>"),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=sweep["Boundary %"], y=sweep["Cost of relief %"], name="Cost of relief %", mode="lines+markers",
                             line=dict(color=AMBER, width=2.4), marker=dict(size=5), hovertemplate="%{y:.2f}%<extra>cost of relief</extra>"),
                  secondary_y=True)
    fig.add_trace(go.Scatter(x=[sweep["Boundary %"].min(), sweep["Boundary %"].max()], y=[cost_of_equity * 100] * 2,
                             mode="lines", name="Cost of equity", line=dict(color=LOSS, dash="dash", width=1.6),
                             hoverinfo="skip"), secondary_y=True)
    if current is not None:
        fig.add_vline(x=current * 100, line=dict(color=FAINT, dash="dot"), annotation_text="your structure")
    _base(fig, 380)
    fig.update_layout(hovermode="x unified")
    costs = sweep["Cost of relief %"].replace([np.inf, -np.inf], np.nan).dropna()
    top = max(2.0 * cost_of_equity * 100, min(float(costs.max()) if len(costs) else 0.0, 4.0 * cost_of_equity * 100)) * 1.1
    fig.update_xaxes(title_text="senior attachment = top of the tranche below it, % of pool")
    fig.update_yaxes(title_text="capital released", secondary_y=False)
    fig.update_yaxes(title_text="cost of relief % per year", secondary_y=True, showgrid=False, range=[0, top],
                     tickmode="auto", nticks=6)
    return fig
