"""Plotly figures for the Monte Carlo pricer."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..charts import ACCENT, AMBER, FAINT, GAIN, INK, LOSS, SERIES, _base

PATH_BLUE = "rgba(53,85,224,0.30)"
PATH_RED = "rgba(200,67,47,0.40)"


def paths_figure(times_years, S, strike=None, barrier=None, touched=None, average: bool = False) -> go.Figure:
    """Sample paths. Paths that touched the barrier are red; ``average=True`` adds the running
    average of the highlighted path (for Asians)."""
    days = np.asarray(times_years) * 365
    fig = go.Figure()
    for i, path in enumerate(S):
        hit = touched is not None and bool(touched[i])
        fig.add_trace(go.Scatter(x=days, y=path, mode="lines", line=dict(width=1.1, color=PATH_RED if hit else PATH_BLUE),
                                 showlegend=False, hoverinfo="skip"))
    first_hit = touched is not None and bool(touched[0])
    fig.add_trace(go.Scatter(x=days, y=S[0], mode="lines", line=dict(width=2.4, color=LOSS if first_hit else ACCENT),
                             name="One path" + (", touched the barrier" if first_hit else "")))
    if average:
        run_avg = np.cumsum(S[0, 1:]) / np.arange(1, S.shape[1])
        fig.add_trace(go.Scatter(x=days[1:], y=run_avg, mode="lines", line=dict(width=2, color=AMBER, dash="dash"),
                                 name="Its running average"))
    if strike is not None:
        fig.add_hline(y=strike, line=dict(color=INK, dash="dot", width=1.2), annotation_text="strike",
                      annotation_position="bottom right")
    if barrier is not None:
        fig.add_hline(y=barrier, line=dict(color=LOSS, dash="dash", width=1.6), annotation_text="barrier",
                      annotation_position="top right", annotation_font_color=LOSS)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=PATH_RED, width=3), name="Touched the barrier"))
    _base(fig, 380)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title_text="days")
    fig.update_yaxes(title_text="price")
    return fig


def payoff_histogram(payoffs, price: float) -> go.Figure:
    y = np.asarray(payoffs, dtype=float)
    zero = float(np.mean(y <= 1e-12))
    pos = y[y > 1e-12]
    fig = go.Figure()
    if len(pos):
        fig.add_trace(go.Histogram(x=pos, nbinsx=60, marker_color="rgba(53,85,224,0.55)", name="Paths that pay",
                                   hovertemplate="payoff %{x:,.2f}: %{y} paths<extra></extra>"))
    fig.add_vline(x=price, line=dict(color=AMBER, width=2), annotation_text=f"average = price {price:,.3f}",
                  annotation_position="top right")
    _base(fig, 380)
    fig.update_layout(hovermode="closest", showlegend=False,
                      title=dict(text=f"{zero:.0%} of paths pay nothing (not shown)", font=dict(size=13)))
    fig.update_xaxes(title_text="discounted payoff of each path")
    fig.update_yaxes(title_text="number of paths")
    return fig


def variance_reduction_figure(table) -> go.Figure:
    t = table.iloc[::-1]
    fig = go.Figure(go.Bar(x=t["Std error"], y=t["Method"], orientation="h", marker_color=[SERIES[i % len(SERIES)] for i in range(len(t))],
                           text=[f"{v:,.1f}× less variance" if v > 1.05 else "baseline" if abs(v - 1) < 0.05 else f"{v:,.2f}×"
                                 for v in t["Variance reduction"]],
                           textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: standard error %{x:.5f}<extra></extra>"))
    _base(fig, 60 + 46 * len(t))
    fig.update_layout(hovermode="closest", showlegend=False, margin=dict(l=10, r=130, t=20, b=10))
    fig.update_xaxes(title_text="standard error at the same number of paths (smaller is better)")
    return fig


def convergence_figure(base, method, benchmark=None, method_label: str = "") -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Estimate with 95% confidence band", "Standard error (log scale)"),
                        horizontal_spacing=0.1)
    for df, label, color, fill in ((base, "Plain", INK, "rgba(125,135,148,0.18)"),
                                   (method, method_label, ACCENT, "rgba(53,85,224,0.18)")):
        if df is None or not len(df) or (label == "Plain" and method_label == "Plain"):
            continue
        fig.add_trace(go.Scatter(x=np.concatenate([df["paths"], df["paths"][::-1]]), y=np.concatenate([df["hi"], df["lo"][::-1]]),
                                 fill="toself", fillcolor=fill, line=dict(width=0), hoverinfo="skip", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["paths"], y=df["estimate"], mode="lines+markers", name=label, line=dict(color=color, width=2),
                                 marker=dict(size=5), hovertemplate="%{x:,} paths: %{y:.4f}<extra>" + label + "</extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["paths"], y=df["stderr"], mode="lines+markers", name=label, showlegend=False,
                                 line=dict(color=color, width=2), marker=dict(size=5),
                                 hovertemplate="%{x:,} paths: %{y:.5f}<extra>" + label + "</extra>"), row=1, col=2)
    ref = base if base is not None and len(base) else method
    if ref is not None and len(ref):
        n0, s0 = float(ref["paths"].iloc[0]), float(ref["stderr"].iloc[0])
        xs = np.array([n0, float(ref["paths"].iloc[-1])])
        fig.add_trace(go.Scatter(x=xs, y=s0 * np.sqrt(n0 / xs), mode="lines", name="1/√N", line=dict(color=FAINT, dash="dash")),
                      row=1, col=2)
    if benchmark is not None:
        fig.add_hline(y=benchmark, line=dict(color=GAIN, dash="dash"), row=1, col=1, annotation_text="formula",
                      annotation_position="top right")
    _base(fig, 400)
    fig.update_layout(hovermode="closest", margin=dict(l=10, r=10, t=40, b=70),
                      legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0))
    fig.update_xaxes(type="log", dtick=1, title_text="paths")
    fig.update_yaxes(type="log", dtick=1, row=1, col=2)
    return fig


def boundary_figure(lsm: dict, binomial: dict | None, strike: float) -> go.Figure:
    fig = go.Figure()
    if binomial is not None and len(binomial["boundary_t"]):
        fig.add_trace(go.Scatter(x=binomial["boundary_t"] * 365, y=binomial["boundary_S"], mode="lines", name="Binomial tree",
                                 line=dict(color=INK, width=2)))
    if len(lsm["boundary_t"]):
        fig.add_trace(go.Scatter(x=lsm["boundary_t"] * 365, y=lsm["boundary_S"], mode="markers", name="Longstaff-Schwartz",
                                 marker=dict(color=AMBER, size=7)))
    fig.add_hline(y=strike, line=dict(color=FAINT, dash="dot"), annotation_text="strike")
    _base(fig, 380)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title_text="days from today")
    fig.update_yaxes(title_text="exercise below this price (put) / above (call)")
    return fig


def correlation_figure(sweep, current_rho: float) -> go.Figure:
    fig = go.Figure()
    lo, hi = sweep["price"] - 1.96 * sweep["stderr"], sweep["price"] + 1.96 * sweep["stderr"]
    fig.add_trace(go.Scatter(x=np.concatenate([sweep["rho"], sweep["rho"][::-1]]), y=np.concatenate([hi, lo[::-1]]),
                             fill="toself", fillcolor="rgba(53,85,224,0.15)", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=sweep["rho"], y=sweep["price"], mode="lines+markers", name="Monte Carlo", line=dict(color=ACCENT, width=2)))
    if sweep["benchmark"].notna().any():
        fig.add_trace(go.Scatter(x=sweep["rho"], y=sweep["benchmark"], mode="lines", name="Formula", line=dict(color=GAIN, dash="dash")))
    fig.add_vline(x=current_rho, line=dict(color=FAINT, dash="dot"), annotation_text="your correlation")
    _base(fig, 360)
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(title_text="correlation between every pair of assets")
    fig.update_yaxes(title_text="price")
    return fig


def model_smile_figure(model_df, user_df, model_name: str) -> go.Figure:
    fig = go.Figure()
    for i, (days, grp) in enumerate(model_df.groupby("days")):
        c = SERIES[i % len(SERIES)]
        fig.add_trace(go.Scatter(x=grp["strike"], y=grp["iv"] * 100, mode="lines", name=f"{model_name}, {days:g} days",
                                 line=dict(color=c, width=2.4), hovertemplate="K %{x:,.2f}: %{y:.2f}%<extra></extra>"))
        u = user_df[user_df["days"] == days]
        fig.add_trace(go.Scatter(x=u["strike"], y=u["iv"] * 100, mode="lines", name=f"Your smile, {days:g} days",
                                 line=dict(color=c, width=1.6, dash="dash"), hovertemplate="K %{x:,.2f}: %{y:.2f}%<extra></extra>"))
    vols = np.concatenate([model_df["iv"].to_numpy(), user_df["iv"].to_numpy()]) * 100
    vols = vols[np.isfinite(vols)]
    _base(fig, 380)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title_text="strike")
    if len(vols):                                  # keep a flat smile from zooming in on rounding noise
        fig.update_yaxes(title_text="implied vol %", range=[float(vols.min()) - 2, float(vols.max()) + 2])
    return fig


def pnl_histogram(pnl, var: float, es: float, alpha: float) -> go.Figure:
    pnl = np.asarray(pnl, dtype=float)
    fig = go.Figure(go.Histogram(x=pnl, nbinsx=80, marker_color="rgba(53,85,224,0.55)",
                                 hovertemplate="P&L %{x:,.0f}: %{y} paths<extra></extra>"))
    fig.add_vline(x=-var, line=dict(color=AMBER, width=2), annotation_text=f"VaR {alpha:.1%}", annotation_position="top left")
    fig.add_vline(x=-es, line=dict(color=LOSS, width=2, dash="dash"), annotation_text="expected shortfall",
                  annotation_position="bottom left")
    fig.add_vline(x=float(pnl.mean()), line=dict(color=INK, dash="dot"), annotation_text="mean", annotation_position="top right")
    _base(fig, 380)
    fig.update_layout(hovermode="closest", showlegend=False)
    fig.update_xaxes(title_text="P&L over the horizon")
    fig.update_yaxes(title_text="number of paths")
    return fig


def pnl_scatter(S_h, pnl, spot: float, n: int = 3000) -> go.Figure:
    """Simulated P&L against the simulated spot: the curvature is why delta-only VaR misleads."""
    idx = np.linspace(0, len(S_h) - 1, min(n, len(S_h))).astype(int)
    fig = go.Figure(go.Scattergl(x=S_h[idx], y=np.asarray(pnl)[idx], mode="markers", marker=dict(size=4, color="rgba(53,85,224,0.35)"),
                                 hovertemplate="spot %{x:,.2f}: P&L %{y:,.0f}<extra></extra>"))
    fig.add_vline(x=spot, line=dict(color=AMBER, dash="dot"), annotation_text="spot today")
    fig.add_hline(y=0, line=dict(color=INK, width=1), opacity=0.5)
    _base(fig, 380)
    fig.update_layout(hovermode="closest", showlegend=False)
    fig.update_xaxes(title_text="simulated spot at the horizon")
    fig.update_yaxes(title_text="P&L")
    return fig
