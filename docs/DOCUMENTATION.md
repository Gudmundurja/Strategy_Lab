# Strategy Lab: technical documentation

This document explains what the Strategy Lab computes, how, and why it was built that way. It covers every
financial concept the code uses, the design decisions behind the models and the app, and the limitations you
should keep in mind when reading its numbers.

**Contents**

1. [Purpose and scope](#1-purpose-and-scope)
2. [Architecture](#2-architecture)
3. [Conventions and units](#3-conventions-and-units)
4. [Options: pricing](#4-options-pricing)
5. [Options: the volatility smile](#5-options-the-volatility-smile)
6. [Options: positions and their analysis](#6-options-positions-and-their-analysis)
7. [Options: sizing and hedging](#7-options-sizing-and-hedging)
8. [Options: the strategy library](#8-options-the-strategy-library)
9. [Monte Carlo pricing](#9-monte-carlo-pricing)
10. [Credit: capital structures](#10-credit-capital-structures)
11. [Credit: the Gaussian copula and tranche risk](#11-credit-the-gaussian-copula-and-tranche-risk)
12. [Credit: the structuring assistant](#12-credit-the-structuring-assistant)
13. [Credit: the cash-flow waterfall](#13-credit-the-cash-flow-waterfall)
14. [Credit: regulatory capital and SRT economics](#14-credit-regulatory-capital-and-srt-economics)
15. [The Streamlit app](#15-the-streamlit-app)
16. [Testing and verification](#16-testing-and-verification)
17. [Limitations](#17-limitations)
18. [Glossary](#18-glossary)
19. [References](#19-references)

---

## 1. Purpose and scope

The Strategy Lab is a learning and first-pass analysis tool with three parts:

* **Options strategy lab.** Build a multi-leg option position, then see its payoff, Greeks, a spot × volatility
  scenario grid, the effect of a volatility smile, and how to size and hedge it.
* **Monte Carlo pricer.** Price exotic, American and multi-asset options by simulation under Black-Scholes,
  Heston or Merton dynamics, compare variance-reduction methods, estimate Greeks three ways, reprice the
  options-lab position and measure its value-at-risk and expected shortfall.
* **Credit structuring lab.** Design a tranched capital structure on a loan pool, then test it three ways:
  tranche risk and pricing in a one-factor Gaussian copula, cash flows in a monthly waterfall, and the
  regulatory capital a bank frees up through a synthetic risk transfer (SRT).

The guiding principle is **transparency over sophistication**. Every model is one you can write down on a page
and check by hand: Black-Scholes, a quadratic smile, the Vasicek loss distribution, a sequential waterfall, the
Basel formulas, and the Monte Carlo models are checked against closed forms wherever one exists. Where a richer
model is standard in practice (local volatility, base correlation, full deal waterfalls), this document says so in
[Limitations](#17-limitations).

It is not a trading, pricing, risk-management or regulatory reporting system, and nothing in it is investment
advice.

---

## 2. Architecture

```
app.py                     Streamlit entry point: page config, navigation, state persistence
views/shared.py            options-lab state shared with the Monte Carlo page
views/strategy_lab.py      options page (UI only)
views/montecarlo_lab.py    Monte Carlo page (UI only)
views/credit_lab.py        credit page (UI only)
derivlab/                  the library: no Streamlit imports anywhere
  pricing.py               Black-Scholes-Merton, Greeks, implied vol, binomial tree
  volatility.py            VolModel: the smile
  position.py              Leg, Market, Position, Analysis
  presets.py               34 strategies with notes
  sizing.py                lots for a budget, delta hedge, exposure
  charts.py                Plotly figures
  fmt.py                   number formatting
  credit/
    structures.py          Tranche, CapitalStructure, templates
    copula.py              Vasicek / Gaussian copula, Monte Carlo, fair spreads
    design.py              structuring assistant, tranche report
    waterfall.py           cash-flow waterfall, breakeven CDR, stress table
    capital.py             IRB, SEC-IRBA, SRT economics
  montecarlo/
    rng.py                 pseudo-random, antithetic and Sobol' normals
    models.py              GBM, Heston, Merton simulators; Brownian bridge
    payoffs.py             European, digital, Asian, barrier, lookback, basket, spread, best/worst-of
    engine.py              the estimator: chunks, control variates, standard errors, convergence
    pricer.py              high-level pricing, benchmarks, method comparison
    analytic.py            closed and semi-closed forms (barriers, lookbacks, Heston, Merton, Stulz...)
    american.py            Longstaff-Schwartz and the binomial exercise boundary
    greeks.py              bump-and-revalue, pathwise and likelihood-ratio Greeks
    multiasset.py          correlated GBM, two-asset and basket benchmarks, correlation sweep
    portfolio.py           repricing positions, model smiles, VaR and expected shortfall
    charts.py              Plotly figures for the Monte Carlo page
tests/                     pytest suite, including headless runs of the app
```

**Decisions**

* **Library first, UI second.** All finance lives in `derivlab`; the `views/` files only collect inputs and
  display results. This keeps the models testable without a browser, reusable in notebooks, and lets the
  "Legs & Python" tab print a position as library code.
* **NumPy vectorisation.** Pricing functions accept scalars or arrays, so a payoff curve over 6,000 spot levels
  is one call rather than a Python loop. The full analysis of a position takes a few milliseconds, which is why
  the options page needs no caching.
* **Plain dataclasses.** `Leg`, `Market`, `VolModel`, `Tranche`, `PoolAssumptions`, `SRTInputs` are small
  dataclasses with validation in `__post_init__` or the constructor. `VolModel` is frozen (immutable), so a
  "shifted" smile is a new object and scenarios never mutate the base market.
* **Plotly figures as the chart interface.** Every function in `charts.py` returns a `plotly.graph_objects.Figure`,
  so the same charts work in Streamlit, Jupyter and static HTML.
* **Dependencies kept to the scientific stack**: NumPy, SciPy, pandas, Plotly, Streamlit, pytest.

---

## 3. Conventions and units

| Quantity | Convention |
| --- | --- |
| Time | Calendar days, 365-day year (`DAYS_PER_YEAR = 365`). `T` in formulas is in years. |
| Rates | Continuously compounded risk-free rate `r` and continuous yield `q` (dividend yield; the foreign rate for FX; a convenience yield for commodities). |
| Volatility | Annualised, as a decimal inside the library (0.25 = 25%), in percent in the UI. |
| Quantities | Signed: positive = long (buy), negative = short (sell). Measured in **contracts per lot**. |
| Multiplier | Units of underlying per contract (100 for US equity options). Applies to underlying legs too. |
| Lots | Number of copies of the strategy. `Position.scaled(lots)` multiplies every quantity. |
| Money figures | Per-unit price × quantity × multiplier. Everything the UI shows includes multiplier and lots. |
| Premium sign | `net_premium > 0` is a credit (cash received), `< 0` a debit (cash paid). |
| Greeks (library, per unit) | Vega and rho per 1.00 change in vol / rate; theta per year. |
| Greeks (position, money) | Delta in units of underlying, gamma per 1.00 spot move, theta per calendar day, vega per vol point, rho per 1% rate move. |
| Credit fractions | Attachment, detachment, PD, LGD are decimals of pool notional inside the library, percent in the UI. |
| Spreads | Decimal inside the library, basis points (bp) in reports. 1 bp = 0.01%. |

---

## 4. Options: pricing

### 4.1 Black-Scholes-Merton

A European call or put on an asset with spot `S`, strike `K`, time to expiry `T`, rate `r`, yield `q` and
volatility `σ`:

```
d1 = [ln(S/K) + (r − q + σ²/2)·T] / (σ√T)        d2 = d1 − σ√T
Call = S·e^(−qT)·N(d1) − K·e^(−rT)·N(d2)
Put  = K·e^(−rT)·N(−d2) − S·e^(−qT)·N(−d1)
```

`N` is the standard normal CDF (`scipy.special.ndtr`). Including `q` makes one formula cover stocks with
dividends, indices, FX (Garman-Kohlhagen, with `q` the foreign rate) and futures (Black-76, with `q = r`).

**Edge cases (decision).** At or after expiry (`T ≤ 0`) the price is intrinsic value. At zero volatility the
price is the discounted intrinsic value of the forward, `max(S·e^(−qT) − K·e^(−rT), 0)` for a call. Both are
the mathematical limits, so P&L curves stay continuous as an option expires.

**Put-call parity** holds by construction: `C − P = S·e^(−qT) − K·e^(−rT)`. It is what makes a synthetic
long (long call + short put) behave like the forward, and a conversion lock in the strike.

### 4.2 Greeks

Closed-form, per unit of underlying (`φ` is the standard normal density):

| Greek | Call | Put | Meaning |
| --- | --- | --- | --- |
| Delta Δ | `e^(−qT)·N(d1)` | `−e^(−qT)·N(−d1)` | change in price per 1.00 move in spot |
| Gamma Γ | `e^(−qT)·φ(d1) / (S·σ·√T)` | same | change in delta per 1.00 move |
| Vega ν | `S·e^(−qT)·φ(d1)·√T` | same | change in price per 1.00 (100 points) of vol |
| Theta Θ | `−S·e^(−qT)·φ(d1)·σ/(2√T) − r·K·e^(−rT)·N(d2) + q·S·e^(−qT)·N(d1)` | `… + r·K·e^(−rT)·N(−d2) − q·S·e^(−qT)·N(−d1)` | change in price per year of time passing |
| Rho ρ | `K·T·e^(−rT)·N(d2)` | `−K·T·e^(−rT)·N(−d2)` | change in price per 1.00 move in rates |

Position Greeks are summed across legs, multiplied by quantity and multiplier, and rescaled to trader units:
theta ÷ 365 (per day), vega ÷ 100 (per vol point), rho ÷ 100 (per 1%). Two derived numbers are shown:

* **Cash delta** = delta × spot: the size of the equivalent stock position in money.
* **Dollar gamma (1%)** = gamma × S² × 0.01: how much cash delta changes for a 1% move in spot.

### 4.3 Implied volatility

`implied_vol` inverts Black-Scholes with Brent's method on `[1e-6, 10]` (0.0001% to 1,000% vol). Before solving
it checks the no-arbitrage bounds (a call is worth at least `max(S·e^(−qT) − K·e^(−rT), 0)` and at most
`S·e^(−qT)`); outside them no volatility can produce the price and the function returns `NaN` instead of a
misleading number.

### 4.4 Binomial tree

`binomial_price` is a Cox-Ross-Rubinstein tree: `u = e^(σ√Δt)`, `d = 1/u`, risk-neutral probability
`p = (e^((r−q)Δt) − d)/(u − d)`, 500 steps by default, with an optional early-exercise check at every node.
It is in the library to show the **early-exercise premium of American puts** and that an American call on a
non-dividend-paying stock is never exercised early. The strategy lab itself prices everything as European.

---

## 5. Options: the volatility smile

### 5.1 Why a smile

Black-Scholes assumes one volatility for every strike. Markets disagree: equity index options show a **put
skew** (low strikes trade at higher implied vol, because crash insurance is in demand and returns have fat
left tails), and FX options show a more symmetric **smile** (both wings above the at-the-money vol). Ignoring
this misprices every strategy whose legs are at different strikes.

### 5.2 The model

`VolModel` uses a quadratic in **standardised moneyness**:

```
x = ln(K / F) / (σ_ATM · √T)                          # strike's distance from the forward, in standard deviations
σ(K, T) = σ_ATM + shift + skew·x + smile·x²            # floored at `floor` (1%)
x is clipped to [−wing, +wing], wing = 4
```

`F = S·e^((r−q)T)` is the forward. Presets: flat; *equity* (`skew = −0.035`, `smile = 0.006`); *FX*
(`skew = −0.003`, `smile = 0.008`); or custom sliders.

**Decisions**

* **Standardised moneyness rather than strike.** Measuring distance in standard deviations makes one set of
  parameters produce a sensible smile for every expiry: a 10%-out-of-the-money strike is far away for a 1-week
  option and close for a 2-year one. A smile in raw strike would need a parameter set per expiry.
* **A quadratic.** Two parameters with obvious meanings (tilt and curvature) are enough to show what skew does
  to strategy prices, and they map to how traders quote smiles (risk reversals measure skew, butterflies measure
  curvature).
* **Flat wings beyond ±4 standard deviations.** A quadratic rises without limit, which gives absurd vols (and
  arbitrage) for far-away strikes. The problem showed up in testing: under a sticky-moneyness equity smile, a
  calendar spread's maximum profit came out as "unlimited", because when spot was pushed to 5× its level the
  far-in-the-money long call was being priced at a 200%+ vol. Holding the smile flat beyond ±4σ fixed this. A
  test now checks, for all 34 presets, that switching on the smile never changes whether risk is unlimited.
* **`shift` is a true parallel move.** Scenario vol shifts add the same number of vol points to every strike.
  An earlier version moved `σ_ATM` itself, which also rescales `x` and changes the shape of the skew; that is
  not what "vol up 5 points" means to a trader.

### 5.3 Sticky strike vs sticky moneyness

When spot moves, what happens to each option's implied vol?

* **Sticky strike.** Each strike keeps its vol. The forward used for moneyness stays at today's spot.
* **Sticky moneyness** (often called sticky delta). The smile slides with spot, so the vol of a fixed strike
  changes as spot moves.

The choice changes the *now* curve, the scenario grid and, importantly, **delta**. Under sticky moneyness the
library computes delta and gamma by bumping spot by 0.1% both ways and repricing with the smile moving
(central finite differences), so they include the term `vega × ∂σ/∂S` (the "smile delta"). With equity skew
this raises a call's delta and makes a put's delta less negative compared with sticky strike. The volatility
smile tab shows both deltas side by side.

### 5.4 Smile effect on cost

The smile tab reprices every leg at the flat ATM vol and compares: `Σ qty × (price on smile − price at ATM vol)
× multiplier`. Positive means the smile makes the position more expensive to put on. Example: with put skew a
bear put spread is cheaper than Black-Scholes suggests, because the lower strike you sell carries the higher
vol.

---

## 6. Options: positions and their analysis

### 6.1 Legs, market and position

* `Leg(kind, qty, strike, days, premium=None)`: `kind` is `call`, `put` or `underlying`. `premium` overrides
  the model price with the price you actually traded at, so P&L reflects your real entry.
* `Market(spot, rate, div_yield, vol, sticky)`.
* `Position(legs, market, multiplier, default_horizon=30)`.

**Entry cost** = `Σ qty × entry price` (model price unless overridden; spot for underlying legs). **Value** at
spot `S` after `d` days with vol shift `Δσ` = `Σ qty × Black-Scholes value` with time to expiry reduced by `d`.
**P&L** = (value − entry cost) × multiplier. With model prices, P&L at inception is exactly zero (tested).

### 6.2 The horizon and multiple expiries

Expiry figures are measured at the **first expiry** (`horizon` = the shortest days to expiry among option
legs; 30 days for a position with no options). Legs that expire later are valued with Black-Scholes at that
date, using their remaining time and the smile.

*Decision:* this is the standard way to read calendar and diagonal spreads (the profit from the short
near-dated option decaying while the long back-month option keeps its time value), and it gives every
position one well-defined "expiry" P&L curve.

### 6.3 Maximum profit, maximum loss and "unlimited"

The P&L at the horizon is evaluated on a grid of 6,000 spot levels from almost zero to `max(5 × spot,
2 × highest strike)`, **with every strike inserted exactly**. At expiry a single-expiry payoff is piecewise
linear with kinks only at the strikes, so its extremes lie at a strike or in a tail; including the strikes
makes those values exact rather than approximate (testing showed a straddle's maximum loss off by about 0.5%
before this).

* Max profit = highest P&L on the grid, or **unlimited** if the P&L is still rising at the top of the grid
  (slope over the last 60 points > 0.001 × multiplier).
* Max loss = lowest P&L, or **unlimited** if still falling.
* A loss bounded only by the underlying going to zero (short put, long stock) is reported as a number, which
  is why the strategy notes distinguish "Large" from "Unlimited".
* If the worst case is a gain (box spread, conversion), max loss shows "None".

**Reward : risk** = max profit ÷ |max loss| when both are finite.

### 6.4 Breakevens

Breakevens are the spot levels where the horizon P&L changes sign, found between grid points by linear
interpolation. Exact zeros on the grid are skipped and handled explicitly (a P&L that touches zero at a strike
then crosses would otherwise be missed), and crossings closer than 0.1% of spot are merged.

### 6.5 Probability of profit

Spot at the horizon is assumed **lognormal under the risk-neutral measure**:

```
ln S_T ~ Normal( ln S + (r − q − σ²/2)·T , σ²·T ),   σ = ATM vol (plus any shift)
```

The breakevens split the price axis into intervals of constant P&L sign. The code checks the sign in the middle
of each interval and adds the exact lognormal probability of every profitable interval. The result is exact up
to the precision of the breakevens (an earlier grid-sum version was only accurate to about 0.2%).

*What it is not.* It is the probability under the pricing measure, which drifts at the risk-free rate and uses
implied vol. Real-world odds differ: the real drift is usually higher and implied vol usually exceeds realised
vol (the volatility risk premium), which is why short-vol strategies tend to show high probabilities of profit
and occasional large losses. Use it to compare strategies, not as a forecast.

### 6.6 The spot × vol scenario grid

`scenario_grid(spot_moves, vol_shifts, days_elapsed)` returns P&L for every combination of a percentage spot
move and a parallel vol shift, valued after `days_elapsed` days (0 = an instant shock). This is the classic way
risk managers view an option book, because Greeks are local and the grid shows what happens in large moves:
gamma and vega interact, and the shape of the payoff dominates.

### 6.7 Curves for the payoff chart

The payoff chart shows the expiry P&L (green above zero, red below) and a dashed "now" curve: the position
marked to model on a chosen day with a chosen vol shift. The gap between them is the time value still in the
options.

---

## 7. Options: sizing and hedging

### 7.1 Lots that fit a loss budget

`size_for_budget(one_lot, budget, basis)`:

* **Max loss basis** (default for defined-risk positions): risk per lot = |max loss at the first expiry|.
* **Stress basis** (automatically used when max loss is unlimited): risk per lot = the worst P&L over spot moves
  within ±range (default ±20%) combined with vol shifts of −5, 0 and +10 points, checked both as an instant
  shock and at the first expiry.
* Lots = ⌊budget ÷ risk per lot⌋. If the position cannot lose in the scenario (a box spread), there is no
  limit.

*Decision:* sizing by maximum loss is how defined-risk option trades are usually sized. For unlimited-risk
trades a maximum loss does not exist, so a stress loss is the honest substitute; the vol shocks are asymmetric
(+10 / −5) because vol tends to jump up in sell-offs.

### 7.2 Delta hedge

`delta_hedge(position, hedge_multiplier)` returns the trade that makes the position delta-neutral: `−delta`
units of the underlying, or `−delta ÷ hedge_multiplier` contracts of a hedge instrument (for example 50 for
E-mini S&P 500 futures). Delta-neutral only holds for small moves at that moment: gamma changes delta as spot
moves and time passes, so a hedged option book must be rebalanced.

### 7.3 Exposure

Option contracts, units of underlying controlled (contracts × multiplier, summed over legs), the notional
(units × spot) and the delta-equivalent position.

---

## 8. Options: the strategy library

34 presets in six categories: bullish, bearish, neutral, volatility, income & hedging, time & arbitrage. Each
has when-to-use, how-to-build, when-to-avoid and a tip, a volatility view, cost and risk labels, and the sign
of delta, gamma, theta and vega at inception.

Legs are written as offsets in percent of spot (`("c", -1, 5)` = sell one call 5% above spot) and an expiry
multiple (calendar and diagonal spreads use 1× and 3× the chosen days). Strikes are rounded to realistic
listed increments: 0.1 below a spot of 20, 0.5 up to 100, 1 up to 400, 5 up to 1,000 and 25 above.

| Category | Strategies |
| --- | --- |
| Bullish | long call, short put, long future, bull call spread, bull put spread, risk reversal, call ratio spread, call backspread, synthetic long |
| Bearish | long put, short call, bear put spread, bear call spread, put ratio spread, put backspread, synthetic short |
| Neutral | long butterfly, broken-wing butterfly, iron condor, iron butterfly, jade lizard |
| Volatility | long and short straddle, long and short strangle |
| Income & hedging | covered call, protective put, collar, put-spread collar, seagull |
| Time & arbitrage | calendar spread, diagonal spread, box spread, conversion |

**The Greek-sign notes are tested.** A test computes the Black-Scholes Greeks of every preset (spot at the
money, 30 days, 25% vol, 4% rates) and checks each `+`, `−` and `0` note. `±` marks Greeks whose sign depends on
the strikes. Writing this test found wrong notes on the ratio spreads and backspreads: with the second strike
10% away, a 1×2 ratio spread is actually *long* gamma at inception, because the at-the-money option has more
gamma than two options 10% out of the money. These presets now use 5% spacing, where the notes hold (short gamma
for ratio spreads, long gamma for backspreads, roughly delta-neutral for both).

Selected concepts:

* **Box spread** (bull call spread + bear put spread, same strikes) pays `K2 − K1` in every scenario, so its
  price is `(K2 − K1)·e^(−rT)`: it is lending at the risk-free rate. P&L is the same at every spot (tested).
* **Conversion** (long stock + long put + short call, same strike) locks in `K` at expiry; its value is
  `K·e^(−rT)`. If calls are rich relative to put-call parity, it earns more than the risk-free rate.
* **Jade lizard**: a short put plus a short call spread, sold for a credit at least the call spread's width, so
  there is no upside risk.
* **Straddle price ≈ 0.8 × σ × √T × S**: the market's expected absolute move, since `√(2/π) ≈ 0.8`.

---

## 9. Monte Carlo pricing

The Monte Carlo pricer (`derivlab/montecarlo`, third page of the app) prices what closed forms cannot:
path-dependent and multi-asset payoffs, early exercise, and models without a smile input. Every price comes
with a 95% confidence interval and, wherever one exists, a formula to check it against.

### 9.1 The principle

Under the risk-neutral measure an option is worth its expected discounted payoff, `V = e^(−rT) E[payoff]`.
Simulate `N` independent price paths, compute each path's discounted payoff `Y_i`, and average:

```
V̂ = (1/N) Σ Y_i          standard error  SE = s / √N         95% interval  V̂ ± 1.96 · SE
```

`s` is the sample standard deviation of the `Y_i`. The law of large numbers makes `V̂` converge to `V`; the
central limit theorem gives the interval. Halving the error takes **four times** the paths, which is why the
variance-reduction methods in 9.4 matter. Monte Carlo's error does not depend on the number of dimensions
(dates × assets), which is what makes it the method of choice for path-dependent and multi-asset products,
while a one-asset European option is priced faster and exactly by a formula.

**Decision: every number has an error bar.** The engine always reports a standard error, and the app shows
the gap to a benchmark formula in standard errors (within about ±2 is agreement).

### 9.2 Price dynamics

All three models are simulated under the risk-neutral drift `r − q` for pricing, or a real-world drift `μ` for
risk (9.11). A test checks that `e^(−(r−q)T) S_T` averages to `S_0` under every model.

**Geometric Brownian motion** (the Black-Scholes world), simulated **exactly** between observation dates:

```
ln S(t+Δt) = ln S(t) + (r − q − σ²/2) Δt + σ √Δt Z
```

**Heston stochastic volatility.** Variance follows its own mean-reverting square-root process:

```
dS/S = (r − q) dt + √v dW₁        dv = κ(θ − v) dt + ξ √v dW₂        corr(dW₁, dW₂) = ρ
```

`v₀` is today's variance, `θ` the long-run variance, `κ` the speed of mean reversion, `ξ` the volatility of
variance and `ρ` the spot-vol correlation. With `ρ < 0` volatility rises when the price falls, which produces
an equity-style skew without any smile input. The **Feller condition** `2κθ ≥ ξ²` keeps the variance away from
zero in continuous time; the app shows whether it holds.

There is no exact one-line scheme, so the code uses the **full-truncation Euler scheme** (Lord, Koekkoek and van
Dijk 2010) on `ln S`, with negative variances truncated to zero inside the drift and diffusion, on a sub-grid of
at most 1/200 year per step. The discretisation leaves a small bias (well inside the tests' 0.5% allowance).
Andersen's QE scheme would allow larger steps.

**Merton jump-diffusion.** GBM plus Poisson jumps (intensity `λ` a year) with normal log sizes `J ~ N(μ_J, δ²)`.
The drift is reduced by `λk`, `k = E[e^J − 1] = e^(μ_J + δ²/2) − 1`, so the price stays a martingale. Simulation
is exact: over a step with `N` jumps the log jump total is `N μ_J + √N δ Z`, so one Poisson draw and one extra
normal per step suffice. Jumps fatten both tails and steepen the short-dated smile.

### 9.3 Random numbers

* **Pseudo-random:** NumPy's PCG64 generator. The same seed gives the same paths, so every result is
  reproducible and two runs can share random numbers on purpose (9.8).
* **Antithetic:** each chunk of draws is `[Z, −Z]`. The pair average is the sample unit for the standard error,
  because the two halves of a pair are not independent.
* **Sobol' (quasi-random):** a low-discrepancy sequence that fills the unit cube evenly, scrambled (Owen) and
  mapped to normals through the inverse normal CDF. Quasi-random points are not independent, so the error comes
  from **16 independently scrambled replications**: `SE = s_replications / √16`. Each replication uses a power of
  two of points, so the path count is rounded up (20,000 paths become 16 × 2,048 = 32,768).
* **Brownian bridge.** With Sobol' points, the first (best-distributed) coordinate sets the end point of each
  Brownian path, the next ones the midpoints, then the quarter points, and so on. The paths have exactly the same
  distribution (tested), but the coordinates that decide most of the payoff get the best dimensions. Without it,
  Sobol' barely helps an Asian option with 126 dates; with it, its variance falls more than 1,000-fold in the
  default example. Heston uses one bridge per Brownian motion; Merton bridges the diffusion part (jump counts stay
  pseudo-random).

### 9.4 Variance reduction

**Antithetic variates.** For a pair `(Y, Y′)` driven by `Z` and `−Z`,
`Var((Y + Y′)/2) = Var(Y)(1 + corr(Y, Y′))/2`. The gain is large for payoffs monotone in the shocks (calls, puts,
forwards) and near zero for symmetric ones (straddles), where `Y` and `Y′` are positively correlated.

**Control variates.** Given a quantity `X` simulated on the same paths with a known mean `E[X]`:

```
V̂_cv = mean(Y) − b (mean(X) − E[X]),     b = Cov(Y, X) / Var(X),     variance reduced by 1 / (1 − corr(Y, X)²)
```

`b` is estimated from the same sample (bias of order 1/N, negligible). The control is chosen automatically:

| Product (model) | Control | Its exact price |
| --- | --- | --- |
| Arithmetic Asian (GBM) | geometric Asian, same strike and fixings | Kemna-Vorst-type closed form (9.6) |
| Digital, barrier, fixed-strike lookback, Asian under Heston/Merton | vanilla option, same strike | the model's formula |
| European, floating-strike lookback, geometric Asian | discounted terminal price | `S_0 e^(−qT)` |
| Arithmetic basket | geometric basket | lognormal closed form |
| Spread | exchange option (K = 0) | Margrabe |
| Best-of / worst-of | discounted sum of terminal prices | `Σ S_i e^(−q_i T)` |

**Quasi-Monte Carlo** errors can fall close to `1/N` (up to log factors) instead of `1/√N` when the payoff
depends mostly on a few "effective" dimensions, which the Brownian bridge arranges.

**How the app compares them.** The Variance reduction section prices with every method at the same path
budget and reports the **variance-reduction factor** `(SE_plain / SE_method)²` (how many times more plain paths
the same accuracy would take) and **efficiency** (that factor adjusted for run time). For an at-the-money
6-month arithmetic Asian call with 126 fixings and 20,000 paths, antithetic gives about 1.8×, the geometric
control about 1,870×, Sobol' with the bridge about 1,800×, and Sobol' plus the control about 30,000×. The
convergence chart shows the estimate and its band shrinking as paths are added, and the standard error on log
axes against a `1/√N` reference line.

### 9.5 Products

Single-asset payoffs are evaluated on the simulated path at its **observation dates** (equally spaced), i.e.
they are *discretely monitored*, as real contracts are.

| Product | Payoff | Notes |
| --- | --- | --- |
| European | `max(φ(S_T − K), 0)` | `φ = +1` call, `−1` put |
| Digital (cash-or-nothing) | `1{φ(S_T − K) > 0}` | pays 1 |
| Asian, arithmetic / geometric | `max(φ(A − K), 0)`, `A` = average of the fixings | today's price is not a fixing |
| Barrier: up/down, in/out | vanilla × 1{touched} or × 1{not touched} | checked today and on every date; in + out = vanilla path by path (tested) |
| Lookback, floating strike | call `S_T − min S`, put `max S − S_T` | extremes include today |
| Lookback, fixed strike | call `max(max S − K, 0)`, put `max(K − min S, 0)` | no benchmark |
| Basket | `max(φ(Σ w_i S_i,T − K), 0)` | 2 or 3 assets |
| Spread | `max(φ(S_1,T − S_2,T − K), 0)` | 2 assets |
| Best-of / worst-of | `max(φ(max_i or min_i S_i,T − K), 0)` | rainbow options |

### 9.6 Benchmarks: closed and semi-closed forms

Each benchmark was validated independently of the simulation (see 9.13).

* **Digital:** `e^(−rT) N(φ d₂)`.
* **Geometric Asian (discrete fixings `t_i = iT/n`):** `ln G` is normal with mean
  `ln S + (r − q − σ²/2) T (n+1)/(2n)` and variance `σ² T (n+1)(2n+1)/(6n²)`, so a Black-type formula is exact.
* **Barriers (Reiner and Rubinstein 1991, as tabulated by Haug):** with
  `μ = (b − σ²/2)/σ²`, `b = r − q`, the knock-in prices are combinations of four terms `A, B, C, D` (e.g. a
  down-and-in call with `K > H` is `C`, with `K < H` it is `A − B + D`). Knock-outs are priced as vanilla minus
  knock-in, which makes in/out parity hold by construction.
* **Discrete monitoring correction (Broadie, Glasserman and Kou 1997):** a barrier checked every `Δt` behaves like
  a continuously monitored barrier shifted away from spot to `H e^(±0.5826 σ √Δt)` (+ for up, − for down). It is
  an approximation: in the tests it agrees with daily-monitored simulation to within about 1-2 standard errors,
  less well when the strike sits near the barrier.
* **Floating-strike lookbacks (Goldman, Sosin and Gatto 1979)**, continuous monitoring. Discrete monitoring sees a
  less extreme minimum or maximum, so the formula is an **upper bound**; simulation approaches it slowly (the gap
  shrinks like `√Δt`; the tests check that 1,000 dates come within 3%).
* **Merton (1976):** a Poisson-weighted sum of Black-Scholes prices,
  `Σ e^(−λ′T)(λ′T)ⁿ/n! · BS(σ_n, r_n)` with `λ′ = λ(1 + k)`, `σ_n² = σ² + nδ²/T`, `r_n = r − λk + n ln(1 + k)/T`.
* **Heston (1993)** through its characteristic function in the "little Heston trap" form (Albrecher et al.
  2007), which avoids branch-cut problems of the logarithm, and **Lewis's (2001) formula**:

  ```
  C = S e^(−qT) − √(S K) e^(−(r+q)T/2) / π · ∫₀^∞ Re[e^(−iuk) φ(u − i/2)] / (u² + ¼) du,     k = ln(K/F)
  ```

  where `φ` is the characteristic function of `ln(S_T/F)`. The same integrator prices Black-Scholes and Merton
  from their characteristic functions; it matches the Black-Scholes formula and the Merton series to about
  1e-14, which validates the integration and its sign conventions.
* **Margrabe (1978)** exchange option (spread with `K = 0`), exact.
* **Kirk (1995)** approximation for spreads with `K ≠ 0`: a Black formula with an effective volatility.
* **Stulz (1982)** options on the maximum or minimum of two assets, using the bivariate normal CDF; puts by
  parity on the extreme. The identity `C_max + C_min = C₁ + C₂` holds to machine precision (tested).
* **Baskets:** the geometric basket is lognormal (exact; used as control); the arithmetic basket is compared with
  **Levy's (1992)** two-moment lognormal approximation.

### 9.7 American options: Longstaff-Schwartz

Early exercise is decided backwards in time, but simulation runs forwards. Longstaff and Schwartz (2001) bridge
the two with regression:

1. Simulate training paths at the exercise dates.
2. At expiry, cash flow = payoff. Step back one date at a time, discounting the cash flows.
3. At each date, regress the discounted future cash flows of **in-the-money** paths on basis functions of the
   state: powers of `S/K` up to the chosen degree (plus `v`, `vS/K`, `v²` under Heston, whose state includes the
   variance). The fitted value estimates the value of continuing.
4. Exercise where the payoff beats the fitted continuation value; replace those paths' cash flows.
5. **Apply the fitted rule to a fresh, independent set of paths** and average the discounted cash flows.

*Decisions.* The two-pass design makes the estimate low-biased (a fitted rule is never better than the optimal
one), instead of the mixed bias of pricing on the training paths. The simulation exercises only on its dates, so
it prices a **Bermudan** option, slightly below the American value. A Cox-Ross-Rubinstein tree with 1,000 steps
and early exercise is the benchmark under GBM; in the tests LSM is within 2% of it.

**Exercise boundary.** For each date the boundary is the price that best separates the fitted
exercise/continue decisions of the training paths (the threshold that agrees with the most decisions). This is
robust to wiggles in the polynomial. Dates with fewer than 50 exercised paths, or whose point jumps far from its
neighbours, are left out; under Heston the boundary depends on variance too and is not drawn. The tree's
boundary is read off the lattice directly.

### 9.8 Monte Carlo Greeks

| Method | Idea | Works for | Weakness |
| --- | --- | --- | --- |
| Bump and revalue, common random numbers | reprice with `S₀ ± 1%` and vol ± 1 point using the **same** random numbers; central differences path by path | any model, any payoff | extra simulations; noisy gamma for payoffs that jump |
| Pathwise | differentiate each path's payoff | continuous payoffs under GBM: Europeans, Asians, lookbacks (delta, vega) | fails for digitals and barriers; no gamma for kinked payoffs |
| Likelihood ratio | differentiate the path density; payoff × score | any payoff under GBM (delta, gamma, vega) | high variance, growing as the first time step shrinks |

Under GBM, `∂S_t/∂S₀ = S_t/S₀` and `∂S_t/∂σ = S_t (ln(S_t/S₀) − (r − q + σ²/2) t)/σ`, so for example the pathwise
delta of a call is `e^(−rT) 1{S_T > K} S_T/S₀`. For likelihood ratio, with `Z_i` the standardised shocks,

```
delta = E[Y · Z₁ / (S₀ σ √Δt₁)]
gamma = E[Y · (Z₁² − 1 − Z₁ σ √Δt₁) / (S₀² σ² Δt₁)]
vega  = E[Y · Σ_i ((Z_i² − 1)/σ − Z_i √Δt_i)]
```

The "Formula" column differentiates the benchmark price numerically where one exists. Without common random
numbers, the noise of two independent simulations would swamp a 1% bump; with them, the bump estimate's error
comes only from the payoff's local behaviour.

### 9.9 Multiple assets

Correlated normals come from the Cholesky factor `L` of the correlation matrix (`Z_corr = Z Lᵀ`); terminal prices
are simulated exactly. The matrix is checked for positive definiteness first, because pairwise correlations
cannot be chosen freely (0.9, 0.9 and −0.9 are inconsistent). The **correlation trade** chart reprices the product
for a common pairwise correlation from −0.9 to 0.95 using the same random numbers at every point: baskets get
cheaper as correlation falls (diversification lowers the basket's volatility), spreads and best-of options get
dearer, worst-of options cheaper. Multi-asset pricing uses correlated GBM only.

### 9.10 Repricing the options-lab position

Every option leg of the position you built is priced by simulation:

* **Under GBM** each leg is simulated at its own smile vol. Since GBM with that vol *is* Black-Scholes, the
  simulation must reproduce the options-lab prices within its confidence interval: a check of the machinery
  (tested).
* **Under Heston or Merton** all legs share one model with no smile input. The model's price, converted back to a
  Black-Scholes implied vol, shows the smile the model produces; the chart compares it with the smile set in the
  options lab. Choosing model parameters to make the two match is **calibration**.

### 9.11 P&L risk: value-at-risk and expected shortfall

1. Simulate spot at the horizon under a **real-world** expected return `μ` (risk is about what will happen, not
   about pricing).
2. **Fully revalue** the position on each path with Black-Scholes on the options-lab smile, with the horizon's
   time decay; options that have expired pay intrinsic value. The P&L is the change in mark-to-model value.
3. Under Heston, optionally move every implied vol by the change in simulated volatility, `√v_h − √v₀`, so spot and
   vol move together (with `ρ < 0`, vol rises in sell-offs). Paths are grouped into 40 quantile bins of the vol
   change for speed.

```
VaR_α = the α-quantile of the loss L = −P&L          ES_α = E[L | L ≥ VaR_α]
```

**Delta-normal VaR** = `z_α · |cash delta| · σ · √h` treats the position as a straight line. Comparing it with full
revaluation shows why that fails for options: it understates the risk of short gamma (a short straddle's
simulated 99% VaR is more than three times its delta-normal VaR in the tests) and overstates it for spreads whose
loss is capped. Expected shortfall measures *how bad* the tail is, not just where it starts, and is coherent
(sub-additive); this is why the Basel market-risk framework (FRTB) replaced 99% VaR with 97.5% ES.

### 9.12 Engineering

* **Chunking.** Paths are generated in chunks of at most 2²² normal numbers (about 32 MB), so 200,000 paths with
  hundreds of steps fit in memory; antithetic pairs and Sobol' blocks are kept intact within chunks.
* **Vectorisation.** Every simulator works on all paths of a chunk at once; only the time loop of the Heston
  scheme is a Python loop.
* **Caching and lazy sections.** The page renders only the selected section, and every simulation is cached on
  its inputs, so switching back and forth is instant. Typical run times at 20,000 paths: a fraction of a second
  under GBM and Merton, one to two seconds for the variance-reduction comparison under Heston.

### 9.13 How the Monte Carlo code is tested

70 dedicated tests, among them: mirror-image antithetic draws; Sobol' normals with the right moments; the
Brownian bridge reproduces independent increments and sets the end point first; the characteristic-function
integrator against Black-Scholes and the Merton series; Heston's limits (`φ(−i) = 1`, zero vol-of-vol gives
Black-Scholes) and put-call parity; in/out barrier parity for formulas and path by path; every sampling method
against Black-Scholes; digitals, geometric Asians and three barrier types against their formulas; Heston and
Merton simulations against their formulas at three strikes and two parameter sets; Heston variance reverting to
its mean; variance-reduction factors above set thresholds; Margrabe and the four Stulz options; the basket control
and Levy approximation; correlation moving basket and spread prices in opposite directions; LSM against the
binomial tree; three Greek methods against Black-Scholes; VaR of a stock against the lognormal quantile; the
Heston spot-vol link; and the path chart's legend.

---

## 10. Credit: capital structures

A securitisation or tranched credit product splits the losses of a loan pool into slices. A tranche with
**attachment** `A` and **detachment** `D` (fractions of pool notional) loses

```
tranche loss = min(L, D) − min(L, A),     as a fraction of its own size: (min(L, D) − min(L, A)) / (D − A)
```

where `L` is the pool loss. Tranches below it are its **subordination** (credit enhancement); the one at the
bottom, attaching at 0%, is the **equity** or **first loss**; the one at the top is the **senior**.

`CapitalStructure` sorts tranches and validates the stack, with plain-English errors: the first must attach at
0%, the last must detach at 100%, no gaps or overlaps, detachment above attachment, everything between 0% and
100%, and unique names (the waterfall and reports look tranches up by name).

Each tranche also carries a **coupon** (the note coupon in the waterfall, or the annual protection premium on
sold tranches in the SRT calculator) and a **holder** (`Retained` or `Sold`, used by the SRT calculator).

**Templates** (illustrative; coupons are indicative, not market quotes): CDX IG and iTraxx Main standard
tranches (0-3-7-15-100% and 0-3-6-12-100%), a bank SRT (retained 0-1% first loss, sold 1-8.5% mezzanine with a
10% premium, retained senior), an auto ABS (equity, classes C, B, A) and a CLO (equity, classes E to A).

---

## 11. Credit: the Gaussian copula and tranche risk

### 11.1 The one-factor model

Each borrower `i` defaults over the horizon when its "asset value"

```
X_i = √ρ · M + √(1 − ρ) · ε_i      falls below   c = N⁻¹(PD)
```

`M` is a common factor (the economy) and `ε_i` is idiosyncratic noise, both standard normal. `ρ` is the
**asset correlation**. Conditional on the factor, defaults are independent with probability

```
p(M) = N( (c − √ρ·M) / √(1 − ρ) )
```

This is the **one-factor Gaussian copula** (Li, 2000) and the same model as the Basel IRB formula (Vasicek;
Gordy, 2003).

### 11.2 Large-pool (Vasicek) distribution

With infinitely many small loans of equal size, the law of large numbers makes the pool loss fraction exactly
`L = LGD × p(M)`. Inverting gives closed forms:

```
P(L ≤ x)          = N( (√(1−ρ)·N⁻¹(x/LGD) − c) / √ρ )
density            = √((1−ρ)/ρ) · exp(−z²/2 + y²/2) / LGD,   y = N⁻¹(x/LGD), z = (√(1−ρ)·y − c)/√ρ
α-quantile         = LGD · N( (c + √ρ·N⁻¹(α)) / √(1−ρ) )
```

The 99.9% quantile is the Basel IRB "unexpected loss" point; the app shows the 1-in-100 and 1-in-1,000 losses.

### 11.3 Tranche expected loss (exact)

Since tranche loss = `min(L, D) − min(L, A)`, tranche expected loss needs `E[min(L, k)]`. `L ≤ k` exactly when
the factor is above `m* = (c − √(1−ρ)·N⁻¹(k/LGD)) / √ρ`, so

```
E[min(L, k)] = k · N(m*) + LGD · ∫_{m*}^{∞} p(m) φ(m) dm
EL_tranche   = ( E[min(L, D)] − E[min(L, A)] ) / (D − A)
```

The remaining integrand is smooth, so composite Gauss-Legendre quadrature (8 × 32 nodes) gives about 12 correct
digits in roughly 0.1 ms.

*Decision:* the first version integrated `P(L > x)` over `[A, D]` with Simpson's rule, which was accurate to
about 3 × 10⁻⁶ relative. Moving the integral to the factor, where the kink sits at a known point, made it exact
for practical purposes and faster. A test checks that tranche expected losses, weighted by thickness, add back
up to the pool's `PD × LGD`.

### 11.4 Other tranche metrics

* **P(loss)** (hit probability) = `P(L > A)`.
* **P(wipeout)** = `P(L ≥ D)`.
* **Loss if hit** = expected loss ÷ P(loss): how bad it is when it happens. Thin tranches lose a lot when hit.

### 11.5 Horizon default probability

The UI asks for an annual PD and a maturity and uses a flat hazard rate: cumulative PD over `t` years =
`1 − (1 − PD)^t`. Tranche probabilities and expected losses are over the whole life.

### 11.6 The correlation trade

Correlation does not change the pool's expected loss (`PD × LGD`), but it redistributes it. Low correlation
concentrates losses around the mean, so the equity is almost certainly hit and senior tranches are safe. High
correlation fattens both tails (many scenarios with almost no defaults, and some with many), so **equity
expected loss falls and senior expected loss rises**. The "EL vs correlation" chart shows this, and it is the
mechanism behind the 2005 correlation crisis and the mispricing of senior CDO tranches before 2008.

### 11.7 Fair spread

The running spread `s` that makes a tranche's protection leg equal its premium leg, with quarterly payments
over the maturity and a flat discount rate `r`:

```
Protection leg = Σ DF(t_i − τ/2) · (EL(t_i) − EL(t_{i−1}))          losses assumed mid-period
Premium leg    = s · Σ τ · DF(t_i) · (1 − (EL(t_i) + EL(t_{i−1}))/2)  = s · RPV01
s              = Protection / RPV01
```

`EL(t)` is the tranche's expected loss up to time `t`, computed from the cumulative PD at `t`. As a sanity
check, for the whole pool (0-100%) the fair spread satisfies the **credit triangle**, spread ≈ hazard rate × LGD
(tested to within 3%).

The report shows the fair spread next to the tranche's coupon. **Excess** = coupon − fair spread is what
investors earn above the modelled expected loss: compensation for unexpected loss, liquidity and model risk.
Real SRT mezzanine premiums sit far above model fair spreads for this reason.

### 11.8 Finite pools by Monte Carlo

Real pools have tens to hundreds of loans, sometimes of very different sizes. `simulate_losses`:

* **Equal sizes:** draw the factor, then the number of defaults from a Binomial(N, p(M)). This is exact and
  very fast.
* **Unequal sizes:** draw each loan's `X_i` directly and sum the defaulted exposure weights, in chunks to limit
  memory. The UI uses lognormal sizes (σ = 1, fixed seed) when "Unequal loan sizes" is on.

A small or lumpy pool has a fatter tail than the large-pool formula suggests (a single default is a visible jump),
which matters most for thin mezzanine tranches. With 5,000 equal loans the simulation matches the closed form
(tested). Seeds are fixed so results are reproducible between reruns.

---

## 12. Credit: the structuring assistant

How much subordination does a tranche need? In this model, a tranche attaching at the `(1 − p)` quantile of the
pool loss distribution takes a loss with probability exactly `p`. `size_by_loss_probability` uses this to build
a structure bottom-up from target loss probabilities (for example mezzanine ≤ 30%, senior ≤ 0.5% over the life):

1. Each attachment = the `(1 − target)` quantile of lifetime pool losses.
2. Attachments are rounded **up** to the next 0.25%, so subordination is never less than required.
3. Targets must fall from junior to senior, and the first-loss piece must not come out empty; otherwise the
   function raises an error that the UI shows as a warning.
4. Coupons are set to each tranche's fair spread; holders follow the classic SRT shape (first loss retained,
   mezzanine sold, senior retained).

This mirrors, in a much simpler form, how rating approaches size tranches against a target probability of loss
or expected loss. Agencies use many more assumptions (loan-level default curves, concentration, cash-flow
stresses, sovereign and counterparty caps), so the output is not a rating.

`attachment_for_target_el` is the expected-loss-based alternative (solved with Brent's method).

---

## 13. Credit: the cash-flow waterfall

### 13.1 Collateral

Monthly, for a pool with balance `B`, annual coupon (WAC), term, amortisation type, annual default rate (CDR),
annual prepayment rate (CPR), loss severity, recovery lag and servicing fee. Annual rates become monthly rates
as `1 − (1 − annual)^(1/12)` (the single monthly mortality, SMM, for prepayments; the monthly default rate, MDR,
for defaults).

Each month, in this order:

1. **Defaults** = performing balance × MDR. Loss = defaults × severity, recognised now. Recovery = defaults ×
   (1 − severity), received `lag` months later.
2. **Interest** = surviving balance × WAC / 12.
3. **Scheduled principal**: level pay (an annuity: equal payments over the remaining term) or bullet (all at
   maturity).
4. **Prepayments** = (surviving balance − scheduled principal) × SMM.

The model runs `term + recovery lag` months so every recovery is collected.

### 13.2 Interest waterfall

1. Servicing fee on the beginning performing balance.
2. Note coupons, **senior first**, on each note's current balance. Any unpaid amount is recorded as an
   interest shortfall.
3. **Excess spread** covers the month's losses: that amount is paid out as principal ("turbo"), so losses are
   absorbed by interest income before they reach the notes.
4. What remains goes to the equity (residual) holder.

### 13.3 Losses and principal

* Losses not covered by excess spread **write down tranches from the bottom up**: equity first.
* Principal (scheduled + prepayments + recoveries + loss cover) pays the notes **sequentially**: the senior
  until it is repaid, then the next, with the equity last. Leftover cash goes to the equity.
* The bottom tranche is the equity: no coupon, it receives residual cash.

**Invariant (tested every month):** total outstanding notes = performing loans + recoveries still to come.
Cash is conserved: interest + principal + recoveries collected = fees + coupons + principal paid + residual.

### 13.4 Outputs

* Per tranche: interest and principal received, residual cash (equity), principal loss %, interest shortfall,
  **weighted average life** (WAL = Σ t × principal_t ÷ Σ principal_t, in years) and **IRR** (annualised, bought
  at par at time 0, solved with Brent's method).
* **Day-one excess spread** = WAC − servicing fee − Σ thickness × coupon over the notes.
* **Breakeven CDR**: the highest constant annual default rate at which the tranche still gets its principal
  back (loses less than 0.01%), found by bisection with every other assumption unchanged. A higher breakeven
  means more protection.
* **Stress grid:** principal loss % by tranche for CDRs from 0% to 20%.

### 13.5 Why sequential pay

Sequential pay is the simplest and most protective structure for the senior notes: it shortens the senior's
life and builds up subordination as the pool amortises. Real deals often switch between pro-rata and
sequential pay on triggers; see Limitations.

---

## 14. Credit: regulatory capital and SRT economics

### 14.1 IRB capital for the pool

The Basel internal ratings-based (IRB) capital requirement per unit of exposure is the 99.9% Vasicek loss minus
expected loss, with a maturity adjustment for corporates:

```
K = [ LGD · N( (N⁻¹(PD) + √R · N⁻¹(0.999)) / √(1 − R) ) − PD · LGD ] × MA
```

* **Correlation R**: corporate `R = 0.12·w + 0.24·(1 − w)` with `w = (1 − e^(−50·PD)) / (1 − e^(−50))` (falls
  from 24% to 12% as PD rises); residential mortgage 15%; qualifying revolving (cards) 4%; other retail
  `0.03·w + 0.16·(1 − w)` with `w = (1 − e^(−35·PD)) / (1 − e^(−35))`.
* **Maturity adjustment** (corporate only): `b = (0.11852 − 0.05478·ln PD)²`,
  `MA = (1 + (M − 2.5)·b) / (1 − 1.5·b)`, with `M` clamped to [1, 5] years.
* PD floor 0.05% (configurable). Risk weight = 12.5 × K; RWA = 12.5 × K × EAD.

The implementation reproduces published reference points: a corporate exposure with PD 1%, LGD 45% and M = 2.5
has a 92.32% risk weight, and PD 0.03% gives 14.44% (tested).

### 14.2 KIRB

For securitisations the pool capital includes expected loss: `KIRB = K + PD × LGD`. It is the capital the bank
would hold if it kept the whole pool, and the anchor of the securitisation framework: tranches below it are
treated as if they absorb the pool's entire capital requirement.

### 14.3 SEC-IRBA

The securitisation internal ratings-based approach (Basel CRE44; EU CRR Article 259) risk-weights a tranche
`[A, D]` with the simplified supervisory formula approach (SSFA):

```
p     = max(0.3,  A_ + B_/N + C_·KIRB + D_·LGD + E_·MT)          (MT = tranche maturity, clamped to [1, 5])
a     = −1 / (p · KIRB)
u     = D − KIRB,    l = max(A − KIRB, 0)
KSSFA = (e^(a·u) − e^(a·l)) / (a·(u − l))
```

Supervisory parameters:

| Exposure | Seniority | Granularity | A_ | B_ | C_ | D_ | E_ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Wholesale | Senior | N ≥ 25 | 0.00 | 3.56 | −1.85 | 0.55 | 0.07 |
| Wholesale | Senior | N < 25 | 0.11 | 2.61 | −2.91 | 0.68 | 0.07 |
| Wholesale | Non-senior | N ≥ 25 | 0.16 | 2.87 | −1.03 | 0.21 | 0.07 |
| Wholesale | Non-senior | N < 25 | 0.22 | 2.35 | −2.46 | 0.48 | 0.07 |
| Retail | Senior | n/a | 0.00 | 0.00 | −7.48 | 0.71 | 0.24 |
| Retail | Non-senior | n/a | 0.00 | 0.00 | −5.78 | 0.55 | 0.27 |

Risk weight:

* `D ≤ KIRB`: **1,250%** (the tranche is effectively deducted from capital).
* `A ≥ KIRB`: `12.5 × KSSFA`.
* `A < KIRB < D`: a blend, `[(KIRB − A)/(D − A)] × 1,250% + [(D − KIRB)/(D − A)] × 12.5 × KSSFA`.
* Floor **15%**. For **STS** securitisations (EU Article 260) `p` is halved (still floored at 0.3) and the
  senior floor is **10%**.

`p` is a non-neutrality parameter: higher `p` means more capital for tranches above KIRB. The senior flag is set
for the top tranche; `N` is the effective number of exposures.

### 14.4 SRT economics

In a synthetic risk transfer the bank keeps the loans but buys credit protection on a slice, typically the
mezzanine, from investors who post cash collateral. `srt_economics` computes:

```
RWA before        = 12.5 × K × EAD
RWA after         = Σ over retained tranches of RW_SEC-IRBA × notional      (sold tranches: 0%)
                    capped at RWA before
Capital released  = (RWA before − RWA after) × capital ratio
Annual cost       = Σ over sold tranches of premium × notional
Cost of relief    = annual cost / capital released
Economic          = cost of relief < cost of equity
```

The default example (EUR 2bn corporate pool, PD 1.2%, LGD 40%, retained 0-1%, sold 1-8.5% at a 10% premium)
gives K ≈ 6.99%, KIRB ≈ 7.47%, RWA falling from about 1.75bn to 0.61bn, about 136m of capital released at a 12%
capital ratio, and a cost of relief of about 11.0%, just below a 12% cost of equity.

### 14.5 How thick should the sold tranche be?

`senior_boundary_sweep` moves the boundary between the senior tranche and the tranche below it and reruns the
economics. The chart shows the core SRT trade-off: while the sold tranche ends below KIRB, the retained senior
still straddles KIRB and relief is small, so the cost of relief is very high. Just above KIRB relief jumps.
Beyond that, the senior hits its 15% floor, so extra premium buys little extra relief and the cost of relief
rises again. The optimum usually sits a little above KIRB.

---

## 15. The Streamlit app

* **Three pages** with `st.navigation`: options strategy lab, Monte Carlo pricer and credit structuring. The
  Monte Carlo page reads the options lab's market and position through `views/shared.py`, so the position you
  build there can be repriced and risk-measured by simulation.
* **Sections instead of tabs on the Monte Carlo page.** Streamlit runs the code of every tab on every rerun, so
  seven tabs of simulations would recompute together. A segmented control renders only the chosen section.
* **Inputs live in session state.** Widget defaults are set once in `st.session_state`, never via a widget's
  `value` argument, to avoid Streamlit's "set twice" warnings. `app.py` re-saves the listed keys (and every
  Monte Carlo input, by prefix) on every run, because Streamlit otherwise forgets the state of widgets that are
  not on screen, which would reset inputs each time you switch pages or sections.
* **Editable tables** (legs and tranches) use `st.data_editor`. The base table is stored separately and the
  editor's key carries a version number: loading a preset, re-centring legs or applying the structuring
  assistant bumps the version, which resets the editor cleanly. The latest edits are saved so they survive a
  page switch. Changing spot does *not* reset the legs (so custom positions are kept); "Re-centre on spot" does.
* **Input parsing is forgiving.** Incomplete rows are ignored, bad rows produce a warning naming the row, and
  an invalid capital structure stops the page with the exact problem.
* **Caching** (`st.cache_data`) is used only where it pays: every Monte Carlo simulation (keyed on plain tuples
  of the model, product and market so results can be hashed), copula Monte Carlo losses and breakeven CDRs
  (each a 40-step bisection over full waterfall runs). Everything else is fast enough to recompute.
* **Strikes and barriers in % of spot** on the Monte Carlo page, so they stay sensible if you change spot from
  100 to 5,000.
* **Sliders keyed by horizon.** Day sliders include the horizon in their key, so changing expiries resets them
  instead of leaving a value outside the new range.
* **Charts** use a fixed colour set (green gain, red loss, blue accent, amber spot and highlights) on
  transparent backgrounds so they read in light and dark themes.

---

## 16. Testing and verification

281 pytest tests (about 40 seconds), also run on an older dependency set (Streamlit 1.50, Plotly 5.24, pandas 2.2,
NumPy 2.1). There, two Monte Carlo page tests are skipped because that version's test harness cannot rerun pages
with segmented controls; the page itself was checked in a browser on that version.

| Area | What is checked |
| --- | --- |
| Pricing | textbook value (S = K = 100, T = 1, r = 5%, σ = 20%: call 10.4506, put 5.5735); put-call parity; scalar vs vectorised; expiry and zero-vol limits; every Greek against finite differences; implied vol round trips and NaN outside bounds; binomial convergence; American put premium; no early exercise of calls without dividends |
| Smile | flat model; ATM vol at the forward; skew and smile shapes; parallel shift; floor; flat wings |
| Positions | closed-form max profit, max loss and breakevens for calls, spreads, straddles, condors; box and conversion lock in the risk-free rate; zero P&L at inception; premium override; linear scaling with multiplier and lots; vega predicts a 1-point vol move; sticky-moneyness delta against finite differences; exact probability of profit for a long future; every preset analyses; **every preset's Greek-sign notes**; the smile never changes whether a preset's risk is unlimited; input validation |
| Sizing | max-loss and stress bases; wider stress means fewer lots; no-loss positions; delta hedge neutralises delta |
| Copula | quantile inverts CDF; density integrates to 1; mean loss = PD × LGD; tranche ELs add up to the pool; correlation moves risk from equity to senior; Monte Carlo converges to the large pool; solvers hit targets; spreads fall with seniority; credit triangle |
| Waterfall | full repayment without defaults; cash conservation; collateral backs the notes every month; losses hit the bottom first; sequential WAL ordering; prepayments shorten WAL; excess spread protects the equity; bullet repayment; breakeven CDR is the edge of loss; stress grid is monotone |
| Capital | Basel reference risk weights; retail formula against an independent implementation; correlations; KIRB; `p` and KSSFA by hand; 1,250% region, 15% and 10% floors; risk weight falls with attachment; the SRT example; sell-everything and keep-everything limits; a thicker sold tranche releases more capital |
| Monte Carlo | 70 tests: see section 9.13 |
| App | every strategy renders; multiplier and lots scale the numbers; smiles and sticky rules; every credit template; Monte Carlo, STS, retail and bullet options; the structuring assistant replaces the structure; every Monte Carlo section under every model; every exotic product with three methods; three-asset products; American calls |

Beyond the tests, every page was run in a headless browser and every tab and section was checked visually. That
review caught and fixed: truncated metric values, overlapping chart labels, unreadable log-axis ticks, a misplaced
cost-of-equity label, `-0.00` IRRs, the "unlimited profit" artefact described in section 5.2, and noisy points
in the Longstaff-Schwartz exercise boundary (section 9.7).

---

## 17. Limitations

**Options**

* European exercise only in the options lab; early exercise and assignment risk on short legs (which breaks box
  spreads) are not modelled there. American options are priced on the Monte Carlo page (section 9.7).
* Continuous dividend yield only: no discrete dividends, borrow costs or hard-to-borrow fees.
* One parametric smile with no term structure of ATM vol (every expiry has the same ATM vol) and no smile
  dynamics beyond the two sticky rules. No local volatility and no arbitrage checks on user-chosen smile
  parameters (extreme custom values can create calendar or butterfly arbitrage). Stochastic volatility and
  jumps appear only on the Monte Carlo page, which does not calibrate them to the smile.
* Flat interest rates, no rate curve.
* No bid-ask spreads, commissions, slippage or margin requirements; "risk budget" sizing ignores margin.
* Probability of profit is risk-neutral, not real-world.
* Scenario grid uses parallel vol shifts only (no skew steepening, no term-structure twists).
* Max profit/loss are measured on a grid up to `max(5 × spot, 2 × top strike)`; "unlimited" is inferred from the
  slope at the top of that range.

**Monte Carlo**

* Heston is simulated with an Euler scheme, which leaves a small discretisation bias; no QE scheme. Merton's
  jump counts stay pseudo-random under Sobol' sampling.
* Pathwise and likelihood-ratio Greeks are implemented for GBM only; other models use bump and revalue.
* Longstaff-Schwartz prices a Bermudan option with a fitted (sub-optimal) rule: a low-biased estimate. The
  drawn exercise boundary is an estimate, and it is not shown under Heston.
* Benchmarks for discretely monitored barriers (shifted barrier) and for Kirk spreads and Levy baskets are
  approximations; floating-strike lookback formulas assume continuous monitoring; fixed-strike lookbacks and
  arithmetic Asians outside GBM have no benchmark.
* Multi-asset options use correlated GBM only (no stochastic vol, jumps or correlation skew).
* VaR and ES come from one model with fixed parameters: no parameter uncertainty, no historical simulation, no
  backtesting. Implied vols are held fixed during the horizon, except for the optional parallel Heston link.
* Constant interest rates and dividend yields throughout; no stochastic rates.

**Credit: copula**

* One factor, one correlation, one PD and one LGD for the whole pool (homogeneous), apart from Monte Carlo loan
  sizes. LGD is deterministic.
* No base-correlation calibration to market tranche quotes, so spreads will not match traded index tranches.
* Flat hazard rate, flat discount rate, no upfront payments, no accrued-premium-on-default details.
* The Gaussian copula has thin joint tails; it famously understated senior tranche risk before 2008.
* Fair spread is a pure expected-loss price with no risk premium.

**Credit: waterfall**

* Constant CDR, CPR and severity over the life (no default or prepayment curves, no seasoning, no vectors).
* No triggers (overcollateralisation or interest-coverage tests), reserve accounts, pro-rata pay, call options,
  interest-rate hedges, revolving or reinvestment periods, or step-up coupons.
* Fixed-rate collateral and notes only.
* Losses recognised at default, recoveries after a fixed lag; no delinquency pipeline.
* The equity IRR assumes purchase at par.

**Credit: capital and SRT**

* Simplified: no significant-risk-transfer tests, no maturity mismatch adjustment between protection and
  loans, no synthetic excess spread, no output floor, no maximum capital requirement cap beyond RWA after ≤ RWA
  before, no SEC-SA or SEC-ERBA fallbacks, no treatment of non-collateralised protection (counterparty risk),
  no supervisory approval process, no IFRS 9 or accounting effects.
* The pool is homogeneous (one PD, LGD, maturity), and the effective number of exposures is an input rather than
  computed from loan sizes.
* "Capital released" uses a single capital ratio; real banks look at several constraints (CET1, leverage,
  large exposures, MREL).
* Regulatory parameters follow the EU CRR implementation of Basel III securitisation rules as written into the
  code; check current rules before relying on any number.

**General**

* Learning and first-pass analysis tool. Not investment advice; not a pricing, risk or regulatory reporting
  system.

---

## 18. Glossary

| Term | Meaning |
| --- | --- |
| Antithetic variates | Pairing each simulated path with its mirror image to cancel part of the noise |
| ATM | At the money: strike at (or near) spot or the forward |
| Attachment / detachment | Pool loss levels at which a tranche starts and stops absorbing losses |
| Bermudan option | Exercisable on a set of dates only; what a simulation with discrete exercise dates prices |
| Breakeven | Underlying price at which P&L is zero |
| Breakeven CDR | Highest constant default rate at which a tranche still repays in full |
| Brownian bridge | Building a Brownian path from its end point inwards; makes quasi-random numbers effective |
| CDR / CPR | Constant (annual) default rate / constant prepayment rate |
| Common random numbers | Reusing the same random draws for two simulations so their difference has low noise |
| Confidence interval (95%) | Estimate ± 1.96 standard errors: where the true price lies with 95% confidence |
| Control variate | A related quantity with a known price used to correct a simulation estimate |
| Credit enhancement | Protection below a tranche: subordination, excess spread, reserves |
| Credit triangle | Spread ≈ default intensity × loss given default |
| EAD | Exposure at default |
| Excess spread | Collateral interest left after fees and note coupons |
| Expected loss (EL) | Probability-weighted average loss: PD × LGD for one loan |
| Expected shortfall (ES) | Average loss in the worst (1 − α) of scenarios; FRTB uses α = 97.5% |
| Feller condition | `2κθ ≥ ξ²`: Heston variance cannot reach zero |
| First loss / equity | The bottom tranche, attaching at 0% |
| Forward | `S·e^((r−q)T)`: the arbitrage-free price for delivery at T |
| Gaussian copula | Model linking defaults through correlated normal variables |
| Greeks | Sensitivities of an option's value: delta, gamma, theta, vega, rho |
| IRB | Basel internal ratings-based approach to credit risk capital |
| Jump-diffusion | A price process with sudden jumps on top of continuous moves (Merton) |
| KIRB | IRB capital of the pool including expected loss |
| LGD / severity | Loss given default: share of exposure lost when a borrower defaults |
| Likelihood ratio method | Greeks by differentiating the probability density instead of the payoff |
| Longstaff-Schwartz (LSM) | Pricing early exercise by regressing continuation values on the simulated state |
| Lot | One unit of a strategy as written |
| Moneyness | How far a strike is from spot or the forward |
| Multiplier | Units of underlying per contract |
| Pathwise method | Greeks by differentiating each simulated path's payoff |
| PD | Probability of default |
| Put-call parity | `C − P = S·e^(−qT) − K·e^(−rT)` for European options |
| Quasi-Monte Carlo / Sobol' | Simulation with evenly spread deterministic points instead of random ones |
| RWA | Risk-weighted assets; capital required = RWA × capital ratio |
| SEC-IRBA | Securitisation IRB approach for tranche risk weights |
| Sequential pay | Principal repays the most senior outstanding note first |
| Smile / skew | Pattern of implied vol across strikes |
| SRT | Significant (synthetic) risk transfer: buying protection on a tranche to cut capital |
| SSFA | Simplified supervisory formula approach (the KSSFA formula) |
| Standard error | Standard deviation of a simulation estimate: sample s.d. / √N for independent paths |
| Sticky strike / moneyness | Assumptions about how implied vols move when spot moves |
| Stochastic volatility | Volatility that follows its own random process (Heston) |
| STS | Simple, transparent and standardised securitisation (EU label with lower capital) |
| Subordination | Tranches below a given tranche that absorb losses first |
| Value-at-risk (VaR) | Loss exceeded with probability 1 − α over a horizon |
| Variance-reduction factor | How many times more plain paths the same accuracy would need |
| Vasicek distribution | Large-pool loss distribution of the one-factor Gaussian model |
| WAC | Weighted average coupon of the collateral |
| WAL | Weighted average life of principal repayments |

---

## 19. References

* Black, F. and Scholes, M. (1973), "The Pricing of Options and Corporate Liabilities", *Journal of Political
  Economy*. Merton, R. (1973), "Theory of Rational Option Pricing", *Bell Journal of Economics*.
* Cox, J., Ross, S. and Rubinstein, M. (1979), "Option Pricing: A Simplified Approach", *Journal of Financial
  Economics*.
* Hull, J., *Options, Futures, and Other Derivatives* (Pearson).
* Natenberg, S., *Option Volatility and Pricing* (McGraw-Hill).
* Gatheral, J. (2006), *The Volatility Surface* (Wiley).
* Vasicek, O. (2002), "The Distribution of Loan Portfolio Value", *Risk*.
* Li, D. (2000), "On Default Correlation: A Copula Function Approach", *Journal of Fixed Income*.
* Gordy, M. (2003), "A Risk-Factor Model Foundation for Ratings-Based Bank Capital Rules", *Journal of
  Financial Intermediation*.
* Basel Committee on Banking Supervision, *An Explanatory Note on the Basel II IRB Risk Weight Functions* (2005).
* Basel Committee on Banking Supervision, Basel Framework, CRE31 (IRB risk weights) and CRE44 (SEC-IRBA).
* Regulation (EU) No 575/2013 (Capital Requirements Regulation), Articles 153-154 and 259-260, as amended by
  Regulation (EU) 2017/2401.

* Boyle, P. (1977), "Options: A Monte Carlo Approach", *Journal of Financial Economics*.
* Glasserman, P. (2003), *Monte Carlo Methods in Financial Engineering* (Springer).
* Broadie, M. and Glasserman, P. (1996), "Estimating Security Price Derivatives Using Simulation", *Management
  Science*.
* Longstaff, F. and Schwartz, E. (2001), "Valuing American Options by Simulation: A Simple Least-Squares
  Approach", *Review of Financial Studies*.
* Heston, S. (1993), "A Closed-Form Solution for Options with Stochastic Volatility", *Review of Financial Studies*.
* Albrecher, H., Mayer, P., Schoutens, W. and Tistaert, J. (2007), "The Little Heston Trap", *Wilmott Magazine*.
* Lewis, A. (2001), "A Simple Option Formula for General Jump-Diffusion and Other Exponential Lévy Processes".
* Lord, R., Koekkoek, R. and van Dijk, D. (2010), "A Comparison of Biased Simulation Schemes for Stochastic
  Volatility Models", *Quantitative Finance*.
* Merton, R. (1976), "Option Pricing When Underlying Stock Returns Are Discontinuous", *Journal of Financial
  Economics*.
* Reiner, E. and Rubinstein, M. (1991), "Breaking Down the Barriers", *Risk*.
* Broadie, M., Glasserman, P. and Kou, S. (1997), "A Continuity Correction for Discrete Barrier Options",
  *Mathematical Finance*.
* Goldman, B., Sosin, H. and Gatto, M. (1979), "Path Dependent Options: Buy at the Low, Sell at the High",
  *Journal of Finance*.
* Kemna, A. and Vorst, A. (1990), "A Pricing Method for Options Based on Average Asset Values", *Journal of
  Banking and Finance*.
* Margrabe, W. (1978), "The Value of an Option to Exchange One Asset for Another", *Journal of Finance*.
* Kirk, E. (1995), "Correlation in the Energy Markets", in *Managing Energy Price Risk* (Risk Publications).
* Stulz, R. (1982), "Options on the Minimum or the Maximum of Two Risky Assets", *Journal of Financial Economics*.
* Levy, E. (1992), "Pricing European Average Rate Currency Options", *Journal of International Money and Finance*.
* Sobol', I. (1967), "On the Distribution of Points in a Cube and the Approximate Evaluation of Integrals",
  *USSR Computational Mathematics and Mathematical Physics*. Owen, A. (1995), "Randomly Permuted (t,m,s)-Nets and
  (t,s)-Sequences".
* Haug, E. G., *The Complete Guide to Option Pricing Formulas* (McGraw-Hill).
* Basel Committee on Banking Supervision, *Minimum Capital Requirements for Market Risk* (FRTB, 2019).
