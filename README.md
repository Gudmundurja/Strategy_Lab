# Strategy Lab

An **options strategy lab** and a **credit structuring lab** in Python: a Streamlit app on top of a small,
tested library (`derivlab`) that you can also use on its own in scripts and notebooks.

## Run it

You need Python 3.10 or newer.

```bash
cd strategy-lab
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser at http://localhost:8501. To check everything works, run `pytest`
(209 tests, about half a minute, including headless runs of both app pages), or take the tour of the library with `python examples/quickstart.py`.

## The options strategy lab

| Tab | What you get |
| --- | --- |
| Strategy & legs | 34 preset strategies with when-to-use notes, or your own legs. Edit any cell, add or delete legs, override the premium you actually traded at. |
| Payoff | P&L at expiry plus the position marked to model on any day before, with an implied-vol shift. Max profit and loss, breakevens, probability of profit, reward to risk. |
| Spot × vol grid | P&L for every combination of spot move and implied-vol shift, as an instant shock or on a later date. Downloadable as CSV. |
| Greeks | Delta, gamma, theta and vega across spot, plus a plain-English table of what each number means for your position. |
| Volatility smile | Flat, equity put skew, FX smile or your own shape. Shows each leg's vol, how much the smile changes the cost of the position, and delta under sticky-strike versus sticky-moneyness. |
| Sizing & hedge | Contract multiplier (1, 10, 50, 100, 1,000 or custom) and lots. How many lots fit a loss budget, sized by max loss or by a stress scenario for unlimited-risk trades. Notional and the delta hedge in the underlying or in futures. |
| Legs & Python | Per-leg prices and Greeks, and the current position written out as `derivlab` code. |

## The credit structuring lab

Design a capital structure in the editor (or start from a template: CDX and iTraxx standard tranches, a bank
SRT, an auto ABS, a CLO), then test it three ways:

| Tab | What you get |
| --- | --- |
| Tranche risk & pricing | One-factor Gaussian copula, as a large-pool closed form (Vasicek) or a finite-pool Monte Carlo with equal or unequal loan sizes. Per tranche: probability of any loss, of a wipeout, expected loss, fair spread and the excess the coupon pays over it. Loss distribution and the correlation trade. A **structuring assistant** sizes tranches to target loss probabilities and loads the result into the editor. |
| Cash-flow waterfall | Monthly waterfall with defaults (CDR), prepayments (CPR), loss severity and recovery lag. Fees, coupons top-down, excess spread covering losses, sequential principal, bottom-up write-downs. WAL, IRR and breakeven default rate per tranche, plus a default-rate stress grid. |
| SRT capital relief | Basel IRB capital (K) and KIRB for the pool, SEC-IRBA risk weights per tranche (with the STS treatment), RWA before and after, capital released, the annual cost of protection and the cost of capital relief compared with the cost of equity. A chart shows how the answer changes as the sold tranche gets thicker. |

## Use the library

```python
from derivlab import Leg, Market, Position, VolModel, presets, sizing

market = Market(spot=5000, rate=0.04, vol=VolModel.equity(atm=0.16), sticky="strike")
legs = presets.get("iron-condor").make_legs(spot=5000, days=45)      # or [Leg("put", -1, 4750, 45), ...]
pos = Position(legs, market, multiplier=100).scaled(10)              # 10 lots of 100

a = pos.analyze()
a.max_profit, a.max_loss, a.breakevens, a.prob_profit, a.greeks
pos.scenario_grid(spot_moves=[-0.1, 0, 0.1], vol_shifts=[-0.05, 0, 0.05])   # DataFrame of P&L
sizing.size_for_budget(Position(legs, market, multiplier=100), budget=50_000)
```

```python
from derivlab.credit import capital, copula, design, waterfall
from derivlab.credit.structures import preset

cs = preset("CDX IG standard tranches")
design.tranche_report(pd_annual=0.01, rho=0.25, lgd=0.6, maturity=5, structure=cs)

deal = design.size_by_loss_probability(0.012, 0.20, 0.40, 3.0, [("Mezzanine", 0.30), ("Senior", 0.005)])
waterfall.run_waterfall(waterfall.PoolAssumptions(cdr=0.03), preset("Auto ABS")).summary
capital.srt_economics(capital.SRTInputs(), preset("Bank SRT (synthetic)"))
```

Every chart in `derivlab.charts` returns a Plotly figure, so it works in Jupyter (`fig.show()`) or as HTML
(`fig.write_html("payoff.html")`).

## Project layout

```
strategy-lab/
├── app.py                   Streamlit entry point (two pages)
├── views/
│   ├── strategy_lab.py      options page
│   └── credit_lab.py        credit page
├── derivlab/                the library
│   ├── pricing.py           Black-Scholes-Merton, Greeks, implied vol, binomial tree
│   ├── volatility.py        the smile model
│   ├── position.py          legs, positions, P&L, Greeks, scenario grid
│   ├── presets.py           34 strategies with notes
│   ├── sizing.py            lots for a budget, delta hedge, exposure
│   ├── charts.py            Plotly figures
│   ├── fmt.py               number formatting
│   └── credit/
│       ├── structures.py    tranches, capital structures, templates
│       ├── copula.py        Vasicek / Gaussian copula, Monte Carlo, fair spreads
│       ├── design.py        structuring assistant and tranche report
│       ├── waterfall.py     cash-flow waterfall, breakeven CDR, stress table
│       └── capital.py       IRB, SEC-IRBA, SRT economics
├── tests/                   pytest suite, including headless runs of the app
└── examples/quickstart.py
```

## Models and conventions

**Options.** Black-Scholes-Merton with continuous rates and dividend yield (use the foreign rate for FX).
Days are calendar days on a 365-day year. Theta is per calendar day, vega per vol point, rho per 1% rate move.
Quantities are contracts per lot, and every money figure includes the multiplier and lots.

**The smile.** `vol(K, T) = atm + shift + skew·x + smile·x²` with `x = ln(K/F) / (atm·√T)`, the strike's distance
from the forward in standard deviations, so one shape works across expiries. Beyond ±4 standard deviations the
smile is held flat so far-away strikes keep sensible vols. A vol shift moves every strike by the same amount. *Sticky strike* keeps each strike's vol when spot moves; *sticky moneyness* slides the smile
with spot, and then delta and gamma include that effect.

**Expiry figures.** Max profit, max loss and breakevens are measured at the first expiry. Legs that expire
later are valued with Black-Scholes on that date. "Unlimited" means the P&L keeps rising or falling as spot
goes to infinity.

**Probability of profit** is risk-neutral: spot at the first expiry is lognormal at the ATM vol and drifts
at the risk-free rate. It is a useful comparison between strategies, not a forecast of real-world odds.

**Credit.** Borrowers default when `√ρ·M + √(1−ρ)·ε < N⁻¹(PD)`. The large-pool loss distribution is
Vasicek's, the model behind the Basel IRB formula. Tranche expected loss is computed exactly by integrating
over the common factor. Fair spreads use a flat hazard rate, quarterly premiums and the large-pool model.

**Waterfall.** Monthly. Defaults hit the performing balance first, then scheduled amortisation, then
prepayments. Losses are recognised at default and recoveries arrive after the lag. The bottom tranche is the
equity: no coupon, it receives the residual cash.

**Capital.** IRB capital from the Basel formula (corporate correlation with maturity adjustment; retail
mortgage, QRRE and other retail). KIRB adds expected loss. SEC-IRBA follows Basel CRE44 / EU CRR Article 259,
with the STS treatment of Article 260 (p halved with a floor of 0.3, senior floor 10%). Sold tranches are
assumed fully cash-collateralised (0% risk weight) and RWA after the transfer is capped at RWA before.

## Limitations

This is a learning and first-pass analysis tool. It is not investment advice, and not a pricing, risk or
regulatory reporting system.

* Options are European and priced off one smile; there are no early exercise, discrete dividends, borrow
  costs, transaction costs or margin rules.
* The copula is one-factor and homogeneous (except the Monte Carlo's loan sizes); real tranches are priced off
  base-correlation skews.
* The waterfall has no triggers, reserve accounts, pro-rata pay, interest rate hedges or reinvestment period.
* The SRT calculator leaves out the significant-risk-transfer tests, maturity mismatch, synthetic excess
  spread, the output floor and supervisory approval.
