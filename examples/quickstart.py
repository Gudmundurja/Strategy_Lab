"""A tour of the derivlab library. Run from the project folder:

    python examples/quickstart.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # works without installing the package

import pandas as pd  # noqa: E402

from derivlab import Market, Position, VolModel, bs_price, implied_vol, presets, sizing  # noqa: E402
from derivlab.credit import capital, design, waterfall  # noqa: E402
from derivlab.credit.structures import preset  # noqa: E402
from derivlab.fmt import compact, money, pct  # noqa: E402

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

print("\n1. Price an option, then back out its implied vol")
c = bs_price("call", 100, 105, 0.5, 0.04, 0.0, 0.22)
print(f"   6-month 105 call at 22% vol: {c:.4f}. Implied vol from that price: "
      f"{implied_vol('call', c, 100, 105, 0.5, 0.04):.2%}")

print("\n2. An iron condor on an index with put skew: 10 lots, multiplier 100")
market = Market(spot=5000, rate=0.04, vol=VolModel.equity(atm=0.16))
legs = presets.get("iron-condor").make_legs(spot=5000, days=45)
for leg in legs:
    print(f"   {leg.side:4s} {abs(leg.qty):g} {leg.kind:4s} {leg.strike:,.0f}  {leg.days:g} days")
pos = Position(legs, market, multiplier=100).scaled(10)
a = pos.analyze()
print(f"   credit {money(a.net_premium)} | max profit {money(a.max_profit)} | max loss {money(a.max_loss)}")
print(f"   breakevens {', '.join(f'{b:,.1f}' for b in a.breakevens)} | probability of profit {pct(a.prob_profit)}")
g = a.greeks
print(f"   delta {g['delta']:,.1f} | gamma {g['gamma']:,.3f} | theta/day {g['theta']:,.0f} | vega/pt {g['vega']:,.0f}")

print("\n3. Spot x vol scenario grid: P&L of an instant shock")
grid = pos.scenario_grid(spot_moves=[-0.10, -0.05, -0.02, 0, 0.02, 0.05, 0.10], vol_shifts=[-0.05, 0, 0.05, 0.10])
print(grid.round(0).to_string())

print("\n4. Sizing: lots that fit a 50,000 loss budget, and the delta hedge in E-mini futures (x50)")
res = sizing.size_for_budget(Position(legs, market, multiplier=100), 50_000)
hedge = sizing.delta_hedge(pos, hedge_multiplier=50)
print(f"   risk per lot {money(res.risk_per_lot)} ({res.basis}) -> {res.lots} lots")
print(f"   to be delta-neutral trade {hedge['contracts']:+.2f} futures ({hedge['units']:+,.0f} index units)")

print("\n5. Tranche risk in the one-factor Gaussian copula (5 years, PD 1%, rho 25%, LGD 60%)")
report = design.tranche_report(pd_annual=0.01, rho=0.25, lgd=0.6, maturity=5, structure=preset("CDX IG standard tranches"))
print(report.round(3).to_string(index=False))

print("\n6. Structure a deal so the senior has at most a 0.5% chance of any loss")
deal = design.size_by_loss_probability(0.012, 0.20, 0.40, 3.0, [("Mezzanine", 0.30), ("Senior", 0.005)])
print(deal.to_frame().round(3).to_string(index=False))

print("\n7. Cash-flow waterfall: auto ABS at a 3% annual default rate")
pool = waterfall.PoolAssumptions(cdr=0.03)
wf = waterfall.run_waterfall(pool, preset("Auto ABS"))
cols = ["Tranche", "Coupon %", "Principal loss %", "WAL (yrs)", "IRR %"]
print(wf.summary[cols].round(2).to_string(index=False))
print(f"   Class A breakeven CDR: {pct(waterfall.breakeven_cdr(pool, preset('Auto ABS'), 'Class A'))}")

print("\n8. Capital relief from a synthetic risk transfer (SEC-IRBA)")
r = capital.srt_economics(capital.SRTInputs(), preset("Bank SRT (synthetic)"))
print(f"   K {pct(r['k'], 2)} | KIRB {pct(r['kirb'], 2)} | RWA {compact(r['rwa_before'])} -> {compact(r['rwa_after'])}")
print(f"   capital released {compact(r['capital_released'])} for {compact(r['annual_cost'])} a year: "
      f"cost of relief {pct(r['cost_of_relief'])} ({'economic' if r['economic'] else 'not economic'} at a 12% cost of equity)")

print("\n9. Monte Carlo: an arithmetic Asian call under Heston stochastic volatility")
from derivlab.montecarlo import GBM, Heston, american, compare_methods, price_option  # noqa: E402
from derivlab.montecarlo.payoffs import Asian  # noqa: E402
from derivlab.pricing import binomial_price  # noqa: E402

res = price_option(Asian("call", 100), Heston(), S0=100, r=0.04, q=0.0, T=0.5, n_obs=126, n_paths=50_000,
                   sampling="sobol", control="auto")
print(f"   price {res.price:.4f} ± {1.96 * res.stderr:.4f} (95%) from {res.n_paths:,} paths, control: {res.control}")

print("\n10. The same option under GBM with every variance-reduction method (same path budget)")
table = compare_methods(Asian("call", 100), GBM(0.25), 100, 0.04, 0.0, 0.5, 126, 20_000)
print(table[["Method", "Price", "Std error", "Variance reduction"]].round(5).to_string(index=False))

print("\n11. An American put: Longstaff-Schwartz against a 1,000-step binomial tree")
lsm = american.lsm_price("put", 100, 100, 1.0, 0.05, 0.0, GBM(0.2), n_ex=50, n_paths=50_000)
print(f"   LSM {lsm['price']:.4f} ± {1.96 * lsm['stderr']:.4f}, tree {binomial_price('put', 100, 100, 1.0, 0.05, 0.0, 0.2, 1000):.4f}, "
      f"European {lsm['european']:.4f}")
