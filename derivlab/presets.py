"""The strategy library: 34 presets with when-to-use notes.

Legs are written as (kind, qty, strike offset in % of spot, expiry multiple),
where kind is 'c' (call), 'p' (put) or 's' (underlying).

``greeks`` gives the sign of delta, gamma, theta and vega at inception for the
default layout (spot at the money, 30 days, positive rates): '+', '-', '0' for
roughly neutral, and '±' where the sign depends on the strikes chosen.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .position import Leg

CATEGORIES = ["Bullish", "Bearish", "Neutral", "Volatility", "Income & hedging", "Time & arbitrage"]


@dataclass(frozen=True)
class Strategy:
    id: str
    name: str
    category: str
    vol_view: str
    cost: str
    risk: str
    legs: tuple
    when: str
    build: str
    avoid: str
    tip: str
    greeks: dict = field(default_factory=dict)
    default_vol: float = 0.25
    default_days: int = 30

    def make_legs(self, spot: float, days: int | None = None) -> list[Leg]:
        days = self.default_days if days is None else days
        step = strike_step(spot)
        out = []
        for kind, qty, k, mult in self.legs:
            if kind == "s":
                out.append(Leg("underlying", qty))
            else:
                strike = round(spot * (1 + k / 100.0) / step) * step
                out.append(Leg("call" if kind == "c" else "put", qty, float(round(strike, 6)), float(round(days * mult))))
        return out


def strike_step(spot: float) -> float:
    if spot >= 1000:
        return 25.0
    if spot >= 400:
        return 5.0
    if spot >= 100:
        return 1.0
    if spot >= 20:
        return 0.5
    return 0.1


def _s(id, name, cat, vol, cost, risk, legs, when, build, avoid, tip, g, **kw):
    legs = tuple((l[0], l[1], l[2] if len(l) > 2 else 0, l[3] if len(l) > 3 else 1) for l in legs)
    return Strategy(id, name, cat, vol, cost, risk, legs, when, build, avoid, tip, dict(zip("dgtv", g)), **kw)


STRATEGIES = [
    _s("long-call", "Long call", "Bullish", "Long vol", "Debit", "Limited", [("c", 1, 0)],
       "You expect a strong rise soon and implied volatility is reasonably priced.", "Buy one call, at or slightly out of the money.",
       "You expect a slow grind higher (theta eats the gain) or implied vol is inflated before an event.",
       "Deep ITM calls behave like stock with less capital; far OTM calls need a big, fast move.", "++-+"),
    _s("long-put", "Long put", "Bearish", "Long vol", "Debit", "Limited", [("p", 1, 0)],
       "You expect a sharp fall, or want insurance against one.", "Buy one put, at or out of the money.",
       "Skew makes downside puts expensive and you only expect a mild decline.",
       "Puts gain from both the fall and the vol spike that usually comes with it.", "-+-+"),
    _s("short-call", "Short (naked) call", "Bearish", "Short vol", "Credit", "Unlimited", [("c", -1, 5)],
       "You expect the price to stay below the strike and implied vol is rich.", "Sell an out-of-the-money call without owning the stock.",
       "Takeover candidates and anything with squeeze risk.", "Prefer a bear call spread: a further-OTM long call caps the loss.", "--+-"),
    _s("short-put", "Short put (cash-secured)", "Bullish", "Short vol", "Credit", "Large", [("p", -1, -5)],
       "You are happy to buy the stock lower and want to be paid while you wait.", "Sell an OTM put, holding cash to buy the shares if assigned.",
       "You could not tolerate owning the stock after a 30% drop.", "Same payoff as a covered call at the same strike.", "+-+-"),
    _s("long-future", "Long future or forward", "Bullish", "No vol exposure", "Margin only", "Large", [("s", 1)],
       "You want full, linear exposure to the underlying or to lock in a purchase price.", "Buy the future or forward.",
       "You cannot meet daily margin calls through an adverse move.", "Linear: gains and losses are symmetric.", "+000"),
    _s("bull-call-spread", "Bull call spread", "Bullish", "Mildly long vol", "Debit", "Limited", [("c", 1, 0), ("c", -1, 10)],
       "You expect a moderate rise to a target price.", "Buy a call at K1, sell a call at a higher K2.",
       "You expect an explosive move far beyond K2.", "Put the short strike at your target.", "+±±±"),
    _s("bear-put-spread", "Bear put spread", "Bearish", "Mildly long vol", "Debit", "Limited", [("p", 1, 0), ("p", -1, -10)],
       "You expect a moderate fall to a target price.", "Buy a put at K2, sell a lower put at K1.",
       "You need crash protection beyond the lower strike.", "With a steep put skew the sold strike is rich, which cheapens the spread.", "-±±±"),
    _s("bull-put-spread", "Bull put spread", "Bullish", "Short vol", "Credit", "Limited", [("p", -1, -5), ("p", 1, -10)],
       "You expect the price to hold above support and implied vol is rich.", "Sell a put at K2, buy a lower put at K1.",
       "Earnings or other gap events inside the expiry window.", "High win rate, capped loss: size by maximum loss.", "+-+-"),
    _s("bear-call-spread", "Bear call spread", "Bearish", "Short vol", "Credit", "Limited", [("c", -1, 5), ("c", 1, 10)],
       "You expect the price to stay below resistance.", "Sell a call at K1, buy a higher call at K2.",
       "Strong momentum or squeeze risk.", "The defined-risk alternative to a naked short call.", "--+-"),
    _s("risk-reversal", "Risk reversal (bullish)", "Bullish", "Skew trade", "Near zero", "Large", [("c", 1, 10), ("p", -1, -10)],
       "You want upside calls funded by selling downside puts.", "Buy an OTM call, sell an OTM put.",
       "You cannot accept being assigned the stock after a crash.", "Also how skew is quoted: vol(25d call) - vol(25d put).", "+±±±"),
    _s("call-ratio-spread", "Call ratio spread (1x2)", "Bullish", "Short vol", "Small debit/credit", "Unlimited", [("c", 1, 0), ("c", -2, 5)],
       "You expect a moderate rise toward the upper strike, not beyond it.", "Buy one call at K1, sell two at a higher K2.",
       "Any chance of a big rally.", "Cheap because you are net short one naked call.", "0-+-"),
    _s("call-backspread", "Call backspread", "Bullish", "Long vol", "Small debit/credit", "Limited", [("c", -1, 0), ("c", 2, 5)],
       "You expect a large rally and want convexity cheaply.", "Sell one call at K1, buy two at a higher K2.",
       "A slow move to the upper strike, the worst outcome.", "Often entered for a small credit, so a fall costs little.", "0+-+"),
    _s("put-ratio-spread", "Put ratio spread (1x2)", "Bearish", "Short vol", "Small debit/credit", "Large", [("p", 1, 0), ("p", -2, -5)],
       "You expect a moderate decline to the lower strike, not a crash.", "Buy one put at K2, sell two at a lower K1.",
       "Crash risk: you end up net short a put.", "Steep skew makes the sold puts rich.", "0-+-"),
    _s("put-backspread", "Put backspread", "Bearish", "Long vol", "Small debit/credit", "Limited", [("p", -1, 0), ("p", 2, -5)],
       "You want cheap, convex crash exposure.", "Sell one put at K2, buy two at a lower K1.",
       "A gentle decline to the lower strike.", "A classic tail-hedge shape.", "0+-+"),
    _s("synthetic-long", "Synthetic long", "Bullish", "No net vol", "Near zero", "Large", [("c", 1, 0), ("p", -1, 0)],
       "You want stock-like exposure without buying the stock.", "Buy a call and sell a put at the same strike.",
       "You are not prepared for stock-like losses.", "Put-call parity: call - put = forward.", "+000"),
    _s("synthetic-short", "Synthetic short", "Bearish", "No net vol", "Near zero", "Unlimited", [("c", -1, 0), ("p", 1, 0)],
       "You want short exposure where borrowing the stock is hard.", "Sell a call and buy a put at the same strike.",
       "Squeeze-prone names.", "Where it trades versus parity reveals the borrow cost.", "-000"),
    _s("covered-call", "Covered call", "Income & hedging", "Short vol", "Stock + credit", "Large", [("s", 1), ("c", -1, 5)],
       "You own the stock, are neutral to mildly bullish and happy to sell at the strike.", "Own the stock, sell an OTM call.",
       "You expect a big rally.", "Equivalent to a short put.", "+-+-"),
    _s("protective-put", "Protective put", "Income & hedging", "Long vol", "Stock + debit", "Limited", [("s", 1), ("p", 1, -5)],
       "You own the stock and want a floor through a risky period.", "Own the stock, buy a put.",
       "Buying it permanently: the volatility risk premium is a long-run drag.", "Same payoff as a call plus cash.", "++-+"),
    _s("collar", "Collar", "Income & hedging", "Skew trade", "Near zero", "Limited", [("s", 1), ("p", 1, -5), ("c", -1, 5)],
       "You hold a concentrated position and want protection without paying premium.", "Own the stock, buy an OTM put, sell an OTM call.",
       "You expect a strong rally beyond the call strike.", "The standard corporate and executive hedge.", "+±±±"),
    _s("put-spread-collar", "Put-spread collar", "Income & hedging", "Skew trade", "Near zero", "Large",
       [("s", 1), ("p", 1, -5), ("p", -1, -15), ("c", -1, 6)],
       "You want protection for a moderate fall and a higher cap than a plain collar.", "Own the stock, buy a put spread, sell an OTM call.",
       "You specifically need crash protection.", "Selling the lower put gives back protection exactly in a crash.", "+±±±"),
    _s("long-straddle", "Long straddle", "Volatility", "Long vol", "Debit", "Limited", [("c", 1, 0), ("p", 1, 0)],
       "You expect a big move but not its direction.", "Buy an ATM call and an ATM put.",
       "Implied vol is already inflated before an event.", "Price is about 0.8 x vol x sqrt(T) x spot: the market's expected move.", "0+-+"),
    _s("short-straddle", "Short straddle", "Volatility", "Short vol", "Credit", "Unlimited", [("c", -1, 0), ("p", -1, 0)],
       "You expect the price to stay pinned and vol to fall.", "Sell an ATM call and an ATM put.",
       "Any event risk.", "The defined-risk version is the iron butterfly.", "0-+-"),
    _s("long-strangle", "Long strangle", "Volatility", "Long vol", "Debit", "Limited", [("c", 1, 5), ("p", 1, -5)],
       "You expect a very large move either way, cheaper than a straddle.", "Buy an OTM call and an OTM put.",
       "You expect a moderate move.", "Cheaper than a straddle but needs a bigger move.", "0+-+"),
    _s("short-strangle", "Short strangle", "Volatility", "Short vol", "Credit", "Unlimited", [("c", -1, 5), ("p", -1, -5)],
       "You expect a range-bound market and rich implied vol.", "Sell an OTM call and an OTM put.",
       "Gap risk.", "Prefer an iron condor so losses are capped.", "0-+-"),
    _s("long-butterfly", "Long call butterfly", "Neutral", "Short vol", "Debit", "Limited", [("c", 1, -5), ("c", -2, 0), ("c", 1, 5)],
       "You expect the price to finish near a specific level.", "Buy K1, sell two K2, buy K3 (equal spacing).",
       "You expect a large move.", "Butterfly prices trace out the implied distribution.", "0-+-"),
    _s("broken-wing-butterfly", "Broken-wing butterfly", "Neutral", "Short vol", "Small debit/credit", "Limited",
       [("c", 1, -5), ("c", -2, 0), ("c", 1, 10)],
       "You expect the price near the body and want no risk on one side.", "A butterfly with one wing moved further out.",
       "A strong trend toward the wide wing.", "Turns a butterfly into a directional, defined-risk trade.", "--+-"),
    _s("iron-condor", "Iron condor", "Neutral", "Short vol", "Credit", "Limited", [("p", 1, -10), ("p", -1, -5), ("c", -1, 5), ("c", 1, 10)],
       "You expect a range-bound market and implied vol is high.", "Sell an OTM put spread and an OTM call spread.",
       "Trending markets and scheduled events.", "Size by maximum loss.", "0-+-"),
    _s("iron-butterfly", "Iron butterfly", "Neutral", "Short vol", "Credit", "Limited", [("p", 1, -5), ("p", -1, 0), ("c", -1, 0), ("c", 1, 5)],
       "You expect the price to stay very close to current levels.", "Sell an ATM straddle, buy an OTM strangle.",
       "Any meaningful move.", "A short straddle with insurance.", "0-+-"),
    _s("calendar-spread", "Calendar spread", "Time & arbitrage", "Long vega, short gamma", "Debit", "Limited", [("c", -1, 0, 1), ("c", 1, 0, 3)],
       "You expect the price to stay near the strike while the near option decays.", "Sell a near-dated call, buy a longer-dated call, same strike.",
       "A big move, or a fall in back-month vol.", "Positions you for a post-event vol crush.", "0-++"),
    _s("diagonal-spread", "Diagonal spread", "Time & arbitrage", "Long vega", "Debit", "Limited", [("c", -1, 5, 1), ("c", 1, -5, 3)],
       "You are mildly bullish and want to collect near-term decay.", "Buy a longer-dated ITM call, sell a near-dated OTM call.",
       "A sharp fall in the stock or in back-month vol.", "Sometimes called a poor man's covered call.", "+-++"),
    _s("jade-lizard", "Jade lizard", "Neutral", "Short vol", "Credit", "Large", [("p", -1, -3), ("c", -1, 4), ("c", 1, 6)],
       "You are neutral to bullish, vol is high, and you want no upside risk.", "Sell an OTM put and an OTM call spread for a credit at least the call spread's width.",
       "You won't own the stock on a sharp fall.", "Check the credit covers the call spread width.", "+-+-", default_vol=0.30),
    _s("seagull", "Seagull (bullish)", "Income & hedging", "Skew trade", "Near zero", "Large", [("c", 1, 0), ("c", -1, 10), ("p", -1, -10)],
       "A buyer wants capped protection against a rise, financed by giving up protection below a floor.", "Buy a call spread, sell an OTM put.",
       "The sold put creates exposure you cannot carry in a collapse.", "Standard in FX and commodity hedging.", "+±±±"),
    _s("box-spread", "Box spread", "Time & arbitrage", "None", "Debit (lending)", "None (European)", [("c", 1, -5), ("c", -1, 5), ("p", 1, 5), ("p", -1, -5)],
       "You want to lend or borrow at an implied rate using listed options.", "Bull call spread plus bear put spread at the same strikes.",
       "American-style options, where early assignment breaks the box.", "Fair value = (K2 - K1) x exp(-rT).", "00+0"),
    _s("conversion", "Conversion", "Time & arbitrage", "None", "Stock + premium", "None (European)", [("s", 1), ("p", 1, 0), ("c", -1, 0)],
       "Calls are rich relative to puts versus put-call parity.", "Buy stock, buy a put, sell a call at the same strike.",
       "Dividends, borrow costs or early assignment are not priced in.", "Locks in the strike: worth K x exp(-rT).", "00+0"),
]

BY_ID = {s.id: s for s in STRATEGIES}


def get(strategy_id: str) -> Strategy:
    return BY_ID[strategy_id]
