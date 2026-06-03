"""
Hold-rule backtest — does "ride the trend" beat "sell at +X%"?

Answers the AMD-at-$350 question directly: how much of a big run do you keep if you
let a trend / trailing-stop rule decide the exit, versus banking a fixed gain early?
Honest walk-forward, no look-ahead: the position decided at close t earns the return
of t+1 (single tradeable lag), costs charged on turnover.

Strategies, per symbol:
  Buy & hold                : own it the whole window (keeps everything, eats full drawdown)
  Sell at +50/75/100%       : the disposition mistake — bank a fixed gain, then sit out
  200d trend long/flat      : long while close > 200d SMA, flat below (re-enters)
  Chandelier 3xATR trail    : long until close < (high since entry - k*ATR) or < 200d; re-enter
                              on a fresh 50&200d uptrend  <- this IS the hold_monitor rule

Reports final equity multiple, CAGR, Sharpe, maxDD, time-in-market, and what % of the
buy & hold run each rule captured. For each take-profit it prints where you'd have sold
and where the stock went after, i.e. what selling early actually cost.

Usage
  python hold_backtest.py                       # AMD NVDA GC=F SI=F, 5y
  python hold_backtest.py --symbols AMD --period max
  python hold_backtest.py --k 3 --atr 22 --cost-bps 2
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
#  data + helpers
# --------------------------------------------------------------------------- #
def fetch_ohlc(symbol: str, period: str) -> pd.DataFrame | None:
    try:
        df = yf.download(symbol, period=period, interval="1d", auto_adjust=True,
                         progress=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(how="all")
        return df if len(df) > 250 else None
    except Exception:
        return None


def atr_series(df: pd.DataFrame, n: int) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def stats(daily_ret: pd.Series, pos: pd.Series | None = None) -> dict:
    r = daily_ret.dropna()
    if len(r) < 30:
        return {}
    n_years = len(r) / TRADING_DAYS
    eq = (1 + r).cumprod()
    mult = float(eq.iloc[-1])
    cagr = mult ** (1 / n_years) - 1
    vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = (r.mean() * TRADING_DAYS) / vol if vol > 0 else np.nan
    dd = float((eq / eq.cummax() - 1).min())
    out = {"Mult": mult, "CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MaxDD": dd}
    if pos is not None:
        out["InMkt"] = float((pos.reindex(r.index) > 0).mean())
    return out


# --------------------------------------------------------------------------- #
#  position generators (target weight known at close t)
# --------------------------------------------------------------------------- #
def pos_trend(close: pd.Series, slow=200) -> pd.Series:
    return (close > close.rolling(slow).mean()).astype(float).fillna(0.0)


def pos_chandelier(df: pd.DataFrame, k=3.0, atr_n=22, fast=50, slow=200) -> pd.Series:
    """The hold_monitor rule as a strategy: ride until the ATR trail or the 200d breaks,
    re-enter on a fresh 50&200d uptrend."""
    close = df["Close"]
    sf = close.rolling(fast).mean().values
    ss = close.rolling(slow).mean().values
    a = atr_series(df, atr_n).values
    c = close.values
    pos = np.zeros(len(c))
    inpos, high = False, -np.inf
    for i in range(len(c)):
        if np.isnan(ss[i]) or np.isnan(a[i]):
            pos[i] = 0.0
            continue
        if not inpos:
            if c[i] > ss[i] and c[i] > sf[i]:
                inpos, high = True, c[i]
        else:
            high = max(high, c[i])
            if c[i] < high - k * a[i] or c[i] < ss[i]:
                inpos = False
        pos[i] = 1.0 if inpos else 0.0
    return pd.Series(pos, index=close.index)


def pos_takeprofit(close: pd.Series, target: float) -> tuple[pd.Series, dict]:
    """Own from the window start; sell for good once up `target` from start. The
    disposition mistake. Returns (pos, info about the sale)."""
    c = close.values
    base = c[0]
    pos = np.ones(len(c))
    info = {"sold": False}
    for i in range(len(c)):
        if c[i] / base - 1 >= target:
            pos[i:] = 0.0
            info = {"sold": True, "date": close.index[i], "price": float(c[i]),
                    "final": float(c[-1]), "left_pct": float(c[-1] / c[i] - 1)}
            break
    return pd.Series(pos, index=close.index), info


def net_ret(pos: pd.Series, ret: pd.Series, cost_bps: float) -> pd.Series:
    pos_lag = pos.shift(1).fillna(0.0)
    turn = (pos - pos.shift(1)).abs().fillna(0.0).shift(1).fillna(0.0)
    return pos_lag * ret - turn * (cost_bps / 1e4)


# --------------------------------------------------------------------------- #
#  run
# --------------------------------------------------------------------------- #
def fmt(name: str, st: dict, bh_mult: float | None = None) -> str:
    if not st:
        return f"  {name:<26} (insufficient data)"
    cap = ""
    if bh_mult and bh_mult > 1:
        cap = f"  cap={100*(st['Mult']-1)/(bh_mult-1):3.0f}%"   # share of buy&hold gain kept
    im = f"  inMkt={100*st['InMkt']:3.0f}%" if "InMkt" in st else ""
    return (f"  {name:<26} x{st['Mult']:5.2f}  CAGR={100*st['CAGR']:+6.1f}%  "
            f"Sharpe={st['Sharpe']:+4.2f}  maxDD={100*st['MaxDD']:6.1f}%{im}{cap}")


def run(symbols: list[str], period: str, k: float, atr_n: int, cost_bps: float) -> None:
    for sym in symbols:
        df = fetch_ohlc(sym, period)
        if df is None:
            print(f"\n=== {sym}: insufficient data ===")
            continue
        close = df["Close"].dropna()
        ret = close.pct_change()
        start, end = close.index.min().date(), close.index.max().date()
        bh = stats(ret)
        bh_mult = bh.get("Mult", 1.0)
        print(f"\n=== {sym} | {start} -> {end} | {len(close)}d | cost={cost_bps:.0f}bps ===")
        print(fmt("Buy & hold", bh))

        # disposition mistake: bank a fixed gain then sit out
        for tp in (0.50, 0.75, 1.00):
            pos, info = pos_takeprofit(close, tp)
            st = stats(net_ret(pos, ret, cost_bps), pos=pos)
            line = fmt(f"Sell at +{tp*100:.0f}% then out", st, bh_mult)
            if info.get("sold"):
                line += (f"\n      -> sold {info['date'].date()} @ {info['price']:.2f}; "
                         f"ran to {info['final']:.2f} = +{info['left_pct']*100:.0f}% left on the table")
            print(line)

        # trend / trail rules (the hold_monitor logic)
        for name, pos in [
            ("200d trend long/flat", pos_trend(close)),
            (f"Chandelier {k:g}xATR trail", pos_chandelier(df, k=k, atr_n=atr_n)),
        ]:
            st = stats(net_ret(pos, ret, cost_bps), pos=pos)
            print(fmt(name, st, bh_mult))
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["AMD", "NVDA", "GC=F", "SI=F"])
    ap.add_argument("--period", default="5y", help="yfinance period (e.g. 3y, 5y, max)")
    ap.add_argument("--k", type=float, default=3.0, help="chandelier ATR multiple")
    ap.add_argument("--atr", type=int, default=22, help="ATR lookback")
    ap.add_argument("--cost-bps", type=float, default=2.0)
    a = ap.parse_args()
    run(a.symbols, a.period, a.k, a.atr, a.cost_bps)


if __name__ == "__main__":
    main()
