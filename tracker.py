"""
Signal tracker -- a LIVE, forward-only paper portfolio of the model's calls.

Clean slate: inception = the day you first run it. Whatever happened before that is
ignored -- we open every name the model currently says BUY/HOLD at the last close and
track the portfolio from there, vs SPY. Each later run is an UPDATE: it reconciles the
ledger against today's signals, opening new BUYs and closing names that flipped to SELL,
keeping every position's original entry date+price. This is real, out-of-sample state --
not recomputed from history -- so the monthly scorecard is an honest forward record.

  ENTRY (BUY/HOLD) : close above its 50d AND 200d  -> open at the last close
  EXIT  (SELL)     : close below its 200d          -> close at the last close

State (committed, so the record persists & survives in git):
  tracker_state.json   inception date + bookkeeping
  tracker_open.csv     positions currently held (original entry date/price kept)
  tracker_closed.csv   completed trades
  tracker_report.md    the read: open book, monthly portfolio-vs-SPY, lifetime

Usage
  python tracker.py                 # update (auto-inceptions on first run)
  python tracker.py --reset         # wipe the ledger and re-inception at today's last close
  python tracker.py --period 2y
"""
from __future__ import annotations

import csv
import json
import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

import sources

ROOT = Path(__file__).parent
FAST, SLOW = 50, 200
STATE = ROOT / "tracker_state.json"
OPEN_F = ROOT / "tracker_open.csv"
CLOSED_F = ROOT / "tracker_closed.csv"
REPORT = ROOT / "tracker_report.md"


def load_universe(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.lower().startswith("symbol,"):
                continue
            p = next(csv.reader([line]))
            if p and p[0].strip():
                rows.append({"symbol": p[0].strip().upper(),
                             "layer": (p[1].strip() if len(p) > 1 else "")})
    return rows


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return df.to_dict("records") if len(df) else []


def _signal(df: pd.DataFrame) -> tuple[str, float, str] | None:
    """Return (signal, last_close, last_date) where signal in {'in','out'} or None."""
    close = df["Close"].dropna()
    if len(close) < SLOW + 1:
        return None
    s_f = close.rolling(FAST).mean().iloc[-1]
    s_s = close.rolling(SLOW).mean().iloc[-1]
    last = float(close.iloc[-1])
    if not (s_s == s_s):
        return None
    if last < s_s:
        sig = "out"                                   # below 200d -> SELL/flat
    elif last > s_s and last > s_f:
        sig = "in"                                    # above 50 & 200d -> BUY/HOLD
    else:
        sig = "between"                               # above 200 but below 50d -> no fresh entry
    return sig, last, close.index[-1].date().isoformat()


def run(symbols: list[str], layers: dict, period: str, reset: bool) -> None:
    prices = sources.fetch_prices(list(dict.fromkeys(symbols + ["SPY"])), period=period)
    spy = prices["SPY"]["Close"].dropna() if "SPY" in prices else pd.Series(dtype=float)

    sigs: dict[str, tuple] = {}
    for s in symbols:
        df = prices.get(s)
        if df is not None and not df.empty:
            r = _signal(df)
            if r:
                sigs[s] = r

    fresh = reset or not STATE.exists()
    open_rows = [] if fresh else _read_csv(OPEN_F)
    closed_rows = [] if fresh else _read_csv(CLOSED_F)
    open_by = {r["symbol"]: r for r in open_rows}

    if fresh:
        inception = dt.date.today().isoformat()
        for s, (sig, last, ldate) in sigs.items():
            if sig == "in":
                open_by[s] = {"symbol": s, "layer": layers.get(s, ""),
                              "entry_date": ldate, "entry": round(last, 2)}
        action = f"INCEPTION {inception} — opened {len(open_by)} positions at last close"
    else:
        state = json.loads(STATE.read_text(encoding="utf-8"))
        inception = state["inception"]
        opened = closed = 0
        for s, (sig, last, ldate) in sigs.items():
            if s in open_by and sig == "out":                 # flipped to SELL -> close
                e = open_by.pop(s)
                ep = float(e["entry"])
                closed_rows.append({**e, "exit_date": ldate, "exit": round(last, 2),
                                    "ret": last / ep - 1,
                                    "days": (dt.date.fromisoformat(ldate)
                                             - dt.date.fromisoformat(str(e["entry_date"]))).days})
                closed += 1
            elif s not in open_by and sig == "in":            # fresh BUY -> open
                open_by[s] = {"symbol": s, "layer": layers.get(s, ""),
                              "entry_date": ldate, "entry": round(last, 2)}
                opened += 1
        action = f"update — opened {opened}, closed {closed}"

    open_rows = list(open_by.values())
    _save(open_rows, closed_rows, inception)
    print(f"[tracker] {action} | {len(open_rows)} open | {len(closed_rows)} closed")
    _report(open_rows, closed_rows, inception, sigs, prices, spy)


def _save(open_rows, closed_rows, inception) -> None:
    pd.DataFrame(open_rows, columns=["symbol", "layer", "entry_date", "entry"]).to_csv(OPEN_F, index=False)
    pd.DataFrame(closed_rows,
                 columns=["symbol", "layer", "entry_date", "entry", "exit_date", "exit", "ret", "days"]
                 ).to_csv(CLOSED_F, index=False)
    STATE.write_text(json.dumps({"inception": inception,
                                 "updated": dt.date.today().isoformat()}, indent=2), encoding="utf-8")


def _pct(x) -> str:
    return f"{x*100:+.0f}%" if x == x else "  -"


def _equity_curve(open_rows, closed_rows, prices, inception):
    """Equal-weight, daily-rebalanced paper portfolio from inception -> now."""
    inc = pd.Timestamp(inception)
    intervals = [(r["symbol"], pd.Timestamp(r["entry_date"]), None) for r in open_rows] + \
                [(r["symbol"], pd.Timestamp(r["entry_date"]), pd.Timestamp(r["exit_date"]))
                 for r in closed_rows]
    if not intervals:
        return None
    # daily returns per symbol, aligned to a common calendar from inception
    rets = {}
    for sym in {i[0] for i in intervals}:
        df = prices.get(sym)
        if df is not None and not df.empty:
            rets[sym] = df["Close"].dropna().pct_change()
    cal = None
    for sym in rets:
        idx = rets[sym].index[rets[sym].index >= inc]
        cal = idx if cal is None else cal.union(idx)
    if cal is None or len(cal) == 0:
        return None
    port = pd.Series(0.0, index=cal)
    for t in cal:
        held = [sym for sym, e, x in intervals if e < t and (x is None or t <= x) and sym in rets]
        if held:
            vals = [rets[sym].get(t, np.nan) for sym in held]
            vals = [v for v in vals if v == v]
            if vals:
                port[t] = float(np.mean(vals))
    return (1 + port).cumprod()


def _report(open_rows, closed_rows, inception, sigs, prices, spy) -> None:
    today = dt.date.today()
    now = pd.Timestamp(today)
    L = [f"# Signal tracker - {today:%Y-%m-%d}",
         f"_LIVE paper portfolio · inception **{inception}** · AI-bottleneck universe · "
         f"signal: long >50&200d, exit <200d · equal-weight vs SPY_",
         ""]

    # mark open book to market
    L.append("## Open positions — model says HOLD")
    L.append("```")
    L.append(f"{'SYM':<6}{'entry':>9}{'on':>12}{'last':>9}{'P&L':>7}{'vsSPY':>7}{'days':>6}")
    book = []
    for r in open_rows:
        sym = r["symbol"]
        last = sigs[sym][1] if sym in sigs else float("nan")
        ep = float(r["entry"])
        ret = last / ep - 1 if last == last else float("nan")
        ed = pd.Timestamp(r["entry_date"])
        spy_r = float(spy.asof(now) / spy.asof(ed) - 1) if len(spy) else float("nan")
        ex = ret - spy_r if (ret == ret and spy_r == spy_r) else float("nan")
        days = (today - ed.date()).days
        book.append((sym, ep, str(r["entry_date"]), last, ret, ex, days))
    for sym, ep, ed, last, ret, ex, days in sorted(book, key=lambda b: b[4] if b[4] == b[4] else -9, reverse=True):
        L.append(f"{sym:<6}{ep:>9.2f}{ed:>12}{last:>9.2f}{_pct(ret):>7}{_pct(ex):>7}{days:>6}")
    L.append("```")
    if book:
        L.append(f"_{len(book)} open · avg P&L {_pct(np.nanmean([b[4] for b in book]))} · "
                 f"avg vs SPY {_pct(np.nanmean([b[5] for b in book]))}_")

    # portfolio equity curve -> monthly vs SPY
    eq = _equity_curve(open_rows, closed_rows, prices, inception)
    L.append("\n## Monthly — equal-weight portfolio vs SPY")
    L.append("```")
    L.append(f"{'month':<9}{'port':>8}{'SPY':>8}{'diff':>8}")
    if eq is not None and len(eq) > 1:
        pm = (1 + eq.pct_change()).resample("ME").prod() - 1
        sm = (1 + spy.reindex(eq.index).pct_change()).resample("ME").prod() - 1
        for m in pm.index:
            p, s = pm.get(m, np.nan), sm.get(m, np.nan)
            L.append(f"{m.strftime('%Y-%m'):<9}{_pct(p):>8}{_pct(s):>8}{_pct(p-s):>8}")
        tot_p = float(eq.iloc[-1] - 1)
        tot_s = float(spy.asof(now) / spy.asof(pd.Timestamp(inception)) - 1) if len(spy) else float("nan")
        L.append("```")
        L.append(f"_since inception: portfolio {_pct(tot_p)} · SPY {_pct(tot_s)} · "
                 f"**{_pct(tot_p-tot_s)} vs SPY**_")
    else:
        L.append("```")
        L.append("_inception today — performance accrues from the next session. Come back tomorrow._")

    # lifetime closed-trade stats
    L.append("\n## Closed trades (lifetime)")
    if closed_rows:
        cdf = pd.DataFrame(closed_rows)
        L += [f"- **{len(cdf)} closed** · win rate **{100*(cdf['ret']>0).mean():.0f}%** · "
              f"avg {_pct(cdf['ret'].mean())} · avg hold {cdf['days'].mean():.0f}d",
              f"- best {_pct(cdf['ret'].max())} ({cdf.loc[cdf['ret'].idxmax(),'symbol']}) · "
              f"worst {_pct(cdf['ret'].min())} ({cdf.loc[cdf['ret'].idxmin(),'symbol']})"]
    else:
        L.append("- _none yet — every position still open (clean slate)._")

    text = "\n".join(L)
    REPORT.write_text(text, encoding="utf-8")
    print("[out] tracker_state.json / tracker_open.csv / tracker_closed.csv / tracker_report.md\n")
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="2y", help="history window for the 50/200d signal")
    ap.add_argument("--reset", action="store_true", help="wipe ledger and re-inception today")
    ap.add_argument("--symbols", nargs="+", help="ad-hoc list; default = ai_universe.csv")
    a = ap.parse_args()
    if a.symbols:
        symbols, layers = [s.upper() for s in a.symbols], {}
    else:
        uni = load_universe(ROOT / "ai_universe.csv")
        symbols = [u["symbol"] for u in uni]
        layers = {u["symbol"]: u["layer"] for u in uni}
    run(symbols, layers, a.period, a.reset)


if __name__ == "__main__":
    main()
