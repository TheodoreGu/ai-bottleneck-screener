"""
Signal tracker -- a live, self-auditing track record of the model's calls.

It turns the trend signal into a paper-trade ledger so you can answer "is this
thing actually any good?" with real, dated numbers instead of vibes:

  ENTRY (model says BUY/HOLD)  : close crosses above its 50d AND 200d  -> record date+price
  EXIT  (model says SELL)      : close drops below its 200d            -> record date+price, P&L

The signal is causal (no look-ahead), so the full ledger is *derived from price
history* each run -- which means you get a track record immediately AND it keeps
growing every month with no state to corrupt. Each closed trade is scored net of
SPY over the identical holding window (did the call beat just owning the index?).

Outputs (committed, so the record persists & is inspectable in git):
  tracker_closed.csv   every completed paper trade
  tracker_open.csv     positions the model currently says HOLD (with unrealised P&L)
  tracker_report.md    the read: open book + monthly hit-rate/return vs SPY + lifetime

Usage
  python tracker.py                      # AI-bottleneck universe, 3y of history
  python tracker.py --period 5y
  python tracker.py --symbols AMD NVDA TLN
"""
from __future__ import annotations

import csv
import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

import sources

ROOT = Path(__file__).parent
FAST, SLOW = 50, 200


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


def _spy_ret(spy: pd.Series, d0, d1) -> float:
    try:
        a, b = spy.asof(d0), spy.asof(d1)
        return float(b / a - 1) if a and b and a > 0 else float("nan")
    except Exception:
        return float("nan")


def segment(df: pd.DataFrame, spy: pd.Series, layer: str, sym: str) -> tuple[list[dict], dict | None]:
    """Split one name's history into trend trades (closed list + current open, if any)."""
    close = df["Close"].dropna()
    if len(close) < SLOW + 5:
        return [], None
    sf = close.rolling(FAST).mean().values
    ss = close.rolling(SLOW).mean().values
    c = close.values
    idx = close.index
    spans, inpos, e_i = [], False, None
    for i in range(len(c)):
        if np.isnan(ss[i]):
            continue
        if not inpos:
            if c[i] > ss[i] and c[i] > sf[i]:
                inpos, e_i = True, i
        elif c[i] < ss[i]:
            spans.append((e_i, i))
            inpos = False
    open_span = (e_i, None) if inpos else None

    def rec(ei, xi):
        ed, ep = idx[ei], float(c[ei])
        closed = xi is not None
        xd, xp = (idx[xi], float(c[xi])) if closed else (idx[-1], float(c[-1]))
        ret = xp / ep - 1
        spy_r = _spy_ret(spy, ed, xd)
        return {"symbol": sym, "layer": layer,
                "entry_date": ed.date().isoformat(), "entry": round(ep, 2),
                "exit_date": xd.date().isoformat(), "exit": round(xp, 2),
                "ret": ret, "days": (xd - ed).days,
                "spy_ret": spy_r, "excess": ret - spy_r if spy_r == spy_r else float("nan"),
                "open": not closed}

    return [rec(e, x) for e, x in spans], (rec(*open_span) if open_span else None)


def run(symbols: list[str], layers: dict, period: str) -> None:
    fetch = list(dict.fromkeys(symbols + ["SPY"]))
    prices = sources.fetch_prices(fetch, period=period)
    spy = prices["SPY"]["Close"].dropna() if "SPY" in prices else pd.Series(dtype=float)
    print(f"[tracker] {len(symbols)} names | prices {len(prices)}/{len(fetch)} | period {period}")

    closed, opens = [], []
    for s in symbols:
        df = prices.get(s)
        if df is None or df.empty:
            continue
        cl, op = segment(df, spy, layers.get(s, ""), s)
        closed.extend(cl)
        if op:
            opens.append(op)

    _write_csv(ROOT / "tracker_closed.csv", closed)
    _write_csv(ROOT / "tracker_open.csv", opens)
    _report(closed, opens, period)


def _write_csv(path: Path, rows: list[dict]) -> None:
    cols = ["symbol", "layer", "entry_date", "entry", "exit_date", "exit",
            "ret", "days", "spy_ret", "excess", "open"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(path, index=False)


def _pct(x) -> str:
    return f"{x*100:+.0f}%" if x == x else "  -"


def _report(closed: list[dict], opens: list[dict], period: str) -> None:
    today = dt.date.today()
    L = [f"# Signal tracker - {today:%Y-%m-%d}",
         f"_universe: AI-bottleneck | signal: long while >200d (enter >50&200d), exit <200d | "
         f"history: {period} | scored net of SPY_",
         ""]

    # ---- open book (the model's current BUY/HOLD calls) ----
    L.append("## Open positions — model currently says HOLD")
    L.append("```")
    L.append(f"{'SYM':<6}{'entry':>9}{'on':>12}{'last':>9}{'unreal':>8}{'vsSPY':>7}{'days':>6}")
    for r in sorted(opens, key=lambda r: r["excess"] if r["excess"] == r["excess"] else -9, reverse=True):
        L.append(f"{r['symbol']:<6}{r['entry']:>9.2f}{r['entry_date']:>12}{r['exit']:>9.2f}"
                 f"{_pct(r['ret']):>8}{_pct(r['excess']):>7}{r['days']:>6}")
    L.append("```")
    if opens:
        avg_un = np.nanmean([r["ret"] for r in opens])
        avg_ex = np.nanmean([r["excess"] for r in opens])
        L.append(f"_{len(opens)} open · avg unrealised {_pct(avg_un)} · avg vs SPY {_pct(avg_ex)}_")

    # ---- monthly track record (closed trades, by exit month) ----
    L.append("\n## Closed trades — by exit month")
    L.append("```")
    L.append(f"{'month':<9}{'n':>4}{'win%':>6}{'avgRet':>8}{'vsSPY':>7}")
    if closed:
        cdf = pd.DataFrame(closed)
        cdf["m"] = pd.to_datetime(cdf["exit_date"]).dt.to_period("M").astype(str)
        for m, g in cdf.groupby("m"):
            L.append(f"{m:<9}{len(g):>4}{100*(g['ret']>0).mean():>5.0f}%"
                     f"{_pct(g['ret'].mean()):>8}{_pct(g['excess'].mean()):>7}")
    L.append("```")

    # ---- lifetime summary ----
    L.append("\n## Lifetime")
    if closed:
        cdf = pd.DataFrame(closed)
        n = len(cdf)
        win = (cdf["ret"] > 0).mean()
        beat = (cdf["excess"] > 0).mean()
        L += [
            f"- **{n} closed trades** · win rate **{win*100:.0f}%** · beat SPY **{beat*100:.0f}%** of the time",
            f"- avg return **{_pct(cdf['ret'].mean())}** (median {_pct(cdf['ret'].median())}) · "
            f"avg excess vs SPY **{_pct(cdf['excess'].mean())}**",
            f"- avg winner {_pct(cdf.loc[cdf['ret']>0,'ret'].mean())} · "
            f"avg loser {_pct(cdf.loc[cdf['ret']<=0,'ret'].mean())} · "
            f"avg hold {cdf['days'].mean():.0f}d",
            f"- best {_pct(cdf['ret'].max())} ({cdf.loc[cdf['ret'].idxmax(),'symbol']}) · "
            f"worst {_pct(cdf['ret'].min())} ({cdf.loc[cdf['ret'].idxmin(),'symbol']})",
        ]
    else:
        L.append("- _no closed trades yet — all signals still open_")
    L.append(f"- {len(opens)} positions currently open (marked-to-market above)")

    text = "\n".join(L)
    (ROOT / "tracker_report.md").write_text(text, encoding="utf-8")
    print("[out] tracker_closed.csv / tracker_open.csv / tracker_report.md\n")
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="3y", help="history window (e.g. 3y, 5y, max)")
    ap.add_argument("--symbols", nargs="+", help="ad-hoc list; default = ai_universe.csv")
    a = ap.parse_args()
    if a.symbols:
        symbols, layers = [s.upper() for s in a.symbols], {}
    else:
        uni = load_universe(ROOT / "ai_universe.csv")
        symbols = [u["symbol"] for u in uni]
        layers = {u["symbol"]: u["layer"] for u in uni}
    run(symbols, layers, a.period)


if __name__ == "__main__":
    main()
