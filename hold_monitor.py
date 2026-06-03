"""
Position hold-monitor  --  the rule that holds for you so emotion doesn't sell.

The point isn't to predict tops. It's to keep you IN a winner while the trend is
intact, and only speak up when a *real* exit line is actually breached -- and to
show you the exact price level that would flip the signal, so you hold to that
line instead of bailing on a scary green day (the AMD-at-$350 problem).

State machine (priority order), per position:
  EXIT   close < 200d AND 200d rolling over     -> trend is broken, step aside
  EXIT   close < 200d                           -> long-trend break
  TRIM   close < chandelier ATR trailing stop   -> the run's giving back too much
  TRIM   >parabolic_ext above 200d, or RSI hot  -> blow-off; bank some, keep a core
  WATCH  close < 50d (but > 200d)               -> first crack, trend still up
  HOLD   above 50d & 200d                       -> ride it

It reuses the project's price/technical plumbing (sources.py, signals.py) and the
same trend-following logic validated in the metals strategy -- applied to single names.

Outputs (./out)
  hold_<date>_full.csv    every level, every position   (audit)
  hold_latest.md          tiny status board             (the read)

Usage
  python hold_monitor.py
  python hold_monitor.py --symbols AMD NVDA          # ad-hoc (no entry/gain)
"""
from __future__ import annotations

import csv
import json
import shutil
import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

import sources
import signals

ROOT = Path(__file__).parent
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)


def load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    cfg = json.loads("\n".join(l for l in text.splitlines() if not l.lstrip().startswith("//")))
    cfg["params"] = {k: v for k, v in cfg["params"].items() if not k.startswith("//")}
    return cfg


def load_holdings(cfg: dict) -> list[dict]:
    fp = ROOT / cfg["holdings_file"]
    rows: list[dict] = []
    with fp.open(newline="", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#") or line.lower().startswith("symbol,"):
                continue
            p = next(csv.reader([line]))
            sym = (p[0] if p else "").strip().upper()
            if not sym:
                continue
            rows.append({
                "symbol": sym,
                "entry": _f(p[1]) if len(p) > 1 else None,
                "date": (p[2].strip() if len(p) > 2 and p[2].strip() else None),
                "benchmark": (p[3].strip().upper() if len(p) > 3 and p[3].strip() else ""),
                "note": (p[4].strip() if len(p) > 4 else ""),
            })
    return rows


def _f(x):
    try:
        return float(str(x).strip())
    except (ValueError, AttributeError):
        return None


def _atr(df: pd.DataFrame, n: int) -> float:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / n, adjust=False).mean().iloc[-1])


def analyze(df: pd.DataFrame, bench: pd.DataFrame | None, h: dict, p: dict) -> dict:
    close = df["Close"].dropna()
    last = float(close.iloc[-1])
    out: dict = {**h, "last": last}
    if len(close) < p["sma_slow"]:
        out.update(state="n/a", reason="insufficient history", cushion=None)
        return out

    sma_f = close.rolling(p["sma_fast"]).mean()
    sma_s = close.rolling(p["sma_slow"]).mean()
    s_f, s_s = float(sma_f.iloc[-1]), float(sma_s.iloc[-1])
    slope = float(s_s - sma_s.iloc[-1 - p["slope_lookback"]]) if len(sma_s) > p["slope_lookback"] else 0.0
    atr = _atr(df, p["atr_n"])

    # trailing-stop high: from entry date if given, else the recent window
    if h.get("date"):
        try:
            seg = close[close.index >= pd.Timestamp(h["date"])]
            hi = float(seg.max()) if len(seg) else float(close.tail(p["high_lookback"]).max())
        except Exception:
            hi = float(close.tail(p["high_lookback"]).max())
    else:
        hi = float(close.tail(p["high_lookback"]).max())
    chand = hi - p["chandelier_k"] * atr
    ext = last / s_s - 1 if s_s else float("nan")
    rsi = signals._rsi(close)

    # relative strength vs benchmark (63d)
    rs = float("nan")
    if bench is not None and len(bench) >= 64 and len(close) >= 64:
        bc = bench["Close"].dropna()
        rs = float(close.iloc[-1] / close.iloc[-64] - 1) - float(bc.iloc[-1] / bc.iloc[-64] - 1)

    out.update(sma_fast=s_f, sma_slow=s_s, slope_slow=slope, atr=atr,
               trail=chand, high=hi, ext=ext, rsi14=rsi, rs_63=rs)

    # ---- decide ----
    if last < s_s and slope <= 0:
        state, reason = "EXIT", f"below {p['sma_slow']}d & it's rolling over -> trend broken"
        line = s_s
    elif last < s_s:
        state, reason = "EXIT", f"closed below {p['sma_slow']}d ({s_s:.2f}) -> long-trend break"
        line = s_s
    elif last < chand:
        state, reason = "TRIM", f"hit ATR trail stop ({chand:.2f}) -> run giving back too much"
        line = chand
    elif (pd.notna(ext) and ext >= p["parabolic_ext"]) or (pd.notna(rsi) and rsi >= p["rsi_hot"]):
        why = f"+{ext*100:.0f}% over {p['sma_slow']}d" if ext >= p["parabolic_ext"] else f"RSI {rsi:.0f}"
        state, reason = "TRIM", f"parabolic ({why}) -> bank ~{p['trim_frac']*100:.0f}%, keep core"
        line = chand
    elif last < s_f:
        state, reason = "WATCH", f"below {p['sma_fast']}d ({s_f:.2f}) -> first crack, {p['sma_slow']}d still up"
        line = s_s
    else:
        state, reason = "HOLD", "above 50d & 200d -> trend intact, ride it"
        line = max(chand, s_s)               # nearest line you'd actually exit on

    out["exit_line"] = line
    out["cushion"] = (last / line - 1) if line and line > 0 else None
    out["state"], out["reason"] = state, reason
    # downgrade a HOLD to WATCH if relative strength has flipped negative
    if state == "HOLD" and pd.notna(rs) and rs < 0:
        out["state"] = "WATCH"
        out["reason"] = "trend up but lagging SPY (RS-) -> watch for momentum roll"
    return out


def run(cfg: dict, args) -> None:
    p = cfg["params"]
    if args.symbols:
        holdings = [{"symbol": s.upper(), "entry": None, "date": None, "benchmark": "", "note": ""}
                    for s in args.symbols]
    else:
        holdings = load_holdings(cfg)
    syms = [h["symbol"] for h in holdings]
    bench_sym = cfg["benchmark"]
    # only treat a clean ticker-like benchmark as a peer (guards against a misaligned CSV col)
    peers = {h["benchmark"] for h in holdings if h["benchmark"] and " " not in h["benchmark"]}
    fetch = list(dict.fromkeys(syms + [bench_sym] + list(peers)))
    prices = sources.fetch_prices(fetch, period="2y")     # 2y so the 200d slope is well-formed
    bench_df = prices.get(bench_sym)
    print(f"[hold] {len(syms)} positions | prices {len(prices)}/{len(fetch)}")

    rows = []
    order = {"EXIT": 0, "TRIM": 1, "WATCH": 2, "HOLD": 3, "n/a": 4}
    for h in holdings:
        df = prices.get(h["symbol"])
        if df is None or df.empty:
            rows.append({**h, "state": "n/a", "reason": "no price data"})
            continue
        b = prices.get(h["benchmark"]) if h["benchmark"] else bench_df
        rows.append(analyze(df, b, h, p))
    rows.sort(key=lambda r: order.get(r.get("state", "n/a"), 9))
    _write_full(rows)
    _write_board(rows, cfg)


def _g(row: dict) -> str:
    e, last = row.get("entry"), row.get("last")
    if e and last:
        return f"{(last/e-1)*100:+.0f}%"
    return "-"


def _write_full(rows: list[dict]) -> None:
    if not rows:
        return
    path = OUT / f"hold_{dt.date.today():%Y-%m-%d}_full.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[out] full -> {path.name}")


def _write_board(rows: list[dict], cfg: dict) -> None:
    today = dt.date.today()
    icon = {"EXIT": "EXIT", "TRIM": "TRIM", "WATCH": "WATCH", "HOLD": "HOLD", "n/a": "n/a"}
    L = [f"# Hold-monitor - {today:%Y-%m-%d}",
         "_stays quiet while the trend holds; the level shown is where the signal flips._",
         "",
         "```",
         f"{'SYM':<7}{'state':<6}{'last':>9}{'gain':>6}{'exit@':>9}{'cush':>6}  reason"]
    for r in rows:
        sym = r["symbol"][:7]
        last = f"{r.get('last', float('nan')):.2f}" if pd.notna(r.get("last")) else "-"
        line = r.get("exit_line")
        line_s = f"{line:.2f}" if line else "-"
        cush = r.get("cushion")
        cush_s = f"{cush*100:+.0f}%" if cush is not None else "-"
        L.append(f"{sym:<7}{icon.get(r.get('state','n/a')):<6}{last:>9}{_g(r):>6}"
                 f"{line_s:>9}{cush_s:>6}  {r.get('reason','')}")
    L.append("```")
    # one line of context per non-HOLD so the board explains itself
    actionable = [r for r in rows if r.get("state") in ("EXIT", "TRIM", "WATCH")]
    if actionable:
        L.append("")
        L.append("**Needs attention:**")
        for r in actionable:
            rs = r.get("rs_63")
            rss = f" · RS {rs*100:+.0f}% vs SPY" if rs is not None and pd.notna(rs) else ""
            L.append(f"- **{r['symbol']}** ({r['state']}): {r.get('reason','')}{rss}"
                     + (f" — {r['note']}" if r.get("note") else ""))
    else:
        L.append("\n_All positions HOLD — trends intact, nothing to do._")

    text = "\n".join(L)
    dated = OUT / f"hold_{today:%Y-%m-%d}.md"
    dated.write_text(text, encoding="utf-8")
    shutil.copyfile(dated, OUT / "hold_latest.md")
    print(f"[out] board -> {dated.name}\n")
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "hold_config.json"))
    ap.add_argument("--symbols", nargs="+", help="ad-hoc symbols (no entry/gain)")
    args = ap.parse_args()
    run(load_config(Path(args.config)), args)


if __name__ == "__main__":
    main()
