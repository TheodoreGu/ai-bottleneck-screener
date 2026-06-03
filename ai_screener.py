"""
AI-bottleneck screener  --  decision support, not execution.

Thesis
------
The binding constraint on the AI build-out has moved OFF the GPU and onto the
physical layer: electricity, grid interconnection, electrical gear, thermal /
liquid cooling, HBM memory, optical interconnect, and (longer-dated) new nuclear.
The companies that sell those "picks beneath the picks" are the bottleneck
beneficiaries. The edge is NOT owning what already tripled -- it's the names in
this basket that (a) smart money already owns and (b) have NOT gone exponential
this year yet, while their uptrend is still intact.

What it ranks (composite 0-100, weights in ai_config.json)
  smartmoney : institutional %  +  Dataroma 13F superinvestors  +  Congress buys
  laggard    : "room to run" -- high when YTD is low, ZERO when parabolic,
               haircut if the 200d trend is broken (avoid falling knives)
  momentum   : trend intact + relative strength + not over-extended

Outputs (./out)
  ai_<date>_full.csv    every name, every column  (audit, never read by the LLM)
  ai_latest_digest.md   tiny ranked digest        (the only thing worth reading)

Usage
  python ai_screener.py
  python ai_screener.py --top 20
  python ai_screener.py --symbols CEG VRT ETN   # ad-hoc, shows all
"""
from __future__ import annotations

import csv
import json
import shutil
import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

import sources
import signals
import smartmoney

ROOT = Path(__file__).parent
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)


def load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    text = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("//"))
    cfg = json.loads(text)
    # drop the "// ..." documentation keys allowed inside the json blocks
    cfg["thresholds"] = {k: v for k, v in cfg["thresholds"].items() if not k.startswith("//")}
    return cfg


def load_universe(cfg: dict) -> list[dict]:
    fp = ROOT / cfg["universe_file"]
    rows: list[dict] = []
    with fp.open(newline="", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.lower().startswith("symbol,"):
                continue
            parts = next(csv.reader([line]))
            sym = (parts[0] if parts else "").strip().upper()
            if not sym:
                continue
            rows.append({"symbol": sym,
                         "layer": (parts[1].strip() if len(parts) > 1 else ""),
                         "note": (parts[2].strip() if len(parts) > 2 else "")})
    return rows


def _ytd(df: pd.DataFrame) -> float:
    """Return YTD total return from the price history, or NaN."""
    close = df["Close"].dropna()
    if not len(close):
        return float("nan")
    yr = dt.date.today().year
    cur = close[close.index.year == yr]
    if not len(cur):
        return float("nan")
    return float(close.iloc[-1] / cur.iloc[0] - 1)


def _clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


def laggard_score(ytd: float, above_sma200, th: dict) -> tuple[float, list[str]]:
    """High when the name has NOT run yet; 0 when parabolic; halved if trend broken."""
    flags: list[str] = []
    if ytd is None or pd.isna(ytd):
        return 0.0, flags
    lag_max, para = th["ytd_laggard_max"], th["ytd_parabolic"]
    # 'room to run' -- 1.0 at/below the laggard band, 0.0 once parabolic
    room = _clamp((para - ytd) / max(1e-9, para - lag_max), 0, 1)
    s = room * 100
    if above_sma200 is False:           # broken trend -> falling-knife haircut
        s *= 0.5
    if ytd <= lag_max:
        flags.append("LAGGARD")
    if ytd >= para:
        flags.append("PARABOLIC")
    return round(s, 1), flags


def momentum_score(row: dict, th: dict) -> tuple[float, list[str]]:
    flags: list[str] = []
    s = 0.0
    if row.get("above_sma50") and row.get("above_sma200"):
        s += 30
        flags.append("UPTRD")
    elif row.get("above_sma200"):
        s += 15
    rs = row.get("rs_63")
    if rs is not None and pd.notna(rs):
        s += _clamp((rs + 0.05) / 0.25 * 30, 0, 30)   # +20% RS ~ full marks
        flags.append("RS+" if rs > 0 else "RS-")
    rsi = row.get("rsi14")
    if rsi is not None and pd.notna(rsi):
        if rsi >= th["rsi_overbought"]:
            flags.append("OB")                         # extended -> no momentum credit
        elif rsi >= 50:
            s += 25                                     # healthy trend
        elif rsi <= th["rsi_low"]:
            s += 10                                     # washed out, possible turn
    p52h = row.get("pct_52w_high")
    if p52h is not None and pd.notna(p52h) and p52h >= -0.05:
        flags.append("52H")
    return round(_clamp(s), 1), flags


def run(cfg: dict, args) -> None:
    if args.symbols:
        universe = [{"symbol": s.upper(), "layer": "", "note": ""} for s in args.symbols]
    else:
        universe = load_universe(cfg)
    symbols = [u["symbol"] for u in universe]
    bench_sym = cfg["benchmark"]
    th, weights = cfg["thresholds"], cfg["scoring_weights"]
    print(f"[universe] {len(symbols)} AI-bottleneck names")

    # ---- prices (batched) ----
    fetch_syms = list(dict.fromkeys(symbols + [bench_sym]))
    prices = sources.fetch_prices(fetch_syms, period="1y")
    bench_df = prices.get(bench_sym)

    # ---- smart-money sources ----
    inst = smartmoney.fetch_institutional(symbols, cfg["cache_hours"] * 2)
    supr = smartmoney.fetch_superinvestors(symbols)
    gov = smartmoney.fetch_congress(symbols, cfg, cfg["congress_lookback_days"], cfg["cache_hours"] * 2)
    gov_state = "FMP on" if gov else "off (no FMP key)"
    print(f"[smartmoney] inst={sum(v.get('inst_pct') is not None for v in inst.values())} "
          f"| dataroma={sum((v.get('dr_count') or 0) > 0 for v in supr.values())} hits "
          f"| congress={gov_state} | prices={len(prices)}/{len(fetch_syms)}")

    rows = []
    for u in universe:
        sym = u["symbol"]
        df = prices.get(sym)
        if df is None or df.empty:
            continue
        row = dict(u)
        row["thesis"] = u.get("note", "")        # keep the universe note (score overwrites 'note')
        row.update(signals.technicals(df, bench_df))
        row["ytd"] = _ytd(df)

        sm, sm_flags, sm_note, sm_fields = smartmoney.smartmoney_score(
            inst.get(sym), supr.get(sym), gov.get(sym), th)
        row.update(sm_fields)
        row["_sm"] = sm
        row["_sm_note"] = sm_note
        row["sm_money"] = (supr.get(sym) or {}).get("sm_money", "")   # who's in it (adders first)

        lag, lag_flags = laggard_score(row.get("ytd"), row.get("above_sma200"), th)
        mom, mom_flags = momentum_score(row, th)
        row["_lag"], row["_mom"] = lag, mom

        row["score"] = round(weights["smartmoney"] * sm
                             + weights["laggard"] * lag
                             + weights["momentum"] * mom, 1)
        row["flags"] = sm_flags + lag_flags + mom_flags
        row["note"] = _note(row, sm_note)
        rows.append(row)

    rows.sort(key=lambda r: r["score"], reverse=True)

    # ---- Stage 2: deeper enrichment for the TOP TIER only (gentle on Dataroma) ----
    # tier = every LAGGARD + the top-N standouts by score; capped.
    standout_n = th.get("fundamentals_top_n", 6)
    tier = [r for r in rows if "LAGGARD" in r["flags"]]
    for r in rows[:standout_n]:
        if r not in tier:
            tier.append(r)
    tier = tier[: th.get("fundamentals_max", 12)]
    print(f"[stage2] tenure + fundamentals for {len(tier)} top-tier names")
    for r in tier:
        sym = r["symbol"]
        holders = (supr.get(sym) or {}).get("holders") or []
        fi = smartmoney.fetch_first_included(sym, holders)
        if fi:
            r.update(fi)
        r["_fund"] = smartmoney.fundamentals_blurb(inst.get(sym))
        irec = inst.get(sym) or {}
        r["sector"] = irec.get("sector")
        r["_tier"] = True
        if fi:                                   # compact tenure flag; full date in the fundamentals section
            r["flags"] = r["flags"] + [f"SINCE'{str(fi['sm_first_sort'][0])[2:]}"]

    _write_full(rows)
    _write_digest(rows, cfg, gov_state, args)


def _note(row: dict, sm_note: str) -> str:
    bits = []                                    # YTD is its own column; don't repeat it here
    rs = row.get("rs_63")
    if rs is not None and pd.notna(rs):
        bits.append(f"RS{rs*100:+.0f}%")
    if sm_note:
        bits.append(sm_note)
    return " ".join(bits)


def _write_full(rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    if "flags" in df:
        df["flags"] = df["flags"].apply(lambda x: " ".join(x) if isinstance(x, list) else x)
    path = OUT / f"ai_{dt.date.today():%Y-%m-%d}_full.csv"
    df.to_csv(path, index=False)
    print(f"[out] full -> {path.name} ({len(df)} rows)")


def _write_digest(rows: list[dict], cfg: dict, gov_state: str, args) -> None:
    th = cfg["thresholds"]
    min_score = th["digest_min_score"]
    max_rows = args.top or th["digest_max_rows"]
    if args.symbols:
        hits = rows[:max_rows]
    else:
        hits = [r for r in rows if r["score"] >= min_score][:max_rows]

    today = dt.date.today()
    L = [f"# AI-bottleneck digest - {today:%Y-%m-%d}",
         f"_thesis: smart-money-owned bottleneck names that haven't gone exponential yet_  "
         f"_(congress: {gov_state})_",
         "",
         "Flags: LAGGARD not-yet-run | PARABOLIC already ran (avoid) | UPTRD >50&200d | "
         "RS+/- vs SPY | OB overbought | 52H near high | SM-INST high institutional | "
         "SM-13F superinvestors hold | SM-NEW fresh 13F initiation | SM-GOV congress buying | "
         "SINCE'YY first smart-money quarter (top tier)",
         "_Smart money column = superinvestor(s) adding it (Dataroma): `*` new position this "
         "quarter, `+` adding to existing; else the largest holder. `·N` = total holders._",
         "",
         "```",
         f"{'SYM':<5}{'layer':<9}{'last':>8}{'YTD':>6}{'scr':>5}  "
         f"{'smart money':<24}flags"]
    for r in hits:
        sym = r["symbol"][:5]
        layer = (r.get("layer") or "")[:8]
        last = f"{r.get('last', float('nan')):.2f}" if pd.notna(r.get("last")) else "-"
        ytd = r.get("ytd")
        ytds = f"{ytd*100:+.0f}%" if ytd is not None and pd.notna(ytd) else "-"
        smoney = (r.get("sm_money") or "")[:23]
        flags = " ".join(r["flags"])[:60]
        L.append(f"{sym:<5}{layer:<9}{last:>8}{ytds:>6}{r['score']:>5.0f}  "
                 f"{smoney:<24}{flags}")
    L.append("```")
    if not hits:
        L.append("\n_(nothing cleared the threshold -- loosen ai_config or quiet tape)_")

    # ---- fundamental summary for the laggard + standout tier ----
    tier = [r for r in rows if r.get("_tier")]
    if tier:
        L.append("")
        L.append("## Fundamentals — laggard + standout tier")
        for r in tier:
            ytd = r.get("ytd")
            ytds = f"{ytd*100:+.0f}%" if ytd is not None and pd.notna(ytd) else "-"
            since = f" · SM since {r['sm_first']}" if r.get("sm_first") else " · SM since n/a"
            tags = " ".join(t for t in r["flags"]
                            if t in ("LAGGARD", "PARABOLIC", "UPTRD", "SM-INST", "SM-13F",
                                     "SM-NEW", "SM-GOV") or t.startswith("RS"))
            fund = r.get("_fund") or "_fundamentals n/a_"
            L.append(f"- **{r['symbol']}** {r.get('layer','')} · score {r['score']:.0f} · "
                     f"YTD {ytds}{since} · {tags}")
            L.append(f"  {fund} — {r.get('thesis','')}")

    text = "\n".join(L)
    dated = OUT / f"ai_{today:%Y-%m-%d}_digest.md"
    dated.write_text(text, encoding="utf-8")
    shutil.copyfile(dated, OUT / "ai_latest_digest.md")
    print(f"[out] digest -> {dated.name}\n")
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "ai_config.json"))
    ap.add_argument("--top", type=int, default=0, help="cap digest rows")
    ap.add_argument("--symbols", nargs="+", help="ad-hoc symbol list; overrides universe, shows all")
    args = ap.parse_args()
    cfg = load_config(Path(args.config))
    run(cfg, args)


if __name__ == "__main__":
    main()
