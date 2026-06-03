"""
Smart-money confirmation signals for the AI-bottleneck screener.

Three independent, FREE-ish sources. Each degrades gracefully: if a source is
unreachable / unconfigured it simply contributes nothing (the name is scored on
whatever else is available), exactly like the IBKR borrow file in the main screener.

  1. INSTITUTIONAL  yfinance .info  -> % of float held by institutions
                    (the broad "smart money owns it" read, refreshes daily)
  2. SUPERINVESTORS dataroma.com    -> how many tracked 13F value-investors hold it
                    (concentrated conviction; 13F is quarterly so cached 7 days)
  3. CONGRESS       FMP API         -> recent buy/sell disclosures by members of
                    Congress for the ticker. Needs a free FMP key in
                    config["fmp_api_key"] or env FMP_API_KEY; OFF (contributes 0)
                    until then. (The old free S3 stock-watcher feeds went private.)

Cache lives in ./cache alongside the main screener's cache.
"""
from __future__ import annotations

import os
import re
import time
import json
import datetime as dt
from pathlib import Path

import requests
import yfinance as yf

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/120 Safari/537.36"}


# --------------------------------------------------------------------------- #
#  tiny cache helpers (mirrors sources.py so behaviour is consistent)
# --------------------------------------------------------------------------- #
def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save(path: Path, obj) -> None:
    try:
        path.write_text(json.dumps(obj), encoding="utf-8")
    except Exception:
        pass


# --------------------------------------------------------------------------- #
#  1. institutional ownership  (yfinance, per ticker, cached)
# --------------------------------------------------------------------------- #
# fundamental fields grabbed from the SAME .info call (free) for the top-tier blurb
_FUND_KEYS = ["sector", "industry", "marketCap", "trailingPE", "forwardPE",
              "revenueGrowth", "earningsGrowth", "grossMargins", "profitMargins",
              "debtToEquity"]


def fetch_institutional(symbols: list[str], cache_hours: float = 24) -> dict[str, dict]:
    """Return {SYM: {inst_pct, insider_pct, <fundamentals>}}.  inst_pct is 0..1 of shares.

    Fundamentals (market cap, P/E, growth, margins, sector) ride along on the same
    yfinance .info call so the top-tier fundamental blurb costs no extra fetches.
    """
    cache = CACHE_DIR / "ownership.json"
    store = _load(cache) or {}
    now = time.time()
    out: dict[str, dict] = {}
    for sym in symbols:
        rec = store.get(sym)
        if rec and (now - rec.get("_ts", 0)) < cache_hours * 3600:
            out[sym] = rec
            continue
        rec = {"inst_pct": None, "insider_pct": None}
        try:
            info = yf.Ticker(sym).info or {}
            rec["inst_pct"] = info.get("heldPercentInstitutions")
            rec["insider_pct"] = info.get("heldPercentInsiders")
            for k in _FUND_KEYS:
                rec[k] = info.get(k)
            summ = info.get("longBusinessSummary") or ""
            rec["summary1"] = summ.split(". ")[0][:200] if summ else None  # 1st sentence
        except Exception:
            pass
        rec["_ts"] = now
        store[sym] = rec
        out[sym] = rec
        time.sleep(0.12)
    _save(cache, store)
    return out


# --------------------------------------------------------------------------- #
#  2. Dataroma superinvestors  (scrape per ticker, cached 7d -> 13F is quarterly)
# --------------------------------------------------------------------------- #
# stock.php rows look like:
#   <a href="/m/holdings.php?m=LPC" ...>Manager</a> ... <td class="buy">Add 40.65%</td>
# A brand-new position this quarter shows activity text == "Buy" (Add/Reduce/Sell otherwise).
_HOLDER_RE = re.compile(
    r'holdings\.php\?m=([A-Za-z0-9]+)"[^>]*>.*?<td class="(buy|sell)">\s*([^<]*?)\s*</td>',
    re.S)


def _scrape_dataroma_stock(sym: str) -> dict | None:
    """Return {dr_count, dr_new, holders:[mgr,...]} for `sym`, or None on failure.

    dr_count = distinct tracked superinvestors holding it.
    dr_new   = how many INITIATED a brand-new position this quarter (activity 'Buy').
    holders  = manager codes (used by fetch_first_included to date the inception).
    """
    url = f"https://www.dataroma.com/m/stock.php?sym={sym}"
    try:
        r = requests.get(url, headers=_UA, timeout=25)
        if r.status_code != 200 or len(r.text) < 500:
            return None
        holders, new = [], 0
        for mgr, _cls, act in _HOLDER_RE.findall(r.text):
            holders.append(mgr)
            if act.strip().lower() == "buy":          # exact 'Buy' = initiation (vs 'Add x%')
                new += 1
        # fall back to a plain code count if the structured parse found nothing
        if not holders:
            codes = set(re.findall(r"holdings\.php\?m=([A-Za-z0-9]+)", r.text))
            return {"dr_count": len(codes), "dr_new": 0, "holders": list(codes)}
        seen = list(dict.fromkeys(holders))
        return {"dr_count": len(seen), "dr_new": new, "holders": seen}
    except Exception:
        return None


def fetch_superinvestors(symbols: list[str], cache_hours: float = 168) -> dict[str, dict]:
    """Return {SYM: {dr_count, dr_new, holders}}.  Cached a week (13F is quarterly)."""
    cache = CACHE_DIR / "dataroma.json"
    store = _load(cache) or {}
    now = time.time()
    out: dict[str, dict] = {}
    for sym in symbols:
        rec = store.get(sym)
        if rec and (now - rec.get("_ts", 0)) < cache_hours * 3600 and "holders" in rec:
            out[sym] = rec
            continue
        scraped = _scrape_dataroma_stock(sym) or {"dr_count": None, "dr_new": None, "holders": []}
        rec = {**scraped, "_ts": now}
        store[sym] = rec
        out[sym] = rec
        time.sleep(0.4)  # be gentle with dataroma
    _save(cache, store)
    return out


def _parse_first_quarter(html: str) -> tuple[int, int] | None:
    """Earliest (year, quarter) in a hist.php table; the oldest row = first reported."""
    quarters = re.findall(r"(20\d\d)\s*(?:&nbsp;?\s*)?Q([1-4])", html)
    if not quarters:
        return None
    ys = [(int(y), int(q)) for y, q in quarters]
    return min(ys)


def fetch_first_included(sym: str, holders: list[str], cache_hours: float = 720) -> dict | None:
    """Earliest quarter `sym` appears in any current holder's history.

    'First time smart money included it' (among tracked superinvestors still holding).
    One hist.php fetch per holder, cached 30 days. Call this ONLY for the top tier.
    Returns {sm_first: 'YYYY Qn', sm_first_sort: [y,q]} or None.
    """
    if not holders:
        return None
    cache = CACHE_DIR / "dataroma_hist.json"
    store = _load(cache) or {}
    now = time.time()
    earliest: tuple[int, int] | None = None
    for mgr in holders:
        key = f"{mgr}:{sym}"
        rec = store.get(key)
        if not (rec and (now - rec.get("_ts", 0)) < cache_hours * 3600):
            yq = None
            try:
                r = requests.get(f"https://www.dataroma.com/m/hist/hist.php?f={mgr}&s={sym}",
                                 headers=_UA, timeout=25)
                if r.status_code == 200:
                    yq = _parse_first_quarter(r.text)
            except Exception:
                yq = None
            rec = {"yq": list(yq) if yq else None, "_ts": now}
            store[key] = rec
            time.sleep(0.35)
        if rec.get("yq"):
            yq = tuple(rec["yq"])
            if earliest is None or yq < earliest:
                earliest = yq
    _save(cache, store)
    if earliest is None:
        return None
    return {"sm_first": f"{earliest[0]} Q{earliest[1]}", "sm_first_sort": list(earliest)}


# --------------------------------------------------------------------------- #
#  fundamental blurb (uses fields already grabbed in fetch_institutional)
# --------------------------------------------------------------------------- #
def _money(x) -> str:
    if x is None:
        return "?"
    for d, s in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(x) >= d:
            return f"${x / d:.1f}{s}"
    return f"${x:.0f}"


def fundamentals_blurb(rec: dict | None) -> str:
    """One-line factual fundamentals from the cached .info fields. No narrative invented."""
    if not rec:
        return ""
    bits = []
    if rec.get("marketCap"):
        bits.append(f"cap {_money(rec['marketCap'])}")
    pe = rec.get("forwardPE") or rec.get("trailingPE")
    if pe and pe > 0:
        bits.append(f"{'fwd' if rec.get('forwardPE') else 'ttm'}P/E {pe:.0f}")
    rg = rec.get("revenueGrowth")
    if rg is not None:
        bits.append(f"rev {rg * 100:+.0f}%")
    eg = rec.get("earningsGrowth")
    if eg is not None:
        bits.append(f"eps {eg * 100:+.0f}%")
    pm = rec.get("profitMargins")
    if pm is not None:
        bits.append(f"net mgn {pm * 100:.0f}%")
    de = rec.get("debtToEquity")
    if de is not None:
        bits.append(f"D/E {de / 100:.1f}" if de > 5 else f"D/E {de:.1f}")  # yf reports as %
    return " · ".join(bits)


# --------------------------------------------------------------------------- #
#  3. Congressional trades  (FMP, per ticker; OFF without a key)
# --------------------------------------------------------------------------- #
_FMP = "https://financialmodelingprep.com/api/v4"


def _fmp_key(cfg: dict) -> str | None:
    return (cfg.get("fmp_api_key") or "").strip() or os.environ.get("FMP_API_KEY")


def _fmp_trades(endpoint: str, sym: str, key: str) -> list[dict]:
    url = f"{_FMP}/{endpoint}?symbol={sym}&apikey={key}"
    try:
        r = requests.get(url, headers=_UA, timeout=25)
        if r.status_code != 200:
            return []
        j = r.json()
        return j if isinstance(j, list) else []
    except Exception:
        return []


def fetch_congress(symbols: list[str], cfg: dict, lookback_days: int = 180,
                   cache_hours: float = 24) -> dict[str, dict]:
    """Return {SYM: {gov_buys, gov_sells, gov_net, gov_last}} for the lookback window.

    Empty dict (every name contributes 0) when no FMP key is configured.
    Counts disclosures by transaction type; 'Purchase'/'buy' vs 'Sale'/'sell'.
    """
    key = _fmp_key(cfg)
    if not key:
        return {}
    cache = CACHE_DIR / "congress.json"
    store = _load(cache) or {}
    now = time.time()
    cutoff = dt.date.today() - dt.timedelta(days=lookback_days)
    out: dict[str, dict] = {}

    def _date(rec) -> dt.date | None:
        for k in ("transactionDate", "disclosureDate", "date"):
            v = rec.get(k)
            if v:
                try:
                    return dt.date.fromisoformat(str(v)[:10])
                except Exception:
                    pass
        return None

    for sym in symbols:
        rec = store.get(sym)
        if rec and (now - rec.get("_ts", 0)) < cache_hours * 3600:
            out[sym] = rec
            continue
        rows = _fmp_trades("senate-trading", sym, key) + \
               _fmp_trades("senate-disclosure", sym, key)   # senate + house feeds
        buys = sells = 0
        last: str | None = None
        for tr in rows:
            d = _date(tr)
            if d is None or d < cutoff:
                continue
            typ = str(tr.get("type") or tr.get("transactionType") or "").lower()
            if "purchase" in typ or "buy" in typ:
                buys += 1
            elif "sale" in typ or "sell" in typ:
                sells += 1
            ds = d.isoformat()
            if last is None or ds > last:
                last = ds
        rec = {"gov_buys": buys, "gov_sells": sells, "gov_net": buys - sells,
               "gov_last": last, "_ts": now}
        store[sym] = rec
        out[sym] = rec
        time.sleep(0.15)
    _save(cache, store)
    return out


# --------------------------------------------------------------------------- #
#  combine into one smart-money sub-score (0..100) + flags + note
# --------------------------------------------------------------------------- #
def smartmoney_score(inst: dict | None, supr: dict | None, gov: dict | None,
                     th: dict) -> tuple[float, list[str], str, dict]:
    """Blend the three sources. Each present source contributes; missing = 0.

    inst : {inst_pct}              supr : {dr_count}            gov : {gov_net,...}
    Returns (score0_100, flags, note, fields) where fields are merged back onto
    the row for the audit CSV.
    """
    flags: list[str] = []
    note_bits: list[str] = []
    fields: dict = {}
    parts: list[float] = []

    # institutional: 0..1 -> reward ownership above a floor, full marks near ~80%
    ip = (inst or {}).get("inst_pct")
    if ip is not None:
        fields["inst_pct"] = ip
        lo, hi = th["inst_pct_floor"], th["inst_pct_full"]
        s = max(0.0, min(100.0, (ip - lo) / max(1e-9, hi - lo) * 100))
        parts.append(s)
        # Yahoo reports institutional holdings as % of float, which can exceed
        # 100% when shares are lent/shorted -> clamp the display at 100%.
        note_bits.append(f"inst{min(ip, 1.0)*100:.0f}%")
        if ip >= th["inst_pct_high"]:
            flags.append("SM-INST")

    # superinvestors: count of tracked 13F holders; full marks at >= dr_full
    dc = (supr or {}).get("dr_count")
    if dc is not None:
        fields["dr_count"] = dc
        s = max(0.0, min(100.0, dc / max(1, th["dr_full"]) * 100))
        # fresh conviction: a brand-new superinvestor position this quarter nudges it up
        dn = (supr or {}).get("dr_new") or 0
        if dn:
            fields["dr_new"] = dn
            s = min(100.0, s + 15)
        parts.append(s)
        if dc > 0:
            note_bits.append(f"13F{dc}" + (f"(+{dn} new)" if dn else ""))
            flags.append("SM-13F")
        if dn:
            flags.append("SM-NEW")

    # congress: net recent buys; each net buy worth a chunk, capped
    if gov:
        net = gov.get("gov_net", 0)
        fields.update({k: gov.get(k) for k in ("gov_buys", "gov_sells", "gov_net", "gov_last")})
        if gov.get("gov_buys") or gov.get("gov_sells"):
            s = max(0.0, min(100.0, 50 + net * 25))   # net 0 -> 50, +2 -> 100
            parts.append(s)
            note_bits.append(f"gov{net:+d}")
            if net > 0:
                flags.append("SM-GOV")

    score = sum(parts) / len(parts) if parts else 0.0
    note = " ".join(note_bits)
    return round(score, 1), flags, note, fields
