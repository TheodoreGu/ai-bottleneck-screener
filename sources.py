"""
Data sources for the screener. Everything here is FREE / no-login:

  * IBKR public borrow file  -> real borrow FEE RATE + shares AVAILABLE
  * yfinance short interest  -> short%float, days-to-cover, MoM SI change
  * yfinance price history   -> batch daily bars for technicals
  * yfinance options         -> IV / open interest (fetched only for a shortlist)

All fetches are cached to ./cache for `cache_hours` so repeated runs in the
same session are instant and gentle on the providers.
"""
from __future__ import annotations

import io
import json
import time
import ftplib
import datetime as dt
from pathlib import Path
from urllib.parse import urlparse

import requests
import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
#  small cache helper
# --------------------------------------------------------------------------- #
def _cache_path(name: str) -> Path:
    return CACHE_DIR / name


def _fresh(path: Path, hours: float) -> bool:
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < hours * 3600


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj), encoding="utf-8")


# --------------------------------------------------------------------------- #
#  IBKR public borrow / short-availability file
# --------------------------------------------------------------------------- #
def _http_get(url: str, timeout: int = 30) -> str:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def _ftp_get(url: str, timeout: int = 30) -> str:
    p = urlparse(url)
    ftp = ftplib.FTP(timeout=timeout)
    ftp.connect(p.hostname, p.port or 21)
    ftp.login(p.username or "anonymous", p.password or "")
    buf = io.BytesIO()
    ftp.retrbinary(f"RETR {p.path.lstrip('/')}", buf.write)
    ftp.quit()
    return buf.getvalue().decode("latin-1")


def parse_borrow_text(text: str) -> dict[str, dict]:
    """
    IBKR file is pipe-delimited:
        #BOF|...timestamp...
        SYM|CUR|NAME|CON|ISIN|REBATERATE|FEERATE|AVAILABLE
        AAPL|USD|APPLE INC|265598|US0378331005|-0.25|0.25|>10000000
        #EOF
    We map columns by the header row when present, else fall back to fixed
    positions, and we coerce AVAILABLE values like ">10000000" / "1,234,500".
    """
    out: dict[str, dict] = {}
    col = {"sym": 0, "fee": 6, "avail": 7, "rebate": 5, "name": 2}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        # header row -> remap columns by name
        upper = [c.strip().upper() for c in parts]
        if "FEERATE" in upper and "AVAILABLE" in upper:
            col = {
                "sym": 0,
                "fee": upper.index("FEERATE"),
                "avail": upper.index("AVAILABLE"),
                "rebate": upper.index("REBATERATE") if "REBATERATE" in upper else 5,
                "name": upper.index("NAME") if "NAME" in upper else 2,
            }
            continue
        if len(parts) <= col["avail"]:
            continue
        sym = parts[col["sym"]].strip().upper()
        if not sym:
            continue

        def _num(x):
            x = x.strip().replace(",", "").lstrip(">").lstrip("<")
            try:
                return float(x)
            except ValueError:
                return None

        out[sym] = {
            "borrow_fee": _num(parts[col["fee"]]),
            "avail": _num(parts[col["avail"]]),
            "rebate": _num(parts[col["rebate"]]) if col["rebate"] < len(parts) else None,
        }
    return out


def fetch_borrow_file(urls: list[str], cache_hours: float, debug: bool = False) -> dict[str, dict]:
    """Return {SYM: {borrow_fee, avail, rebate}}.  Empty dict if all sources fail."""
    cache = _cache_path("borrow.json")
    if _fresh(cache, cache_hours):
        cached = _load_json(cache)
        if cached:
            return cached

    text = None
    for url in urls:
        try:
            text = _ftp_get(url) if url.lower().startswith("ftp") else _http_get(url)
            if text and len(text) > 100:
                if debug:
                    print(f"[borrow] fetched {len(text)} bytes from {url}")
                    print("\n".join(text.splitlines()[:4]))
                break
        except Exception as e:  # noqa: BLE001
            if debug:
                print(f"[borrow] {url} failed: {e}")
            text = None

    if not text:
        print("[borrow] WARNING: IBKR borrow file unreachable -> "
              "using Yahoo short-interest only (no live fee/availability).")
        return {}

    parsed = parse_borrow_text(text)
    _save_json(cache, parsed)
    return parsed


# --------------------------------------------------------------------------- #
#  yfinance short interest (per ticker, cached)
# --------------------------------------------------------------------------- #
_SI_KEYS = [
    "sharesShort", "shortRatio", "shortPercentOfFloat",
    "sharesShortPriorMonth", "floatShares", "sharesOutstanding", "averageVolume",
]


def fetch_short_interest(symbols: list[str], cache_hours: float) -> dict[str, dict]:
    cache = _cache_path("short_interest.json")
    store = _load_json(cache) or {}
    now = time.time()
    out: dict[str, dict] = {}

    for sym in symbols:
        rec = store.get(sym)
        if rec and (now - rec.get("_ts", 0)) < cache_hours * 3600:
            out[sym] = rec
            continue
        try:
            info = yf.Ticker(sym).info or {}
            rec = {k: info.get(k) for k in _SI_KEYS}
        except Exception:
            rec = {k: None for k in _SI_KEYS}
        rec["_ts"] = now
        store[sym] = rec
        out[sym] = rec
        time.sleep(0.15)  # be polite to Yahoo

    _save_json(cache, store)
    return out


# --------------------------------------------------------------------------- #
#  price history (batch) + options (targeted)
# --------------------------------------------------------------------------- #
def fetch_prices(symbols: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
    """One batched download for the whole universe. Returns {sym: OHLCV frame}."""
    if not symbols:
        return {}
    data = yf.download(
        tickers=" ".join(symbols),
        period=period,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    out: dict[str, pd.DataFrame] = {}
    if isinstance(data.columns, pd.MultiIndex):
        for sym in symbols:
            if sym in data.columns.get_level_values(0):
                df = data[sym].dropna(how="all")
                if len(df):
                    out[sym] = df
    else:  # single symbol -> flat columns
        df = data.dropna(how="all")
        if len(df):
            out[symbols[0]] = df
    return out


def fetch_options(symbol: str, expiries: int = 2) -> dict | None:
    """Pull nearest `expiries` chains. Returns aggregated calls/puts + spot."""
    try:
        tk = yf.Ticker(symbol)
        all_exps = tk.options
        if not all_exps:
            return None
        # skip stale front-week quotes: prefer expirations >= 21 days out
        today = dt.date.today()
        dated = [(e, (dt.date.fromisoformat(e) - today).days) for e in all_exps]
        mid = [e for e, d in dated if d >= 21]
        exps = (mid or list(all_exps))[:expiries]
        calls, puts = [], []
        for e in exps:
            ch = tk.option_chain(e)
            calls.append(ch.calls)
            puts.append(ch.puts)
        calls = pd.concat(calls, ignore_index=True)
        puts = pd.concat(puts, ignore_index=True)
        hist = tk.history(period="5d")
        spot = float(hist["Close"].iloc[-1]) if len(hist) else None
        return {"calls": calls, "puts": puts, "spot": spot, "expiries": list(exps)}
    except Exception:
        return None
