"""
Signal computation + composite scoring.

Four sub-scores (each 0-100), combined with per-row weight normalisation so a
missing data source (e.g. IBKR borrow file unreachable) does NOT deflate the
total — only the available components count.

  borrow     : how expensive / constrained the borrow is        (IBKR file)
  squeeze    : crowded short + building pressure + an up-spark   (Yahoo SI + px)
  options    : premium richness & positioning extremes          (Yahoo options)
  technical  : momentum / relative strength / actionable extremes (px)
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
#  technical indicators
# --------------------------------------------------------------------------- #
def _rsi(close: pd.Series, n: int = 14) -> float:
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    ru = up.ewm(alpha=1 / n, adjust=False).mean()
    rd = dn.ewm(alpha=1 / n, adjust=False).mean()
    rs = ru / rd.replace(0, np.nan)
    val = 100 - 100 / (1 + rs)
    return float(val.iloc[-1]) if len(val) and pd.notna(val.iloc[-1]) else float("nan")


def _hv(close: pd.Series, n: int = 20) -> float:
    lr = np.log(close / close.shift(1)).dropna()
    if len(lr) < n:
        return float("nan")
    return float(lr.tail(n).std() * math.sqrt(252) * 100)  # annualised %


def technicals(df: pd.DataFrame, bench: pd.DataFrame | None) -> dict:
    close = df["Close"].dropna()
    out: dict = {"last": float(close.iloc[-1]) if len(close) else float("nan")}
    if len(close) < 20:
        out.update(hv20=float("nan"))
        return out

    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else float("nan")
    sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else float("nan")
    hi = close.tail(252).max()
    lo = close.tail(252).min()

    out["rsi14"] = _rsi(close)
    out["hv20"] = _hv(close)
    out["above_sma20"] = bool(close.iloc[-1] > sma20) if pd.notna(sma20) else None
    out["above_sma50"] = bool(close.iloc[-1] > sma50) if pd.notna(sma50) else None
    out["above_sma200"] = bool(close.iloc[-1] > sma200) if pd.notna(sma200) else None
    out["pct_52w_high"] = float(close.iloc[-1] / hi - 1) if hi else float("nan")  # <=0
    out["pct_52w_low"] = float(close.iloc[-1] / lo - 1) if lo else float("nan")   # >=0

    # 63-day (~3mo) return and relative strength vs benchmark
    if len(close) >= 64:
        out["ret_63"] = float(close.iloc[-1] / close.iloc[-64] - 1)
    else:
        out["ret_63"] = float("nan")
    if bench is not None and len(bench) >= 64 and pd.notna(out["ret_63"]):
        bc = bench["Close"].dropna()
        bret = float(bc.iloc[-1] / bc.iloc[-64] - 1)
        out["rs_63"] = out["ret_63"] - bret
    else:
        out["rs_63"] = float("nan")

    # volume z-score (today vs 20d)
    vol = df["Volume"].dropna()
    if len(vol) >= 20:
        m, s = vol.tail(20).mean(), vol.tail(20).std()
        out["vol_z"] = float((vol.iloc[-1] - m) / s) if s else float("nan")
    else:
        out["vol_z"] = float("nan")

    # ATR% (14)
    if len(df) >= 15:
        hl = df["High"] - df["Low"]
        hc = (df["High"] - df["Close"].shift()).abs()
        lc = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
        out["atr_pct"] = float(atr / close.iloc[-1]) if close.iloc[-1] else float("nan")
    else:
        out["atr_pct"] = float("nan")
    return out


# --------------------------------------------------------------------------- #
#  borrow + short interest
# --------------------------------------------------------------------------- #
def borrow_signals(borrow: dict | None, si: dict | None) -> dict:
    out = {"borrow_fee": None, "avail": None, "short_pct_float": None,
           "dtc": None, "si_change": None}
    if borrow:
        out["borrow_fee"] = borrow.get("borrow_fee")
        out["avail"] = borrow.get("avail")
    if si:
        out["short_pct_float"] = si.get("shortPercentOfFloat")
        out["dtc"] = si.get("shortRatio")
        cur, prior = si.get("sharesShort"), si.get("sharesShortPriorMonth")
        if cur and prior:
            out["si_change"] = (cur - prior) / prior
    return out


# --------------------------------------------------------------------------- #
#  options
# --------------------------------------------------------------------------- #
def options_signals(opt: dict | None, hv20: float) -> dict:
    out = {"atm_iv": None, "iv_hv": None, "put_call_oi": None, "skew": None}
    if not opt or opt.get("spot") is None:
        return out
    spot = opt["spot"]
    if spot < 2.0:  # sub-$2: yfinance option IV/OI is unreliable -> skip options
        return out
    calls, puts = opt["calls"], opt["puts"]

    def _atm_iv(df):
        if df.empty or "impliedVolatility" not in df:
            return float("nan")
        i = (df["strike"] - spot).abs().idxmin()
        return float(df.loc[i, "impliedVolatility"])

    # sanitise: drop implausible yfinance IV quotes (<2% or >500% are bad data)
    def _ok(v):
        return pd.notna(v) and 0.02 < v < 5.0

    civ, piv = _atm_iv(calls), _atm_iv(puts)
    ivs = [v for v in (civ, piv) if _ok(v)]
    if ivs:
        atm_iv = float(np.mean(ivs)) * 100  # %
        out["atm_iv"] = atm_iv
        if pd.notna(hv20) and hv20 > 0:
            out["iv_hv"] = atm_iv / hv20

    # put/call open interest (need real OI on both sides to be meaningful)
    coi = float(calls["openInterest"].fillna(0).sum()) if "openInterest" in calls else 0.0
    poi = float(puts["openInterest"].fillna(0).sum()) if "openInterest" in puts else 0.0
    if coi > 0 and poi > 0:
        out["put_call_oi"] = poi / coi

    # skew: ~10% OTM put IV minus ~10% OTM call IV (vol points)
    def _iv_near(df, target):
        if df.empty or "impliedVolatility" not in df:
            return float("nan")
        i = (df["strike"] - target).abs().idxmin()
        return float(df.loc[i, "impliedVolatility"])

    p_otm = _iv_near(puts, spot * 0.90)
    c_otm = _iv_near(calls, spot * 1.10)
    if _ok(p_otm) and _ok(c_otm):
        out["skew"] = (p_otm - c_otm) * 100
    return out


# --------------------------------------------------------------------------- #
#  scoring
# --------------------------------------------------------------------------- #
def _clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


def score(row: dict, th: dict, weights: dict) -> tuple[float, list[str], str]:
    flags: list[str] = []
    subs: dict[str, float] = {}

    # ---- borrow sub-score (IBKR) ----
    fee, avail = row.get("borrow_fee"), row.get("avail")
    if fee is not None or avail is not None:
        s = 0.0
        if fee is not None:
            # piecewise: 0..htb -> 0..40, htb..extreme -> 40..100
            htb, ext = th["borrow_fee_htb"], th["borrow_fee_extreme"]
            if fee <= htb:
                s += _clamp(fee / htb * 40, 0, 40)
            else:
                s += _clamp(40 + (fee - htb) / (ext - htb) * 60, 40, 100)
            if fee >= th["borrow_fee_htb"]:
                flags.append("HTB")
            if fee >= th["borrow_fee_extreme"]:
                flags.append("FEE!")
        if avail is not None:
            if avail <= 0:
                s += 30
            elif avail < th["avail_low"]:
                s += 30 * (1 - avail / th["avail_low"])
            if avail < th["avail_low"]:
                flags.append("THIN")
        subs["borrow"] = _clamp(s)

    # ---- squeeze sub-score (crowding + spark) ----
    spf, dtc, sic = row.get("short_pct_float"), row.get("dtc"), row.get("si_change")
    if spf is not None or dtc is not None:
        s = 0.0
        if spf is not None:
            s += _clamp(spf / (2 * th["short_pct_float_high"]) * 40, 0, 40)
            if spf >= th["short_pct_float_high"]:
                flags.append("CROWD")
        if dtc is not None:
            s += _clamp(dtc / (2 * th["dtc_high"]) * 25, 0, 25)
        if sic is not None and sic > 0:
            s += _clamp(sic / 0.5 * 20, 0, 20)
            if sic >= th["si_rise_pct"]:
                flags.append("UP-SI")
        # the spark: crowded short starting to lift
        if row.get("above_sma20") and 50 <= (row.get("rsi14") or 0) <= 72:
            s += 15
        subs["squeeze"] = _clamp(s)

    # ---- options sub-score ----
    iv_hv, pc, skew = row.get("iv_hv"), row.get("put_call_oi"), row.get("skew")
    if iv_hv is not None or pc is not None:
        s = 0.0
        if iv_hv is not None:
            s += _clamp(abs(iv_hv - 1.0) * 60, 0, 50)
            if iv_hv >= th["ivhv_rich"]:
                flags.append("IV-RICH")
            elif iv_hv <= th["ivhv_cheap"]:
                flags.append("IV-CHEAP")
        if pc is not None:
            s += _clamp(abs(pc - 1.0) * 20, 0, 25)
            if pc >= 1.5:
                flags.append("PUTS")
        if skew is not None and skew > 0:
            s += _clamp(skew / 20 * 25, 0, 25)
        subs["options"] = _clamp(s)

    # ---- technical sub-score ----
    s = 0.0
    rs, rsi = row.get("rs_63"), row.get("rsi14")
    if rs is not None and pd.notna(rs):
        s += _clamp((rs + 0.10) / 0.30 * 40, 0, 40)  # +20% RS ~ full marks
        flags.append("RS+" if rs > 0 else "RS-")
    if row.get("above_sma50") and row.get("above_sma200"):
        s += 30
        flags.append("UPTRD")
    if rsi is not None and pd.notna(rsi):
        if rsi <= th["rsi_low"]:
            s += 15
            flags.append("OS")
        elif rsi >= th["rsi_high"]:
            s += 15
            flags.append("OB")
    vz = row.get("vol_z")
    if vz is not None and pd.notna(vz) and vz >= th["vol_z_spike"]:
        s += 15
        flags.append("VOL")
    p52h = row.get("pct_52w_high")
    if p52h is not None and pd.notna(p52h) and p52h >= -0.03:
        flags.append("52H")
    subs["technical"] = _clamp(s)

    # ---- cross-asset context (informational flags, no score weight) ----
    cp = row.get("cot_pctile")
    if cp is not None and pd.notna(cp):
        if cp >= 85:
            flags.append("COT-LONG!")     # specs crowded long -> contrarian caution
        elif cp <= 15:
            flags.append("COT-SHORT!")    # specs washed out -> contrarian bullish
    if row.get("news_halt"):
        flags.append("HALT")
    if row.get("news_raise"):
        flags.append("RAISE")             # capital raise -> dilution risk
    elif row.get("news_recent_ps"):
        flags.append("NEWS")
    rp = row.get("rs_peer")
    if rp is not None and pd.notna(rp):
        flags.append("RP+" if rp > 0 else "RP-")   # relative strength vs peer/metal

    # ---- additive weighted composite ----
    # Missing data contributes 0 (NOT normalised away) so a name with only one
    # weak signal can't outrank a genuine borrow/squeeze setup.
    total = sum(weights[k] * v for k, v in subs.items())

    return round(total, 1), flags, _note(row)


def _note(row: dict) -> str:
    bits = []
    fee, avail = row.get("borrow_fee"), row.get("avail")
    if fee is not None:
        a = "" if avail is None else f" avail{_h(avail)}"
        bits.append(f"fee{fee:.1f}%{a}")
    spf, dtc, sic = row.get("short_pct_float"), row.get("dtc"), row.get("si_change")
    si_bits = []
    if spf is not None:
        si_bits.append(f"{spf*100:.0f}%fl")
    if dtc is not None:
        si_bits.append(f"DTC{dtc:.1f}")
    if sic is not None:
        si_bits.append(f"SI{sic*100:+.0f}%")
    if si_bits:
        bits.append(" ".join(si_bits))
    iv_hv, pc = row.get("iv_hv"), row.get("put_call_oi")
    opt_bits = []
    if iv_hv is not None:
        opt_bits.append(f"IV/HV{iv_hv:.2f}")
    if pc is not None:
        opt_bits.append(f"PC{pc:.2f}")
    if opt_bits:
        bits.append(" ".join(opt_bits))
    rs = row.get("rs_63")
    if rs is not None and pd.notna(rs):
        bits.append(f"RS{rs*100:+.0f}%")
    rp = row.get("rs_peer")
    if rp is not None and pd.notna(rp):
        bits.append(f"v{row.get('peer_label','peer')}{rp*100:+.0f}%")
    cp = row.get("cot_pctile")
    if cp is not None and pd.notna(cp):
        bits.append(f"COT{cp:.0f}pct")
    if row.get("news_last_head"):
        age = row.get("news_last_age")
        bits.append(f"news{'' if age is None else f' {age}d'}: {row['news_last_head']}")
    return " | ".join(bits)


def _h(n: float) -> str:
    if n is None:
        return "?"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.0f}k"
    return f"{n:.0f}"
