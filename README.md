# AI-bottleneck screener

Thesis-driven, token-efficient stock screener. Ranks the AI build-out
"picks-beneath-the-picks" — power generation, grid/transformers, electrical gear,
thermal/liquid cooling, HBM memory, optical interconnect, and new nuclear — surfacing
the names that **smart money already owns** and that **haven't gone exponential this
year** (the asymmetric, not-yet-run setups).

This is the cloud-runnable copy of the `ai_screener.py` module from Ted's local
`screener/` project, curated so a scheduled remote agent can clone and run it against
public data with no local dependencies.

## Run

```bash
python -m venv .venv && . .venv/bin/activate     # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python ai_screener.py                            # full basket
python ai_screener.py --top 20                   # cap digest rows
python ai_screener.py --symbols CEG VRT ETN      # ad-hoc list
```

Writes `out/ai_<date>_full.csv` (audit) and `out/ai_latest_digest.md` (the small,
ranked read). Both are gitignored.

## Composite score (0–100, weights in `ai_config.json`)

| sub-score  | weight | captures |
|------------|--------|----------|
| smartmoney | 0.40 | institutional % held + Dataroma 13F superinvestor count + congressional buys |
| laggard    | 0.32 | "room to run": high when YTD is low, **0 when parabolic**, halved if 200d trend broken |
| momentum   | 0.28 | trend intact (50&200d) + relative strength vs SPY + not over-extended |

**Flags:** `LAGGARD` not-yet-run · `PARABOLIC` already ran (avoid) · `UPTRD` >50&200d ·
`RS+/-` vs SPY · `OB` overbought · `52H` near high · `SM-INST` high institutional ·
`SM-13F` superinvestors hold · `SM-GOV` congress buying.

## Data sources (free, no login)

- **Prices / technicals** — Yahoo (yfinance).
- **Institutional %** — Yahoo `heldPercentInstitutions` (% of float; clamped at 100% in display).
- **Superinvestors** — count of tracked 13F value-investors on dataroma.com (cached 7d).
- **Congress** — FMP per-symbol endpoint; **off** until a free key is set in
  `ai_config.json` (`fmp_api_key`) or env `FMP_API_KEY`. Scored on the other two until then.

Decision-support only — not a buy list, no order routing.
