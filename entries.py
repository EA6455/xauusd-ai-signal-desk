"""Tactical entry engine — multi-confluence pullback setups on 15m gold.

Philosophy: "high win rate" comes from CONFLUENCE + patience, not from more
indicators. A setup only triggers when most of these align:
  1. 1H trend aligned (EMA structure + price side)
  2. 15m trend aligned (EMA 9 vs 21)
  3. Price pulled back into the EMA21 zone (not chasing an overextended move)
  4. RSI in the trigger window 38–62 (room to run, not chasing)
  5. MACD histogram turning in the trade direction
  6. Momentum close beyond EMA 9
  7. Volatility at/above its recent median (avoids dead chop)
  8. Confirmation candle (closing in the favorable part of the bar)

Levels are ATR-based: SL = 1.8 x ATR, TP1 = 0.9 x ATR (win-rate tilted),
TP2 = 2.0 x ATR. The dashboard shows the measured historical hit rate —
no guarantees, markets change.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np

import ml

MIN_CHECKS_TRIGGER = 8          # A+ (8/8) triggers; 7/8 = "forming", no alert
SL_ATR = 1.5
TP1_ATR = 3.0
TP2_ATR = 4.5
MAX_WAIT_BARS = 20              # setup resolves within 20 x 15m = 5 hours

CHECKS = ["1H trend aligned", "15m trend (EMA 9/21)", "Pullback to EMA21 zone",
          "RSI in trigger window", "MACD hist turning", "Momentum close vs EMA9",
          "Volatility above median", "Confirmation candle"]

_stats_cache = {"key": None, "stats": None}

SESSION_LABEL = {"dead": "dead hours (21–01 UTC)", "asia": "Asian session",
                 "london": "London session", "ny-overlap": "London–NY overlap",
                 "ny-late": "late NY session"}


def session_of(ts):
    """Trading-session bucket for a UTC timestamp."""
    hr = datetime.fromtimestamp(ts, timezone.utc).hour
    if hr >= 21 or hr < 1:
        return "dead"
    if hr < 7:
        return "asia"
    if hr < 12:
        return "london"
    if hr < 16:
        return "ny-overlap"
    return "ny-late"


# ------------------------------------------------------------------ helpers
def htf_trend_from_15m(candles, i):
    """Trend of the LAST COMPLETED 1H bar before 15m index i (no look-ahead).

    Resamples the 15m closes to 1H and checks the EMA 9/21 structure + side.
    """
    t = candles[i]["t"]
    hour = (t // 3600) * 3600          # 1H bucket this bar belongs to
    buckets = {}
    for k in range(i + 1):             # only data up to i
        hb = (candles[k]["t"] // 3600) * 3600
        if hb < hour:                  # only COMPLETED 1H bars
            buckets[hb] = candles[k]["c"]      # last 15m close in the hour
    if len(buckets) < 25:
        return 0
    closes = np.array([v for _, v in sorted(buckets.items())], float)
    e9, e21 = ml.ema(closes, 9), ml.ema(closes, 21)
    if e9[-1] > e21[-1] and closes[-1] > e21[-1]:
        return 1
    if e9[-1] < e21[-1] and closes[-1] < e21[-1]:
        return -1
    return 0


def htf_trend_live(candles_1h):
    """1H trend for the live evaluation (uses the freshest 1H data)."""
    if not candles_1h or len(candles_1h) < 25:
        return 0
    c = np.array([k["c"] for k in candles_1h], float)
    e9, e21 = ml.ema(c, 9), ml.ema(c, 21)
    if e9[-1] > e21[-1] and c[-1] > e21[-1]:
        return 1
    if e9[-1] < e21[-1] and c[-1] < e21[-1]:
        return -1
    return 0


def checks_at(p, i, trend):
    """(long_checks[8], short_checks[8]) at bar i — strictly causal."""
    c, e9, e21 = p["c"], p["e9"], p["e21"]
    hist, rng = p["macd_hist"], p["h"][i] - p["l"][i]
    atr_i = p["atr"][i]
    pull = abs(c[i] - e21[i]) <= 1.3 * atr_i
    rsi_win = 38.0 <= p["rsi"][i] <= 62.0
    lo = max(0, i - 49)
    vol_ok = (atr_i / c[i]) >= float(np.median(p["atr"][lo:i + 1] / c[lo:i + 1]))
    conf_l = rng > 0 and (c[i] - p["l"][i]) / rng >= 0.40
    conf_s = rng > 0 and (p["h"][i] - c[i]) / rng >= 0.40
    longs = [trend == 1, e9[i] > e21[i], pull, rsi_win,
             hist[i] > hist[i - 1], c[i] > e9[i], vol_ok, conf_l]
    shorts = [trend == -1, e9[i] < e21[i], pull, rsi_win,
              hist[i] < hist[i - 1], c[i] < e9[i], vol_ok, conf_s]
    return longs, shorts


def _grade(n):
    if n >= 8:
        return "A+"
    if n == 7:
        return "A"          # forming — shown, but no alert / no trigger
    return None


# ------------------------------------------------------------------ live eval
def evaluate(candles15, candles_1h):
    """Current tactical setup on the latest 15m bar."""
    if not candles15 or len(candles15) < 60:
        return None
    p = ml.indicator_pack(candles15)
    i = len(candles15) - 1
    trend = htf_trend_live(candles_1h) if candles_1h else 0
    longs, shorts = checks_at(p, i, trend)
    nl, ns = sum(longs), sum(shorts)
    d = 1 if nl >= ns else -1
    n = max(nl, ns)
    checks = longs if d == 1 else shorts
    grade = _grade(n)
    atr_i = float(p["atr"][i])
    entry = float(p["c"][i])
    e21 = float(p["e21"][i])
    sl_d = SL_ATR * atr_i
    tp_d1, tp_d2 = TP1_ATR * atr_i, TP2_ATR * atr_i
    sess = session_of(int(candles15[i]["t"]))
    return dict(
        direction="LONG" if d == 1 else "SHORT",
        passed=int(n), grade=grade or "—",
        active=(grade == "A+" and sess != "dead"),
        session=sess, sessionLabel=SESSION_LABEL[sess],
        entry=round(entry, 2),
        entryZone=[round(min(entry, e21) - 0.15 * atr_i, 2),
                   round(max(entry, e21) + 0.15 * atr_i, 2)],
        sl=round(entry - d * sl_d, 2),
        tp1=round(entry + d * tp_d1, 2),
        tp2=round(entry + d * tp_d2, 2),
        rr1=round(tp_d1 / sl_d, 2), rr2=round(tp_d2 / sl_d, 2),
        atr=round(atr_i, 2),
        barTime=int(candles15[i]["t"]),
        checks=[dict(label=CHECKS[k], ok=bool(checks[k])) for k in range(8)],
        trend1h=["BEARISH", "NEUTRAL", "BULLISH"][trend + 1],
    )


# ------------------------------------------------------------------ backtest
def _resolve(c, p, i, d):
    """Outcome of a setup triggered at bar i, direction d. Conservative:
    if SL and TP are touched in the same bar, it counts as a LOSS."""
    entry = c[i]
    sl = entry - d * SL_ATR * p["atr"][i]
    tp = entry + d * TP1_ATR * p["atr"][i]
    end = min(len(c), i + 1 + MAX_WAIT_BARS)
    for j in range(i + 1, end):
        if d == 1:
            if p["l"][j] <= sl:
                return "loss", -1.0
            if p["h"][j] >= tp:
                return "win", TP1_ATR / SL_ATR
        else:
            if p["h"][j] >= sl:
                return "loss", -1.0
            if p["l"][j] <= tp:
                return "win", TP1_ATR / SL_ATR
    if end < len(c):                     # timed out — mark-to-market
        r = d * (c[end - 1] - entry) / (SL_ATR * p["atr"][i])
        return ("win" if r > 0 else "loss"), float(r)
    return None, None                    # ran out of data


def backtest_stats(candles15, min_grade="A"):
    """Historical performance of the tactical setups on the loaded 15m data."""
    if not candles15 or len(candles15) < 200:
        return None
    key = f"{len(candles15)}:{candles15[-1]['t']}:{min_grade}"
    if _stats_cache["key"] == key:
        return _stats_cache["stats"]
    need = {"A": 7, "A+": 8}[min_grade]
    p = ml.indicator_pack(candles15)
    c = p["c"]
    n = len(c)
    wins = losses = 0
    r_sum = 0.0
    last_dir, cooldown = 0, 0
    for i in range(60, n - 1):
        if cooldown > 0:
            cooldown -= 1
            continue
        trend = htf_trend_from_15m(candles15, i)
        longs, shorts = checks_at(p, i, trend)
        nl, ns = sum(longs), sum(shorts)
        d, best = (1, nl) if nl >= ns else (-1, ns)
        if best < need:
            continue
        res, r = _resolve(c, p, i, d)
        if res is None:
            continue
        if res == "win":
            wins += 1
        else:
            losses += 1
        r_sum += r
        last_dir, cooldown = d, 4        # don't re-trigger every bar
    total = wins + losses
    stats = dict(setups=total, wins=wins, losses=losses,
                 winRate=round(wins / total, 3) if total else None,
                 avgR=round(r_sum / total, 3) if total else None,
                 window="last 60 days of 15m bars",
                 tpAtr=TP1_ATR, slAtr=SL_ATR, minChecks=need)
    _stats_cache.update(key=key, stats=stats)
    return stats


def recent_setups(candles15, lookback=380):
    """Chronological list of full-confluence (8/8) setups with resolved outcomes.
    Used for chart markers and the live tracker's session statistics."""
    out = []
    if not candles15 or len(candles15) < 200:
        return out
    p = ml.indicator_pack(candles15)
    c = p["c"]
    n = len(c)
    cooldown = 0
    for i in range(max(60, n - lookback), n - 1):
        if cooldown > 0:
            cooldown -= 1
            continue
        trend = htf_trend_from_15m(candles15, i)
        longs, shorts = checks_at(p, i, trend)
        nl, ns = sum(longs), sum(shorts)
        d, best = (1, nl) if nl >= ns else (-1, ns)
        if best < 8:
            continue
        res, r = _resolve(c, p, i, d)
        out.append(dict(
            t=int(candles15[i]["t"]), i=i, dir="LONG" if d == 1 else "SHORT",
            entry=round(float(c[i]), 1),
            sl=round(float(c[i] - d * SL_ATR * p["atr"][i]), 1),
            tp1=round(float(c[i] + d * TP1_ATR * p["atr"][i]), 1),
            outcome=res, r=(round(r, 2) if r is not None else None),
            session=session_of(int(candles15[i]["t"]))))
        cooldown = 4
    return out


def session_stats(setups):
    """Win rate / avg R per session from a recent_setups() list."""
    buckets = {}
    for s in setups:
        if s["outcome"] is None:
            continue
        b = buckets.setdefault(s["session"], [0, 0, 0.0])
        b[0] += 1
        b[1] += 1 if s["outcome"] == "win" else 0
        b[2] += s["r"] or 0.0
    out = {}
    for k, (tot, wins, rsum) in buckets.items():
        out[k] = dict(setups=tot, winRate=round(wins / tot, 3), avgR=round(rsum / tot, 2))
    return out
