"""AI Trader Note — turns the raw indicator state into a human-sounding
trader's desk commentary: market recap, structure read, key levels, and a
trade plan, written the way a desk trader would say it.

The wording varies (deterministically, per 15-minute bar) so it doesn't read
like a template, but the numbers are always the live computed ones.
"""
from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone

import ml


# ------------------------------------------------------------------ helpers
def _pick(bucket, options):
    h = int(hashlib.md5(str(bucket).encode()).hexdigest(), 16)
    return options[h % len(options)]


def _swings(candles, k=3, lookback=220):
    """Swing highs/lows from the most recent `lookback` candles."""
    cs = candles[-lookback:]
    highs, lows = [], []
    for i in range(k, len(cs) - k):
        win = cs[i - k:i + k + 1]
        if cs[i]["h"] == max(x["h"] for x in win):
            highs.append(cs[i]["h"])
        if cs[i]["l"] == min(x["l"] for x in win):
            lows.append(cs[i]["l"])
    return highs, lows


def _levels(candles, price, k=3, candles_htf=None, day_hi=None, day_lo=None):
    """Nearest resistance levels above and support levels below `price`.

    Blends 15m swings (day-trade scale), 1H swings (bigger picture) and the
    day's own high/low — then keeps the two nearest on each side, which is
    what an intraday trader actually watches.
    """
    try:
        highs, lows = _swings(candles, k=k)
        if candles_htf:
            h2, l2 = _swings(candles_htf, k=2, lookback=240)
            highs, lows = highs + h2, lows + l2
        if day_hi:
            highs.append(day_hi)
        if day_lo:
            lows.append(day_lo)
        # keep the two NEAREST levels on each side, at least $2 apart
        def nearest_sorted(vals, above):
            if above:
                cands = sorted(set(round(v, 1) for v in vals if v > price + 1.5))
            else:
                cands = sorted(set(round(v, 1) for v in vals if v < price - 1.5),
                               reverse=True)
            out = []
            for v in cands:
                if not out or abs(out[-1] - v) >= 2.0:
                    out.append(v)
                if len(out) == 2:
                    break
            return out
        resist = nearest_sorted(highs, True)
        support = nearest_sorted(lows, False)
        return ([dict(price=v, kind="R") for v in resist]
                + [dict(price=v, kind="S") for v in support])
    except Exception:  # noqa: BLE001
        return []


def _structure(candles):
    """HH/HL vs LH/LL read on the last few swings."""
    try:
        cs = candles[-160:]
        k = 3
        sh, sl = [], []
        for i in range(k, len(cs) - k):
            win = cs[i - k:i + k + 1]
            if cs[i]["h"] == max(x["h"] for x in win):
                sh.append(cs[i]["h"])
            if cs[i]["l"] == min(x["l"] for x in win):
                sl.append(cs[i]["l"])
        if len(sh) < 2 or len(sl) < 2:
            return "mixed", "range-bound price action"
        hh = sh[-1] > sh[-2]
        hl = sl[-1] > sl[-2]
        if hh and hl:
            return "up", "a clean sequence of higher highs and higher lows"
        if not hh and not hl:
            return "down", "lower highs and lower lows — sellers in control"
        return "mixed", "a choppy mix of swings — no clean structure"
    except Exception:  # noqa: BLE001
        return "mixed", "range-bound price action"


def _session():
    h = datetime.now(timezone.utc).hour
    if 0 <= h < 6:
        return "Asian hours", "thin liquidity"
    if 6 <= h < 12:
        return "the London session", "European flows"
    if 12 <= h < 16:
        return "the London–New York overlap", "the busiest tape of the day"
    if 16 <= h < 21:
        return "the New York session", "US flows"
    return "late New York hours", "fading liquidity"


def _fmt(p):
    return f"{p:,.1f}"


# ------------------------------------------------------------------ the note
def build_note(c15, c60, c1d, entry, signal, price):
    """Returns dict(paras=[..], levels=[..], updatedAt, session, bias)."""
    bucket = int(c15[-1]["t"] // 900 if c15 else time.time() // 900)
    t = _pick(bucket, ["Right now,", "At the moment,", "Current tape —",
                       "Desk view:"])
    sess, flow = _session()
    paras, lines = [], []

    # ---------- paragraph 1: the tape ----------
    day_o = day_h = day_l = prev_c = None
    if c1d and len(c1d) >= 2:
        day_o, day_h, day_l = c1d[-1]["o"], c1d[-1]["h"], c1d[-1]["l"]
        prev_c = c1d[-2]["c"]
    if prev_c:
        chg = (price - prev_c) / prev_c * 100
        rng = (day_h - day_l) or 1
        pos = (price - day_l) / rng
        if pos > 0.75:
            pos_txt = _pick(bucket, ["pressing the highs", "trading near the top of the range",
                                     "grinding into the high of the day"])
        elif pos < 0.25:
            pos_txt = _pick(bucket, ["sitting near the lows", "bouncing off session lows",
                                     "holding the bottom of the range"])
        else:
            pos_txt = _pick(bucket, ["mid-range", "chopping through the middle of the day's range",
                                     "consolidating in the middle of the range"])
        direction = "up" if chg >= 0 else "down"
        paras.append(
            f"{t} gold is {direction} {abs(chg):.2f}% on the day at {_fmt(price)}, "
            f"{pos_txt} ({_fmt(day_l)} – {_fmt(day_h)}). We're in {sess} with {flow}.")
    else:
        paras.append(f"{t} gold is trading at {_fmt(price)} in {sess} with {flow}.")

    # ---------- paragraph 2: trend & momentum ----------
    trend = entry.get("trend1h", "NEUTRAL") if entry else "NEUTRAL"
    p = ml.indicator_pack(c15)
    i = len(c15) - 1
    rsi = float(p["rsi"][i])
    hist = float(p["macd_hist"][i])
    e9, e21 = float(p["e9"][i]), float(p["e21"][i])
    atr = float(p["atr"][i])
    st, st_txt = _structure(c15)

    if trend == "BULLISH":
        tr_txt = _pick(bucket, ["the 1H trend is up", "1H structure stays bullish",
                                "the higher-timeframe trend favors buyers"])
        tr_dir = "up"
    elif trend == "BEARISH":
        tr_txt = _pick(bucket, ["the 1H trend is down", "1H structure stays bearish",
                                "the higher-timeframe trend favors sellers"])
        tr_dir = "down"
    else:
        tr_txt = "the 1H picture is undecided"
        tr_dir = "flat"

    if e9 > e21:
        ema_txt = _pick(bucket, ["15m EMAs are bullish (9 over 21)",
                                 "the 15m is in a bullish EMA regime"])
    else:
        ema_txt = _pick(bucket, ["15m EMAs are bearish (9 under 21)",
                                 "the 15m is in a bearish EMA regime"])
    if rsi > 70:
        rsi_txt = "RSI is stretched overbought"
    elif rsi > 55:
        rsi_txt = "momentum is with the buyers"
    elif rsi >= 45:
        rsi_txt = "momentum is neutral"
    elif rsi > 30:
        rsi_txt = "momentum is with the sellers" if rsi < 40 else "momentum leans bearish"
    else:
        rsi_txt = "RSI is washed-out oversold"
    macd_txt = ("MACD histogram is turning up" if hist > 0
                else "MACD histogram is rolling over")
    if tr_dir == st:
        struct_join = _pick(bucket, ["and 15m confirms it with",
                                     "and the 15m is playing along with"])
    elif tr_dir != "flat" and st != "mixed":
        struct_join = "while the 15m flips the script with"
    else:
        struct_join = "while the 15m shows"
    ema_sent = ema_txt[0].upper() + ema_txt[1:]
    paras.append(f"On the chart, {tr_txt} {struct_join} {st_txt}. {ema_sent}, "
                 f"{rsi_txt} at {rsi:.0f}, {macd_txt}. Volatility (ATR) is {_fmt(atr)} "
                 f"per 15m bar — {'plenty of movement to work with' if atr > 4 else 'a relatively calm tape'}.")

    # ---------- paragraph 3: levels ----------
    lvls = _levels(c15, price, k=3, candles_htf=c60, day_hi=day_h, day_lo=day_l)
    above = sorted([x["price"] for x in lvls if x["kind"] == "R"])
    below = sorted([x["price"] for x in lvls if x["kind"] == "S"], reverse=True)
    lvl_parts = []
    if above:
        lvl_parts.append("resistance at " + " then ".join(_fmt(v) for v in above))
    if below:
        lvl_parts.append("support at " + " then ".join(_fmt(v) for v in below))
    if lvl_parts:
        nearest = above[0] if above else below[0]
        dist = abs(nearest - price)
        if dist <= 12 * atr:
            lvl_src = ("the day's high/low are doing the work"
                       if (day_h and abs(nearest - day_h) < 0.5)
                       or (day_l and abs(nearest - day_l) < 0.5)
                       else "old swing levels above/below")
            paras.append(f"Levels I'm watching: {' and '.join(lvl_parts)} — {lvl_src}. "
                         f"First test is {_fmt(nearest)}, ${dist:.1f} away "
                         f"(~{dist / atr:.1f} ATR).")
        else:
            paras.append(f"Levels I'm watching: {' and '.join(lvl_parts)} — but they're all "
                         f"a stretch from here; the day's own high/low "
                         f"({_fmt(day_h)} / {_fmt(day_l)}) is the first real test.")
    else:
        paras.append("No clean swing levels nearby — price is in discovery mode; "
                     "round numbers do the work.")

    # ---------- paragraph 4: the plan ----------
    if entry and entry.get("active"):
        d = "long" if entry["direction"] == "LONG" else "short"
        paras.append(f"Setup: an A+ {d} setup is live from the "
                     f"{_fmt(entry['entryZone'][0])}–{_fmt(entry['entryZone'][1])} zone. "
                     f"Targets {_fmt(entry['tp1'])} (1:{entry['rr1']:.0f}) and {_fmt(entry['tp2'])} "
                     f"for the runner; the thesis is dead below {_fmt(entry['sl'])} for longs"
                     f"{'.' if d == 'long' else ' (mirror the logic for shorts).'}")
    elif entry and entry.get("passed") == 7:
        missing = [c["label"] for c in entry.get("checks", []) if not c["ok"]]
        mtxt = missing[0].lower() if missing else "the last confirmation"
        paras.append(f"Plan: we're one check away from an A+ setup ({entry['passed']}/8) — "
                     f"still waiting on {mtxt}. Let price come to the zone; the moment the "
                     f"checklist fills, the desk note will flag it. No checklist, no trade.")
    else:
        n = entry.get("passed", "?") if entry else "?"
        paras.append(f"Plan: no clean SNR retest on the tape right now ({n}/8 checks). "
                     f"That's fine — { _pick(bucket, ['patience is the position', 'flat is a position', 'the best trades are the ones you wait for'])}. "
                     f"When structure, momentum and location line up 8/8, the alert fires.")

    bias = signal.get("type", "HOLD") if signal else "HOLD"
    paras.append(f"Bias: {bias} while price holds this structure. "
                 f"— AI desk note, auto-generated; not financial advice.")

    return dict(paras=paras, levels=lvls, updatedAt=int(time.time()),
                session=sess, bias=bias)
