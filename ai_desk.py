"""AI Analyst Desk — an ensemble of independent models.

Each analyst reads the SAME chart with a DIFFERENT methodology and votes
bullish / bearish / neutral with a confidence and a one-line human reason.
The consensus meter aggregates the votes (confidence-weighted).

Analysts:
  neural    – the trained ML model (P(up) on the 15m horizon)
  momentum  – EMA stack + RSI on 15m, confirmed by the 60m stack
  structure – swing pivots (HH/HL vs LH/LL) on 60m and 1d
  snr       – the SNR zone engine's own read (fresh zone in play?)
  meanrev   – statistical stretch: z-score vs the 20-bar mean + RSI extremes
  volrange  – ATR regime + position inside the 24h range (breakout pressure)
  flow      – live order-flow: drift of the exchange book over 5m/15m
  macro     – dollar index + US 10Y yield pressure on gold
"""
from __future__ import annotations

import time

import ml
import wsfeed


def _closes(c):
    return [float(k["c"]) for k in c]


def _v(verdict, conf, note):
    return dict(verdict=verdict, conf=round(max(0.0, min(0.95, conf)), 2), note=note)


# ------------------------------------------------------------------ analysts
def a_neural(sig):
    p = sig.get("pUp")
    typ = (sig.get("type") or "HOLD").upper()
    if p is None:
        return _v("neutral", 0.3, "model warming up — no probability yet")
    edge = abs(p - 0.5) * 2.0
    if typ == "LONG":
        return _v("bullish", 0.35 + 0.6 * edge,
                  f"trained model P(up) {p * 100:.0f}% on the 15m horizon")
    if typ == "SHORT":
        return _v("bearish", 0.35 + 0.6 * edge,
                  f"trained model P(up) {p * 100:.0f}% — leaning lower")
    return _v("neutral", 0.3 + 0.3 * edge,
              f"model P(up) {p * 100:.0f}% — no edge either way")


def a_momentum(c15, c60):
    if len(c15) < 30 or len(c60) < 30:
        return _v("neutral", 0.3, "waiting for candle history")
    p15 = ml.indicator_pack(c15)
    p60 = ml.indicator_pack(c60)
    c = _closes(c15)[-1]
    e9, e21 = p15["e9"][-1], p15["e21"][-1]
    e9h, e21h = p60["e9"][-1], p60["e21"][-1]
    rsi = p15["rsi"][-1]
    s = 0.0
    s += 1.0 if e9 > e21 else -1.0
    s += 0.5 if c > e9 else -0.5
    s += 1.0 if e9h > e21h else -1.0
    s += 0.5 if rsi > 55 else (-0.5 if rsi < 45 else 0.0)
    stack = "9>21" if e9 > e21 else "9<21"
    hstack = "9>21" if e9h > e21h else "9<21"
    note = f"EMA {stack} on 15m, {hstack} on 60m · RSI {rsi:.0f}"
    if s >= 1.5:
        return _v("bullish", 0.4 + 0.11 * s, note + " — trend is up")
    if s <= -1.5:
        return _v("bearish", 0.4 + 0.11 * -s, note + " — trend is down")
    return _v("neutral", 0.35, note + " — mixed")


def _swing_trend(c, lb=2):
    """(trend, description) from the last two confirmed swing highs/lows."""
    if len(c) < 2 * lb + 5:
        return 0, "insufficient history"
    h = [float(k["h"]) for k in c]
    l = [float(k["l"]) for k in c]
    ph, pl = [], []
    for i in range(lb, len(c) - lb):
        if h[i] == max(h[i - lb:i + lb + 1]):
            ph.append(h[i])
        if l[i] == min(l[i - lb:i + lb + 1]):
            pl.append(l[i])
    if len(ph) < 2 or len(pl) < 2:
        return 0, "no clean swings yet"
    hh, hl = ph[-1] > ph[-2], pl[-1] > pl[-2]
    if hh and hl:
        return 1, "higher highs & higher lows"
    if not hh and not hl:
        return -1, "lower highs & lower lows"
    return 0, "compressed / indecisive swings"


def a_structure(c60, c1d):
    if not c60 or not c1d:
        return _v("neutral", 0.3, "waiting for higher-timeframe data")
    t60, d60 = _swing_trend(c60)
    td, dd = _swing_trend(c1d, lb=1)
    s = t60 + 2.0 * td
    note = f"60m: {d60} · daily: {dd}"
    if s >= 1.5:
        return _v("bullish", 0.45 + 0.1 * s, note + " — buyers in control")
    if s <= -1.5:
        return _v("bearish", 0.45 + 0.1 * -s, note + " — sellers in control")
    return _v("neutral", 0.4, note)


def a_snr(ent):
    if not ent or not ent.get("zoneSide"):
        return _v("neutral", 0.3, "no fresh SNR zone in play — stand aside")
    side = ent["zoneSide"]
    passed = ent.get("passed", 0)
    grade = ent.get("grade") or "—"
    if side == "demand":
        note = f"pullback into fresh demand {ent['entryZone'][0]:,.0f}–{ent['entryZone'][1]:,.0f}"
        if ent.get("active"):
            return _v("bullish", 0.75, note + f" — {grade} retest in progress")
        return _v("bullish", 0.4 + 0.05 * passed, note + f" ({passed}/8 checks)")
    note = f"rally into fresh supply {ent['entryZone'][0]:,.0f}–{ent['entryZone'][1]:,.0f}"
    if ent.get("active"):
        return _v("bearish", 0.75, note + f" — {grade} retest in progress")
    return _v("bearish", 0.4 + 0.05 * passed, note + f" ({passed}/8 checks)")


def a_meanrev(c15):
    if len(c15) < 40:
        return _v("neutral", 0.3, "waiting for candle history")
    c = _closes(c15)
    win = c[-20:]
    mean = sum(win) / len(win)
    var = sum((x - mean) ** 2 for x in win) / len(win)
    sd = var ** 0.5 or 1e-9
    z = (c[-1] - mean) / sd
    rsi = ml.indicator_pack(c15)["rsi"][-1]
    if z > 1.8 and rsi > 68:
        return _v("bearish", min(0.85, 0.4 + 0.18 * z),
                  f"{z:.1f}σ above the 20-bar mean, RSI {rsi:.0f} — stretched, favours a pullback")
    if z < -1.8 and rsi < 32:
        return _v("bullish", min(0.85, 0.4 - 0.18 * z),
                  f"{-z:.1f}σ below the 20-bar mean, RSI {rsi:.0f} — washed out, favours a bounce")
    return _v("neutral", 0.35,
              f"{z:+.1f}σ vs the 20-bar mean, RSI {rsi:.0f} — fairly valued")


def a_volrange(c15):
    if len(c15) < 220:
        return _v("neutral", 0.3, "waiting for candle history")
    p = ml.indicator_pack(c15)
    atr_now = p["atr"][-1]
    atrs = [a for a in p["atr"][-200:] if a == a]
    if not atrs:
        return _v("neutral", 0.3, "volatility data warming up")
    rank = sum(1 for a in atrs if a <= atr_now) / len(atrs)
    hi = max(float(k["h"]) for k in c15[-96:])
    lo = min(float(k["l"]) for k in c15[-96:])
    pos = 0.5 if hi <= lo else (float(c15[-1]["c"]) - lo) / (hi - lo)
    expanding = rank > 0.66
    quiet = rank < 0.33
    note = f"ATR {atr_now:.1f} ({rank * 100:.0f}th pct), price at {pos * 100:.0f}% of the 24h range"
    if expanding and pos > 0.78:
        return _v("bullish", 0.5 + 0.4 * (pos - 0.78), note + " — expansion to the upside")
    if expanding and pos < 0.22:
        return _v("bearish", 0.5 + 0.4 * (0.22 - pos), note + " — expansion to the downside")
    if quiet:
        return _v("neutral", 0.4, note + " — compression building, breakout pending")
    return _v("neutral", 0.35, note)


def a_flow():
    try:
        now = time.time()
        m0 = wsfeed.mid(5.0)
        m5 = wsfeed.mid_at(now - 300, 90.0)
        m15 = wsfeed.mid_at(now - 900, 120.0)
        if m0 is None:
            return _v("neutral", 0.3, "tick feed warming up")
        v5 = (m0 - m5) if m5 is not None else None
        v15 = (m0 - m15) if m15 is not None else None
        if v5 is None and v15 is None:
            return _v("neutral", 0.3, "tick history still building")
        thr = 1.2
        v = v5 if v5 is not None else v15
        note = []
        if v5 is not None:
            note.append(f"{v5:+.2f} in 5m")
        if v15 is not None:
            note.append(f"{v15:+.2f} in 15m")
        txt = " · ".join(note) + " on the exchange book"
        if v is not None and v > thr:
            return _v("bullish", min(0.8, 0.4 + 0.12 * v), txt + " — buyers pressing")
        if v is not None and v < -thr:
            return _v("bearish", min(0.8, 0.4 - 0.12 * v), txt + " — sellers pressing")
        return _v("neutral", 0.35, txt + " — balanced")
    except Exception:  # noqa: BLE001
        return _v("neutral", 0.3, "order-flow feed unavailable")


def a_macro(fund):
    if not fund:
        return _v("neutral", 0.3, "macro feed unavailable")
    dxy = fund.get("dxy") or {}
    y10 = fund.get("us10y") or {}
    s = 0.0
    dp = dxy.get("chgPct")
    yp = y10.get("chgPct")
    if dp is not None:
        if dp <= -0.2:
            s += 1
        elif dp >= 0.2:
            s -= 1
    if yp is not None:
        if yp <= -1.0:
            s += 1
        elif yp >= 1.0:
            s -= 1
    bits = []
    if dp is not None:
        bits.append(f"DXY {dp:+.2f}%")
    if yp is not None:
        bits.append(f"US10Y {yp:+.2f}%")
    txt = " · ".join(bits) or "macro data thin"
    if s >= 1:
        return _v("bullish", 0.45 + 0.15 * s, txt + " — tailwind for gold")
    if s <= -1:
        return _v("bearish", 0.45 + 0.15 * -s, txt + " — headwind for gold")
    return _v("neutral", 0.35, txt + " — neutral for gold")


# ------------------------------------------------------------------ ensemble
ROSTER = [
    ("neural", "🧠", "Neural ML", a_neural),
    ("momentum", "⚡", "Momentum", a_momentum),
    ("structure", "🏔️", "Structure", a_structure),
    ("snr", "🎯", "SNR Purest", a_snr),
    ("meanrev", "🔮", "Mean Reversion", a_meanrev),
    ("volrange", "🌊", "Vol & Range", a_volrange),
    ("flow", "🖨️", "Order Flow", a_flow),
    ("macro", "🌍", "Macro", a_macro),
]


def build(c15, c60, c1d, ent, sig, fund):
    """Full desk: per-analyst verdicts + confidence-weighted consensus."""
    args = {
        "neural": (sig,),
        "momentum": (c15, c60),
        "structure": (c60, c1d),
        "snr": (ent,),
        "meanrev": (c15,),
        "volrange": (c15,),
        "flow": (),
        "macro": (fund,),
    }
    analysts = []
    for key, icon, name, fn in ROSTER:
        try:
            v = fn(*args[key])
        except Exception:  # noqa: BLE001
            v = _v("neutral", 0.3, "model unavailable")
        analysts.append(dict(key=key, icon=icon, name=name,
                             verdict=v["verdict"], conf=v["conf"], note=v["note"]))
    bull = sum(1 for a in analysts if a["verdict"] == "bullish")
    bear = sum(1 for a in analysts if a["verdict"] == "bearish")
    neut = len(analysts) - bull - bear
    wsum = sum(a["conf"] for a in analysts)
    score = (sum(a["conf"] for a in analysts if a["verdict"] == "bullish")
             - sum(a["conf"] for a in analysts if a["verdict"] == "bearish")) / (wsum or 1.0)
    if score > 0.45:
        label = "STRONG BULLISH"
    elif score > 0.15:
        label = "BULLISH"
    elif score >= -0.15:
        label = "MIXED"
    elif score >= -0.45:
        label = "BEARISH"
    else:
        label = "STRONG BEARISH"
    return dict(analysts=analysts,
                consensus=dict(score=round(score, 2), label=label,
                               bull=bull, bear=bear, neutral=neut,
                               n=len(analysts), asOf=int(time.time())))
