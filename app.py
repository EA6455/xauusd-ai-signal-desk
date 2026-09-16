"""XAUUSD · AI Signal Desk — live web app.

Serves a dashboard with the gold price, an AI buy/sell signal per timeframe,
an alert feed (signal flips + custom price levels) and optional Telegram push.

Run:  python app.py   →  http://localhost:7860
"""
from __future__ import annotations

import calendar
import hashlib
import itertools
import json
import os
import threading
import time

import numpy as np
from flask import Flask, Response, jsonify, render_template, request

import ai_desk
import data
import entries
import fundamentals
import llm_desk
import ml
import narrative
import wsfeed

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, "models")
STATE_PATH = os.path.join(BASE, "state.json")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_MIX = 0.45                    # weight of the ML model in the composite score
RULES_MIX = 1.0 - MODEL_MIX
BUY_TH, SELL_TH = 0.20, -0.20       # composite-score thresholds
WINDOW = {"1s": 240, "5s": 240, "15s": 240, "30s": 240, "1m": 240, "5m": 200,
          "15m": 180, "30m": 200, "60m": 180, "2h": 180, "4h": 180, "1d": 240,
          "1w": 160, "1M": 180, "1Y": 40}   # bars shown on the chart

# payload rebuild cadence per tf (seconds) — fast for live tfs, relaxed for
# slow ones (the forming candle still updates from ticks on every tf)
REBUILD_S = {"1s": 8, "5s": 10, "15s": 15, "30s": 20, "1m": 30, "5m": 45,
             "15m": 45, "30m": 45, "60m": 60, "2h": 120, "4h": 180, "1d": 600,
             "1w": 1200, "1M": 1800, "1Y": 3600}

def _tg_secret(fname):
    """Bot token / chat id from a local file next to the app (not committed),
    so self-hosted instances work without environment plumbing."""
    try:
        with open(os.path.join(BASE, fname)) as f:
            return f.read().strip()
    except OSError:
        return ""


TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "") or _tg_secret("telegram_token")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "") or _tg_secret("telegram_chat")

# ------------------------------------------------------------------ version
def app_version():
    """Fingerprint of the app code — changes whenever app.py or the template
    is edited. Open tabs compare it and reload themselves automatically."""
    h = hashlib.md5()
    for f in (os.path.join(BASE, "app.py"),
              os.path.join(BASE, "templates", "index.html")):
        try:
            h.update(str(os.path.getmtime(f)).encode())
        except OSError:
            pass
    return h.hexdigest()[:8]


# ------------------------------------------------------------------ state
STATE_LOCK = threading.Lock()
STATE = dict(alerts=[], priceAlerts=[], lastSig={}, lastPrice=None,
             liveTrade=None, tradeHistory=[], lastSigT={}, lastSetup=None,
             lastSetupT=0.0, eventAlerted=[], newsSeen=None, brokerOffset=0.0,
             llmBriefDay="", asiaBO=None)
try:
    with open(STATE_PATH) as f:
        _loaded = json.load(f)
    STATE.update({k: v for k, v in _loaded.items() if k in STATE})
except Exception:  # noqa: BLE001
    pass

_id_counter = itertools.count(int(time.time() * 1000) % 1_000_000_000)


def _next_id():
    return next(_id_counter)


def _save_state():
    try:
        tmp = STATE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(STATE, f)
        os.replace(tmp, STATE_PATH)
    except OSError:
        pass


TG_CHATS = [c.strip() for c in TG_CHAT.split(",") if c.strip()]


def _notify(text):
    """Send a Telegram message to every configured chat (private and/or group).
    TELEGRAM_CHAT_ID may be a comma-separated list of chat ids."""
    if not (TG_TOKEN and TG_CHATS):
        return
    import urllib.request
    for chat in TG_CHATS:
        try:
            req = urllib.request.Request(
                f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                data=json.dumps({"chat_id": chat, "text": text}).encode(),
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10)
            print(f"[tg] sent to {chat}: {text[:50]!r}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[tg] FAILED to {chat}: {e}", flush=True)


# ------------------------------------------------------------------ models
models = {}
for _tf in data.TFS:
    try:
        models[_tf] = ml.Model.load(os.path.join(MODELS_DIR, f"{_tf}.json"))
    except Exception:  # noqa: BLE001
        models[_tf] = ml.Model()

_tried_train = set()


def ensure_model(tf, candles):
    """If no trained model exists for this timeframe, train one on the fly."""
    if models[tf].available or tf in _tried_train:
        return
    _tried_train.add(tf)
    try:
        m = ml.train_from_candles(candles)
        if m.available:
            m.save(os.path.join(MODELS_DIR, f"{tf}.json"))
            models[tf] = m
    except Exception:  # noqa: BLE001
        pass


# ----------------------------------------------------------- signal engine
def rules_score(p, i):
    """Weighted vote of classic technical rules, in [-1, 1]."""
    c = p["c"]
    v_trend = 1.0 if p["e9"][i] > p["e21"][i] else -1.0
    v_sma = 1.0 if c[i] > p["s20"][i] else -1.0
    v_macd = float(np.clip(p["macd_hist"][i] / c[i] * 100.0 * 15.0, -1.0, 1.0))
    r = p["rsi"][i]
    v_rsi = 1.0 if r < 30 else (-1.0 if r > 70 else 0.0)
    z = (c[i] - p["s20"][i]) / (p["sd20"][i] + 1e-9)
    v_bb = float(np.clip(-z / 2.5, -1.0, 1.0)) * 0.6
    return (1.0 * v_trend + 0.8 * v_sma + 0.8 * v_macd + 0.6 * v_rsi + 0.5 * v_bb) / 3.7


def indicator_rows(p, i):
    c = p["c"]
    r = p["rsi"][i]
    z = (c[i] - p["s20"][i]) / (p["sd20"][i] + 1e-9)

    def stance(v):
        return 1 if v > 0.15 else (-1 if v < -0.15 else 0)

    return [
        dict(name="Trend · EMA 9/21", value=f"{p['e9'][i]:.2f} / {p['e21'][i]:.2f}",
             stance=stance(1.0 if p["e9"][i] > p["e21"][i] else -1.0),
             note="golden-cross regime" if p["e9"][i] > p["e21"][i] else "death-cross regime"),
        dict(name="MACD hist (12,26,9)", value=f"{p['macd_hist'][i]:+.2f}",
             stance=stance(float(np.clip(p['macd_hist'][i] / c[i] * 100 * 15, -1, 1))),
             note="momentum"),
        dict(name="RSI (14)", value=f"{r:.1f}",
             stance=1 if r < 30 else (-1 if r > 70 else 0),
             note="oversold" if r < 30 else ("overbought" if r > 70 else "neutral")),
        dict(name="Price vs SMA 20", value=f"{z:+.2f}σ",
             stance=stance(1.0 if c[i] > p["s20"][i] else -1.0), note="band position"),
        dict(name="ATR (14)", value=f"{p['atr'][i]:.2f} ({p['atr'][i] / c[i] * 100:.2f}%)",
             stance=0, note="volatility"),
    ]


def alerts_snapshot():
    with STATE_LOCK:
        return STATE["alerts"][:30]


FLIP_REARM_MIN = 180              # same-direction re-alert needs 3h of quiet


def maybe_alert(tf, sig_type, price, score, conf, atr=None):
    """Fire an alert when the composite signal flips to BUY or SELL.
    Sticky regimes mean this is a genuine reversal; a same-direction repeat
    within FLIP_REARM_MIN is suppressed, and every alert carries reference
    SL / TP levels (1.5xATR stop, 1:2 RR target) so it is actionable."""
    if sig_type not in ("BUY", "SELL"):
        with STATE_LOCK:
            if STATE["lastSig"].get(tf) not in (None, "HOLD"):
                STATE["lastSig"][tf] = "HOLD"
                _save_state()
        return
    with STATE_LOCK:
        prev = STATE["lastSig"].get(tf)
        last_t = STATE.get("lastSigT", {}).get(tf, 0)
        if prev == sig_type:
            return
        now = time.time()
        if prev is None:
            # first stance after a (re)start is just "where we are" — prime
            # the state silently instead of messaging the phone
            STATE["lastSig"][tf] = sig_type
            STATE.setdefault("lastSigT", {})[tf] = now
            _save_state()
            return
        if now - last_t < FLIP_REARM_MIN * 60:
            # too soon after the last alert from this timeframe: track the
            # stance silently so no stale alert fires later
            STATE["lastSig"][tf] = sig_type
            _save_state()
            return
        STATE.setdefault("lastSigT", {})[tf] = now
        STATE["lastSig"][tf] = sig_type
        first = prev is None
        d = 1 if sig_type == "BUY" else -1
        levels = ""
        if atr and atr > 0:
            sl = price - d * 1.5 * atr
            tp = price + d * 3.0 * atr
            levels = (f" · SL {sl:,.1f} · TP {tp:,.1f} (1:2 ref)")
        a = dict(id=_next_id(), time=int(time.time()), tf=tf, type=sig_type,
                 price=round(float(price), 2), score=round(float(score), 3),
                 confidence=round(float(conf), 3),
                 msg=f"XAUUSD {tf} {sig_type} @ {price:,.2f} · "
                     f"conf {conf * 100:.0f}% · score {score:+.2f}{levels}"
                     + (" · initial stance" if first else " · signal flip"))
        STATE["alerts"].insert(0, a)
        del STATE["alerts"][100:]
        _save_state()
    # web-feed only — the phone is reserved for A+ entries, TP/SL results
    # and user-armed price alerts (no chart-flip messages)


SETUP_COOLDOWN_S = 2 * 3600         # min gap between ENTRY alerts


def _event_blackout():
    """High-impact event window: 30 min before start -> 15 min after.
    Returns the event title while inside the window, else None."""
    try:
        for e in fundamentals.upcoming(hours=0.5, impacts=("High",)):
            if e.get("minutesTo", 999) <= 30:
                return e.get("title") or "high-impact event"
    except Exception:  # noqa: BLE001
        pass
    now = time.time()
    for key in (STATE.get("eventAlerted") or [])[-10:]:
        try:
            ts = int(str(key).split(":", 1)[0])
        except ValueError:
            continue
        if 0 <= now - ts <= 15 * 60:
            return str(key).split(":", 1)[-1]
    return None


_postpone_mem = {}


def _postpone_note(ent, title):
    """Web-feed note: a live setup exists but alerting is postponed."""
    if not ent or not ent.get("touching"):
        return
    key = (ent.get("zoneKey") or "") + ":" + str(ent.get("grade"))
    now = time.time()
    if _postpone_mem.get(key) and now - _postpone_mem[key] < 3600:
        return
    _postpone_mem[key] = now
    _web_alert("WAIT", f"⏸ {ent.get('grade', 'setup')} {ent['direction']} signal "
                       f"postponed — {title} event window")


_sweep_alerted = {"keys": []}


def maybe_sweep_alert(sw):
    """Liquidity-sweep watch: web feed only (recent 60d stats below the phone
    bar — flip PHONE=True to also send cards). One alert per sweep event."""
    PHONE = True
    if not sw or sw.get("grade") not in ("A+", "B+"):
        return
    if _event_blackout():
        return
    key = f"sweep:{sw.get('anchor')}:{sw.get('barTime')}"
    if key in _sweep_alerted["keys"]:
        return
    _sweep_alerted["keys"] = (_sweep_alerted["keys"] + [key])[-40:]
    _web_alert("SWEEP", f"🧹 {sw['grade']} SWEEP {sw['direction']} reclaim @ "
                        f"{sw['entry']:,.2f} · swept {sw['zone'][0]:,.1f}–"
                        f"{sw['zone'][1]:,.1f} · SL {sw['sl']:,.2f} · "
                        f"TP1 {sw['tp1']:,.2f} · {sw['passed']}/4")
    if not PHONE:
        return
    if time.time() - STATE.get("lastSetupT", 0) < SETUP_COOLDOWN_S:
        return
    with STATE_LOCK:
        STATE["lastSetupT"] = time.time()
    side_lbl = "Support" if sw["zoneSide"] == "demand" else "Resistance"
    icon = "🟢" if sw["direction"] == "LONG" else "🔴"
    warn = "" if sw["grade"] == "A+" else \
        f"\n⚠ {sw['passed']}/4 confluence — reduced quality: smaller size or skip"
    _notify(f"{icon} {sw['grade']} SWEEP {sw['direction']} SIGNAL\n\n"
            f"📊 Timeframe: 15M\n"
            f"💰 Symbol: XAUUSD\n"
            f"📍 Setup: Liquidity Sweep · {side_lbl}\n\n"
            f"🎯 Entry: {sw['entry']:,.2f}\n"
            f"🛑 SL: {sw['sl']:,.2f}\n"
            f"🎯 TP: {sw['tp1']:,.2f}\n\n"
            f"⭐ SNR Rating: {sw['grade']} SWEEP{warn}")


def _asia_bo_watch():
    """London breakout of the Asian range: breakout close -> retest ->
    confirmation -> one card per day. Web feed + phone (event-gated)."""
    now = time.time()
    if entries.session_of(now) != "london":
        return
    today = time.strftime("%Y-%m-%d", time.gmtime())
    st = STATE.get("asiaBO") or {}
    if st.get("date") != today:
        st = dict(date=today, stage="wait", dir=None, level=None,
                  hi=None, lo=None, bi=None)
        STATE["asiaBO"] = st
        _save_state()
    if st.get("stage") == "done":
        return
    try:
        c15 = data.get_candles("15m").get("candles") or []
    except Exception:  # noqa: BLE001
        return
    if len(c15) < 260:
        return
    day0 = calendar.timegm(time.strptime(today, "%Y-%m-%d"))
    a0, a1 = day0 + 3600, day0 + 7 * 3600
    asia = [k for k in c15 if a0 <= k["t"] < a1]
    if len(asia) < 8:
        return
    hi, lo = max(k["h"] for k in asia), min(k["l"] for k in asia)
    h = [k["h"] for k in c15[-40:]]
    l = [k["l"] for k in c15[-40:]]
    cc = [k["c"] for k in c15[-40:]]
    atr = ml.atr(h, l, cc, 14)[-1] or 1.0
    if not (0.4 * atr <= hi - lo <= 10 * atr):
        STATE["asiaBO"] = dict(st, stage="done")
        _save_state()
        return
    london = [k for k in c15 if k["t"] >= a1]
    if not london:
        return
    if st["stage"] == "wait":
        for k in london[:-1]:                    # closed bars only
            if k["c"] > hi or k["c"] < lo:
                br = "LONG" if k["c"] > hi else "SHORT"
                STATE["asiaBO"] = dict(st, stage="broken", dir=br,
                                       level=round(hi if br == "LONG" else lo, 1),
                                       hi=hi, lo=lo, bi=k["t"])
                _save_state()
                break
    st = STATE["asiaBO"]
    if st["stage"] != "broken":
        return
    level, dr = st["level"], st["dir"]
    tol = 0.25 * atr
    for k in london[:-1]:
        if k["t"] <= (st.get("bi") or 0):
            continue
        if dr == "LONG":
            if k["c"] < level - 1.5 * atr:       # failed breakout
                STATE["asiaBO"] = dict(st, stage="wait", dir=None, level=None, bi=None)
                _save_state()
                return
            if k["l"] <= level + tol and k["c"] > k["o"] and k["c"] > level:
                entry = k["c"]
                break
        else:
            if k["c"] > level + 1.5 * atr:
                STATE["asiaBO"] = dict(st, stage="wait", dir=None, level=None, bi=None)
                _save_state()
                return
            if k["h"] >= level - tol and k["c"] < k["o"] and k["c"] < level:
                entry = k["c"]
                break
        if k["t"] - (st.get("bi") or 0) > 12 * 900:   # no retest in 12 bars
            STATE["asiaBO"] = dict(st, stage="done")
            _save_state()
            return
    else:
        return
    STATE["asiaBO"] = dict(st, stage="done")
    _save_state()
    d = 1 if dr == "LONG" else -1
    sl = entry - d * 1.0 * atr
    tp = entry + d * 2.0 * atr
    _web_alert("ENTRY", f"London breakout {dr} · Asia range {lo:.1f}–{hi:.1f} · "
                        f"retest {level:.1f}")
    if not _event_blackout():
        icon = "🟢" if dr == "LONG" else "🔴"
        _notify(f"{icon} LONDON BREAKOUT {dr} SIGNAL\n\n"
                f"📊 Timeframe: 15M\n"
                f"💰 Symbol: XAUUSD\n"
                f"📍 Setup: Asia Range Breakout\n\n"
                f"🎯 Entry: {entry:,.2f}\n"
                f"🛑 SL: {sl:,.2f}\n"
                f"🎯 TP: {tp:,.2f}\n\n"
                f"📐 Asia range: {lo:,.1f} – {hi:,.1f}\n"
                f"⭐ SNR Rating: BREAKOUT")


_zw_mem = {"t": 0.0, "keys": []}


def maybe_zone_watch(ent):
    """⏳ Heads-up: price just tapped a fresh SNR zone but the confluence
    isn't complete yet — 'setup forming, watch this zone'. One per zone,
    at least 45 minutes apart, live sessions only, event-gated."""
    if not ent or not ent.get("zoneSide") or not ent.get("touching"):
        return
    if ent.get("active") or ent.get("grade") in ("A+", "B+", "C+"):
        return                                  # full cards already cover it
    if ent.get("passed", 0) < 4:
        return                                  # too little going for it
    if ent.get("session") == "dead" or _event_blackout():
        return
    key = ent.get("zoneKey") or f"{ent['direction']}:{ent['barTime']}"
    now = time.time()
    if key in _zw_mem["keys"] or now - _zw_mem["t"] < 45 * 60:
        return
    _zw_mem["keys"] = (_zw_mem["keys"] + [key])[-40:]
    _zw_mem["t"] = now
    side = ent["zoneSide"]
    lbl = "Support" if side == "demand" else "Resistance"
    lo, hi = ent["entryZone"][0], ent["entryZone"][1]
    _web_alert("WATCH", f"setup forming · {side} {lo:,.1f}–{hi:,.1f} · "
                        f"{ent['passed']}/8 checks")
    _notify(f"⏳ SETUP FORMING — XAUUSD\n\n"
            f"📍 Price is testing a fresh {lbl} zone\n"
            f"📐 Zone: {lo:,.1f} – {hi:,.1f}\n"
            f"🧩 {ent['passed']}/8 confluence so far\n\n"
            f"⏳ Waiting for the confirmation candle\n"
            f"📈 A+ / B+ signal card follows if it completes\n\n"
            f"💰 XAUUSD · 15M")


_armed_mem = {"t": 0.0, "keys": []}


def maybe_armed_alert(ent):
    """🎯 Zone armed: a fresh SNR zone has 6/8+ confluence in place but price
    has NOT returned yet (the missing check is the first retest). Phone
    heads-up so the zone can be watched / a price alert placed there — this
    is NOT an entry signal; the entry card follows if the retest confirms.
    One alert per zone, 60-min global gap. Deliberately fires during event
    windows too (a zone arming is a standing fact, not an entry)."""
    if not ent or not ent.get("zoneKey") or not ent.get("zoneSide"):
        return
    if ent.get("touching") or ent.get("active"):
        return                                  # retest cards already cover it
    if ent.get("grade") in ("A+", "B+", "C+"):
        return
    if ent.get("passed", 0) < 6:
        return                                  # only C+/B+ quality zones arm
    key = "armed:" + ent["zoneKey"]
    now = time.time()
    if key in _armed_mem["keys"] or now - _armed_mem["t"] < 60 * 60:
        return
    _armed_mem["keys"] = (_armed_mem["keys"] + [key])[-60:]
    _armed_mem["t"] = now
    side = ent["zoneSide"]
    lbl = "Support" if side == "demand" else "Resistance"
    lo, hi = ent["entryZone"][0], ent["entryZone"][1]
    quality = "B+" if ent["passed"] >= 7 else "C+"
    _web_alert("ARMED", f"🎯 zone armed · {side} {lo:,.1f}–{hi:,.1f} · "
                        f"{ent['passed']}/8 · waiting for retest")
    _notify(f"🎯 ZONE ARMED — XAUUSD\n\n"
            f"📍 Fresh {lbl} zone ({ent['direction']} bias)\n"
            f"📐 Zone: {lo:,.1f} – {hi:,.1f}\n"
            f"🧩 {ent['passed']}/8 confluence in place ({quality} quality if it holds)\n"
            f"⏳ Waiting for price to return\n\n"
            f"⚠️ Not an entry yet — signal card follows when price retests the zone\n\n"
            f"💰 XAUUSD · 15M")


def maybe_setup_alert(ent):
    """Fire an alert on SNR retests, once per ZONE+GRADE (a setup that stays
    live for hours must not re-alert every 15 minutes). All grades go to the
    phone (B+/C+ carry quality warnings). Phone-worthy grades share the
    SETUP_COOLDOWN_S anti-spam gap."""
    if not ent or not ent.get("touching"):
        return
    grade = ent.get("grade")
    if grade not in ("A+", "B+", "C+"):
        return
    key = (ent.get("zoneKey") or f"{ent['direction']}:{ent['barTime']}") + ":" + grade
    phone = grade in ("A+", "B+", "C+")
    with STATE_LOCK:
        if STATE.get("lastSetup") == key:
            return
        if phone and time.time() - STATE.get("lastSetupT", 0) < SETUP_COOLDOWN_S:
            return
        STATE["lastSetup"] = key
        if phone:
            STATE["lastSetupT"] = time.time()
        typ = "ENTRY_BUY" if ent["direction"] == "LONG" else "ENTRY_SELL"
        a = dict(id=_next_id(), time=int(time.time()), tf="15m", type=typ,
                 price=ent["entry"], score=round(ent["passed"] / 8.0, 2),
                 confidence=round(ent["passed"] / 8.0, 2),
                 msg=(f"SNR {grade} {ent['direction']} retest @ {ent['entry']:,.2f} · "
                      f"zone {ent['entryZone'][0]:,.1f}–{ent['entryZone'][1]:,.1f} · "
                      f"SL {ent['sl']:,.2f} · TP1 {ent['tp1']:,.2f} · "
                      f"{ent['passed']}/8 SNR checks"))
        STATE["alerts"].insert(0, a)
        del STATE["alerts"][100:]
        _save_state()
    if not phone:
        return
    # phone message: the classic SNR signal card (ONE message per zone+grade)
    side = ent.get("zoneSide") or ("demand" if ent["direction"] == "LONG" else "supply")
    setup_lbl = "Support" if side == "demand" else "Resistance"
    icon = "🟢" if ent["direction"] == "LONG" else "🔴"
    if grade == "A+":
        warn = ""
    elif grade == "B+":
        warn = f"\n⚠ {ent['passed']}/8 confluence — reduced quality: smaller size or skip"
    else:
        warn = f"\n⚠ {ent['passed']}/8 confluence — watchlist quality: tiny size or paper trade"
    htf_line = ""
    if ent.get("htf"):
        h = ent["htf"]
        htf_line = (f"\n📐 HTF: {h['tf']} "
                    f"{'demand' if side == 'demand' else 'supply'} "
                    f"{h['bottom']:,.0f}–{h['top']:,.0f}")
    _notify(
        f"{icon} {grade} {ent['direction']} SIGNAL\n\n"
        f"📊 Timeframe: 15M\n"
        f"💰 Symbol: XAUUSD\n"
        f"📍 Setup: {setup_lbl}{htf_line}\n\n"
        f"🎯 Entry: {ent['entry']:,.2f}\n"
        f"🛑 SL: {ent['sl']:,.2f}\n"
        f"🎯 TP: {ent['tp1']:,.2f}\n\n"
        f"⭐ SNR Rating: {grade}{warn}")


def _close_trade(tr, result, r, price):
    tr.update(status="closed", result=result, r=round(r, 2),
              closedAt=int(time.time()), closePrice=round(float(price), 2))
    STATE["tradeHistory"].insert(0, dict(tr))
    del STATE["tradeHistory"][25:]


def ensure_trade(ent):
    """Open a tracked position when an A+ setup fires (one at a time)."""
    if not ent or not ent.get("active"):
        return
    key = ent.get("zoneKey") or f"{ent['direction']}:{ent['barTime']}"
    with STATE_LOCK:
        lt = STATE.get("liveTrade")
        if lt and lt.get("status") not in ("closed", None):
            return
        STATE["liveTrade"] = dict(
            key=key, dir=ent["direction"], entry=ent["entry"], sl=ent["sl"],
            tp1=ent["tp1"], tp2=ent["tp2"], atr=ent["atr"],
            session=ent.get("session"), openedAt=int(time.time()),
            barTime=ent["barTime"], status="open", result=None, r=0.0)
        _save_state()


def update_trade_tracker():
    """Manage the live signal with a pro scheme: half off at TP1 (+1.0R banked),
    stop to breakeven, runner to TP2 (+2.5R total). Alerts on every event."""
    price, _src = tick_spot()
    if price is None:
        return
    fired = []
    with STATE_LOCK:
        tr = STATE.get("liveTrade")
        if not tr or tr.get("status") == "closed":
            return
        tr_dir = tr["dir"]
        d = 1 if tr["dir"] == "LONG" else -1
        entry, sl, tp1, tp2 = tr["entry"], tr["sl"], tr["tp1"], tr["tp2"]
        sl_dist = abs(entry - sl) or 1.0
        now = int(time.time())
        # 5h timeout (20 x 15m bars)
        timed_out = now - tr["openedAt"] > 20 * 15 * 60
        if tr["status"] == "open":
            if (d == 1 and price >= tp1) or (d == -1 and price <= tp1):
                tr["status"] = "tp1hit"; tr["tp1At"] = now
                fired.append(("TP1_HIT", f"TP1 hit at {tp1:,.1f} · half banked +1.0R · stop moved to breakeven {entry:,.1f} · runner targets {tp2:,.1f}"))
            elif (d == 1 and price <= sl) or (d == -1 and price >= sl):
                _close_trade(tr, "loss", -1.0, price)
                fired.append(("SL_HIT", f"Stopped out at {sl:,.1f} · -1.0R · next setup will come, patience"))
            elif timed_out:
                r = 0.5 * d * (price - entry) / sl_dist
                _close_trade(tr, "timeout", r, price)
                fired.append(("TIMEOUT", f"5h timeout — closed at market {price:,.1f} ({r:+.2f}R on the runner half)"))
        if tr.get("status") == "tp1hit":
            if (d == 1 and price >= tp2) or (d == -1 and price <= tp2):
                _close_trade(tr, "win", 2.5, price)
                fired.append(("TP2_HIT", f"TP2 hit at {tp2:,.1f} · runner closed · total +2.5R on the signal 🎯"))
            elif (d == 1 and price <= entry) or (d == -1 and price >= entry):
                _close_trade(tr, "be", 1.0, price)
                fired.append(("BE_STOP", f"Runner stopped at breakeven {entry:,.1f} · signal finishes +1.0R (TP1 banked)"))
            elif timed_out:
                r = 1.0 + 0.5 * d * (price - entry) / sl_dist
                _close_trade(tr, "timeout", r, price)
                fired.append(("TIMEOUT", f"5h timeout — runner closed at market {price:,.1f} · total {r:+.2f}R"))
        if fired:
            _save_state()
    for typ, msg in fired:
        with STATE_LOCK:
            a = dict(id=_next_id(), time=int(time.time()), tf="15m", type=typ,
                     price=round(float(price), 2), score=0.0, confidence=None,
                     msg=tr_dir + " signal · " + msg)
            STATE["alerts"].insert(0, a)
            del STATE["alerts"][100:]
            _save_state()
        # phone: result card in the same style as the A+ signal card
        icon = {"TP1_HIT": "✅", "TP2_HIT": "🎯", "SL_HIT": "🛑",
                "BE_STOP": "⚖️", "TIMEOUT": "⏱"}.get(typ, "•")
        head = {"TP1_HIT": "TP1 HIT", "TP2_HIT": "TP2 HIT", "SL_HIT": "SL HIT",
                "BE_STOP": "BREAKEVEN STOP", "TIMEOUT": "TIMEOUT · 5h"}.get(typ, typ)
        ent_p, sl_p, tp1_p, tp2_p = tr["entry"], tr["sl"], tr["tp1"], tr["tp2"]
        if typ == "TP1_HIT":
            body = (f"🎯 Entry: {ent_p:,.2f}\n"
                    f"✅ TP1 reached: {tp1_p:,.2f} (+1.0R banked)\n"
                    f"🛑 SL moved to breakeven {ent_p:,.2f}\n"
                    f"🎯 TP2 runner: {tp2_p:,.2f}\n\n"
                    f"⭐ Half position closed")
        elif typ == "TP2_HIT":
            body = (f"🎯 Entry: {ent_p:,.2f}\n"
                    f"✅ TP1: {tp1_p:,.2f} (+1.0R)\n"
                    f"🎯 TP2: {tp2_p:,.2f} (+1.5R runner)\n\n"
                    f"⭐ Trade closed · Total +2.5R 🎉")
        elif typ == "SL_HIT":
            body = (f"🎯 Entry: {ent_p:,.2f}\n"
                    f"🛑 SL: {sl_p:,.2f} (-1.0R)\n\n"
                    f"⭐ Trade closed · next setup will come — patience")
        elif typ == "BE_STOP":
            body = (f"🎯 Entry: {ent_p:,.2f}\n"
                    f"✅ TP1 banked: {tp1_p:,.2f} (+1.0R)\n"
                    f"⚖️ Runner stopped at breakeven {ent_p:,.2f}\n\n"
                    f"⭐ Trade closed · Total +1.0R")
        else:  # TIMEOUT
            r = tr.get("r") or 0.0
            body = (f"🎯 Entry: {ent_p:,.2f}\n"
                    f"💰 Closed at market: {tr.get('closePrice', price):,.2f}\n\n"
                    f"⭐ Trade closed · Total {r:+.2f}R")
        _notify(f"{icon} A+ {tr_dir} — {head}\n\n"
                f"📊 Timeframe: 15M\n"
                f"💰 Symbol: XAUUSD\n\n"
                f"{body}")


def check_price_alerts(prev, new):
    if not prev or not new:
        return
    fired = []
    with STATE_LOCK:
        for pa in STATE["priceAlerts"]:
            if pa.get("triggeredAt"):
                continue
            lvl = pa["level"]
            if (prev - lvl) * (new - lvl) < 0 or new == lvl:
                pa["triggeredAt"] = int(time.time())
                up = new > lvl
                fired.append(dict(id=_next_id(), time=pa["triggeredAt"], tf="—",
                                  type="PRICE_UP" if up else "PRICE_DOWN",
                                  price=round(float(new), 2), score=0.0, confidence=None,
                                  msg=f"XAUUSD crossed {lvl:,.2f} "
                                      f"{'↑' if up else '↓'} · now {new:,.2f}"))
        for a in fired:
            STATE["alerts"].insert(0, a)
        del STATE["alerts"][100:]
        if fired:
            _save_state()
    for a in fired:
        _notify("⏰ " + a["msg"])


def spot_reference():
    """Best available spot XAU/USD: gold-api.com, else OKX PAXG ticker."""
    s = data.get_spot(max_age=30)
    if s:
        return s["price"], "gold-api.com"
    try:
        j = data._http_json("https://www.okx.com/api/v5/market/ticker?instId="
                            + data.OKX_INSTRUMENT, timeout=8)
        return float(j["data"][0]["last"]), "OKX PAXG"
    except Exception:  # noqa: BLE001
        return None, None


_tick_mem = {"t": 0.0, "price": None, "src": None}
_okx_tick_mem = {"t": 0.0, "price": None}
_basis_mem = {"t": 0.0, "value": 0.0, "valid": False, "ema": None, "bwarm": 0}
_quote_mem = {"t": 0.0, "price": None}


def _okx_last():
    now = time.time()
    if now - _okx_tick_mem["t"] < 20 and _okx_tick_mem["price"]:
        return _okx_tick_mem["price"]
    try:
        j = data._http_json("https://www.okx.com/api/v5/market/ticker?instId="
                            + data.OKX_INSTRUMENT, timeout=6)
        p = float(j["data"][0]["last"])
        _okx_tick_mem.update(t=now, price=p)
        return p
    except Exception:  # noqa: BLE001
        return None


def _yahoo_gc_quote():
    """(last price, market time) for GC=F — the market time also tells us
    whether the gold market is open or closed."""
    now = time.time()
    if now - _quote_mem["t"] < 1.5 and _quote_mem["price"]:
        return _quote_mem["price"], _quote_mem.get("mtime")
    if not data.yahoo_ok():
        return None, None
    for host in ("query1", "query2"):
        try:
            meta = data._http_json("https://" + host + ".finance.yahoo.com/v8/"
                                   "finance/chart/GC=F?interval=1m&range=1d"
                                   )["chart"]["result"][0]["meta"]
            p = float(meta["regularMarketPrice"])
            if p > 100:
                data.note_yahoo_result(True)
                _quote_mem.update(t=now, price=p, mtime=meta.get("regularMarketTime"))
                return p, _quote_mem["mtime"]
        except Exception:  # noqa: BLE001
            continue
    data.note_yahoo_result(False)
    return None, None


_anchor_mem = {"offset": 0.0, "t": 0.0, "ema": None, "last_ts": None,
               "warm": 0, "ema_f": None, "warm_f": 0, "last_f_ts": 0.0}
_frozen_mem = {"price": None}


def _feed_epoch(iso):
    """Epoch seconds from a gold-api updatedAt string (or None)."""
    try:
        from datetime import datetime
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except Exception:  # noqa: BLE001
        return None

try:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")

    def _gold_market_open():
        """CME Globex gold hours (ET): Sun 18:00 → Fri 17:00, daily 17:00–18:00 break."""
        dt = datetime.now(_ET)
        wd, hm = dt.weekday(), dt.hour * 60 + dt.minute
        if wd == 5:
            return False                       # Saturday
        if wd == 4 and hm >= 17 * 60:
            return False                       # Friday after the close
        if wd == 6 and hm < 18 * 60:
            return False                       # Sunday before the open
        if 0 <= wd <= 3 and 17 * 60 <= hm < 18 * 60:
            return False                       # daily maintenance break
        return True
except Exception:  # noqa: BLE001  — no tz data: assume open, ws feed still runs
    def _gold_market_open():
        return True


def _tick_spot_raw(max_age=0.8):
    """Freshest spot XAU/USD for the live tick line (market level, no
    per-broker adjustment).

    Priority:
      1. OKX websocket mid-price (tick-by-tick) + a spot anchor offset  ← realtime
      2. Live GC=F quote minus the current futures basis
      3. gold-api.com spot
      4. OKX REST last price
    When COMEX is closed (stale futures timestamp) the price freezes at the
    last close instead of following the 24/7 crypto token.
    """
    now = time.time()
    if _tick_mem["price"] and now - _tick_mem["t"] < max_age:
        return _tick_mem["price"], _tick_mem["src"]

    spot = data.get_spot(max_age=12)          # cheap; internally cached
    q, _qm = _yahoo_gc_quote()
    basis_ok = _basis_mem["valid"] and now - _basis_mem["t"] < 180

    # gold market closed? (CME clock — NOT Yahoo's unreliable timestamp)
    if not _gold_market_open():
        if _frozen_mem["price"] is None:
            if spot:
                _frozen_mem["price"] = spot["price"]
            elif q and basis_ok:
                _frozen_mem["price"] = round(q - _basis_mem["value"], 2)
            elif _tick_mem["price"]:
                _frozen_mem["price"] = _tick_mem["price"]
        if _frozen_mem["price"] is not None:
            _tick_mem.update(t=now, price=_frozen_mem["price"], src="market closed")
            return _tick_mem["price"], _tick_mem["src"]
    _frozen_mem["price"] = None

    price, src = None, None

    # 1. real-time websocket mid, pinned to spot XAU via the anchor offset.
    #    The offset is measured TIMESTAMP-MATCHED: gold-api's (slow, ~30s)
    #    quote is paired with the book mid at the quote's own as-of time,
    #    so the anchor carries no staleness bias — the displayed price
    #    follows the live book instead of trailing the slow spot feed.
    wm, _wt = wsfeed.mid(max_age=30)
    if wm:
        # ---- sample A: timestamp-matched gold-api premium (level truth, slow)
        matched_ts = None
        anchor_target = None
        if spot and spot.get("updatedAt"):
            ts = _feed_epoch(spot["updatedAt"])
            if ts and now - ts <= 120:
                m_then = wsfeed.mid_at(ts)
                if m_then:
                    anchor_target = spot["price"] - m_then
                    matched_ts = ts
        if anchor_target is None and spot:
            anchor_target = spot["price"] - wm
        if anchor_target is not None and abs(anchor_target) <= 8.0:
            if matched_ts is not None and matched_ts != _anchor_mem.get("last_ts"):
                # AUTO-SYNC TO THE REAL MARKET: each NEW spot print re-locks
                # the level hard (a just-published quote IS the market at that
                # instant); older re-samples only nudge it gently.
                prev = _anchor_mem.get("ema")
                warm = _anchor_mem.get("warm", 0)
                quote_age = max(0.0, now - matched_ts)
                if prev is None:
                    ema = anchor_target
                else:
                    alpha = 0.35 if warm < 8 else (0.55 if quote_age <= 20 else 0.08)
                    ema = prev + alpha * (anchor_target - prev)
                _anchor_mem.update(ema=ema, warm=warm + 1, last_ts=matched_ts)
            elif matched_ts is None:
                if _anchor_mem.get("ema") is None:
                    _anchor_mem["ema"] = anchor_target
                    _anchor_mem["warm"] = _anchor_mem.get("warm", 0) + 1
        # ---- sample B: real-time futures premium (GC=F is near-instant; the
        #      futures basis is stable, so q - basis tracks spot with no lag)
        if q and basis_ok:
            fs = (q - _basis_mem["value"]) - wm
            if abs(fs) <= 8.0 and now - _anchor_mem.get("last_f_ts", 0) >= 5:
                prev = _anchor_mem.get("ema_f")
                warm = _anchor_mem.get("warm_f", 0)
                alpha = 0.35 if warm < 12 else 0.15
                ema_f = fs if prev is None else prev + alpha * (fs - prev)
                _anchor_mem.update(ema_f=ema_f, warm_f=warm + 1, last_f_ts=now)
        # ---- blend: level truth (A) + real-time tracking (B).
        # The futures level (B) reacts within milliseconds; gold-api (A) trails
        # the true spot by 30-60s. So B gets the larger weight, and even more
        # while A's quote is aging — the displayed price sticks to the LIVE
        # market instead of trailing the slow spot feed.
        ea, ef = _anchor_mem.get("ema"), _anchor_mem.get("ema_f")
        if ea is not None and ef is not None:
            ga_age = 999.0
            if spot and spot.get("updatedAt"):
                _ts = _feed_epoch(spot["updatedAt"])
                if _ts:
                    ga_age = now - _ts
            # the fresher the spot print, the more it IS the market;
            # as it ages the instant futures level carries the movement
            if ga_age <= 20:
                w_f = 0.40
            elif ga_age <= 45:
                w_f = 0.60
            else:
                w_f = 0.72
            _anchor_mem["offset"] = (1.0 - w_f) * ea + w_f * ef
        elif ea is not None:
            _anchor_mem["offset"] = ea
        elif ef is not None:
            _anchor_mem["offset"] = ef
        _anchor_mem["t"] = now
        off = _anchor_mem["offset"]
        if abs(off) <= 8.0:
            price, src = round(wm + off, 2), "realtime feed"
    # 2. live futures quote minus basis
    if price is None and q and basis_ok:
        price, src = round(q - _basis_mem["value"], 2), "live futures - basis"
    # 3. spot feed
    if price is None and spot:
        price, src = spot["price"], "gold-api.com"
    # 4. OKX REST last
    if price is None:
        p = _okx_last()
        if p:
            price, src = p, "OKX PAXG"

    if price is not None:
        _tick_mem.update(t=now, price=price, src=src)
    return price, src


def tick_spot(max_age=0.8, broker=True):
    """Live price as displayed. `broker=True` adds the user's broker-sync
    offset so every shown number (ticks, alerts, tracker) matches THEIR
    broker exactly; internal candle work uses broker=False."""
    p, src = _tick_spot_raw(max_age)
    if p is not None and broker:
        bo = STATE.get("brokerOffset") or 0.0
        if bo:
            p = round(p + bo, 2)
    return p, src


def build_payload(tf, d, symbol="XAUUSD"):
    raw = d["candles"]
    gold = symbol == "XAUUSD"
    sym_cfg = data.SYMBOLS[symbol]

    # ---- spot-align: remove the futures basis so prices match XAU/USD spot
    spot_price, spot_src = (spot_reference() if gold else (None, None))
    adjust = 0.0
    if spot_price:
        adjust = raw[-1]["c"] - spot_price
        if not (0 < adjust < 150):        # sanity guard (normal contango only)
            adjust = 0.0
    if adjust:
        candles = [dict(t=k["t"], o=k["o"] - adjust, h=k["h"] - adjust,
                        l=k["l"] - adjust, c=k["c"] - adjust) for k in raw]
        source = d["source"] + " · spot-aligned"
    else:
        candles = raw
        source = d["source"]
    # smooth the basis slowly (roll drifts over days) so the real-time
    # futures anchor sample isn't contaminated by any single spot quote
    prev_b = _basis_mem.get("ema")
    bw = _basis_mem.get("bwarm", 0)
    b_ema = adjust if prev_b is None else prev_b + (0.2 if bw < 10 else 0.05) * (adjust - prev_b)
    _basis_mem.update(t=time.time(), value=round(b_ema, 2), ema=b_ema,
                      valid=bool(adjust), bwarm=bw + 1)

    # live-tick patch: the FORMING candle and the payload price follow the
    # realtime feed (ws book + matched anchor), not the candle source's
    # last close — so /api/data never lags /api/tick and the UI's periodic
    # reload can't drag the displayed price backwards.
    try:
        _tp, _tsrc = (tick_spot(broker=False) if gold else (None, None))
        if _tp and candles:
            _lc = candles[-1]
            _lc["c"] = float(_tp)
            if _tp > _lc["h"]:
                _lc["h"] = float(_tp)
            if _tp < _lc["l"]:
                _lc["l"] = float(_tp)
    except Exception:  # noqa: BLE001
        pass

    if gold:
        ensure_model(tf, candles)
        model = models[tf]
    else:
        model = ml.Model()

    X, idxs, p = ml.build_features(candles)
    c = p["c"]
    n = len(c)
    probs = model.predict_proba(X) if (model.available and len(X)) else np.full(len(X), 0.5)

    scores = np.full(n, np.nan)
    for j in range(len(idxs)):
        i = int(idxs[j])
        rs = rules_score(p, i)
        if model.available:
            scores[i] = MODEL_MIX * (2.0 * probs[j] - 1.0) + RULES_MIX * rs
        else:
            scores[i] = rs

    # Sticky regime (hysteresis): once BUY, stay BUY until the SELL threshold
    # is crossed (and vice versa) — a score hovering around one threshold no
    # longer flip-flops the stance, so alerts only fire on genuine reversals.
    sig = np.zeros(n, int)
    v = ~np.isnan(scores)
    regime = 0
    for i in range(n):
        if v[i] and scores[i] >= BUY_TH:
            regime = 1
        elif v[i] and scores[i] <= SELL_TH:
            regime = -1
        sig[i] = regime

    # historical signal flips (markers on the chart)
    markers = []
    prev = 0
    for i in range(n):
        if sig[i] != 0 and sig[i] != prev:
            markers.append(dict(i=i, t=int(candles[i]["t"]), p=float(c[i]),
                                type="BUY" if sig[i] == 1 else "SELL",
                                score=float(scores[i])))
            prev = sig[i]

    # forward-horizon hit rate of those historical signals
    wins = tot = 0
    for m in markers:
        if m["i"] + ml.HORIZON < n:
            ret = c[m["i"] + ml.HORIZON] - c[m["i"]]
            tot += 1
            if (m["type"] == "BUY" and ret > 0) or (m["type"] == "SELL" and ret < 0):
                wins += 1

    # current signal
    i = n - 1
    cur = int(sig[i])
    cur_score = float(scores[i]) if not np.isnan(scores[i]) else 0.0
    p_up = float(probs[-1]) if len(probs) else 0.5
    sig_type = {1: "BUY", -1: "SELL", 0: "HOLD"}[cur]
    conf = min(0.95, 0.5 + abs(cur_score) * 0.45)
    j = i
    while j >= 0 and int(sig[j]) == cur:
        j -= 1
    since = int(candles[j + 1]["t"]) if j >= 0 else int(candles[0]["t"])

    # 24h / previous-day change
    if tf == "1d":
        ref = c[-2] if n >= 2 else c[0]
    else:
        t0 = candles[-1]["t"]
        ref = None
        for k in range(n - 2, -1, -1):
            if candles[k]["t"] <= t0 - 86400:
                ref = c[k]
                break
        ref = c[0] if ref is None else ref
    chg = float(c[-1] - ref)
    chg_pct = float(chg / ref * 100.0)

    W = WINDOW.get(tf, 180)
    s0 = max(0, n - W)
    win = candles[s0:]
    w_markers = [dict(k=m["i"] - s0, t=m["t"], p=round(m["p"], 2),
                      type=m["type"], score=round(m["score"], 3))
                 for m in markers if m["i"] >= s0][-40:]

    payload = dict(
        tf=tf,
        sym=symbol, symName=sym_cfg["name"], dec=sym_cfg.get("dec", 2),
        version=app_version(),
        source=source, stale=bool(d.get("stale")),
        fetchedAt=int(d["fetchedAt"]), serverNow=int(time.time()),
        price=round(float(c[-1]), 2), chg=round(chg, 2), chgPct=round(chg_pct, 3),
        spot=dict(price=round(spot_price, 2), source=spot_src) if spot_price else None,
        basis=round(adjust, 2),
        candles=[dict(t=int(k["t"]), o=round(k["o"], 2), h=round(k["h"], 2),
                      l=round(k["l"], 2), c=round(k["c"], 2)) for k in win],
        emas=dict(e9=[round(float(x), 2) for x in p["e9"][s0:]],
                  e21=[round(float(x), 2) for x in p["e21"][s0:]]),
        scores=[None if np.isnan(s) else round(float(s), 3) for s in scores[s0:]],
        markers=w_markers,
        hitRate=dict(wins=wins, total=tot),
        signal=dict(type=sig_type, confidence=round(conf, 3),
                    score=round(cur_score, 3), pUp=round(p_up, 3), since=since),
        indicators=indicator_rows(p, i),
        model=dict(available=model.available,
                   name=model.d.get("name", "—"),
                   testAcc=model.d.get("test_acc"),
                   testBalAcc=model.d.get("test_bal_acc"),
                   n=model.d.get("n"), horizon=model.d.get("horizon"),
                   trainedAt=model.d.get("trained_at")),
        alerts=alerts_snapshot(),
        priceAlerts=None,   # filled by refresh() under the state lock
    )
    try:
        payload["fundamentals"] = fundamentals.snapshot()
    except Exception:  # noqa: BLE001
        payload["fundamentals"] = None
    if tf == "15m" and gold:
        try:
            d1h = data.get_candles("60m")
            ent = entries.evaluate(candles, d1h.get("candles"))
            payload["entry"] = ent
            payload["entryStats"] = entries.backtest_stats(candles, "A+")
            setups = entries.recent_setups(candles)
            W = WINDOW.get(tf, 180)
            s0 = max(0, len(candles) - W)
            payload["setups"] = [dict(k=m["i"] - s0, t=m["t"], dir=m["dir"],
                                      p=m["entry"], outcome=m["outcome"])
                                 for m in setups if m["i"] >= s0][-30:]
            payload["sessionStats"] = entries.session_stats(setups)
            payload["radar"] = entries.zone_radar(
                candles, d1h.get("candles"), price=payload["price"])
            # 4H confluence ladder: 15m setup inside a 4H zone = HTF grade
            try:
                d4h = data.get_candles("4h")
                if ent and ent.get("zoneSide"):
                    hz = entries.htf_zone(d4h.get("candles") or [],
                                          ent["zoneSide"], float(payload["price"]))
                    if hz:
                        ent["htf"] = dict(tf="4H",
                                          top=round(float(hz["top"]), 1),
                                          bottom=round(float(hz["bottom"]), 1))
            except Exception:  # noqa: BLE001
                pass
            # liquidity sweeps: live event + honest per-grade stats
            try:
                events = entries.scan_sweeps(candles)
                n15 = len(candles)
                live = [s for s in events if s["i"] >= n15 - 5]
                payload["sweep"] = entries.sweep_view(live[-1]) if live else None

                def _sst(ss):
                    w = sum(1 for s in ss if s["outcome"] == "win")
                    r = sum(s["r"] or 0.0 for s in ss)
                    return dict(setups=len(ss),
                                winRate=round(w / len(ss), 3) if ss else None,
                                avgR=round(r / len(ss), 3) if ss else None)
                payload["sweepStats"] = dict(A=_sst([s for s in events
                                                     if s["grade"] == "A+"]),
                                             B=_sst([s for s in events
                                                     if s["grade"] == "B+"]))
            except Exception:  # noqa: BLE001
                payload["sweep"] = None
                payload["sweepStats"] = None
            maybe_armed_alert(ent)       # arming heads-up: even during events
            blk = _event_blackout()
            if blk:
                _postpone_note(ent, blk)
            else:
                ensure_trade(ent)
                maybe_setup_alert(ent)
                maybe_zone_watch(ent)
                if payload.get("sweep"):
                    maybe_sweep_alert(payload["sweep"])
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            payload["entry"] = None
            payload["entryStats"] = None
    maybe_alert(tf, sig_type, float(c[-1]), cur_score, conf,
                atr=float(p["atr"][i]) if p["atr"][i] else None)
    return payload


# ------------------------------------------------------------------ refresh
class TFState:
    def __init__(self):
        self.payload = None
        self.lock = threading.Lock()


tf_states = {}


def _st(symbol, tf):
    """Per-(symbol, timeframe) build state, created on demand."""
    key = (symbol, tf)
    if key not in tf_states:
        tf_states[key] = TFState()
    return tf_states[key]


_panel_views = {}          # (symbol, tf) -> last time a visitor requested it


def refresh(tf, force=False, allow_build=True, symbol="XAUUSD"):
    if tf not in data.TFS:
        return dict(error=f"unknown timeframe {tf}")
    if symbol not in data.SYMBOLS:
        return dict(error=f"unknown symbol {symbol}")
    st = _st(symbol, tf)
    with st.lock:
        d = data.get_candles(tf, force=force, symbol=symbol)
        if not d.get("candles"):
            if st.payload is not None:
                st.payload["stale"] = True
                return st.payload
            return dict(error="data source unavailable — retrying", tf=tf)
        candles = d.get("candles") or []
        last_bar = candles[-1]["t"] if candles else None
        # rebuild only when it matters: first build, a true refetch with a NEW
        # bar, or >45s since the last build — and never hold the tf lock
        # during the (CPU-heavy) build, so user requests never queue behind it
        need = (st.payload is None or d.get("changed")
                or (d.get("fetchedAt") != getattr(st, "lastFetch", None)
                    and (last_bar != getattr(st, "lastBar", None)
                         or time.time() - getattr(st, "builtAt", 0.0)
                         > REBUILD_S.get(tf, 45))))
        building = getattr(st, "building", False)
    if need and not building and allow_build:
        st.building = True
        try:
            payload = build_payload(tf, d, symbol)
            with st.lock:
                st.lastFetch = d.get("fetchedAt")
                st.lastBar = last_bar
                st.builtAt = time.time()
                prev_price = STATE.get("lastPrice")
                st.payload = payload
                if symbol == "XAUUSD":
                    check_price_alerts(prev_price, payload["price"])
                with STATE_LOCK:
                    STATE["lastPrice"] = payload["price"]
                    _save_state()
        finally:
            st.building = False
    note = get_note()
    desk = get_ai_desk()                      # outside the tf lock — it can
    with st.lock:                           # take a moment when its 4-minute
        with STATE_LOCK:                    # cache expires
            if st.payload is not None:
                st.payload["alerts"] = STATE["alerts"][:30]
                st.payload["priceAlerts"] = STATE["priceAlerts"]
                st.payload["note"] = note
                st.payload["aiDesk"] = desk
                st.payload["tracker"] = dict(live=STATE.get("liveTrade"),
                                             history=STATE.get("tradeHistory", [])[:8])
        return st.payload


_bg_started = False


# ------------------------------------------------------------------ fundamentals
NEWS_FWD_MIN_GAP = 15 * 60        # max one forwarded news per 15 minutes
EVENT_PRE_MIN = 30                # pre-alert this many minutes before an event


def _web_alert(typ, msg, price=None):
    with STATE_LOCK:
        a = dict(id=_next_id(), time=int(time.time()), tf="—", type=typ,
                 price=price, score=0.0, confidence=None, msg=msg)
        STATE["alerts"].insert(0, a)
        del STATE["alerts"][100:]
        _save_state()
    return a


def fundamental_watch():
    """Fundamental alerts: a pre-alert shortly before high-impact USD events
    (CPI, FOMC, NFP...) and forwarding of gold-relevant breaking news from
    WatcherGuru. Both rate-limited and deduped — no spam."""
    now = time.time()
    # ---- high-impact event pre-alerts ----
    try:
        for e in fundamentals.upcoming(hours=1, impacts=("High",)):
            key = f"{e['ts']}:{e['title']}"
            if 0 <= e["minutesTo"] <= EVENT_PRE_MIN and key not in STATE["eventAlerted"]:
                STATE["eventAlerted"].append(key)
                del STATE["eventAlerted"][:-60]
                _save_state()
                when = time.strftime("%H:%M", time.gmtime(e["ts"]))
                fc = (f" · fc {e['forecast']} · prev {e['previous']}"
                      if (e["forecast"] or e["previous"]) else "")
                _web_alert("EVENT", f"{e['title']} ({e['country']}) in ~"
                           f"{e['minutesTo']} min · {when} UTC{fc}")
                _notify(f"📅 HIGH-IMPACT EVENT — {e['title']}\n\n"
                        f"⏰ Starts in ~{e['minutesTo']} min ({when} UTC)\n"
                        f"🌍 Currency: {e['country']}{fc}\n\n"
                        f"💰 XAUUSD — expect volatility")
    except Exception:  # noqa: BLE001
        pass
    # ---- news forwarding (gold-relevant, WatcherGuru + Google News) ----
    try:
        seen = STATE.get("newsSeen") or {"lastTs": 0, "lastFwd": 0.0, "keys": []}
        items = list(fundamentals.watcher_headlines() or [])
        try:
            items += [(ts, ttl) for ts, ttl, _s in fundamentals.google_news(limit=12)]
        except Exception:  # noqa: BLE001
            pass
        if not items:
            return
        newest = max(ts for ts, _t in items)
        if not seen.get("lastTs"):
            # first run after a start: set the baseline, no history dump
            STATE["newsSeen"] = dict(seen, lastTs=newest)
            _save_state()
            return
        if now - (seen.get("lastFwd") or 0) < NEWS_FWD_MIN_GAP:
            STATE["newsSeen"] = dict(seen, lastTs=newest)
            _save_state()
            return

        def _similar(a, b):
            """Same story from a different outlet? (word overlap)"""
            wa = {w for w in a.lower().split() if len(w) > 3}
            wb = {w for w in b.lower().split() if len(w) > 3}
            return bool(wa and wb) and len(wa & wb) / len(wa | wb) > 0.5

        keys = seen.get("keys") or []
        for ts, txt in sorted(items, key=lambda x: -x[0]):
            if ts <= seen["lastTs"] or now - ts > 45 * 60:
                continue                      # old news
            if any(txt == k or _similar(txt, k) for k in keys):
                continue                      # already forwarded / same story
            if fundamentals.gold_relevant(txt):
                keys = keys[-30:] + [txt]
                STATE["newsSeen"] = dict(lastTs=newest, lastFwd=now, keys=keys)
                _save_state()
                _web_alert("NEWS", txt[:140])
                _notify(f"📰 GOLD-RELEVANT NEWS\n\n{txt[:400]}")
                break
        else:
            STATE["newsSeen"] = dict(lastTs=newest)
            _save_state()
    except Exception:  # noqa: BLE001
        pass


_ping_mem = {"t": 0.0}


def _self_keepalive():
    """Ping our own public URL every 5 minutes so the free-tier service
    never spins down (a cold start costs the visitor ~50 seconds)."""
    url = os.environ.get("RENDER_EXTERNAL_URL")
    if not url:
        return                                # not on Render — nothing to do
    now = time.time()
    if now - _ping_mem["t"] < 300:
        return
    _ping_mem["t"] = now
    try:
        import urllib.request
        urllib.request.urlopen(url.rstrip("/") + "/api/health", timeout=20).read()
    except Exception:  # noqa: BLE001
        pass


def _background_loop():
    while True:
        try:
            _self_keepalive()
        except Exception:  # noqa: BLE001
            pass
        try:
            fundamentals.snapshot()           # keep news/macro/calendar caches
        except Exception:  # noqa: BLE001    # warm so user requests never wait
            pass
        for tf in data.TFS:
            try:
                refresh(tf)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.25)
        # keep forex pairs a visitor is actually watching warm (10 min)
        now2 = time.time()
        stale_views = [k for k, ts in _panel_views.items() if now2 - ts > 600]
        for k in stale_views:
            _panel_views.pop(k, None)
        for (sym, tf) in list(_panel_views):
            if sym == "XAUUSD":
                continue
            try:
                refresh(tf, symbol=sym)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.4)
        try:
            update_trade_tracker()
        except Exception:  # noqa: BLE001
            pass
        try:
            fundamental_watch()
        except Exception:  # noqa: BLE001
            pass
        try:
            _asia_bo_watch()
        except Exception:  # noqa: BLE001
            pass
        try:
            _llm_cycle()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(6)


def start_background():
    global _bg_started
    wsfeed.start()                      # real-time websocket gold feed
    if _bg_started:
        return
    _bg_started = True
    try:
        import fcntl
        # one background loop per machine (gunicorn may fork several workers)
        _bg_lock_file = open(os.path.join(BASE, "bg.lock"), "w")
        fcntl.flock(_bg_lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except Exception:  # noqa: BLE001  — another worker already owns the loop
        return
    threading.Thread(target=_background_loop, daemon=True).start()


# ------------------------------------------------------------- trader note
_note_cache = {"t": 0.0, "note": None}


_desk_cache = {"t": 0.0, "desk": None}


def get_ai_desk(force=False):
    """AI Analyst Desk — 8 independent models voting on the same chart.
    Rebuilt every 4 minutes (or on demand)."""
    now = time.time()
    if not force and _desk_cache["desk"] and now - _desk_cache["t"] < 240:
        return _desk_cache["desk"]
    try:
        c15 = data.get_candles("15m").get("candles")
        c60 = data.get_candles("60m").get("candles")
        c1d = data.get_candles("1d").get("candles")
        if not (c15 and c60 and c1d):
            return _desk_cache["desk"]
        spot_price, _ = spot_reference()
        if spot_price:
            adj = c15[-1]["c"] - spot_price
            if 0 < adj < 150:
                c15 = [dict(t=k["t"], o=k["o"] - adj, h=k["h"] - adj,
                            l=k["l"] - adj, c=k["c"] - adj) for k in c15]
                c60 = [dict(t=k["t"], o=k["o"] - adj, h=k["h"] - adj,
                            l=k["l"] - adj, c=k["c"] - adj) for k in c60]
                c1d = [dict(t=k["t"], o=k["o"] - adj, h=k["h"] - adj,
                            l=k["l"] - adj, c=k["c"] - adj) for k in c1d]
        ent = None
        try:
            ent = entries.evaluate(c15, c60)
        except Exception:  # noqa: BLE001
            pass
        sig = {}
        try:
            p15 = tf_states["15m"].payload
            if p15:
                sig = p15.get("signal", {})
        except Exception:  # noqa: BLE001
            pass
        fund = None
        try:
            fund = (fundamentals.snapshot() or {}).get("macro")
        except Exception:  # noqa: BLE001
            pass
        desk = ai_desk.build(c15, c60, c1d, ent, sig, fund)
        desk["llm"] = llm_desk.snapshot()
        _desk_cache.update(t=now, desk=desk)
        return desk
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        return _desk_cache["desk"]


def get_note(force=False):
    """Human-style desk commentary, rebuilt every 4 minutes (or on demand)."""
    now = time.time()
    if not force and _note_cache["note"] and now - _note_cache["t"] < 240:
        return _note_cache["note"]
    try:
        c15 = data.get_candles("15m").get("candles")
        c60 = data.get_candles("60m").get("candles")
        c1d = data.get_candles("1d").get("candles")
        if not (c15 and c60 and c1d):
            return _note_cache["note"]
        # spot-align like the chart
        spot_price, _ = spot_reference()
        raw = c15
        if spot_price:
            adj = raw[-1]["c"] - spot_price
            if 0 < adj < 150:
                c15 = [dict(t=k["t"], o=k["o"] - adj, h=k["h"] - adj,
                            l=k["l"] - adj, c=k["c"] - adj) for k in raw]
                c60 = [dict(t=k["t"], o=k["o"] - adj, h=k["h"] - adj,
                            l=k["l"] - adj, c=k["c"] - adj) for k in c60]
                c1d = [dict(t=k["t"], o=k["o"] - adj, h=k["h"] - adj,
                            l=k["l"] - adj, c=k["c"] - adj) for k in c1d]
        ent = None
        try:
            ent = entries.evaluate(c15, c60)
        except Exception:  # noqa: BLE001
            pass
        price = float(c15[-1]["c"])
        sig = {}
        try:
            p15 = tf_states["15m"].payload
            if p15:
                sig = p15.get("signal", {})
        except Exception:  # noqa: BLE001
            pass
        note = narrative.build_note(c15, c60, c1d, ent, sig, price)
        _note_cache.update(t=now, note=note)
        return note
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        return _note_cache["note"]


def _llm_context():
    """Market snapshot text the external AI models read."""
    bits = []
    try:
        p, _src = tick_spot(broker=False)
        if p:
            bits.append(f"Spot XAU/USD now: {p:,.2f}")
    except Exception:  # noqa: BLE001
        pass
    d = _desk_cache.get("desk") or {}
    if d:
        c = d.get("consensus", {})
        bits.append(f"Local 8-model consensus: {c.get('label')} "
                    f"(score {c.get('score')}, {c.get('bull')} bullish / "
                    f"{c.get('bear')} bearish / {c.get('neutral')} neutral)")
        for a in d.get("analysts", []):
            bits.append(f"- {a['name']}: {a['verdict']} (conf {a['conf']}) — {a['note']}")
    try:
        f = fundamentals.snapshot() or {}
        m = f.get("macro") or {}
        if m.get("dxy"):
            bits.append(f"DXY {m['dxy'].get('chgPct', 0):+.2f}% today")
        if m.get("us10y"):
            bits.append(f"US 10Y yield {m['us10y'].get('chgPct', 0):+.2f}% today")
        nx = f.get("nextHigh") or {}
        if nx.get("title"):
            bits.append(f"Next high-impact USD event: {nx['title']}")
    except Exception:  # noqa: BLE001
        pass
    return "XAUUSD market snapshot:\n" + "\n".join(bits)


def _daily_brief():
    """One morning Telegram card: local desk consensus + external AI reads."""
    d = _desk_cache.get("desk") or {}
    if not d:
        return
    c = d.get("consensus", {})
    lines = ["🌅 AI DAILY BRIEF — XAUUSD", ""]
    try:
        p, _ = tick_spot(broker=False)
        if p:
            lines.append(f"⚡ Price: {p:,.2f}")
    except Exception:  # noqa: BLE001
        pass
    lines.append(f"🧠 Local desk (8 models): {c.get('label')} — "
                 f"{c.get('bull')}B / {c.get('bear')}S / {c.get('neutral')}N")
    for a in [x for x in d.get("analysts", []) if x["verdict"] != "neutral"][:2]:
        lines.append(f"   {a['icon']} {a['name']}: {a['note']}")
    try:
        for s in llm_desk.snapshot().get("seats", []):
            lines.append(f"{s['icon']} {s['name']}: {s['verdict']} — {s['note']}")
    except Exception:  # noqa: BLE001
        pass
    try:
        m = (fundamentals.snapshot() or {}).get("macro") or {}
        if m.get("dxy") or m.get("us10y"):
            lines.append(f"🌍 DXY {m.get('dxy', {}).get('chgPct', 0):+.2f}% · "
                         f"US10Y {m.get('us10y', {}).get('chgPct', 0):+.2f}%")
    except Exception:  # noqa: BLE001
        pass
    lines += ["", "⚠ Analysis only — not financial advice"]
    _notify("\n".join(lines))


def _llm_cycle():
    """Called every background loop: refresh external AI seats (24/7, own
    15-min cadence, non-blocking) and send the one daily morning brief."""
    if llm_desk.configured() and llm_desk.due():
        try:
            txt = _llm_context()
            threading.Thread(target=llm_desk.refresh, args=(txt,),
                             daemon=True).start()
        except Exception:  # noqa: BLE001
            pass
    g = time.gmtime()
    day = time.strftime("%Y-%m-%d", g)
    if g.tm_hour >= 6 and STATE.get("llmBriefDay") != day:
        with STATE_LOCK:
            if STATE.get("llmBriefDay") != day:
                STATE["llmBriefDay"] = day
                _save_state()
        _daily_brief()


# ------------------------------------------------------------------ flask
app = Flask(__name__)


@app.after_request
def add_cors(resp):
    """Allow the sandboxed live-preview iframe (null origin) to call the API."""
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, ngrok-skip-browser-warning"
    return resp


@app.route("/", methods=["OPTIONS"])
@app.route("/<path:path>", methods=["OPTIONS"])
def cors_preflight(path=""):
    return "", 204


@app.route("/")
def index():
    start_background()
    payload = None
    try:
        payload = refresh("60m")
        if payload is not None:
            payload["note"] = get_note()
            payload["aiDesk"] = get_ai_desk()
    except Exception:  # noqa: BLE001
        pass
    return render_template("index.html", payload=payload)


@app.route("/api/data")
def api_data():
    tf = request.args.get("tf", "60m")
    if tf not in data.TFS:
        return jsonify(error=f"unknown timeframe {tf}"), 400
    symbol = request.args.get("symbol", "XAUUSD")
    if symbol not in data.SYMBOLS:
        return jsonify(error=f"unknown symbol {symbol}"), 400
    force = request.args.get("force") == "1"
    start_background()
    _panel_views[(symbol, tf)] = time.time()
    # user requests never trigger heavy builds — the background loop owns
    # them; users always get the cached payload with fresh STATE attached.
    # Exception: a brand-new (symbol, tf) combo builds once so the first
    # visitor isn't left waiting on the loop.
    first_time = _st(symbol, tf).payload is None
    return jsonify(refresh(tf, force=force, allow_build=first_time, symbol=symbol))


@app.route("/api/price-alerts", methods=["POST"])
def add_price_alert():
    try:
        level = float((request.get_json(force=True, silent=True) or {}).get("level", 0))
    except (TypeError, ValueError):
        return jsonify(error="invalid level"), 400
    if not (0 < level < 100000):
        return jsonify(error="level out of range"), 400
    with STATE_LOCK:
        if len(STATE["priceAlerts"]) >= 10:
            return jsonify(error="max 10 price alerts"), 400
        STATE["priceAlerts"].append(dict(id=_next_id(), level=level,
                                         created=int(time.time()), triggeredAt=None))
        _save_state()
        return jsonify(priceAlerts=STATE["priceAlerts"])


@app.route("/api/price-alerts/<int:pid>", methods=["DELETE"])
def del_price_alert(pid):
    with STATE_LOCK:
        STATE["priceAlerts"] = [x for x in STATE["priceAlerts"] if x["id"] != pid]
        _save_state()
        return jsonify(priceAlerts=STATE["priceAlerts"])


_watch_cache = {"t": 0.0, "items": None}


@app.route("/api/watchlist")
def api_watchlist():
    """Live price + day change for every symbol (watchlist panel)."""
    now = time.time()
    if _watch_cache["items"] and now - _watch_cache["t"] < 20:
        return jsonify(asOf=_watch_cache["t"], items=_watch_cache["items"])
    items = []
    try:
        p, _src = tick_spot()
        if p:
            chg = None
            try:
                pl = tf_states.get(("XAUUSD", "60m")).payload
                if pl:
                    chg = pl.get("chg")
            except Exception:  # noqa: BLE001
                pass
            items.append(dict(sym="XAUUSD", name="Gold", dec=2, price=round(p, 2),
                              chg=round(chg, 2) if chg is not None else None,
                              chgPct=round(chg / (p - chg) * 100, 2)
                              if chg is not None and p != chg else None))
    except Exception:  # noqa: BLE001
        pass
    for sym, cfg in data.SYMBOLS.items():
        if sym == "XAUUSD":
            continue
        try:
            q = data.get_quote(sym, max_age=50)
            if not q:
                continue
            p, _ts, prev = q[0], q[1], q[2]
            chg = p - prev if prev else None
            items.append(dict(sym=sym, name=cfg["name"], dec=cfg.get("dec", 2),
                              price=round(p, cfg.get("dec", 2)),
                              chg=round(chg, cfg.get("dec", 2)) if chg is not None else None,
                              chgPct=round(chg / prev * 100, 2) if chg is not None and prev else None))
        except Exception:  # noqa: BLE001
            continue
    if items:
        _watch_cache.update(t=now, items=items)
    return jsonify(asOf=now, items=items)


@app.route("/api/tick")
def api_tick():
    symbol = request.args.get("symbol", "XAUUSD")
    if symbol not in data.SYMBOLS:
        return jsonify(error=f"unknown symbol {symbol}"), 400
    if symbol != "XAUUSD":
        q = data.get_quote(symbol)
        if not q:
            return jsonify(price=None, serverNow=int(time.time()),
                           source="unavailable", sym=symbol)
        p, ts = q[0], q[1]
        age = time.time() - ts
        dec = data.SYMBOLS[symbol].get("dec", 2)
        return jsonify(price=round(p, dec), serverNow=int(time.time()),
                       source=("market closed" if age > 3600 else "live quote"),
                       sym=symbol)
    price, src = tick_spot()
    return jsonify(price=round(price, 2) if price else None,
                   source=src, serverNow=int(time.time()),
                   version=app_version())


@app.route("/api/entry")
def api_entry():
    """Tactical 15m entry setup — independent of the selected chart tab."""
    start_background()
    p = refresh("15m")
    return jsonify(entry=p.get("entry"), entryStats=p.get("entryStats"),
                   price=p.get("price"), fetchedAt=p.get("fetchedAt"))


@app.route("/api/stream")
def api_stream():
    """Server-Sent Events: push every price update to the browser instantly.
    No browser polling — the server polls the quote once (globally cached)
    and fans the value out to every connected client."""
    def gen():
        started = time.time()
        last_sent = None
        last_yield = time.time()
        while True:
            try:
                p, src = tick_spot(max_age=0.05)
            except Exception:  # noqa: BLE001
                p, src = None, None
            now = time.time()
            if p is not None and (last_sent is None or abs(p - last_sent) >= 0.01):
                last_sent = p
                last_yield = now
                yield ("data: " + json.dumps(
                    dict(price=round(p, 2), source=src, t=int(now))) + "\n\n")
            if now - started > 3300:      # ~55 min; EventSource auto-reconnects
                break
            if now - last_yield > 14:     # idle keepalive for proxies
                last_yield = now
                yield ": keepalive\n\n"
            # wake the INSTANT the exchange book changes (event-driven push)
            wsfeed.wait_for_change(15.0)
    resp = Response(gen(), mimetype="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


@app.route("/api/wait")
def api_wait():
    """Realtime long-poll: the request stays open until the live price actually
    MOVES (or a 20s keepalive). The exchange book wakes us instantly, so the
    only latency left is network travel time — no timers in the chain."""
    try:
        last = float(request.args.get("last", "0"))
    except (TypeError, ValueError):
        last = 0.0
    deadline = time.time() + 20.0
    while True:
        p, src = tick_spot(max_age=0.05)
        now = time.time()
        if p is not None and abs(p - last) >= 0.01:
            return jsonify(price=round(p, 2), source=src, t=int(now),
                           version=app_version())
        if now >= deadline:
            p, src = tick_spot()
            return jsonify(price=round(p, 2) if p else None, source=src,
                           t=int(now), version=app_version(), keepalive=True)
        wsfeed.wait_for_change(min(0.5, deadline - now))


@app.route("/api/broker-sync", methods=["POST"])
def api_broker_sync():
    """Match the displayed prices to the user's broker: send the price you
    see at your broker (or a raw offset) and every number on the desk shifts
    to that feed. |offset| is capped at $25."""
    j = request.get_json(force=True, silent=True) or {}
    offset = None
    try:
        if j.get("offset") is not None:
            offset = float(j["offset"])
        elif j.get("price") is not None:
            p, _s = _tick_spot_raw()
            if p:
                offset = float(j["price"]) - p
    except (TypeError, ValueError):
        offset = None
    if offset is None or abs(offset) > 25:
        return jsonify(error="send {price: what your broker shows} or {offset}"), 400
    with STATE_LOCK:
        STATE["brokerOffset"] = round(offset, 2)
        _save_state()
    p, _s = tick_spot()
    return jsonify(offset=round(offset, 2), price=p)


@app.route("/api/broker-sync", methods=["DELETE"])
def api_broker_sync_clear():
    with STATE_LOCK:
        STATE["brokerOffset"] = 0.0
        _save_state()
    return jsonify(offset=0.0)


@app.route("/api/health")
def health():
    return jsonify(ok=True, tfs=list(data.TFS),
                   telegram=bool(TG_TOKEN and TG_CHAT))


# Start the realtime feed + background loop at import time so WSGI servers
# (gunicorn in the Dockerfile) get it too — start_background() is idempotent
# and an flock keeps it to ONE loop per machine even with multiple workers.
start_background()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, threaded=True, debug=False)
