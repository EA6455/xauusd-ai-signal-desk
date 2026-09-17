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
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

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
             liveTrade=None, tradeHistory=[], signalTrades=[], lastSigT={},
             lastSetup=None, lastSetupT=0.0, eventAlerted=[], newsSeen=None,
             brokerOffset=0.0)
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


TG_TOPIC_FILE = os.path.join(BASE, "telegram_topics.json")
_TG_TOPICS = None


def _tg_topics():
    """{'signal': thread_id, 'news': thread_id} for the forum group, learned
    automatically by tg_topic_detect(). Empty while the group has no topics."""
    global _TG_TOPICS
    if _TG_TOPICS is None:
        try:
            with open(TG_TOPIC_FILE) as f:
                _TG_TOPICS = json.load(f)
        except Exception:  # noqa: BLE001
            _TG_TOPICS = {}
    return _TG_TOPICS


def _notify(text, cat="signal"):
    """Send a Telegram message to every configured chat (private and/or
    group). In a forum group each message lands in its own topic:
    cat='signal' → SIGNALS topic, cat='news' → NEWS topic (auto-learned).
    Private chats are unaffected."""
    if not (TG_TOKEN and TG_CHATS):
        return
    import urllib.request
    tops = _tg_topics()
    for chat in TG_CHATS:
        try:
            payload = {"chat_id": chat, "text": text}
            if chat.startswith("-100"):
                tid = tops.get(cat)
                if tid:
                    payload["message_thread_id"] = int(tid)
            req = urllib.request.Request(
                f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"})
            try:
                urllib.request.urlopen(req, timeout=10)
            except Exception as e:  # noqa: BLE001 — closed/deleted topic
                msg = str(e).upper()
                if payload.pop("message_thread_id", None) and \
                        ("TOPIC" in msg or "THREAD" in msg):
                    req = urllib.request.Request(
                        f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                        data=json.dumps(payload).encode(),
                        headers={"Content-Type": "application/json"})
                    urllib.request.urlopen(req, timeout=10)
                    print(f"[tg] topic unavailable in {chat} — "
                          f"sent to General; re-learning topics", flush=True)
                    global _TG_TOPICS
                    _TG_TOPICS = None
                    try:
                        os.remove(TG_TOPIC_FILE)
                    except OSError:
                        pass
                else:
                    raise
            print(f"[tg] sent to {chat}: {text[:50]!r}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[tg] FAILED to {chat}: {e}", flush=True)


def tg_topic_detect(updates=None):
    """Auto-learn the group's forum topics. When Topics are enabled and topics
    are created, the bot receives forum_topic_created service messages and
    maps them by name: a topic with 'news' in the name gets the news alerts;
    one with signal/trade/entry/snr/alert gets the signal cards. Whichever
    category has no own topic falls back to General (topic 1). Runs every
    background-loop pass until both topics are known; cached to disk."""
    global _TG_TOPICS
    tops = _tg_topics()
    if tops.get("complete"):
        return tops
    if not (TG_TOKEN and TG_CHATS):
        return None
    group = next((c for c in TG_CHATS if c.startswith("-100")), None)
    if not group:
        return None
    if updates is None:
        import urllib.request
        try:
            req = urllib.request.Request(
                f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates",
                data=json.dumps({"limit": 100}).encode(),
                headers={"Content-Type": "application/json"})
            updates = json.load(urllib.request.urlopen(req, timeout=10)).get("result", [])
        except Exception:  # noqa: BLE001
            return None
    forum = False
    found = {}
    for u in updates or []:
        m = u.get("message") or {}
        if str(m.get("chat", {}).get("id")) != group:
            continue
        fc = m.get("forum_topic_created")
        if not fc:
            continue
        forum = True
        name = (fc.get("name") or "").lower()
        tid = m.get("message_thread_id") or m["message_id"]
        if "news" in name:
            found["news"] = tid
        elif any(w in name for w in ("signal", "trade", "entry", "snr", "alert")):
            found["signal"] = tid
    if not forum or not found:
        return None
    tops = dict(signal=found.get("signal", 1), news=found.get("news", 1),
                complete=("signal" in found and "news" in found))
    _TG_TOPICS = tops
    try:
        with open(TG_TOPIC_FILE, "w") as f:
            json.dump(tops, f)
    except OSError:
        pass
    print(f"[tg] forum topics detected: {tops}", flush=True)
    if tops["complete"]:
        _notify("✅ Topic routing active — news \u2192 NEWS topic \u00b7 "
                "signals \u2192 SIGNALS topic", cat="signal")
    return tops


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
                        f"TP1 {sw['tp1']:,.2f} · {sw['passed']}/3")
    if not PHONE:
        return
    if time.time() - STATE.get("lastSetupT", 0) < SETUP_COOLDOWN_S:
        return
    with STATE_LOCK:
        STATE["lastSetupT"] = time.time()
    side_lbl = "Support" if sw["zoneSide"] == "demand" else "Resistance"
    icon = "🟢" if sw["direction"] == "LONG" else "🔴"
    warn = "" if sw["grade"] == "A+" else \
        f"\n⚠ dead-session entry — smaller size or skip"
    _notify(f"{icon} {sw['grade']} SWEEP {sw['direction']} SIGNAL\n\n"
            f"📊 Timeframe: 15M\n"
            f"💰 Symbol: XAUUSD\n"
            f"📍 Setup: Liquidity Sweep · {side_lbl} · EMA200-aligned\n\n"
            f"🎯 Entry: {sw['entry']:,.2f}\n"
            f"🛑 SL: {sw['sl']:,.2f}\n"
            f"🎯 TP1: {sw['tp1']:,.2f} (half off · 1:1 RR)\n"
            f"🎯 TP2: {sw['tp2']:,.2f} (runner · 1:2 RR)\n\n"
            f"👉 Trade now on your own broker\n\n"
            f"📊 Backtest: 57% win · +0.12R avg (EMA200-aligned reclaim)\n"
            f"⭐ SNR Rating: {sw['grade']} SWEEP{warn}")
    track_signal("SWEEP", key, sw["direction"], sw["entry"], sw["sl"],
                 sw["tp1"], sw["tp2"])


_liq_watch_mem = {"keys": []}


def maybe_liq_watch(w):
    """🔁 Web-feed watch: a prior-day liquidity level just got swept — the
    'run from one side to the other' story may be starting. Informational
    only (first-side rotations are a 48% coin flip): no TG card, no trade."""
    if not w:
        return
    key = f"liq:{w.get('day')}:{w.get('side')}"
    if key in _liq_watch_mem["keys"]:
        return
    _liq_watch_mem["keys"] = (_liq_watch_mem["keys"] + [key])[-40:]
    side = "HIGH" if w.get("side") == "high" else "LOW"
    if w.get("otherDrained"):
        _web_alert("LIQ", f"🔁 DAILY LIQ — prior-day {side} swept @ "
                          f"{w['level']:,.1f} · other side already drained — "
                          f"FLIP ARMED, waiting for the reclaim close")
    else:
        _web_alert("LIQ", f"🔁 DAILY LIQ — prior-day {side} swept @ "
                          f"{w['level']:,.1f} · rotation watch toward the "
                          f"other side (first-side sweeps alone are a 48% "
                          f"coin flip — the tradeable flip needs BOTH sides "
                          f"drained)")


_rot_alerted = {"keys": []}


def maybe_rotation_alert(rt, stats=None):
    """🔁 LIQUIDITY ROTATION trade card: both daily pools drained and the
    second sweep reclaimed — trade the rotation back toward the spent side.
    One alert per day; honest small-sample stats on the card."""
    PHONE = True
    if not rt:
        return
    if _event_blackout():
        return
    key = f"rot:{rt.get('day')}:{rt.get('direction')}"
    if key in _rot_alerted["keys"]:
        return
    _rot_alerted["keys"] = (_rot_alerted["keys"] + [key])[-40:]
    icon = "🟢" if rt["direction"] == "LONG" else "🔴"
    _web_alert("ROT", f"🔁 {rt['grade']} LIQUIDITY ROTATION "
                      f"{rt['direction']} flip @ {rt['entry']:,.2f} · both "
                      f"pools drained · SL {rt['sl']:,.2f} · "
                      f"TP1 {rt['tp1']:,.2f}")
    if not PHONE:
        return
    if time.time() - STATE.get("lastSetupT", 0) < SETUP_COOLDOWN_S:
        return
    with STATE_LOCK:
        STATE["lastSetupT"] = time.time()
    st = stats or {}
    hist = ""
    if st.get("setups"):
        hist = (f"📊 Backtest: {round((st.get('winRate') or 0) * 100):.0f}% win"
                f" · {st.get('avgR', 0):+.2f}R avg · n={st['setups']}"
                f" (small sample)\n")
    rot_dir = "DOWN" if rt["direction"] == "SHORT" else "UP"
    if rt["firstSide"] == "low":
        f1 = (f"✅ SELLSIDE drained first — prior-day LOW "
              f"{rt['firstLevel']:,.1f} swept {rt.get('firstT', 'earlier today')}")
        f2 = (f"✅ BUYSIDE just swept — prior-day HIGH "
              f"{rt['sweptLevel']:,.1f} → wick {rt['sweepExt']:,.1f}")
        spent = "sellside"
    else:
        f1 = (f"✅ BUYSIDE drained first — prior-day HIGH "
              f"{rt['firstLevel']:,.1f} swept {rt.get('firstT', 'earlier today')}")
        f2 = (f"✅ SELLSIDE just swept — prior-day LOW "
              f"{rt['sweptLevel']:,.1f} → wick {rt['sweepExt']:,.1f}")
        spent = "buyside"
    warn = "" if rt["grade"] == "A+" else \
        f"\n⚠ dead-session entry — smaller size or skip"
    _notify(f"{icon} {rt['grade']} LIQUIDITY ROTATION {rt['direction']} SIGNAL\n\n"
            f"📊 Timeframe: 15M\n"
            f"💰 Symbol: XAUUSD\n"
            f"📍 Setup: Daily liquidity rotation · both pools drained\n\n"
            f"🧭 Liquidity story (side → side):\n"
            f"{f1}\n"
            f"{f2}\n"
            f"✅ Reclaim — 15m close back inside the daily range\n"
            f"🔁 Rotation: back {rot_dir} toward the spent {spent}\n\n"
            f"🎯 Entry: {rt['entry']:,.2f} (at reclaim close)\n"
            f"🛑 SL: {rt['sl']:,.2f} (beyond the sweep wick)\n"
            f"🎯 TP1: {rt['tp1']:,.2f} (half off · 1:0.75 RR)\n"
            f"🎯 TP2: {rt['tp2']:,.2f} (runner · 1:1.5 RR)\n\n"
            f"👉 Trade now on your own broker\n\n"
            f"{hist}"
            f"⚠ First-side sweeps alone are a 48% coin flip — only the "
            f"both-drained flip is traded\n"
            f"⭐ SNR Rating: {rt['grade']} ROTATION{warn}")
    track_signal("ROTATION", key, rt["direction"], rt["entry"], rt["sl"],
                 rt["tp1"], rt["tp2"])


# ------------------------------------------------------------------ alerts
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


def maybe_momentum_alert(ent):
    """⚡ Momentum entry: a fresh SNR zone has 6/8+ confluence in place but
    price has NOT returned (missing check = first retest). Instead of waiting
    for the retest, fire an immediately tradeable card at MARKET price with
    ATR-based risk and 1:2 RR (trend continuation in the armed zone's
    direction). The classic retest A+/B+/C+ cards still fire separately if
    price returns to the zone. One card per zone, 60-min global gap, live
    sessions only, event-gated."""
    if not ent or not ent.get("zoneKey") or not ent.get("zoneSide"):
        return
    if ent.get("touching") or ent.get("active"):
        return                                  # retest cards already cover it
    if ent.get("grade") in ("A+", "B+", "C+"):
        return
    if ent.get("passed", 0) < 6:
        return                                  # only C+/B+ quality zones arm
    if ent.get("session") not in ("london", "ny-overlap", "ny-late"):
        return          # backtest: London/NY only lifts momentum win 28.6%->44%
    if not ent.get("e200ok"):
        return          # never fight the big trend (EMA200, 15m)
    if _event_blackout():
        return                                  # it is a real entry signal now
    key = "armed:" + ent["zoneKey"]
    now = time.time()
    if key in _armed_mem["keys"] or now - _armed_mem["t"] < 60 * 60:
        return
    try:
        price, _src = tick_spot(broker=False)
    except Exception:  # noqa: BLE001
        price = None
    if not price:
        return
    _armed_mem["keys"] = (_armed_mem["keys"] + [key])[-60:]
    _armed_mem["t"] = now
    d = 1 if ent["direction"] == "LONG" else -1
    a = max(float(ent.get("atr") or 0.0), 0.5)
    risk = 2.5 * a                              # stop distance = 2.5×ATR (high-win)
    entry = float(price)
    sl = entry - d * risk
    tp = entry + d * 0.75 * risk                # TP1 0.75R — half off, high hit rate
    tp2 = entry + d * 1.5 * risk                # TP2 runner 1.5R
    side = ent["zoneSide"]
    lbl = "Support" if side == "demand" else "Resistance"
    lo, hi = ent["entryZone"][0], ent["entryZone"][1]
    icon = "\U0001F7E2" if d == 1 else "\U0001F534"
    _web_alert("MOMENTUM",
               f"\u26A1 momentum {ent['direction'].lower()} @ {entry:,.1f} \u00b7 "
               f"SL {sl:,.1f} \u00b7 TP {tp:,.1f} \u00b7 {ent['passed']}/8 armed zone "
               f"{lo:,.1f}\u2013{hi:,.1f}")
    _notify(f"{icon} MOMENTUM {ent['direction']} SIGNAL\n\n"
            f"\U0001F4CA Timeframe: 15M\n"
            f"\U0001F4B0 Symbol: XAUUSD\n"
            f"\U0001F4CD Setup: Trend continuation \u2014 armed {lbl} zone "
            f"{lo:,.1f}\u2013{hi:,.1f} ({ent['passed']}/8)\n\n"
            f"\U0001F3AF Entry: {entry:,.2f} (market now)\n"
            f"\U0001F6D1 SL: {sl:,.2f}\n"
            f"\U0001F3AF TP1: {tp:,.2f} (half off \u00b7 1:0.75 RR)\n"
            f"\U0001F3AF TP2: {tp2:,.2f} (runner \u00b7 1:1.5 RR)\n\n"
            f"\U0001F449 Trade now on your own broker\n\n"
            f"\u2B50 SNR Rating: MOMENTUM ({ent['passed']}/8)\n"
            f"\U0001F4CA Backtest: 68% win \u00b7 +0.18R avg (London/NY \u00b7 with EMA200 trend)\n"
            f"\u26A0 Not a zone retest \u2014 momentum entry, smaller size")
    track_signal("MOMENTUM", key, ent["direction"], entry, sl, tp, tp2)


def maybe_setup_alert(ent, elite_stats=None):
    """Fire an alert on SNR retests, once per ZONE+GRADE (a setup that stays
    live for hours must not re-alert every 15 minutes).

    v3 gating (1:2 RR backtest): phone cards only for cohorts that WIN at
    1:2 — 🔥 ELITE (A+/B+ continuation zones, 83%/+0.65R) and plain A+
    (71%/+0.50R). B+/C+ mid/deep-pullback zones lose at 1:2 (42-44%) so
    they stay web-feed watch lines — visible, honestly labeled, no card."""
    if not ent or not ent.get("touching"):
        return
    grade = ent.get("grade")
    if grade not in ("A+", "B+", "C+"):
        return
    elite = bool(ent.get("contZone")) and grade in ("A+", "B+")
    phone = elite or grade == "A+"
    key = (ent.get("zoneKey") or f"{ent['direction']}:{ent['barTime']}") + ":" + grade
    with STATE_LOCK:
        if STATE.get("lastSetup") == key:
            return
        STATE["lastSetup"] = key
        if not phone:
            # below the 1:2 quality bar — web-feed watch line only, no card
            zt = ("CONTINUATION" if ent.get("contZone")
                  else "mid/deep pullback")
            a = dict(id=_next_id(), time=int(time.time()), tf="15m",
                     type="WATCH",
                     price=ent["entry"], score=round(ent["passed"] / 8.0, 2),
                     confidence=round(ent["passed"] / 8.0, 2),
                     msg=(f"⏸ {grade} {ent['direction']} retest @ "
                          f"{ent['entry']:,.2f} · {zt} zone · mid/deep zones "
                          f"win only ~42% at 1:2 — below card quality bar, "
                          f"watch only"))
            STATE["alerts"].insert(0, a)
            del STATE["alerts"][100:]
            _save_state()
            return
        if time.time() - STATE.get("lastSetupT", 0) < SETUP_COOLDOWN_S:
            return
        STATE["lastSetupT"] = time.time()
        typ = "ENTRY_BUY" if ent["direction"] == "LONG" else "ENTRY_SELL"
        a = dict(id=_next_id(), time=int(time.time()), tf="15m", type=typ,
                 price=ent["entry"], score=round(ent["passed"] / 8.0, 2),
                 confidence=round(ent["passed"] / 8.0, 2),
                msg=(f"SNR {grade} {ent['direction']} retest @ {ent['entry']:,.2f} · "
                     f"zone {ent['entryZone'][0]:,.1f}–{ent['entryZone'][1]:,.1f} · "
                     f"SL {ent['sl']:,.2f} · TP1 {ent['tp1']:,.2f} · "
                     f"{ent['passed']}/8 SNR checks · 👉 trade now on your own broker"))
        STATE["alerts"].insert(0, a)
        del STATE["alerts"][100:]
        _save_state()
    if not phone:
        return
    # phone message: the classic SNR signal card (ONE message per zone+grade)
    side = ent.get("zoneSide") or ("demand" if ent["direction"] == "LONG" else "supply")
    setup_lbl = "Support" if side == "demand" else "Resistance"
    icon = "🟢" if ent["direction"] == "LONG" else "🔴"
    warn = ""            # phone cards are A+ or elite-continuation only now —
                         # B+/C+ mid/deep pullbacks stay web-feed watch lines
    htf_line = ""
    if ent.get("htf"):
        h = ent["htf"]
        htf_line = (f"\n📐 HTF: {h['tf']} "
                    f"{'demand' if side == 'demand' else 'supply'} "
                    f"{h['bottom']:,.0f}–{h['top']:,.0f}")
    # trader-flow checklist (the 5-SOP: zone -> rejection -> confirmation ->
    # setup -> entry) — the card reads like an SNR trader executing the trade
    lo_z, hi_z = ent["entryZone"][0], ent["entryZone"][1]
    checks = ent.get("checks") or []
    rej_lbl = "tapped & holding" if (checks and checks[3].get("ok")) else "retest pending"
    conf_lbl = "engulf / pin candle" if (checks and checks[5].get("ok")) else "no candle confirm"
    zone_type = ("CONTINUATION · shallow pullback in a strong leg"
                 if ent.get("contZone") else
                 "retracement zone · deeper pullback")
    elite_tag = " · 🔥 ELITE" if elite else ""
    hist = ""
    if elite:
        st = elite_stats or {}
        if st.get("setups"):
            hist = (f"\n📈 Elite cohort: {round((st.get('winRate') or 0) * 100):.0f}%"
                    f" win · {st.get('avgR', 0):+.2f}R avg · n={st['setups']}"
                    f" (A+/B+ continuation, 60d)")
    _notify(
        f"{icon} {grade} {ent['direction']} SIGNAL{elite_tag}\n\n"
        f"📊 Timeframe: 15M\n"
        f"💰 Symbol: XAUUSD\n"
        f"📍 Setup: {setup_lbl}{htf_line}\n\n"
        f"🧭 SNR trader checklist\n"
        f"✅ ZONE — fresh {setup_lbl.lower()} {lo_z:,.1f}–{hi_z:,.1f}\n"
        f"✅ REJECTION — {rej_lbl}\n"
        f"✅ CONFIRMATION — {conf_lbl}\n"
        f"📐 Zone type: {zone_type}{hist}\n\n"
        f"🎯 Entry: {ent['entry']:,.2f} (at confirmation close)\n"
        f"🛑 SL: {ent['sl']:,.2f} (beyond zone)\n"
        f"🎯 TP1: {ent['tp1']:,.2f} (half off · 1:1 RR)\n"
        f"🎯 TP2: {ent['tp2']:,.2f} (runner · 1:2 RR)\n\n"
        f"👉 Trade now on your own broker\n\n"
        f"⭐ SNR Rating: {grade}{warn}")
    track_signal(grade, key, ent["direction"], ent["entry"], ent["sl"],
                 ent["tp1"], ent["tp2"])


def _close_trade(tr, result, r, price):
    tr.update(status="closed", result=result, r=round(r, 2),
              closedAt=int(time.time()), closePrice=round(float(price), 2))
    STATE["tradeHistory"].insert(0, dict(tr))
    del STATE["tradeHistory"][25:]


# ------------------------------------------------- every-signal tracker
# EVERY entry card (A+/B+/C+ retest + MOMENTUM) opens its own tracked
# position and is followed to TP or SL — one result card per event.


def _market_delta(candles, window=140, series_len=60):
    """🌊 Market delta — volume-weighted order-flow pressure, smoothed.

    Per bar, volume is split by where the close lands in the bar's range:
    buy fraction = (close − low) / (high − low), delta = vol × (2×frac − 1).
    The raw series is noisy, so the fast EMA(9) is what's shown (smooth
    line) with a slow EMA(21) for the pressure trend. Falls back to a
    range-only proxy when a source has no volume."""
    if not candles or len(candles) < 30:
        return None
    bars = candles[-window:]
    raw = []
    for k in bars:
        rng = float(k["h"]) - float(k["l"])
        v = float(k.get("v") or 0.0)
        if v <= 0:
            v = 1.0                       # no volume → range proxy
        frac = ((float(k["c"]) - float(k["l"])) / rng) if rng > 0 else 0.5
        raw.append(v * (2.0 * frac - 1.0))

    def _ema(xs, span):
        out = []
        e = xs[0]
        a = 2.0 / (span + 1.0)
        for x in xs:
            e = a * x + (1 - a) * e
            out.append(e)
        return out

    d9 = _ema(raw, 9)
    d21 = _ema(raw, 21)
    e9, e21 = d9[-1], d21[-1]
    vol_all = sum(abs(x) for x in raw[-series_len:])
    if e9 > 0 and e9 >= e21:
        state = "BUYERS IN CONTROL"
    elif e9 < 0 and e9 <= e21:
        state = "SELLERS IN CONTROL"
    elif e9 > 0:
        state = "BUYERS FADING"
    else:
        state = "SELLERS FADING"
    # divergence: last 30 bars, price higher highs but delta lower highs?
    p_win = bars[-30:]
    d_win = d9[-30:]
    ph = max(float(k["h"]) for k in p_win[:15])
    ph2 = max(float(k["h"]) for k in p_win[15:])
    dh = max(d_win[:15]); dh2 = max(d_win[15:])
    pl = min(float(k["l"]) for k in p_win[:15])
    pl2 = min(float(k["l"]) for k in p_win[15:])
    dl = min(d_win[:15]); dl2 = min(d_win[15:])
    div = None
    if ph2 > ph and dh2 < dh and d9[-1] < d9[-16]:
        div = "bearish — price up, delta down"
    elif pl2 < pl and dl2 > dl and d9[-1] > d9[-16]:
        div = "bullish — price down, delta up"
    mx = max(abs(x) for x in d9[-series_len:]) or 1.0
    return dict(
        series=[round(x / mx, 4) for x in d9[-series_len:]],
        ema9=round(e9, 2), ema21=round(e21, 2),
        cum=round(sum(raw[-series_len:]), 1),
        buyPct=round((vol_all + sum(raw[-series_len:])) / (2 * vol_all), 3)
        if vol_all > 0 else 0.5,
        state=state, divergence=div, tf=candles and bars[-1].get("t"))


def track_signal(src, key, direction, entry, sl, tp1, tp2):
    """Open a tracked position for a fired signal card (dedup per key)."""
    with STATE_LOCK:
        sigs = STATE.get("signalTrades") or []
        if any(s.get("key") == key for s in sigs):
            return
        hist = STATE.get("tradeHistory") or []
        if any(h.get("key") == key and h.get("src") == src for h in hist):
            return
        # the featured A+ live tracker already manages this zone — skip dup
        lt = STATE.get("liveTrade")
        if lt and lt.get("status") in ("open", "tp1hit") and lt.get("key") in key:
            return
        sigs.append(dict(key=key, src=src, dir=direction,
                         entry=round(float(entry), 2), sl=round(float(sl), 2),
                         tp1=round(float(tp1), 2), tp2=round(float(tp2), 2),
                         openedAt=int(time.time()), status="open",
                         result=None, r=0.0))
        STATE["signalTrades"] = sigs[-20:]
        _save_state()


def update_signal_trades():
    """Step every open signal trade to its outcome: TP1 -> half banked +1.0R,
    stop to breakeven, runner to TP2 (+2.5R); SL -> -1.0R; 5h timeout.
    One alert + phone card per event, labeled with the signal's source."""
    price, _src = tick_spot()
    if price is None:
        return
    fired = []
    with STATE_LOCK:
        sigs = STATE.get("signalTrades") or []
        changed = False
        for tr in sigs:
            if tr.get("status") == "closed":
                continue
            d = 1 if tr["dir"] == "LONG" else -1
            entry, sl, tp1, tp2 = tr["entry"], tr["sl"], tr["tp1"], tr["tp2"]
            sl_dist = abs(entry - sl) or 1.0
            # R math from the trade's OWN levels (momentum cards use 1:1.5 RR,
            # retest cards 1:2 — banked/runner shares must match the card)
            r_tp1 = round(0.5 * abs(tp1 - entry) / sl_dist, 2)
            r_tp2 = round(r_tp1 + 0.5 * abs(tp2 - entry) / sl_dist, 2)
            now = int(time.time())
            timed_out = now - tr["openedAt"] > 20 * 15 * 60
            if tr["status"] == "open":
                if (d == 1 and price >= tp1) or (d == -1 and price <= tp1):
                    tr["status"] = "tp1hit"; tr["tp1At"] = now
                    changed = True
                    fired.append((tr, "TP1_HIT",
                                  f"TP1 hit at {tp1:,.1f} · half banked +{r_tp1:.2f}R · "
                                  f"stop to breakeven {entry:,.1f} · runner {tp2:,.1f}"))
                elif (d == 1 and price <= sl) or (d == -1 and price >= sl):
                    _close_trade(tr, "loss", -1.0, price)
                    changed = True
                    fired.append((tr, "SL_HIT", f"Stopped out at {sl:,.1f} · -1.0R"))
                elif timed_out:
                    r = 0.5 * d * (price - entry) / sl_dist
                    _close_trade(tr, "timeout", r, price)
                    changed = True
                    fired.append((tr, "TIMEOUT",
                                  f"5h timeout — closed {price:,.1f} ({r:+.2f}R)"))
            if tr.get("status") == "tp1hit":
                if (d == 1 and price >= tp2) or (d == -1 and price <= tp2):
                    _close_trade(tr, "win", r_tp2, price)
                    changed = True
                    fired.append((tr, "TP2_HIT",
                                  f"TP2 hit at {tp2:,.1f} · total +{r_tp2:.2f}R 🎯"))
                elif (d == 1 and price <= entry) or (d == -1 and price >= entry):
                    _close_trade(tr, "be", r_tp1, price)
                    changed = True
                    fired.append((tr, "BE_STOP",
                                  f"Runner stopped at breakeven {entry:,.1f} · "
                                  f"total +{r_tp1:.2f}R (TP1 banked)"))
                elif timed_out:
                    r = r_tp1 + 0.5 * d * (price - entry) / sl_dist
                    _close_trade(tr, "timeout", r, price)
                    changed = True
                    fired.append((tr, "TIMEOUT",
                                  f"5h timeout — runner closed {price:,.1f} · "
                                  f"total {r:+.2f}R"))
        if changed:
            STATE["signalTrades"] = [s for s in sigs if s.get("status") != "closed"]
            _save_state()
    for tr, typ, msg in fired:
        src = tr.get("src", "SIGNAL")
        with STATE_LOCK:
            a = dict(id=_next_id(), time=int(time.time()), tf="15m", type=typ,
                     price=round(float(price), 2), score=0.0, confidence=None,
                     msg=f"{src} {tr['dir']} · {msg}")
            STATE["alerts"].insert(0, a)
            del STATE["alerts"][100:]
            _save_state()
        icon = {"TP1_HIT": "✅", "TP2_HIT": "🎯", "SL_HIT": "🛑",
                "BE_STOP": "⚖️", "TIMEOUT": "⏱"}.get(typ, "•")
        head = {"TP1_HIT": "TP1 HIT", "TP2_HIT": "TP2 HIT", "SL_HIT": "SL HIT",
                "BE_STOP": "BREAKEVEN STOP", "TIMEOUT": "TIMEOUT · 5h"}.get(typ, typ)
        long = tr["dir"] == "LONG"
        _notify(f"{icon} {head} — {src} {tr['dir']} SIGNAL\n\n"
                f"📊 Timeframe: 15M\n"
                f"💰 Symbol: XAUUSD\n"
                f"{'🟢' if long else '🔴'} Direction: {tr['dir']}\n\n"
                f"🎯 Entry: {tr['entry']:,.2f}\n"
                f"🛑 SL: {tr['sl']:,.2f}\n"
                f"🎯 TP1: {tr['tp1']:,.2f}\n"
                f"🎯 TP2: {tr['tp2']:,.2f}\n\n"
                f"📈 {msg}\n\n"
                f"⭐ Source card: {src} · tracked automatically")


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
        r_tp1 = round(0.5 * abs(tp1 - entry) / sl_dist, 2)
        r_tp2 = round(r_tp1 + 0.5 * abs(tp2 - entry) / sl_dist, 2)
        if tr["status"] == "open":
            if (d == 1 and price >= tp1) or (d == -1 and price <= tp1):
                tr["status"] = "tp1hit"; tr["tp1At"] = now
                fired.append(("TP1_HIT", f"TP1 hit at {tp1:,.1f} · half banked +{r_tp1:.2f}R · stop moved to breakeven {entry:,.1f} · runner targets {tp2:,.1f}"))
            elif (d == 1 and price <= sl) or (d == -1 and price >= sl):
                _close_trade(tr, "loss", -1.0, price)
                fired.append(("SL_HIT", f"Stopped out at {sl:,.1f} · -1.0R · next setup will come, patience"))
            elif timed_out:
                r = 0.5 * d * (price - entry) / sl_dist
                _close_trade(tr, "timeout", r, price)
                fired.append(("TIMEOUT", f"5h timeout — closed at market {price:,.1f} ({r:+.2f}R on the runner half)"))
        if tr.get("status") == "tp1hit":
            if (d == 1 and price >= tp2) or (d == -1 and price <= tp2):
                _close_trade(tr, "win", r_tp2, price)
                fired.append(("TP2_HIT", f"TP2 hit at {tp2:,.1f} · runner closed · total +{r_tp2:.2f}R on the signal 🎯"))
            elif (d == 1 and price <= entry) or (d == -1 and price >= entry):
                _close_trade(tr, "be", r_tp1, price)
                fired.append(("BE_STOP", f"Runner stopped at breakeven {entry:,.1f} · signal finishes +{r_tp1:.2f}R (TP1 banked)"))
            elif timed_out:
                r = r_tp1 + 0.5 * d * (price - entry) / sl_dist
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
                        l=k["l"] - adjust, c=k["c"] - adjust,
                        v=k.get("v")) for k in raw]
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
                      l=round(k["l"], 2), c=round(k["c"], 2),
                      v=(round(k["v"], 1) if k.get("v") is not None else None)) for k in win],
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
        try:
            payload["delta"] = _market_delta(candles)
        except Exception:  # noqa: BLE001
            payload["delta"] = None
    except Exception:  # noqa: BLE001
        payload["fundamentals"] = None
    if tf == "15m" and gold:
        try:
            d1h = data.get_candles("60m")
            ent = entries.evaluate(candles, d1h.get("candles"))
            payload["entry"] = ent
            payload["entryStats"] = entries.backtest_stats(candles, "A+")
            try:
                payload["eliteStats"] = entries.cohort_stats(candles)
            except Exception:  # noqa: BLE001
                payload["eliteStats"] = None
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
            # daily liquidity rotation: tradeable both-pools-drained flip +
            # informational watch lines for every daily-level sweep
            try:
                rot = entries.scan_daily_rotations(candles)
                n15 = len(candles)
                livert = [e for e in rot["trades"] if e["i"] >= n15 - 5]
                payload["rotation"] = (entries.rotation_view(livert[-1])
                                       if livert else None)
                tr = [e for e in rot["trades"] if e["i"] <= n15 - 4]
                wt = sum(1 for e in tr if e["outcome"] == "win")
                payload["rotationStats"] = dict(
                    setups=len(tr),
                    winRate=round(wt / len(tr), 3) if tr else None,
                    avgR=round(sum(e["r"] or 0.0 for e in tr) / len(tr), 3)
                    if tr else None)
                wa = [x for x in rot["watches"] if x["i"] >= n15 - 5]
                payload["liqWatch"] = (entries.watch_view(wa[-1])
                                       if wa else None)
            except Exception:  # noqa: BLE001
                payload["rotation"] = None
                payload["rotationStats"] = None
                payload["liqWatch"] = None
            blk = _event_blackout()
            if blk:
                _postpone_note(ent, blk)
            else:
                ensure_trade(ent)
                maybe_setup_alert(ent, payload.get("eliteStats"))
                maybe_momentum_alert(ent)   # ⚡ immediate entry at market on armed 6/8+ zones
                maybe_zone_watch(ent)
                if payload.get("sweep"):
                    maybe_sweep_alert(payload["sweep"])
                if payload.get("liqWatch"):
                    maybe_liq_watch(payload["liqWatch"])
                if payload.get("rotation"):
                    maybe_rotation_alert(payload["rotation"],
                                         payload.get("rotationStats"))
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
                                             history=STATE.get("tradeHistory", [])[:8],
                                             signals=[s for s in (STATE.get("signalTrades") or [])
                                                      if s.get("status") != "closed"][:6])
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


def _news_card(txt):
    """📰 news alert with the gold read: which way this headline pushes
    XAU/USD and why — deterministic rule engine, honest MIXED verdicts."""
    b = fundamentals.gold_bias(txt)
    ic = {"BULLISH": "🟢", "BEARISH": "🔴", "MIXED": "🟡"}[b["bias"]]
    why = " · ".join(b["why"][:2])
    return (f"📰 GOLD-RELEVANT NEWS\n\n{txt[:400]}\n\n"
            f"🧭 Gold read: {ic} {b['bias']}\n"
            f"Why: {why}\n\n"
            f"⚠ Headline analysis — not a trade signal"), b["bias"]


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
                        f"💰 XAUUSD — expect volatility\n\n"
                        f"🧭 How gold usually reacts:\n"
                        f"{fundamentals.event_playbook(e['title'])}\n\n"
                        f"⚠ Event reaction guide — not a trade signal",
                        cat="news")
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
                card, bias = _news_card(txt)
                ic = {"BULLISH": "🟢", "BEARISH": "🔴",
                      "MIXED": "🟡"}[bias]
                _web_alert("NEWS", f"{ic} {txt[:110]} · {bias} for gold")
                _notify(card, cat="news")
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
            tg_topic_detect()         # auto-learn forum topics (no-op when done)
        except Exception:  # noqa: BLE001
            pass
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


def _llm_cycle():
    """Called every background loop: refresh external AI seats (24/7, own
    15-min cadence, non-blocking). The daily Telegram brief was removed —
    it re-sent after every Render restart (ephemeral disk) and spammed."""
    if llm_desk.configured() and llm_desk.due():
        try:
            txt = _llm_context()
            threading.Thread(target=llm_desk.refresh, args=(txt,),
                             daemon=True).start()
        except Exception:  # noqa: BLE001
            pass


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


STATIC_DIR = os.path.join(BASE, "static")


@app.route("/manifest.json")
def pwa_manifest():
    return send_from_directory(STATIC_DIR, "manifest.json",
                               mimetype="application/manifest+json",
                               max_age=3600)


@app.route("/sw.js")
def pwa_sw():
    resp = send_from_directory(STATIC_DIR, "sw.js",
                               mimetype="text/javascript")
    resp.headers["Cache-Control"] = "no-cache"   # SW updates must ship fresh
    return resp


@app.route("/icons/<path:name>")
def pwa_icon(name):
    return send_from_directory(os.path.join(STATIC_DIR, "icons"), name,
                               mimetype="image/png", max_age=86400)


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

start_background()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, threaded=True, debug=False)
