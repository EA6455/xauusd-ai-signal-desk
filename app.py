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
_BOOT_T = time.time()             # uptime for the member digest

# The desk announces its own updates to the group (DEVELOP topic): every
# deployed version posts its changelog there automatically on boot.
SYSTEM_VERSION = "2.9.12"
SYSTEM_CHANGELOG = {
    "2.9.12": [
        "Lossless discipline: AI-trader calls are hard-blocked against "
        "the 60m trend regime (the exact cohort that produced both "
        "recorded losses — backtest: counter-trend 37% win vs with-trend "
        "50%), entry orders expire after 4h instead of 12h so stale "
        "theses can never fill far from validity, no new auto cards "
        "while 4 calls are open, and junk journal records are purged "
        "from the scorecard",
    ],
    "2.9.11": [
        "Permanent TV lock: the level pusher now runs as its own "
        "always-on service (separate IP, immune to scanner bans and to "
        "research-machine reboots); fixed a cooldown bug that made the "
        "desk hammer the scanner while throttled; mutual keepalive "
        "pings keep both free services awake 24/7",
    ],
    "2.9.10": [
        "Always-matched prices: the research machine now streams "
        "TradingView's level to the production desk every 3s (separate "
        "IP, immune to scanner throttling) — combined with the desk's "
        "own gentle 6s poll and 429 cooldowns, the TV hard-lock never "
        "goes stale",
    ],
    "2.9.8": [
        "Self-tuning TradingView poll: backs off instantly when the "
        "scanner throttles, ramps up slowly when it doesn't — the level "
        "lock runs at the fastest cadence TradingView allows from this "
        "network (diagnostics in /api/feed)",
    ],
    "2.9.7": [
        "Zero-lag TV match: the TradingView poll now runs every 2s and "
        "HARD-LOCKS the level to their exact number on every poll — "
        "between polls only the tape's motion moves the price, so the "
        "desk and TradingView read the same number together",
    ],
    "2.9.6": [
        "While TradingView's feed is reachable it carries the price level "
        "100% (futures leg demoted to fallback) — the displayed number "
        "locks to the TV level and moves at tape speed",
    ],
    "2.9.5": [
        "ONE price space: the MTF matrix card and the AI trader brief now "
        "use the same spot-aligned candles as the chart — every level on "
        "the desk matches TradingView's price space",
    ],
    "2.9.4": [
        "Prices now MATCH TradingView: the level anchors to TradingView's "
        "own spot feed (OANDA:XAUUSD via their public scanner, polled "
        "every 6s) with gold-api as backup — chart candles re-aligned to "
        "the same spot level; live delta vs TV visible in /api/feed",
    ],
    "2.9.3": [
        "Consolidated-index pricing: median of venue levels + the fastest "
        "feed's live motion — the displayed price now moves at tape speed "
        "(several updates per second) while staying pinned to the "
        "cross-market level, with a $3 spike guard",
    ],
    "2.9.2": [
        "Price engine: displayed spot recomputed at 4Hz in the background "
        "— the live stream now pushes at tape density instead of waiting "
        "on slow REST quotes between updates",
    ],
    "2.9.1": [
        "Realtime price tape rebuilt as a 4-stream aggregate (OKX PAXG + "
        "OKX XAUT + Binance + Bybit) — TradingView-style, median-merged, "
        "several updates per second instead of one every ~5s",
        "Zero-timer chain: every tick wakes the stream instantly; "
        "/api/feed shows live per-stream density",
    ],
    "2.9.0": [
        "Every-timeframe signal analysis: 15m · 1H · 2H · 4H · 1D scanned "
        "every minute — trend, zones and setup state per TF",
        "BEST SIGNAL alerts to the group the moment a setup reaches "
        "tradeable quality (6/8 arming, live full-house zones, 4+ TF "
        "confluence) — no more waiting for the perfect 8/8",
        "Exact market delta on every alert: EMA9/EMA21 values, buy-volume "
        "%, pressure state, divergence — the same numbers that gate the "
        "engine",
        "Freshness proven: every analysis carries the age of the last "
        "closed bar (analysis never late)",
        "MTF Signal Matrix card on the panel with the live per-TF view",
    ],
    "2.8.0": [
        "The AI desk now informs members itself: every deploy posts its "
        "changelog here automatically",
        "DESK UPDATE digest every 6h: engine record, AI trader scorecard, "
        "research, calendar, system health",
        "One voice: only the production desk posts to the group — no more "
        "duplicate digests or test noise from research machines",
        "AI trader accountability: every call tracked & scored against real "
        "price action, honest win% / avgR on the panel",
        "AI trader best-signal card + hourly auto-refresh (2.7.x)",
    ],
}

STATE = dict(alerts=[], priceAlerts=[], lastSig={}, lastPrice=None,
             liveTrade=None, tradeHistory=[], signalTrades=[], lastSigT={},
             lastSetup=None, lastSetupT=0.0, eventAlerted=[], newsSeen=None,
             brokerOffset=0.0, aiTrades=[], sysVer=None)
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
# Baked-in topic map for the forum group (SIGNAL 59 / NEWS 60 / DEVELOP 164,
# verified open on 2026-09-17). Render's free tier wipes the learned topic
# file on every redeploy — without this default the map is lost and signal
# cards silently fall back to the General topic. The on-disk file (learned or
# shipped) still overrides, and tg_topic_detect() still re-learns if a
# topic ever goes away.
DEFAULT_TG_TOPICS = {"signal": 59, "news": 60, "develop": 164,
                     "complete": True}
_TG_TOPICS = None


def _tg_topics():
    """{'signal': thread_id, 'news': thread_id} for the forum group, learned
    automatically by tg_topic_detect(). Falls back to DEFAULT_TG_TOPICS when
    nothing has been learned yet (fresh deploy), so signal cards always land
    in the SIGNAL topic."""
    global _TG_TOPICS
    if _TG_TOPICS is None:
        try:
            with open(TG_TOPIC_FILE) as f:
                _TG_TOPICS = json.load(f)
        except Exception:  # noqa: BLE001
            _TG_TOPICS = dict(DEFAULT_TG_TOPICS)
    return _TG_TOPICS


def _desk_speaker():
    """Only the production desk (DESK_SPEAKER=1, set on Render) sends
    automatic Telegram updates. Research/sandbox machines run the same
    loops but stay silent — members hear ONE desk, not echoes."""
    return os.environ.get("DESK_SPEAKER") == "1"


def _notify(text, cat="signal"):
    """Send a Telegram message to every configured chat (private and/or
    group). In a forum group each message lands in its own topic:
    cat='signal' → SIGNALS topic, cat='news' → NEWS topic (auto-learned).
    Private chats are unaffected."""
    if not (TG_TOKEN and TG_CHATS):
        return
    if not _desk_speaker():
        return          # one voice: only the production desk posts
    import urllib.request
    tops = _tg_topics()
    for chat in TG_CHATS:
        if cat == "develop" and not chat.startswith("-100"):
            continue          # develop digests: DEVELOP topic in the group only
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
                    # park on General (topic 1) and mark incomplete so
                    # tg_topic_detect() resumes re-learning new topic IDs
                    _TG_TOPICS = {"signal": 1, "news": 1, "develop": 1,
                                  "complete": False}
                    try:
                        with open(TG_TOPIC_FILE, "w") as f:
                            json.dump(_TG_TOPICS, f)
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
        elif any(w in name for w in ("develop", "research", "backtest", "lab")):
            found["develop"] = tid
        elif any(w in name for w in ("signal", "trade", "entry", "snr", "alert")):
            found["signal"] = tid
    if not forum or not found:
        return None
    tops = dict(signal=found.get("signal", 1), news=found.get("news", 1),
                develop=found.get("develop", 1),
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


def _stop_cap_ok(entry, sl):
    """1000-pip hard stop cap ($100 on gold): a signal whose stop is wider
    than this can never reach the phone — max possible loss per signal."""
    return abs(float(entry) - float(sl)) <= entries.MAX_STOP_DIST


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
    act = "BUY" if sw["direction"] == "LONG" else "SELL"
    if not _stop_cap_ok(sw["entry"], sw["sl"]):
        _web_alert("SWEEP", f"⏸ sweep skipped — stop distance "
                            f"${abs(sw['entry'] - sw['sl']):,.0f} exceeds the "
                            f"1000-pip cap")
        return
    _notify(f"{icon} {act} · XAUUSD 🧹\n\n"
            f"🎯 Entry: {sw['entry']:,.2f}\n"
            f"🛑 SL: {sw['sl']:,.2f}\n"
            f"🎯 TP1: {sw['tp1']:,.2f}\n"
            f"🎯 TP2: {sw['tp2']:,.2f} (2R)")
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
    """🔁 LIQUIDITY ROTATION: both daily pools drained and the second sweep
    reclaimed. Web-feed only — its researched exit profile is TP2 1.5R
    (75% win), which is BELOW the 2RR+ phone bar, and at 1:2 it wins only
    56%. The rotation box in the web UI still shows the full story."""
    PHONE = False
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
    if not _stop_cap_ok(entry, sl):
        _web_alert("MOMENTUM", f"⏸ momentum skipped — stop distance "
                               f"${risk:,.0f} exceeds the 1000-pip cap")
        return
    tp = entry + d * 0.75 * risk                # TP1 0.75R — half off, high hit rate
    tp2 = entry + d * 2.0 * risk                # TP2 runner 2.0R — 2RR+ phone bar
    lo, hi = ent["entryZone"][0], ent["entryZone"][1]
    icon = "🟢" if d == 1 else "🔴"
    act = "BUY" if d == 1 else "SELL"
    _web_alert("MOMENTUM",
               f"⚡ momentum {ent['direction'].lower()} @ {entry:,.1f} · "
               f"SL {sl:,.1f} · TP {tp:,.1f} · {ent['passed']}/8 armed zone "
               f"{lo:,.1f}–{hi:,.1f} · 68% win profile · not a zone retest")
    _notify(f"{icon} {act} · XAUUSD ⚡\n\n"
            f"🎯 Entry: {entry:,.2f}\n"
            f"🛑 SL: {sl:,.2f}\n"
            f"🎯 TP1: {tp:,.2f}\n"
            f"🎯 TP2: {tp2:,.2f} (2R)")
    track_signal("MOMENTUM", key, ent["direction"], entry, sl, tp, tp2)


def maybe_setup_alert(ent, elite_stats=None):
    """Fire an alert on SNR retests, once per ZONE+GRADE (a setup that stays
    live for hours must not re-alert every 15 minutes).

    v3 gating (1:2 RR backtest): phone cards only for the DELTA-CONFIRMED
    cohort — (A+ or A+/B+ continuation zone, 83%/+0.65R) AND the smoothed
    market delta pushing the same way at entry: 92% win · +0.76R (n=13).
    Delta-against A+/elite setups (40-50%) and B+/C+ mid/deep zones (42-44%)
    stay web-feed watch lines — visible, honestly labeled, no card."""
    if not ent or not ent.get("touching"):
        return
    grade = ent.get("grade")
    if grade not in ("A+", "B+", "C+"):
        return
    # self-upgrade knobs: the research engine promotes challengers by
    # switching this gate/exit automatically (see _maybe_upgrade)
    upg = ((STATE.get("research") or {}).get("upgrade") or {})
    gate = upg.get("gate") or "delta"
    exitr = upg.get("exit") or [1.0, 2.0]
    elite = bool(ent.get("contZone")) and grade in ("A+", "B+")
    base = elite or grade == "A+"
    if gate == "cont-only":
        base = elite
    phone = base and (gate != "delta" or bool(ent.get("deltaOK")))
    if gate == "session":
        phone = phone and ent.get("session") in ("london", "ny-overlap",
                                                 "ny-late")
    if gate == "none" and grade == "C+":
        phone = False          # no-delta-gate variant never trades C+ zones
    if exitr != [1.0, 2.0]:
        # promoted exit profile — recompute TPs as pure R multiples
        _d = 1 if ent["direction"] == "LONG" else -1
        _risk = abs(float(ent["entry"]) - float(ent["sl"]))
        ent["tp1"] = round(float(ent["entry"]) + _d * exitr[0] * _risk, 2)
        ent["tp2"] = round(float(ent["entry"]) + _d * exitr[1] * _risk, 2)
    if phone and not _stop_cap_ok(ent["entry"], ent["sl"]):
        phone = False
        cap_skip = True
    else:
        cap_skip = False
    key = (ent.get("zoneKey") or f"{ent['direction']}:{ent['barTime']}") + ":" + grade
    with STATE_LOCK:
        if STATE.get("lastSetup") == key:
            return
        STATE["lastSetup"] = key
        if not phone:
            # below the quality bar — web-feed watch line only, no card
            if cap_skip:
                why = (f"stop distance ${abs(ent['entry'] - ent['sl']):,.0f} "
                       "exceeds the 1000-pip cap")
            elif base:
                why = ("delta not confirmed — order flow "
                       f"{ent.get('deltaState') or 'flat'} · delta-against "
                       "setups win only 40-50%")
            else:
                why = "mid/deep pullback zone · wins only ~42% at 1:2"
            a = dict(id=_next_id(), time=int(time.time()), tf="15m",
                     type="WATCH",
                     price=ent["entry"], score=round(ent["passed"] / 8.0, 2),
                     confidence=round(ent["passed"] / 8.0, 2),
                     msg=(f"⏸ {grade} {ent['direction']} retest @ "
                          f"{ent['entry']:,.2f} · {why} — below card quality "
                          f"bar, watch only"))
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
    # phone message: MINIMAL by request — direction, entry, SL, TPs only.
    # The full trader-flow detail (zone, rejection, confirmation, delta,
    # zone type, cohort stats) stays visible in the web UI's SNR Entry card.
    icon = "🟢" if ent["direction"] == "LONG" else "🔴"
    act = "BUY" if ent["direction"] == "LONG" else "SELL"
    tag = " · 🔥" if elite else ""
    _notify(
        f"{icon} {act} · XAUUSD{tag}\n\n"
        f"🎯 Entry: {ent['entry']:,.2f}\n"
        f"🛑 SL: {ent['sl']:,.2f}\n"
        f"🎯 TP1: {ent['tp1']:,.2f}\n"
        f"🎯 TP2: {ent['tp2']:,.2f} ({exitr[1]:g}R)")
    track_signal(grade, key, ent["direction"], ent["entry"], ent["sl"],
                 ent["tp1"], ent["tp2"])


def _close_trade(tr, result, r, price):
    tr.update(status="closed", result=result, r=round(r, 2),
              closedAt=int(time.time()), closePrice=round(float(price), 2))
    STATE["tradeHistory"].insert(0, dict(tr))
    del STATE["tradeHistory"][25:]
    _ae_close(tr)        # settle the matching autonomous position, if any


# ------------------------------------------------- autonomous trading
# The system TRADES by itself: every fired signal card is executed
# automatically on the engine account — paper mode now (real live prices,
# exact 1% risk sizing, zero money at risk); a live cent-account MT5
# adapter plugs into the same engine when broker access is connected.
# Hard guards: max 1 open position, max 5 trades/day, daily loss pause.

AE_SOURCES = ("A+", "B+", "MOMENTUM", "SWEEP")   # rotation stays web-only


def _ae():
    ae = STATE.get("autoexec")
    if not ae:
        ae = dict(enabled=True, mode="paper", riskPct=1.0,
                  startBalance=100.0, balance=100.0,
                  positions=[], closed=[],
                  day=dict(d="", trades=0, pnl=0.0, paused=False),
                  guards=dict(maxDailyLossPct=3.0, maxTradesPerDay=5,
                              maxOpen=1),
                  totalTrades=0, totalPnl=0.0)
        with STATE_LOCK:
            STATE["autoexec"] = ae
            _save_state()
    return ae


def _ae_equity(ae, px=None):
    """Balance + unrealized P&L of open autonomous positions."""
    eq = ae.get("balance", 0.0)
    if ae.get("positions"):
        if px is None:
            try:
                px, _s = tick_spot(broker=False)
            except Exception:  # noqa: BLE001
                px = None
        if px:
            for p in ae["positions"]:
                d = 1 if p.get("dir") == "LONG" else -1
                eq += d * (px - p["entry"]) * p.get("lots", 0) * 100.0
    return round(eq, 2)


def _ae_rollover(ae):
    """Reset daily counters at UTC midnight + post the day's summary."""
    today = time.strftime("%Y-%m-%d", time.gmtime())
    day = ae.get("day") or {}
    if day.get("d") == today:
        return
    if day.get("d"):
        prev = [c for c in (ae.get("closed") or [])
                if time.strftime("%Y-%m-%d",
                                 time.gmtime(c.get("closedAt", 0)))
                == day["d"]]
        if prev:
            wins = sum(1 for c in prev if (c.get("pnl") or 0) > 0)
            pnl = round(sum(c.get("pnl") or 0 for c in prev), 2)
            _notify(f"🤖 AUTO-TRADE daily summary · {day['d']}\n\n"
                    f"trades {len(prev)} · wins {wins} "
                    f"({100 * wins // len(prev)}%)\n"
                    f"day P&L {pnl:+.2f} $\n"
                    f"💵 balance ${ae.get('balance', 0):,.2f}\n"
                    f"all-time {ae.get('totalPnl', 0):+.2f} $ over "
                    f"{ae.get('totalTrades', 0)} trades", cat="develop")
    ae["day"] = dict(d=today, trades=0, pnl=0.0, paused=False)


def _ae_open(src, key, direction, entry, sl, tp1, tp2):
    """Autonomously execute a fired signal: open a position sized to
    riskPct of the account balance. Paper mode fills fractional lots at
    the signal price; live cent mode scales x100 with a 0.01 lot minimum."""
    ae = _ae()
    if not ae.get("enabled") or src not in AE_SOURCES:
        return
    _ae_rollover(ae)
    day = ae["day"]
    g = ae.get("guards") or {}
    if day.get("pnl", 0) <= -abs(g.get("maxDailyLossPct", 3.0)) / 100.0 \
            * ae.get("startBalance", 100.0):
        day["paused"] = True          # self-healing loss brake
    if day.get("paused") or day.get("trades", 0) >= g.get(
            "maxTradesPerDay", 5):
        return
    if len(ae.get("positions") or []) >= g.get("maxOpen", 1):
        return
    if any(p.get("key") == key for p in ae.get("positions") or []):
        return
    try:
        entry = float(entry)
        stop = abs(entry - float(sl))
    except (TypeError, ValueError):
        return
    if stop <= 0:
        return
    risk_amt = round(ae["balance"] * ae.get("riskPct", 1.0) / 100.0, 2)
    lots = risk_amt / (stop * 100.0)          # XAUUSD: 1 lot = 100 oz
    if ae.get("mode") == "live":
        lots = max(0.01, round(lots * 100, 2))   # cent account: x100
    else:
        lots = round(lots, 4)                    # paper: fractional ok
    pos = dict(id=_next_id(), key=key, src=src, dir=direction,
               entry=round(entry, 2), sl=round(float(sl), 2),
               tp1=round(float(tp1), 2), tp2=round(float(tp2), 2),
               lots=lots, riskAmt=risk_amt, openedAt=int(time.time()))
    pos["live"] = _mt5_exec_live(pos, stop)   # real MT5 accounts, if any
    _cloud_dirty()
    ae.setdefault("positions", []).append(pos)
    day["trades"] += 1
    ae["totalTrades"] = ae.get("totalTrades", 0) + 1
    with STATE_LOCK:
        STATE["autoexec"] = ae
        _save_state()
    ic = "🟢" if direction == "LONG" else "🔴"
    act = "BUY" if direction == "LONG" else "SELL"
    live_txt = ""
    if pos.get("live"):
        parts = []
        for e in pos["live"]:
            if e.get("ticket"):
                parts.append(f"{e['login']} · {e['lots']} lots · "
                             f"#{e['ticket']}")
            else:
                parts.append(f"{e['login']} · ERR {e.get('err', '')[:40]}")
        live_txt = "\n🌐 LIVE: " + " | ".join(parts)
    _notify(f"🤖 AUTO-TRADE · {act} {lots} lots @ {entry:,.2f}\n\n"
            f"🛑 SL {pos['sl']:,.2f}\n"
            f"🎯 TP1 {pos['tp1']:,.2f}\n"
            f"🎯 TP2 {pos['tp2']:,.2f}\n\n"
            f"💵 risk ${risk_amt:.2f} ({ae.get('riskPct', 1):g}%) · "
            f"{ae.get('mode', 'paper')} · equity "
            f"${_ae_equity(ae):,.2f}{live_txt}", cat="signal")


def _ae_close(tr):
    """Settle the autonomous position matching a closed tracked signal."""
    ae = STATE.get("autoexec")
    if not ae or not ae.get("enabled"):
        return
    key, src = tr.get("key"), tr.get("src")
    pos = None
    for p in ae.get("positions") or []:
        if p.get("key") == key and p.get("src") == src:
            pos = p
            break
    if not pos:
        return
    ae["positions"] = [p for p in ae["positions"] if p is not pos]
    r = tr.get("r") if isinstance(tr.get("r"), (int, float)) else 0.0
    pnl = round(r * pos.get("riskAmt", 0.0), 2)
    ae["balance"] = round(ae.get("balance", 0.0) + pnl, 2)
    ae["totalPnl"] = round(ae.get("totalPnl", 0.0) + pnl, 2)
    day = ae.get("day") or {}
    day["pnl"] = round(day.get("pnl", 0.0) + pnl, 2)
    ae["closed"] = (ae.get("closed") or [])[-49:] + [dict(
        pos, closedAt=int(time.time()), r=round(r, 2), pnl=pnl,
        result=tr.get("result"), balance=ae["balance"])]
    g = ae.get("guards") or {}
    if day.get("pnl", 0) <= -abs(g.get("maxDailyLossPct", 3.0)) / 100.0 \
            * ae.get("startBalance", 100.0):
        day["paused"] = True
    with STATE_LOCK:
        STATE["autoexec"] = ae
        _save_state()
    cloud_save(force=True)
    ic = "✅" if pnl >= 0 else "❌"
    extra = ("\n🛑 daily loss limit hit — auto-trading paused until "
             "tomorrow" if day.get("paused") else "")
    _notify(f"🤖 AUTO-TRADE CLOSED {ic}\n\n"
            f"{pos.get('src')} {pos.get('dir')} {pos.get('lots')} lots · "
            f"{r:+.2f}R → {pnl:+.2f} $\n\n"
            f"💵 balance ${ae['balance']:,.2f} · today "
            f"{day.get('pnl', 0):+.2f} ${extra}", cat="signal")
    _mt5_close_live(pos)     # also close the matching real-account tickets


# ------------------------------------------------- live MT5 execution
# People connect their own MT5 account in the web panel (login + server +
# password — Exness fully supported) and the bot trades it for real.
# A Render server cannot run the MT5 terminal itself, so accounts are
# bridged through the MetaApi cloud (metaapi.cloud — free plan available),
# which runs the terminal for us and exposes a REST API. Credentials are
# encrypted at rest and only ever used to connect the owner's account.

MTA_DOMAINS = ("agiliumtrade.agiliumtrade.ai", "metaapi.cloud")
MTA_PROV = "https://mt-provisioning-api-v1."
MTA_CLIENT = "https://mt-client-api-v1."
_mt5_host_mem = {"t": 0.0, "host": None}   # dynamic client-api hostname


def _mt5_client_host(token):
    """MetaApi migrates API hostnames (the original agiliumtrade.ai domain
    was retired). The official SDK resolves the current client-api host
    from the provisioning API — we do the same, cached 24h, with static
    fallbacks."""
    now = time.time()
    if _mt5_host_mem["host"] and now - _mt5_host_mem["t"] < 86400:
        return _mt5_host_mem["host"]
    for dom in MTA_DOMAINS:
        try:
            j = _mt5_req("GET", MTA_PROV + dom +
                         "/users/current/servers/mt-client-api", token)
            host = (j or {}).get("hostname")
            if host:
                _mt5_host_mem.update(t=now, host=host)
                return host
        except Exception:  # noqa: BLE001
            continue
    return "mt-client-api-v1." + MTA_DOMAINS[0]


def _mt5_prov_url():
    return MTA_PROV + MTA_DOMAINS[0]


EXNESS_SERVERS = ("Exness-Real", "Exness-Cent", "Exness-MT5Trial",
                  "Exness-Real14", "Exness-Cent8")


def _mt5_sys_token():
    """The system-hosted MetaApi bridge token (env METAAPI_TOKEN). People
    never need their own token — they paste MT5 login/password/server."""
    return os.environ.get("METAAPI_TOKEN") or ""


def _enc(s):
    """Best-effort encryption at rest (XOR stream keyed by the bot)."""
    import base64
    if not s:
        return ""
    k = hashlib.sha256((TG_TOKEN or "xauusd").encode()).digest()
    b = bytes(c ^ k[i % len(k)] for i, c in enumerate(s.encode()))
    return base64.b64encode(b).decode()


def _dec(s):
    import base64
    if not s:
        return ""
    try:
        k = hashlib.sha256((TG_TOKEN or "xauusd").encode()).digest()
        b = base64.b64decode(s)
        return bytes(c ^ k[i % len(k)] for i, c in enumerate(b)).decode()
    except Exception:  # noqa: BLE001
        return ""


class _Mt5Error(Exception):
    pass


def _mt5_req(method, path, token, body=None, timeout=25, prov=False,
             txn=None):
    """One MetaApi REST call. auth-token header; JSON error bodies raise
    with the server's own message. `path` is the path only when `prov`
    is set (provisioning API), else a full URL."""
    import urllib.request
    url = _mt5_prov_url() + path if prov else path
    headers = {"auth-token": token, "Content-Type": "application/json",
               "Accept": "application/json",
               "User-Agent": "xauusd-ai-desk/1.0"}
    if txn:
        headers["transaction-id"] = txn
    req = urllib.request.Request(
        url, data=json.dumps(body).encode() if body is not None else None,
        headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        try:
            msg = (json.loads(e.read().decode()) or {}).get(
                "message", str(e))
        except Exception:  # noqa: BLE001
            msg = str(e)
        raise _Mt5Error(f"{e.code}: {msg}") from None


def _mt5_accounts():
    return STATE.get("mt5") or []


def _mt5_provision_thread(acc, tries=None):
    """Connect a submitted account end-to-end: create it on MetaApi
    (retries while automatic broker detection runs — 202s), wait for the
    cloud terminal to deploy, pull balance/equity, mark connected."""
    tok = _dec(acc["tokenEnc"])
    acc["tries"] = (acc.get("tries") or 0) + 1 if tries is None else tries
    try:
        if not acc.get("accountId"):       # retries reuse the created one
            # reuse an existing bridge record for this login+server so
            # retries never leave duplicate accounts upstream (free tier
            # allows very few) — disconnect deletes the record, so a
            # reuse only ever sees the same credentials
            try:
                for ex in _mt5_req("GET", "/users/current/accounts", tok,
                                   prov=True):
                    if (str(ex.get("login")) == str(acc["login"])
                            and ex.get("server") == acc["server"]
                            and ex.get("platform", "mt5") == "mt5"):
                        acc["accountId"] = ex["_id"]
                        break
            except Exception:  # noqa: BLE001  list unavailable — create
                pass
        if not acc.get("accountId"):
            import uuid
            txn = uuid.uuid4().hex
            created = None
            for attempt in range(10):      # 202 -> same txn, retry
                try:
                    created = _mt5_req(
                        "POST", "/users/current/accounts", tok,
                        dict(login=str(acc["login"]),
                             password=_dec(acc["pwEnc"]),
                             name=f"xauusd-ai {acc['login']}",
                             server=acc["server"], platform="mt5",
                             magic=20260918,
                             type="cloud-g1",          # free-tier compatible
                             reliability="regular",
                             symbol="XAUUSD",
                             keywords=[acc.get("broker") or "Exness"]),
                        prov=True, txn=txn)
                    break
                except _Mt5Error as e:
                    if "202" in str(e) or "retry" in str(e).lower():
                        time.sleep(min(60, 15 * (attempt + 1)))
                        continue
                    raise
            if not created or not created.get("id"):
                raise RuntimeError("account creation returned no id")
            acc["accountId"] = created["id"]
        # cloud terminals do NOT auto-deploy via REST — deploy explicitly
        try:
            _mt5_req("POST", f"/users/current/accounts/{acc['accountId']}"
                             "/deploy", tok, timeout=90, prov=True)
        except _Mt5Error as e:
            s = str(e)
            if "403" in s or "top up" in s:
                acc["tries"] = 99          # billing wall — stop retrying
                raise RuntimeError(
                    "MetaApi cloud cannot deploy the live trading "
                    "terminal on its free plan — activate the 7-day "
                    "trial or subscribe at metaapi.cloud (once), then "
                    "press Connect again. Paper trading still works.")
            if "already deployed" not in s.lower():
                raise
        # wait for the cloud terminal to connect to the broker
        last = None
        for i in range(90):                # up to ~7.5 minutes
            time.sleep(5)
            try:
                st = _mt5_req("GET", f"/users/current/accounts/"
                                     f"{acc['accountId']}", tok, prov=True)
                last = st.get("connectionStatus")
            except _Mt5Error as e:
                if "404" in str(e):
                    acc.pop("accountId", None)   # gone — re-create next try
                    raise RuntimeError("stale bridge account — will "
                                       "re-create")
                continue
            except Exception:  # noqa: BLE001  timeout / transient
                continue
            if last == "CONNECTED":
                break
            if last == "DISCONNECTED" and i > 12:
                raise RuntimeError(
                    "broker rejected the connection — check the MT5 "
                    "login, password and server name, then press "
                    "Connect again")
        if last != "CONNECTED":
            raise RuntimeError(f"terminal state: {last or 'unreachable'} "
                               f"after 7.5 min — will keep retrying")
        _mt5_sync_acc(acc)
        acc["state"] = "connected"
        acc["err"] = None
    except Exception as e:  # noqa: BLE001
        acc["state"] = "error"
        acc["err"] = str(e)[:400]
    with STATE_LOCK:
        STATE["mt5"] = _mt5_accounts()
        _save_state()
    cloud_save(force=True)
    if acc["state"] == "connected":
        _notify(f"🔗 MT5 CONNECTED · {acc.get('label') or acc['login']}\n\n"
                f"🏦 {acc['server']}\n"
                f"💵 balance {acc.get('currency', '')} "
                f"{acc.get('balance', 0):,.2f}\n"
                f"🤖 the bot now trades this account automatically at "
                f"{acc.get('riskPct', 1):g}% risk", cat="signal")
    else:
        _notify(f"⚠ MT5 CONNECT FAILED · {acc.get('label') or acc['login']}"
                f"\n{acc.get('err')}", cat="signal")


def _mt5_retry_stuck():
    """Self-healing: re-attempt accounts stuck in error (e.g. after an
    API outage) up to 5 times, ~10 minutes apart. Runs from the loop."""
    now = time.time()
    for acc in _mt5_accounts():
        if acc.get("state") != "error":
            continue
        if (acc.get("tries") or 0) >= 5:
            continue
        if now - acc.get("lastTryT", 0) < 600:
            continue
        acc["lastTryT"] = int(now)
        threading.Thread(target=_mt5_provision_thread, args=(acc,),
                         daemon=True).start()


def _mt5_sync_acc(acc):
    host = _mt5_client_host(_dec(acc["tokenEnc"]))
    info = _mt5_req("GET", f"{MTA_CLIENT}{host}/users/current/accounts/"
                           f"{acc.get('accountId')}/account-information",
                    _dec(acc["tokenEnc"]))
    acc["balance"] = round(float(info.get("balance") or 0), 2)
    acc["equity"] = round(float(info.get("equity") or 0), 2)
    acc["currency"] = info.get("currency") or "USD"
    acc["leverage"] = info.get("leverage")
    return info


_mt5_spec_mem = {}          # accountId -> (t, spec)


def _mt5_get_spec(acc):
    """The broker's own XAUUSD contract spec: exact contract size, min
    volume, volume step — so sizing fits the person's real account."""
    key = acc.get("accountId")
    if not key:
        return None
    now = time.time()
    hit = _mt5_spec_mem.get(key)
    if hit and now - hit[0] < 600:
        return hit[1]
    try:
        host = _mt5_client_host(_dec(acc["tokenEnc"]))
        spec = _mt5_req("GET", f"{MTA_CLIENT}{host}/users/current/accounts/"
                               f"{key}/symbols/XAUUSD/specification",
                        _dec(acc["tokenEnc"]))
        _mt5_spec_mem[key] = (now, spec)
        return spec
    except Exception:  # noqa: BLE001  — fall back to standard gold spec
        return None


def _mt5_lots(acc, stop_dist):
    """Risk-sized lots using the account's EXACT balance and the broker's
    own contract spec. Works for real, demo, trial, standard and cent
    accounts (cent currencies like USc scale x100 automatically)."""
    bal = float(acc.get("equity") or acc.get("balance") or 0)
    if bal <= 0 or stop_dist <= 0:
        return None
    spec = _mt5_get_spec(acc) or {}
    contract = float(spec.get("contractSize") or 100)
    min_lot = float(spec.get("minVolume") or 0.01)
    step = float(spec.get("volumeStep") or 0.01)
    max_lot = float(spec.get("maxVolume") or 100)
    cur = (acc.get("currency") or "USD").upper()
    scale = 100.0 if cur.endswith("C") and cur != "USDC" else 1.0
    loss_per_lot = stop_dist * contract * scale     # account currency
    risk_amt = bal * (acc.get("riskPct", 1.0) / 100.0)
    lots = risk_amt / loss_per_lot
    lots = int(lots / step) * step                  # round DOWN to step
    if lots < min_lot:
        if min_lot * loss_per_lot > risk_amt * 1.5:
            return None        # even the minimum lot risks too much — skip
        lots = min_lot
    return round(min(lots, max_lot), 2)


def _mt5_exec_live(pos, stop_dist):
    """Fire the same signal on every connected live account."""
    out = []
    for acc in _mt5_accounts():
        if acc.get("state") != "connected":
            continue
        lots = _mt5_lots(acc, stop_dist)
        if not lots:
            continue
        try:
            host = _mt5_client_host(_dec(acc["tokenEnc"]))
            j = _mt5_req(
                "POST", f"{MTA_CLIENT}{host}/users/current/accounts/"
                        f"{acc['accountId']}/trade",
                _dec(acc["tokenEnc"]),
                dict(actionType="ORDER_TYPE_BUY" if pos["dir"] == "LONG"
                     else "ORDER_TYPE_SELL",
                     symbol="XAUUSD", volume=lots,
                     stopLoss=pos["sl"], takeProfit=pos["tp2"],
                     comment="xauusd-ai"))
            if j.get("numericCode") not in (10009, 10008):
                raise RuntimeError(j.get("message") or j.get("stringCode")
                                   or "trade rejected")
            out.append(dict(accId=acc["id"], login=acc["login"],
                            server=acc["server"], lots=lots,
                            ticket=str(j.get("positionId")
                                       or j.get("orderId") or "")))
        except Exception as e:  # noqa: BLE001
            out.append(dict(accId=acc["id"], login=acc["login"],
                            server=acc["server"], lots=lots,
                            err=str(e)[:120]))
    return out


def _mt5_close_live(pos):
    """Close the real-account tickets belonging to a settled position."""
    for e in pos.get("live") or []:
        if not e.get("ticket") or e.get("err"):
            continue
        acc = next((a for a in _mt5_accounts()
                    if a.get("id") == e.get("accId")), None)
        if not acc or not acc.get("accountId"):
            continue
        try:
            host = _mt5_client_host(_dec(acc["tokenEnc"]))
            _mt5_req("POST", f"{MTA_CLIENT}{host}/users/current/accounts/"
                             f"{acc['accountId']}/trade",
                     _dec(acc["tokenEnc"]),
                     dict(actionType="POSITION_CLOSE_ID",
                          positionId=str(e["ticket"])))
            e["closed"] = True
        except Exception as ex:  # noqa: BLE001
            e["closeErr"] = str(ex)[:120]


# ------------------------------------------------- every-signal tracker
# EVERY entry card (A+/B+/C+ retest + MOMENTUM) opens its own tracked
# position and is followed to TP or SL — one result card per event.


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
    _ae_open(src, key, direction, entry, sl, tp1, tp2)   # autonomous exec


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
        # R math from the trade's OWN levels (all phone signals are 2R+)
        _risk = abs(ent_p - sl_p) or 1.0
        _rr1 = abs(tp1_p - ent_p) / _risk
        _rr2 = abs(tp2_p - ent_p) / _risk
        if typ == "TP1_HIT":
            body = (f"🎯 Entry: {ent_p:,.2f}\n"
                    f"✅ TP1 reached: {tp1_p:,.2f}\n"
                    f"🛑 SL moved to breakeven {ent_p:,.2f}\n"
                    f"🎯 TP2 runner: {tp2_p:,.2f}\n\n"
                    f"⭐ Half position closed")
        elif typ == "TP2_HIT":
            body = (f"🎯 Entry: {ent_p:,.2f}\n"
                    f"✅ TP1: {tp1_p:,.2f}\n"
                    f"🎯 TP2: {tp2_p:,.2f} ({_rr2:.1f}R)\n\n"
                    f"⭐ Trade closed · Total +{0.5 * _rr1 + 0.5 * _rr2:.2f}R 🎉")
        elif typ == "SL_HIT":
            body = (f"🎯 Entry: {ent_p:,.2f}\n"
                    f"🛑 SL: {sl_p:,.2f} (-1.0R)\n\n"
                    f"⭐ Trade closed · next setup will come — patience")
        elif typ == "BE_STOP":
            body = (f"🎯 Entry: {ent_p:,.2f}\n"
                    f"✅ TP1 banked: {tp1_p:,.2f}\n"
                    f"⚖️ Runner stopped at breakeven {ent_p:,.2f}\n\n"
                    f"⭐ Trade closed · Total +{0.5 * _rr1:.2f}R")
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


_tv_spot_mem = {"price": None, "t": 0.0, "n": 0, "tape_ref": None}

TV_SPOT_TFS = ["OANDA:XAUUSD", "FOREXCOM:XAUUSD"]


def _tv_spot(force=False):
    """Level truth straight from TradingView's own public scanner — the
    exact number their XAUUSD chart shows (OANDA spot). Also snapshots
    the tape mid at poll time (tape_ref) so the displayed price can be
    re-pinned to TV's number and carry only tape MOTION between polls."""
    now = time.time()
    if not force and _tv_spot_mem["price"] and now - _tv_spot_mem["t"] < 4:
        return _tv_spot_mem["price"]
    import urllib.request
    try:
        body = json.dumps({
            "symbols": {"tickers": TV_SPOT_TFS, "query": {"types": []}},
            "columns": ["close"]}).encode()
        req = urllib.request.Request(
            "https://scanner.tradingview.com/global/scan", data=body,
            headers={"Content-Type": "application/json",
                     "User-Agent": "Mozilla/5.0",
                     "Origin": "https://www.tradingview.com",
                     "Referer": "https://www.tradingview.com/"})
        with urllib.request.urlopen(req, timeout=4) as r:
            j = json.load(r)
        closes = [row["d"][0] for row in j.get("data") or []
                  if isinstance(row.get("d"), list) and row["d"]
                  and isinstance(row["d"][0], (int, float))]
        if closes:
            closes.sort()
            _tv_spot_mem.update(price=closes[len(closes) // 2], t=now,
                                n=_tv_spot_mem.get("n", 0) + 1,
                                tape_ref=wsfeed.mid(max_age=60)[0])
    except Exception as e:  # noqa: BLE001  — throttled / unreachable
        _tv_spot_mem["err"] = str(e)[:80]
    return _tv_spot_mem["price"]


def _tv_spot_loop():
    """Keep the TradingView level fresh from THIS machine: poll gently
    (6s — within TradingView's public-scanner tolerance), and when they
    429-throttle, cool down hard (1 → 5 → 15 min) before trying again.
    The research machine's pusher (see _tv_pusher) keeps the lock alive
    from a different IP while this one is in cooldown."""
    base = 8.0
    delay = base
    while True:
        t0 = _tv_spot_mem["t"]
        try:
            _tv_spot(force=True)
        except Exception:  # noqa: BLE001
            pass
        ok = _tv_spot_mem["t"] > t0     # fresh sample landed? (the call
        if ok:                          # returns the STALE price on errors)
            delay = base
        else:
            delay = 60.0 if delay <= base else min(900.0, delay * 3)
        _tv_spot_mem["pollDelay"] = round(delay, 1)
        time.sleep(delay)


PROD_URL = "https://render-trading-chart-desk.onrender.com"


def _pusher_keepalive():
    """Ping the TV level pusher service every 5 minutes so its free
    instance never idles to sleep (and its 3s pushes keep THIS desk
    awake in return — mutual keepalive between the two services)."""
    import urllib.request
    url = os.environ.get("PUSHER_URL", "").rstrip("/")
    while True:
        if url:
            try:
                urllib.request.urlopen(url + "/health", timeout=8).read()
            except Exception:  # noqa: BLE001
                pass
        time.sleep(300)


def _tv_pusher():
    """Research-machine role: poll TradingView's scanner from THIS
    machine's IP (not datacenter-throttled) every 3s and push the level
    to the production desk, which hard-locks its price to it. Keeps the
    prod lock exact even while TradingView throttles prod's own IP.
    Runs only when TV_PUSHER=1 (the research machine's env)."""
    import urllib.request
    while True:
        try:
            body = json.dumps({
                "symbols": {"tickers": TV_SPOT_TFS, "query": {"types": []}},
                "columns": ["close"]}).encode()
            req = urllib.request.Request(
                "https://scanner.tradingview.com/global/scan", data=body,
                headers={"Content-Type": "application/json",
                         "User-Agent": "Mozilla/5.0",
                         "Origin": "https://www.tradingview.com",
                         "Referer": "https://www.tradingview.com/"})
            with urllib.request.urlopen(req, timeout=6) as r:
                j = json.load(r)
            closes = [row["d"][0] for row in j.get("data") or []
                      if isinstance(row.get("d"), list) and row["d"]
                      and isinstance(row["d"][0], (int, float))]
            if closes:
                closes.sort()
                lvl = closes[len(closes) // 2]
                sec = os.environ.get("TV_PUSH_SECRET") or ""
                preq = urllib.request.Request(
                    PROD_URL + "/api/tv/push",
                    data=json.dumps(dict(secret=sec, price=lvl,
                                         t=int(time.time()))).encode(),
                    headers={"Content-Type": "application/json"}, method="POST")
                urllib.request.urlopen(preq, timeout=8).read()
        except Exception:  # noqa: BLE001  — prod busy / unreachable
            pass
        time.sleep(3)


def spot_reference():
    """Best available spot XAU/USD: TradingView's own feed first (the
    reference users compare against), then gold-api.com, else OKX."""
    tv = _tv_spot_mem["price"]
    if tv and time.time() - _tv_spot_mem["t"] < 45:
        return tv, "tradingview"
    s = data.get_spot(max_age=30)
    if s:
        return s["price"], "gold-api.com"
    try:
        j = data._http_json("https://www.okx.com/api/v5/market/ticker?instId="
                            + data.OKX_INSTRUMENT, timeout=8)
        return float(j["data"][0]["last"]), "OKX PAXG"
    except Exception:  # noqa: BLE001
        return None, None
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
        # ---- sample T: TradingView HARD LOCK (resample & hold): every
        #      fresh TV poll re-pins the displayed level to their EXACT
        #      number; between polls only the tape's motion is applied.
        #      delta vs TV = 0 by construction at each poll.
        tv_lock = None
        tvp, tvt = _tv_spot_mem["price"], _tv_spot_mem["t"]
        tref = _tv_spot_mem.get("tape_ref")
        if tvp and tref and now - tvt < 30:
            off_tv = tvp - tref
            if abs(off_tv) <= 8.0:
                _anchor_mem.update(offset=off_tv, t=now, ema=off_tv,
                                   ema_f=None, warm=99, last_ts=tvt)
                tv_lock = off_tv
        elif tvp and now - tvt < 30 and not tref:
            fs_tv = tvp - wm                     # tape ref missing: soft lock
            if abs(fs_tv) <= 8.0:
                prev = _anchor_mem.get("ema")
                ema = fs_tv if prev is None else prev + 0.5 * (fs_tv - prev)
                _anchor_mem.update(ema=ema, warm=max(
                    _anchor_mem.get("warm", 0), 99), last_ts=tvt)
        if tv_lock is None:
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
                tv_fresh = (_tv_spot_mem["price"] is not None
                             and now - _tv_spot_mem["t"] < 30)
                if tv_fresh:
                    w_f = 0.0           # TV IS the level; futures only as fallback
                elif ga_age <= 20:
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
        if off is not None and abs(off) <= 8.0:
            price, src = round(wm + off, 2), ("tradingview-locked"
                                              if tv_lock is not None
                                              else "realtime feed")
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
            payload["delta"] = entries.delta_view(candles)
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
            try:
                if ent:
                    _d9, _d21 = entries.delta_series(candles)
                    ent["deltaOK"] = entries.delta_align(
                        _d9, len(candles) - 1,
                        ent.get("direction") or "LONG")
                    ent["deltaState"] = (payload.get("delta") or {}) \
                        .get("state")
            except Exception:  # noqa: BLE001
                pass
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


# ------------------------------------------------------- 24/7 AI research
RESEARCH_SCAN_S = 60         # full research scan every 60 seconds
RESEARCH_STATUS_S = 30       # live status message edited every 30 seconds
RESEARCH_DIGEST_S = 3600     # full digest posted every hour


def _fmt_r(st):
    if not st or not st.get("n"):
        return "n=0"
    return (f"n={st['n']} · {round((st['win'] or 0) * 100)}% · "
            f"{st['avgR']:+.2f}R")


def _tg_post_topic(text, thread_id):
    """Send to the forum group topic; return the new message_id or None."""
    if not (TG_TOKEN and TG_CHATS):
        return None
    group = next((c for c in TG_CHATS if c.startswith("-100")), None)
    if not group:
        return None
    import urllib.request
    payload = {"chat_id": group, "text": text,
               "message_thread_id": int(thread_id)}
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=10))
        return (r.get("result") or {}).get("message_id")
    except Exception:  # noqa: BLE001
        return None


def _tg_edit(chat, message_id, text):
    import urllib.request
    payload = {"chat_id": chat, "message_id": int(message_id), "text": text}
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{TG_TOKEN}/editMessageText",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=10)


def _research_scan(force_data=False):
    """One research pass over the freshest 15m data: backtest production +
    challenger variants, learn out-of-sample, store results. Runs every
    minute, 24/7 — caches make each pass cheap."""
    try:
        d = data.get_candles("15m", force=force_data).get("candles")
    except Exception:  # noqa: BLE001
        d = None
    if not d or len(d) < 400:
        return None
    r = STATE.get("research") or {}
    if not r.get("sessionStartT"):
        r = dict(sessionStartT=d[-1]["t"], cycles=0, scans=0,
                 startedAt=int(time.time()))
    res = entries.research_variants(d, since_ts=r["sessionStartT"])
    if not res:
        return None
    r["scans"] = r.get("scans", 0) + 1
    r["lastScanT"] = int(time.time())
    r["last"] = res
    with STATE_LOCK:
        STATE["research"] = r
        _save_state()
    return r, res


def _research_digest(r, res):
    """Hourly full digest -> DEVELOP topic only."""
    r["cycles"] = r.get("cycles", 0) + 1
    r["lastDigestT"] = int(time.time())
    with STATE_LOCK:
        STATE["research"] = r
        _save_state()
    p = res["production"]
    lines = [
        f"🔬 AI RESEARCH · digest {r['cycles']} · "
        f"{r.get('scans', 0)} scans · {res['bars']:,} bars · 15m",
        "",
        "🏭 PRODUCTION (delta-confirmed 1:2 · 1000-pip cap)",
        f"window: {_fmt_r(p['full'])}",
        f"OOS since session start: {_fmt_r(p['oos'])}",
        "",
        "⚖ CHALLENGERS (window | out-of-sample)",
    ]
    for v in res["variants"]:
        lines.append(f"• {v['name']}: {_fmt_r(v['full'])} | "
                     f"{_fmt_r(v['oos'])}")
    lines += [
        "",
        "🩺 ENGINE HEALTH",
        f"• sweeps A+/B+: {_fmt_r(res['sweeps'])}",
        f"• rotation flips: {_fmt_r(res['rotation'])}",
    ]
    cands = []
    po = p["oos"]
    if po.get("n") and po.get("avgR") is not None:
        for v in res["variants"]:
            vo = v["oos"]
            if vo.get("n", 0) >= 3 and vo.get("avgR") is not None and \
                    vo["avgR"] > po["avgR"] + 0.15:
                cands.append(f"{v['name']} (oos {vo['avgR']:+.2f}R vs "
                             f"prod {po['avgR']:+.2f}R · n={vo['n']})")
    if cands:
        lines += ["", "🔬 CANDIDATES — beating production "
                  "out-of-sample:"]
        lines += [f"  → {c}" for c in cands]
    else:
        lines += ["", "🔬 candidates: none yet "
                  "(needs OOS n≥3 and +0.15R edge)"]
    upg = _upgrade_state(r)
    lines += ["",
              f"⚙ live config: {upg.get('gate', 'delta')}-gate · exit "
              f"{upg['exit'][0]:g}:{upg['exit'][1]:g}"]
    if upg.get("pending"):
        lines.append(f"⏳ pending upgrade: {upg['pending']['name']} "
                     f"({upg['pending'].get('streak', 1)}/{UPGRADE_STREAK} "
                     "checks confirmed)")
    for h in reversed((upg.get("history") or [])[-3:]):
        when = time.strftime("%d %b %H:%M", time.gmtime(h["ts"]))
        if h.get("name") == "rollback":
            lines.append(f"↩ {when} rollback → gate {h['gate']} · exit "
                         f"{h['exit'][0]:g}:{h['exit'][1]:g} "
                         f"(live {h.get('liveAvgR', 0):+.2f}R)")
        else:
            lines.append(f"⬆ {when} upgrade: {h['name']} → gate "
                         f"{h['gate']} · exit {h['exit'][0]:g}:{h['exit'][1]:g}")
    _notify("\n".join(lines), cat="develop")
    _maybe_upgrade(r, res)          # research findings update the system


def _research_status():
    """Keep ONE live status message in the DEVELOP topic showing the AI is
    online 24/7: uptime timer, scan countdown, activity, live results.
    Edited every 30s (no message spam) — recreated if it gets deleted."""
    r = STATE.get("research") or {}
    if not r.get("startedAt") or not r.get("last"):
        return
    res = r["last"]
    now = time.time()
    up = int(now - r.get("startedAt", now))
    uh, rem = divmod(up, 3600)
    um, us = divmod(rem, 60)
    last_scan = int(now - r.get("lastScanT", now))
    next_scan = max(0, RESEARCH_SCAN_S - last_scan)
    try:
        px, _s = tick_spot(broker=False)
    except Exception:  # noqa: BLE001
        px = None
    p = (res.get("production") or {}).get("full") or {}
    oos = (res.get("production") or {}).get("oos") or {}
    cands = 0
    if oos.get("n") and oos.get("avgR") is not None:
        cands = sum(1 for v in res.get("variants", [])
                    if (v.get("oos") or {}).get("n", 0) >= 3
                    and (v.get("oos") or {}).get("avgR") is not None
                    and v["oos"]["avgR"] > oos["avgR"] + 0.15)
    lines = [
        "🟢 AI ONLINE 24/7 · RESEARCH MODE",
        f"⏱ uptime {uh:02d}:{um:02d}:{us:02d}",
        f"🔁 scans {r.get('scans', 0)} · digests "
        f"{r.get('cycles', 0)} · next scan in {next_scan}s",
        f"⚙ activity: scanning {res.get('bars', 0):,} bars · "
        f"{len(res.get('variants', []))} challengers · OOS learning",
    ]
    if px:
        lines.append(f"💰 XAUUSD {px:,.1f} (live)")
    if p.get("n"):
        lines.append(f"🏭 production "
                     f"{round((p.get('win') or 0) * 100)}% · "
                     f"{p.get('avgR', 0):+.2f}R · n={p['n']}")
    if oos.get("n"):
        lines.append(f"🔬 OOS {round((oos.get('win') or 0) * 100)}% "
                     f"· {oos.get('avgR', 0):+.2f}R · n={oos['n']}")
    lines.append(f"🧪 candidates beating production: {cands}")
    upg = r.get("upgrade") or {}
    cfg = f"⚙ live config: {upg.get('gate', 'delta')}-gate · exit " \
          f"{(upg.get('exit') or [1, 2])[0]}:{(upg.get('exit') or [1, 2])[1]}"
    if upg.get("pending"):
        cfg += f" · ⏳ {upg['pending']['name']} " \
               f"({upg['pending'].get('streak', 1)}/{UPGRADE_STREAK})"
    lines.append(cfg)
    ld = r.get("lastDigestT")
    if ld:
        nxt = max(0, (RESEARCH_DIGEST_S - int(now - ld)) // 60)
        lines.append(f"📊 next full digest in {nxt} min")
    else:
        lines.append("📊 first digest posting with first scan")
    intel = r.get("intel") or {}
    if intel.get("headlines"):
        lines.append(f"🌐 outside intel · "
                     f"{intel.get('nSources', 1)} sources: "
                     f"{intel['headlines'][0][:55]}")
    else:
        lines.append("🌐 outside intel: scanning news feeds")
    llmst = (r.get("llm") or {}).get("state")
    if llmst == "active":
        lines.append(f"🧠 LLM researcher: "
                     f"{(r.get('llm') or {}).get('model', 'AI')} · active")
    elif llmst == "error":
        lines.append("🧠 LLM researcher: provider error — retrying hourly")
    else:
        lines.append("🧠 LLM researcher: add OPENAI_API_KEY to activate")
    ae = STATE.get("autoexec") or {}
    if ae.get("enabled"):
        eq = _ae_equity(ae, px)
        st = ae.get("startBalance") or 100.0
        lines.append(f"🤖 auto-trade · {ae.get('mode', 'paper').upper()} · "
                     f"equity ${eq:,.2f} ({(eq / st - 1) * 100:+.1f}%) · "
                     f"today {(ae.get('day') or {}).get('pnl', 0):+.2f}$"
                     + (" · PAUSED" if (ae.get("day") or {}).get("paused")
                        else ""))
    lines.append(f"🌂 {time.strftime('%H:%M:%S', time.gmtime(now))}"
                 " UTC · this status updates live every 30s")
    txt = "\n".join(lines)

    sm = r.get("statusMsg") or {}
    if sm.get("id"):
        try:
            _tg_edit(sm["chat"], sm["id"], txt)
            return
        except Exception:  # noqa: BLE001 — deleted/edited message: recreate
            pass
    mid = _tg_post_topic(txt, _tg_topics().get("develop") or 1)
    if mid:
        r["statusMsg"] = dict(
            chat=next((c for c in TG_CHATS if c.startswith("-100")), ""),
            id=mid)
        with STATE_LOCK:
            STATE["research"] = r
            _save_state()


INTEL_SCAN_S = 900          # outside-intel self-research every 15 min
LLM_RESEARCH_S = 3600       # external-AI research analysis every hour


def _research_intel():
    """Self-directed OUTSIDE research: what the world is doing to gold —
    fresh headlines (with the engine's own bullish/bearish read), the
    high-impact event calendar, and the macro backdrop. Posts a report to
    the DEVELOP topic when there is NEW material; always refreshes the
    intel cache that feeds the live status message. Runs 24/7."""
    try:
        news = fundamentals.wide_news(limit=24)
    except Exception:  # noqa: BLE001
        news = []
    fresh = [(ts, t, s) for ts, t, s in (news or [])
             if fundamentals.gold_relevant(t)][:8]
    nsrc = len({s for _ts, _t, s in news or []}) or 1
    try:
        evs = fundamentals.upcoming(hours=24, impacts=("High",))
    except Exception:  # noqa: BLE001
        evs = []
    try:
        mac = fundamentals.macro_snapshot() or {}
    except Exception:  # noqa: BLE001
        mac = {}
    try:
        wmac = fundamentals.wide_macro() or {}
    except Exception:  # noqa: BLE001
        wmac = {}
    now = int(time.time())
    r = STATE.get("research") or {}
    intel = r.get("intel") or {}
    posted = intel.get("posted") or []
    new = [(ts, t, s) for ts, t, s in fresh
           if t.strip().lower() not in posted]
    intel.update(t=now, posted=(posted + [t.strip().lower()
                                          for _ts, t, _s in new])[-40:],
                 headlines=[t for _ts, t, _s in fresh],
                 nSources=nsrc,
                 events=[dict(title=e.get("title"), minutesTo=e.get("minutesTo"),
                              country=e.get("country")) for e in evs[:3]],
                 dxy=(mac.get("dxy") or {}).get("price"),
                 us10y=(mac.get("us10y") or {}).get("price"),
                 gc=(wmac.get("gc") or {}).get("price"),
                 si=(wmac.get("si") or {}).get("price"),
                 ratio=wmac.get("ratio"),
                 spx=(wmac.get("spx") or {}),
                 oil=(wmac.get("cl") or {}))
    r["intel"] = intel
    r["lastIntelT"] = now
    with STATE_LOCK:
        STATE["research"] = r
        _save_state()
    if not new:
        return False
    lines = [f"\U0001F310 OUTSIDE INTEL \u00b7 {nsrc} online sources \u00b7 "
             f"{time.strftime('%H:%M', time.gmtime(now))} UTC", ""]
    for _ts, t, _s in new[:4]:
        b = fundamentals.gold_bias(t)
        ic = {"BULLISH": "\U0001F7E2", "BEARISH": "\U0001F534",
              "MIXED": "\U0001F7E1"}[b["bias"]]
        lines.append(f"{ic} {t[:110]}")
    if evs:
        lines += ["", "\U0001F4C5 next high-impact:"]
        for e in evs[:3]:
            mn = e.get("minutesTo")
            when = f"{mn//60}h{mn%60:02d}m" if mn and mn >= 60 else f"{mn}m"
            lines.append(f"\u2022 {e.get('title')} ({e.get('country')}) "
                         f"in {when}")
    dxy = intel.get("dxy"); us10y = intel.get("us10y")
    if dxy or us10y:
        m = []
        if dxy:
            m.append(f"DXY {dxy:.2f}")
        if us10y:
            m.append(f"US10Y {us10y:.2f}%")
        lines += ["", "\U0001F4B5 macro: " + " \u00b7 ".join(m) +
                  " (dollar/yields up = headwind for gold)"]
    spx = intel.get("spx") or {}
    oil = intel.get("oil") or {}
    ext = []
    if intel.get("ratio"):
        ext.append(f"Au/Ag ratio {intel['ratio']}")
    if spx.get("price"):
        ext.append(f"S&P 500 {spx['price']:,.0f} "
                   f"{spx.get('chgPct', 0):+.1f}%")
    if oil.get("price"):
        ext.append(f"WTI {oil['price']:.1f} {oil.get('chgPct', 0):+.1f}%")
    if ext:
        lines.append("\U0001F30D risk board: " + " \u00b7 ".join(ext))
    lines += ["", "\U0001F9EA research note: outside info feeds the "
              "backtest context \u2014 full analysis in the hourly digest"]
    _notify("\n".join(lines), cat="develop")
    return True


def _llm_researcher():
    """External AI researcher (OpenAI / Anthropic / Gemini / Groq): reads
    the research table + outside intel and posts its own analysis to the
    DEVELOP topic. Auto-activates the moment a provider key exists in the
    environment (set OPENAI_API_KEY etc.); without a key it reports that
    in the live status message. Runs hourly, 24/7, and remembers its own
    previous IDEA lines so each brief builds on the last (self-buffing)."""
    import llm_desk
    conf = llm_desk.configured()
    now = int(time.time())
    r = STATE.get("research") or {}
    if not conf:
        r["llm"] = dict(state="no-key", lastT=now)
        with STATE_LOCK:
            STATE["research"] = r
            _save_state()
        return False
    k = conf[0]
    p = llm_desk.PROVIDERS[k]
    key = os.environ.get(p["key_env"])
    model = os.environ.get(p["model_env"]) or p["model"]
    res = r.get("last") or {}
    intel = r.get("intel") or {}

    def _fr(st):
        if not st or not st.get("n"):
            return "n=0"
        return f"{st['n']} trades, {round((st['win'] or 0) * 100)}% win, {st['avgR']:+.2f}R"

    prod = res.get("production") or {}
    brief = [
        "GOLD (XAUUSD) 15m SNR strategy research brief.",
        f"Production (delta-confirmed, 1:2 RR, 1000-pip cap): window {_fr(prod.get('full'))}, "
        f"out-of-sample {_fr(prod.get('oos'))}.",
        "Challengers (window | out-of-sample):",
    ]
    for v in res.get("variants", []):
        brief.append(f"- {v['name']}: {_fr(v.get('full'))} | {_fr(v.get('oos'))}")
    if intel.get("headlines"):
        brief.append("Outside headlines from " + str(
            intel.get("nSources", 1)) + " online sources: "
            + "; ".join(intel["headlines"][:8]))
    if intel.get("events"):
        brief.append("Upcoming high-impact events: " + "; ".join(
            f"{e['title']} in {e['minutesTo']}m" for e in intel["events"]))
    mac_line = []
    if intel.get("dxy"):
        mac_line.append(f"DXY {intel['dxy']}")
    if intel.get("us10y"):
        mac_line.append(f"US10Y {intel['us10y']}%")
    if intel.get("ratio"):
        mac_line.append(f"gold/silver ratio {intel['ratio']}")
    spx = intel.get("spx") or {}
    if spx.get("price"):
        mac_line.append(f"S&P500 {spx['price']} ({spx.get('chgPct', 0):+.1f}%)")
    oil = intel.get("oil") or {}
    if oil.get("price"):
        mac_line.append(f"WTI {oil['price']} ({oil.get('chgPct', 0):+.1f}%)")
    if mac_line:
        brief.append("Macro/risk board: " + ", ".join(mac_line))
    upg = r.get("upgrade") or {}
    cfg = (f"Current live system config: {upg.get('gate', 'delta')}-gate, "
           f"exit ratio {upg.get('exit', [1.0, 2.0])[0]:g}:"
           f"{upg.get('exit', [1.0, 2.0])[1]:g}")
    if upg.get("pending"):
        cfg += (f", pending upgrade {upg['pending'].get('name')} "
                f"({upg['pending'].get('streak', 1)} consecutive checks)")
    brief.append(cfg)
    ideas = (r.get("llm") or {}).get("ideas") or []
    if ideas:
        brief.append("Your earlier research ideas (follow up — keep the "
                     "ones holding up, refine or replace the rest): "
                     + " | ".join(ideas[-3:]))
    sys_txt = ("You are the always-on research desk of a quantitative gold "
               "(XAUUSD) SNR trading system. Analyze the brief and reply "
               "with compact plain text, max 120 words, three short labeled "
               "lines: MATTER: what matters most for the strategy now; "
               "IDEA: one concrete backtest idea to test next; RISK: one "
               "risk to watch. No markdown, no JSON.")
    txt, err = None, None
    try:
        try:
            txt = llm_desk._CALLS[k](key, model, "\n".join(brief), sys_txt)
        except TypeError:  # older provider signature without sys prompt
            txt = llm_desk._CALLS[k](key, model, "\n".join(brief))
    except Exception as e:  # noqa: BLE001  bad key / no credits / network
        err = str(e)[:200]
    txt = (txt or "").strip()[:900]
    if not txt:
        r["llm"] = dict(state="error", provider=k, model=model,
                        lastT=now, err=err or "empty reply")
        with STATE_LOCK:
            STATE["research"] = r
            _save_state()
        return False
    _notify(f"\U0001F9E0 LLM RESEARCHER \u00b7 {p['name']}\n\n{txt}",
            cat="develop")
    # self-buffing: remember its own IDEA lines so every hourly brief makes
    # the AI follow up on its previous research — knowledge compounds 24/7
    idea = None
    for ln in txt.splitlines():
        s = ln.strip()
        if s.upper().startswith("IDEA") and ":" in s:
            idea = s.split(":", 1)[1].strip()[:140]
            break
    llm_st = dict(state="active", provider=k, model=model, lastT=now)
    if idea:
        llm_st["ideas"] = (((r.get("llm") or {}).get("ideas") or [])
                           + [idea])[-5:]
    r["llm"] = llm_st
    with STATE_LOCK:
        STATE["research"] = r
        _save_state()
    # the researcher also refreshes its trader call — best signal
    threading.Thread(target=ai_trader, kwargs={"auto": True},
                     daemon=True).start()
    return True


# ---------------------------------------------------- AI TRADER
# The external-AI research desk's trader mode: reads the same data the
# desk sees (candles, SNR zones, market delta, 8-model consensus, macro,
# events, live record) and calls its single best trade like a senior
# discretionary gold trader — direction, entry, stop, target, style,
# confidence and a trader's thesis. Strictly validated against the live
# 15m ATR (sane stop, real RR, entry near market). Never auto-executed:
# the gated SNR engine owns live orders. Panel button (manual, always
# posts) + hourly auto-refresh alongside the research brief (auto posts
# only NEW actionable calls, max one per 45 min).

_AI_TRADER_SYS = (
    """You are a senior discretionary XAUUSD (gold) trader with 15 years
of screen time, trading your own account. Read the desk data below and
call your SINGLE best trade for the next few hours. Think like a
trader: location first (fresh SNR zones, structure, HTF trend), then
confirmation (delta pressure, structure shift), then risk (ATR-based
stop, reward at least 1.5x risk when possible). Reply with ONLY compact
JSON, no markdown fences:
{"direction":"buy|sell|none","entry":<price>,"stop":<price>,
"target":<price>,"confidence":0.0-1.0,"style":"scalp|intraday|swing",
"thesis":"max 45 words, trader language",
"invalidation":"max 15 words"}
Use "none" honestly when there is no A+ location right now - real
traders stand aside. Entry may be market or a limit at a zone; stop and
target must fit the direction and the ATR given."""
)

_ai_trader_mem = dict(t=0.0, sig=None, busy=False, lastErr=None,
                      lastPost=dict(t=0.0, key=""))


def _ai_trader_brief():
    """Everything a pro trader would look at, as compact text."""
    try:
        c15 = data.get_candles("15m").get("candles") or []
        c60 = data.get_candles("60m").get("candles") or []
        c1d = data.get_candles("1d").get("candles") or []
    except Exception:  # noqa: BLE001
        return None, None, None
    if len(c15) < 220 or len(c60) < 220:
        return None, None, None
    try:
        spot, _src = tick_spot(broker=False)
    except Exception:  # noqa: BLE001
        spot = None
    spot = float(spot or c15[-1]["c"])
    # every bar and zone in the brief lives in spot space (like the chart)
    c15, _a15 = _align_to_spot(c15, spot)
    c60, _a60 = _align_to_spot(c60, spot)
    c1d, _a1d = _align_to_spot(c1d, spot)
    trs = []
    for i in range(len(c15) - 15, len(c15) - 1):
        h, l, pc = (float(c15[i]["h"]), float(c15[i]["l"]),
                    float(c15[i - 1]["c"]))
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    atr = round(sum(trs) / max(1, len(trs)), 2) or 2.0

    def rows(cl, k):
        out = []
        for c in cl[-k:]:
            hh = time.strftime("%H:%M", time.gmtime(c["t"]))
            out.append(f"{hh} O{float(c['o']):.1f} H{float(c['h']):.1f} "
                       f"L{float(c['l']):.1f} C{float(c['c']):.1f}")
        return "; ".join(out)

    lines = [
        "UTC now: " + time.strftime("%a %H:%M", time.gmtime()),
        "Session: " + str(entries.session_of(time.time())),
        f"XAUUSD spot: {spot:,.2f} · 15m ATR(14): {atr}",
        f"15m last 8 bars: {rows(c15, 8)}",
        f"1H last 6 bars: {rows(c60, 6)}",
        f"1D last 4 bars: {rows(c1d, 4)}",
    ]
    try:
        ent = entries.evaluate(c15, c60)
    except Exception:  # noqa: BLE001
        ent = None
    if ent:
        lines.append(
            f"Desk SNR setup: {ent['direction']} grade {ent['grade']} "
            f"(active={ent['active']}, touching={ent['touching']}) "
            f"entry~{ent['entry']} SL {ent['sl']} TP1 {ent['tp1']} "
            f"TP2 {ent['tp2']} · 1H {ent['trend1h']}")
    else:
        lines.append("Desk SNR setup: none armed (zones may be arming)")
    try:
        zones = entries.zone_radar(c15, c60, price=spot, max_zones=6)
        if zones:
            lines.append("SNR zones near price (nearest first): " + "; ".join(
                f"{z['tf']} {z['side']} {z['bottom']:.0f}-{z['top']:.0f} "
                f"{z['status']} ({z['passed']}/5)" for z in zones))
    except Exception:  # noqa: BLE001
        pass
    try:
        dv = entries.delta_view(c15)
        if dv:
            d = (f"Market delta: {dv['state']}, "
                 f"{round(dv['buyPct'] * 100)}% buy volume")
            if dv.get("divergence"):
                d += f", divergence {dv['divergence']}"
            lines.append(d)
    except Exception:  # noqa: BLE001
        pass
    try:
        dsk = get_ai_desk() or {}
        c = dsk.get("consensus") or {}
        if c:
            lines.append(f"8-model consensus: {c.get('label')} "
                         f"(score {float(c.get('score') or 0):+.2f})")
        for s in ((dsk.get("llm") or {}).get("seats") or []):
            lines.append(f"External AI {s['name']}: {s['verdict']} "
                         f"conf {s['conf']}")
    except Exception:  # noqa: BLE001
        pass
    try:
        f = fundamentals.snapshot() or {}
        m = f.get("macro") or {}
        mac = []
        for k, lbl in (("dxy", "DXY"), ("us10y", "US10Y")):
            if m.get(k):
                mac.append(f"{lbl} {m[k].get('chgPct', 0):+.2f}%")
        if mac:
            lines.append("Macro: " + ", ".join(mac) + " (up = gold headwind)")
        nx = f.get("nextHigh") or {}
        if nx.get("title"):
            lines.append(f"Next high-impact event: {nx['title']} "
                         f"in {nx.get('minutesTo', '?')}m")
    except Exception:  # noqa: BLE001
        pass
    try:
        res = ((STATE.get("research") or {}).get("last") or {})
        prod = (res.get("production") or {}).get("full") or {}
        if prod.get("n"):
            lines.append(f"Desk live record: {prod['n']} trades, "
                         f"{round((prod.get('win') or 0) * 100)}% win, "
                         f"{prod.get('avgR', 0):+.2f}R avg")
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(lines), spot, atr


def _ai_trader_parse(txt):
    if not txt:
        return None
    import re as _re
    m = _re.search(r"\{.*\}", txt, _re.S)
    if not m:
        return None
    try:
        j = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None
    return j if isinstance(j, dict) else None


def _ai_trader_check(j, spot, atr):
    """Validate + normalize the trader's call. None if it does not hold
    up against the live market (sane stop, real RR, entry near price)."""
    if not j:
        return None
    d = str(j.get("direction") or "").lower()
    if d == "long":
        d = "buy"
    elif d == "short":
        d = "sell"
    if d not in ("buy", "sell", "none"):
        return None
    style = str(j.get("style") or "intraday").lower()
    style = style if style in ("scalp", "intraday", "swing") else "intraday"
    try:
        conf = min(0.95, max(0.05, float(j.get("confidence"))))
    except (TypeError, ValueError):
        conf = 0.5
    thesis = str(j.get("thesis") or j.get("note") or "").strip()[:300]
    inval = str(j.get("invalidation") or "").strip()[:160]
    if d == "none":
        return dict(direction="none", conf=round(conf, 2), style=style,
                    thesis=thesis or "no A+ location — standing aside",
                    invalidation=inval)
    try:
        entry, stop, target = (float(j.get("entry")), float(j.get("stop")),
                               float(j.get("target")))
    except (TypeError, ValueError):
        return None
    if spot <= 0:
        return None
    if abs(entry - spot) > 0.010 * spot:       # >1% from market = fantasy
        return None
    risk = (entry - stop) if d == "buy" else (stop - entry)
    rew = (target - entry) if d == "buy" else (entry - target)
    if risk <= 0 or rew <= 0:
        return None
    a = atr if atr and atr > 0 else max(2.0, spot * 0.001)
    if not 0.25 * a <= risk <= 4.0 * a:        # sane vs live volatility
        return None
    rr = rew / risk
    if rr < 1.1:                               # a pro's best call pays
        return None
    return dict(direction=d, entry=round(entry, 2), stop=round(stop, 2),
                target=round(target, 2), rr=round(rr, 1),
                riskAtr=round(risk / a, 2), conf=round(conf, 2),
                style=style, thesis=thesis, invalidation=inval)


def _ai_trader_post(sig):
    """Signal-topic card, styled like the desk's own signal cards."""
    m = sig.get("model") or "AI"
    if sig["direction"] == "none":
        _notify(f"\U0001F9E0 AI TRADER · {m}\n\n"
                f"\u26aa STAND ASIDE — {sig.get('thesis') or 'no A+ location'}\n\n"
                f"\u26a0 AI research opinion — not the auto-trade engine",
                cat="signal")
        return
    arrow = "\U0001F7E2 BUY" if sig["direction"] == "buy" else "\U0001F534 SELL"
    conf_line = (f"\u26a1 {sig['confluence']}\n\n"
                 if sig.get("confluence") else "\n")
    _notify(
        f"\U0001F9E0 AI TRADER · {m}\n\n"
        f"{arrow} · XAUUSD · {sig.get('style')}\n\n"
        f"\U0001F3AF Entry {sig['entry']:,.2f}\n"
        f"\U0001F6D1 SL {sig['stop']:,.2f}\n"
        f"\U0001F3C1 TP {sig['target']:,.2f}\n\n"
        + conf_line +
        f"R:R 1:{sig['rr']:g} · risk {sig['riskAtr']:g}×ATR · "
        f"confidence {round(sig['conf'] * 100)}%\n"
        f"\U0001F5E3 {sig.get('thesis') or '—'}\n"
        f"\u274C dead if: {sig.get('invalidation') or '—'}\n\n"
        f"\u26a0 AI research opinion — not the auto-trade engine",
        cat="signal")


AI_TRADE_MAX = 50          # tracked AI calls kept (newest first)
AI_FILL_BARS = 16          # bars a limit/stop entry gets to fill (4h —
                           # an intraday thesis is stale after that)
AI_HOLD_BARS = 48          # bars after fill before mark-to-market (12h)


def _ai_trader_record(sig):
    """Track an actionable AI call so the desk can score it honestly.
    Deduped (same open call re-pressed within 30min is skipped), capped,
    persisted to the shared cloud state."""
    trades = STATE.setdefault("aiTrades", [])
    last = trades[0] if trades else None
    if (last and last.get("status") == "open"
            and last.get("direction") == sig["direction"]
            and abs((last.get("entry") or 0) - sig["entry"]) < 0.05
            and time.time() - (last.get("t") or 0) < 1800):
        return
    with STATE_LOCK:
        trades.insert(0, dict(
            t=sig["t"], direction=sig["direction"], entry=sig["entry"],
            stop=sig["stop"], target=sig["target"], rr=sig["rr"],
            conf=sig["conf"], style=sig.get("style"), spot=sig.get("spot"),
            model=sig.get("model"), thesis=(sig.get("thesis") or "")[:200],
            status="open", fillT=None, exitT=None, r=None))
        del trades[AI_TRADE_MAX:]
        _save_state()
    cloud_save(force=True)


def _ai_trader_resolve():
    """Score open AI calls against REAL 15m bars. Conservative rules,
    same as the engine's backtests: entry fills on touch (marketable
    entries fill on the first bar after the call); if the fill bar can
    also be a stop-out it IS one; stop before target on the same bar is
    a loss; no fill in 12h or no exit in 12h after fill = marked to
    market. Wins pay the call's own R:R."""
    try:
        candles = data.get_candles("15m").get("candles") or []
    except Exception:  # noqa: BLE001
        return
    if len(candles) < 30:
        return
    changed = False
    for tr in [t for t in (STATE.get("aiTrades") or [])
               if t.get("status") == "open"]:
        bars = [c for c in candles if c["t"] >= (tr.get("t") or 0) + 900]
        if not bars:
            continue                          # call newer than the data
        buy = tr["direction"] == "buy"
        entry, stop, target = (float(tr["entry"]), float(tr["stop"]),
                               float(tr["target"]))
        risk = abs(entry - stop) or 1.0
        spot = float(tr.get("spot") or entry)
        marketable = (entry <= spot) if buy else (entry >= spot)
        status = None
        r = None
        filled = False
        j = 0                                 # bars since fill
        for i, c in enumerate(bars):
            hi, lo = float(c["h"]), float(c["l"])
            if not filled:
                if buy:                  # limit/market fills on a low
                    hit = (lo <= entry) if marketable else (hi >= entry)
                else:                    # stop entries fill on a break
                    hit = (hi >= entry) if marketable else (lo <= entry)
                if hit:
                    filled = True
                    tr["fillT"] = c["t"]
                    if (lo <= stop if buy else hi >= stop):
                        status, r = "loss", -1.0
                    elif (hi >= target if buy else lo <= target):
                        status, r = "win", float(tr.get("rr") or 0)
                elif i >= AI_FILL_BARS:
                    status, r = "nofill", 0.0
                if status:
                    break
                continue
            j += 1
            if (lo <= stop if buy else hi >= stop):
                status, r = "loss", -1.0
                break
            if (hi >= target if buy else lo <= target):
                status, r = "win", float(tr.get("rr") or 0)
                break
            if j >= AI_HOLD_BARS:
                close = float(c["c"])
                r = ((close - entry) if buy else (entry - close)) / risk
                status = "timeout"
                break
        if status:
            tr["status"] = status
            tr["r"] = round(r, 2)
            tr["exitT"] = int(time.time())
            changed = True
            if status in ("win", "loss", "timeout"):
                st = _ai_trader_stats()
                emoji = {"win": "\u2705", "loss": "\u274c",
                         "timeout": "\u23f1"}[status]
                _notify(
                    f"\U0001F3AF AI TRADER CALL {status.upper()} {emoji}\n\n"
                    f"{tr['direction'].upper()} {entry:,.2f} \u00b7 "
                    f"SL {stop:,.2f} \u00b7 TP {target:,.2f}\n"
                    f"result {tr['r']:+.2f}R \u00b7 record: {st['n']} "
                    f"calls, {st['winPct'] or 0}% win, "
                    f"{st['avgR'] or 0:+.2f}R avg", cat="signal")
    if changed:
        with STATE_LOCK:
            _save_state()
        cloud_save(force=True)


def _ai_trader_stats():
    """Honest scorecard: only resolved calls count (open + no-fill
    excluded)."""
    rs = [t.get("r") for t in (STATE.get("aiTrades") or [])
          if t.get("status") in ("win", "loss", "timeout")
          and t.get("direction") in ("buy", "sell")
          and isinstance(t.get("r"), (int, float))]
    if not rs:
        return dict(n=0, winPct=None, avgR=None)
    return dict(n=len(rs),
                winPct=round(100 * sum(1 for r in rs if r > 0) / len(rs)),
                avgR=round(sum(rs) / len(rs), 2))


def _ai_trades_merge(remote, local):
    """Cloud merge for tracked AI calls: union by call key; a resolved
    copy beats an open one; local wins ties; newest first, cap 50."""
    def key(t):
        return f"{t.get('t')}|{t.get('direction')}|{t.get('entry')}"

    def rank(t):
        return 1 if t.get("status") in ("win", "loss", "timeout",
                                        "nofill") else 0

    m = {}
    for t in (remote or []):
        if t.get("direction") in ("buy", "sell"):
            m[key(t)] = t
    for t in (local or []):
        if t.get("direction") not in ("buy", "sell"):
            continue
        k = key(t)
        if k not in m or rank(t) >= rank(m[k]):
            m[k] = t
    return sorted(m.values(), key=lambda t: t.get("t") or 0,
                  reverse=True)[:AI_TRADE_MAX]


_ai_regime_mem = {"t": 0.0, "v": 0}


def _ai_regime():
    """60m trend regime for the AI-trader gate: EMA20 vs EMA50 on hourly
    candles with a dead-band (0.25 x 14-bar ATR) so flat, undecided
    markets allow both directions. +1 bull, -1 bear, 0 undecided.
    Backtest (71 setups, 2.3 months): with-trend 50% / +0.00R vs
    counter-trend 37% / -0.28R — and BOTH recorded AI losses were
    counter-trend sells in a BULL 60m regime."""
    if time.time() - _ai_regime_mem["t"] < 60:
        return _ai_regime_mem["v"]
    v = 0
    try:
        cl = data.get_candles("60m").get("candles") or []
        if len(cl) >= 60:
            c = [float(k["c"]) for k in cl[-120:]]
            k20, k50 = 2 / 21, 2 / 51
            e20 = e50 = c[0]
            for x in c:
                e20 += k20 * (x - e20)
                e50 += k50 * (x - e50)
            h = [float(k["h"]) for k in cl[-14:]]
            l = [float(k["l"]) for k in cl[-14:]]
            atr14 = sum(hi - lo for hi, lo in zip(h, l)) / 14
            if abs(e20 - e50) > 0.25 * max(atr14, 0.01):
                v = 1 if e20 > e50 else -1
    except Exception:  # noqa: BLE001
        v = 0
    _ai_regime_mem.update(t=time.time(), v=v)
    return v


def ai_trader(force=False, auto=False):
    """Generate the AI trader's best signal. force=True re-runs even if
    a fresh call exists (the panel button); auto=True is the hourly
    refresh which posts only NEW actionable calls."""
    import llm_desk
    if _ai_trader_mem["busy"]:
        return _ai_trader_mem.get("sig")
    if (not force and _ai_trader_mem["sig"]
            and time.time() - _ai_trader_mem["t"] < 900):
        return _ai_trader_mem["sig"]
    conf = llm_desk.configured()
    if not conf:
        _ai_trader_mem["lastErr"] = "no AI provider key configured"
        return None
    _ai_trader_mem.update(busy=True)
    try:
        k = conf[0]
        p = llm_desk.PROVIDERS[k]
        key = os.environ.get(p["key_env"])
        model = os.environ.get(p["model_env"]) or p["model"]
        brief, spot, atr = _ai_trader_brief()
        if not brief:
            _ai_trader_mem["lastErr"] = "market data not warm yet"
            return None
        call = llm_desk._CALLS[k]
        sys_txt = (_AI_TRADER_SYS + f" Current spot {spot:,.2f}, 15m ATR "
                   f"{atr} — entry/stop/target must be consistent.")
        txt = sig = None
        for _attempt in (1, 2):               # one corrective retry
            try:
                txt = call(key, model, brief, sys_txt)
            except Exception as e:  # noqa: BLE001
                _ai_trader_mem["lastErr"] = str(e)[:200]
                return None
            sig = _ai_trader_check(_ai_trader_parse(txt), spot, atr)
            if sig:
                break
            brief += ("\n\nYour previous reply was rejected: it must be "
                      "compact JSON with direction/entry/stop/target "
                      "consistent with the spot and ATR (stop 0.25-4 ATR "
                      "away, reward >= 1.1x risk), or direction \"none\". "
                      "Call your best trade again.")
        if not sig:
            sig = dict(direction="none", conf=0.5, style="intraday",
                       thesis="model reply did not pass the risk checks — "
                              "standing aside", invalidation="")
        sig.update(t=int(time.time()), provider=p["name"], model=model,
                   spot=round(spot, 2), atr=atr)
        # ---- v2.9.12 REGIME GATE: never trade against the 60m trend.
        #      The losing cohort of the AI journal (both losses) and of
        #      the 71-setup backtest. Blocked calls stand aside with the
        #      reason visible in the panel.
        if sig["direction"] in ("buy", "sell"):
            rg = _ai_regime()
            against = (sig["direction"] == "buy" and rg < 0) or \
                      (sig["direction"] == "sell" and rg > 0)
            if against:
                sig = dict(direction="none", conf=sig.get("conf"),
                           style=sig.get("style"),
                           thesis=(f"{(sig.get('thesis') or '')[:140]} "
                                   "— DESK BLOCK: counter-trend vs the 60m "
                                   "regime (losing cohort: 37% win / "
                                   "-0.28R). Standing aside."),
                           invalidation=sig.get("invalidation", ""))
        try:                                   # confluence with the desk
            _d = get_ai_desk() or {}
            _cl = str(((_d.get("consensus") or {}).get("label"))
                      or "").lower()
            if sig["direction"] == "buy" and "bull" in _cl:
                sig["confluence"] = "8-model consensus bullish"
            elif sig["direction"] == "sell" and "bear" in _cl:
                sig["confluence"] = "8-model consensus bearish"
        except Exception:  # noqa: BLE001
            pass
        if sig["direction"] != "none":
            try:
                _ai_trader_record(sig)
            except Exception:  # noqa: BLE001  — never kill the call
                pass
        _ai_trader_mem.update(t=sig["t"], sig=sig, lastErr=None)
        open_n = sum(1 for t in (STATE.get("aiTrades") or [])
                      if t.get("status") == "open")
        crowded = open_n >= 4
        if force or (auto and sig["direction"] != "none" and not crowded):
            pk = f"{sig['direction']}:{sig.get('entry')}:{sig.get('stop')}"
            lp = _ai_trader_mem["lastPost"]
            if force or (pk != lp["key"] or time.time() - lp["t"] > 2700):
                _ai_trader_post(sig)
                _ai_trader_mem["lastPost"] = dict(t=time.time(), key=pk)
        return sig
    finally:
        _ai_trader_mem["busy"] = False


# ------------------------------------------------- MTF SIGNAL SCAN
# Every-timeframe analysis (15m/1H/2H/4H/1D) running every minute: per-TF
# trend, the zones near price, the 15m setup state and the EXACT market
# delta. The best signal across all TFs is alerted to the group the
# moment it becomes tradeable — members no longer wait for the perfect
# 8/8 A+ to know what the desk sees. Every alert carries the age of the
# last closed bar so freshness is provable.

def _align_to_spot(cl, spot):
    """Shift raw (futures-space) candles onto the spot level — the same
    instantaneous adjust the main chart uses, so every TF, zone level
    and AI-trader number lives in ONE price space (spot, matching
    TradingView). Returns (aligned_candles, adjust)."""
    if not cl or not spot:
        return cl, 0.0
    adj = float(cl[-1]["c"]) - float(spot)
    if not (0 < adj < 150):            # sanity: normal contango only
        return cl, 0.0
    out = [dict(t=k["t"], o=float(k["o"]) - adj, h=float(k["h"]) - adj,
                l=float(k["l"]) - adj, c=float(k["c"]) - adj)
           for k in cl]
    return out, adj


MTF_TFS = ["15m", "60m", "2h", "4h", "1d"]
MTF_TREND = {1: "BULL", -1: "BEAR", 0: "RANGE"}
_mtf_mem = dict(lastAlerts={}, lastGlobal=0.0, lastScan=None)


def _mtf_scan():
    """One full multi-timeframe pass. Cheap (packs are cached) and safe."""
    try:
        spot, _src = tick_spot(broker=False)
    except Exception:  # noqa: BLE001
        spot = None
    tfs = {}
    c15 = c60 = None
    for tf in MTF_TFS:
        try:
            cl = data.get_candles(tf).get("candles") or []
        except Exception:  # noqa: BLE001
            cl = []
        if len(cl) < 220:
            continue
        cl, _adj = _align_to_spot(cl, spot)     # ONE price space: spot
        try:
            pack = entries._snr_pack(cl)
        except Exception:  # noqa: BLE001
            continue
        i_last = pack["n"] - 2
        trend = MTF_TREND.get(int(pack["struct"][i_last]), "RANGE")
        # zone detection in CANDLE space (same as the engine's own radar
        # and the panel) — never mix spot quotes with candle-space zones
        px = float(cl[-1]["c"])
        try:
            zones = entries.zone_radar(cl, None, price=px, max_zones=3)
        except Exception:  # noqa: BLE001
            zones = []
        for z in zones:
            z["tf"] = tf                      # relabel with the real TF
        row = dict(tf=tf, trend=trend, close=round(float(cl[-1]["c"]), 2),
                   zones=zones)
        tfs[tf] = row
        if tf == "15m":
            c15 = cl
        if tf == "60m":
            c60 = cl
    if not tfs:
        return None
    out = dict(t=int(time.time()), tfs=tfs,
               spot=round(float(spot or 0), 2) or None)
    # exact delta on the freshest 15m data
    dv = None
    if c15:
        try:
            dv = entries.delta_view(c15)
        except Exception:  # noqa: BLE001
            dv = None
        out["delta"] = dv and dict(
            ema9=dv.get("ema9"), ema21=dv.get("ema21"),
            buyPct=dv.get("buyPct"), state=dv.get("state"),
            divergence=dv.get("divergence"), cum=dv.get("cum"))
        out["barAge"] = int(max(0, time.time() - c15[-2]["t"]))
    # 15m setup (full checks) — the engine's own view
    ent = None
    if c15 and c60:
        try:
            ent = entries.evaluate(c15, c60)
        except Exception:  # noqa: BLE001
            ent = None
    if ent:
        out["setup"] = dict(
            direction=ent.get("direction"), passed=ent.get("passed", 0),
            grade=ent.get("grade") or "", touching=ent.get("touching"),
            active=ent.get("active"),
            entry=ent.get("entry"), sl=ent.get("sl"),
            tp1=ent.get("tp1"), tp2=ent.get("tp2"),
            session=ent.get("sessionLabel"),
            missing=[c["label"] for c in ent.get("checks") or []
                     if not c["ok"]])
    # ---- pick the BEST signal across all TFs ----
    best = None
    def better(cand):
        return best is None or cand["score"] > best["score"]
    if ent and not ent.get("touching") and not ent.get("active") \
            and (ent.get("passed") or 0) >= 6 \
            and ent.get("grade") in (None, "", "\u2014") \
            and ent.get("entry") is not None:
        cand = dict(kind="arming", tf="15m",
                    side=("demand" if ent.get("direction") == "LONG"
                          else "supply"),
                    direction=ent.get("direction") or "LONG",
                    passed=ent["passed"], entry=ent["entry"],
                    sl=ent.get("sl") or 0, tp1=ent.get("tp1") or 0,
                    tp2=ent.get("tp2") or 0,
                    missing=ent.get("checks") or [],
                    score=ent["passed"] + 2)
        if better(cand):
            best = cand
    for tf, row in tfs.items():
        for z in row["zones"]:
            if z["status"] == "live":
                side_dir = "LONG" if z["side"] == "demand" else "SHORT"
                cand = dict(kind="live", tf=tf, side=z["side"],
                            direction=side_dir, passed=z["passed"],
                            lo=z["bottom"], hi=z["top"], zone=z,
                            score=10 + z["passed"])
                if better(cand):
                    best = cand
    bulls = [tf for tf, r in tfs.items() if r["trend"] == "BULL"]
    bears = [tf for tf, r in tfs.items() if r["trend"] == "BEAR"]
    align = None
    if len(bulls) >= 4:
        align = dict(direction="LONG", tfs=bulls)
    elif len(bears) >= 4:
        align = dict(direction="SHORT", tfs=bears)
    out["bias"] = dict(bulls=len(bulls), bears=len(bears),
                       total=len(tfs), align=align and align["direction"])
    if align and best is None:
        want = "demand" if align["direction"] == "LONG" else "supply"
        for tf, row in tfs.items():
            for z in row["zones"]:
                if z["side"] == want and z["passed"] >= 4 \
                        and z["status"] != "tested":
                    cand = dict(kind="biaszone", tf=tf, side=z["side"],
                                direction=align["direction"],
                                passed=z["passed"], lo=z["bottom"],
                                hi=z["top"], zone=z,
                                score=6 + z["passed"])
                    if better(cand):
                        best = cand
    out["best"] = best
    return out


def _mtf_alert(scan):
    """Post the best signal to the group immediately. Deduped per zone /
    kind (45 min) with a global gap (20 min) so members get signal
    quality, never spam. The 8/8 A+ phone cards stay separate."""
    b = (scan or {}).get("best")
    if not b:
        return
    spot = scan.get("spot") or (b.get("entry") or 0)
    if b["kind"] == "live":
        key = f"live:{b['tf']}:{b['lo']:.0f}-{b['hi']:.0f}"
        head = (f"\U0001F534 LIVE ZONE · {b['tf']} "
                f"{'DEMAND' if b['side'] == 'demand' else 'SUPPLY'}\n\n"
                f"Full-house retest right now — {b['passed']}/5 checks\n"
                f"Zone {b['lo']:,.1f} \u2013 {b['hi']:,.1f} "
                f"(spot {spot:,.2f})")
    elif b["kind"] == "arming":
        key = f"arming:{b['entry']:.0f}:{b['direction']}"
        head = (f"\u23f3 BEST SIGNAL ARMING · 15m {b['direction']}\n\n"
                f"{b['passed']}/8 confluence \u2014 tradeable quality\n"
                f"Entry zone {b['entry']:,.2f} \u00b7 SL {b['sl']:,.2f}\n"
                f"TP1 {b['tp1']:,.2f} \u00b7 TP2 {b['tp2']:,.2f}\n"
                f"Missing: " + " \u00b7 ".join(
                    c["label"] for c in b["missing"][:3]))
    else:
        key = f"bias:{b['tf']}:{b['lo']:.0f}-{b['hi']:.0f}"
        head = (f"\u26a1 MTF CONFLUENCE · {b['direction']}\n\n"
                f"{scan['bias']['bulls'] if b['direction'] == 'LONG' else scan['bias']['bears']}"
                f"/{scan['bias']['total']} timeframes aligned "
                f"{'bullish' if b['direction'] == 'LONG' else 'bearish'}\n"
                f"Best zone: {b['tf']} "
                f"{'demand' if b['side'] == 'demand' else 'supply'} "
                f"{b['lo']:,.1f} \u2013 {b['hi']:,.1f} "
                f"({b['passed']}/5)")
    now = time.time()
    if _mtf_mem["lastAlerts"].get(key, 0) and \
            now - _mtf_mem["lastAlerts"][key] < 2700:
        return
    if now - _mtf_mem["lastGlobal"] < 1200:
        return
    # exact delta line — the same numbers that gate the engine
    dline = ""
    dv = scan.get("delta")
    if dv:
        e9, e21 = dv.get("ema9"), dv.get("ema21")
        dline = (f"\n\U0001F4CA EXACT DELTA: EMA9 {e9:+.3f} \u00b7 "
                 f"EMA21 {e21:+.3f} \u00b7 "
                 f"{round((dv.get('buyPct') or 0.5) * 100)}% buy volume"
                 f" \u00b7 {dv.get('state', '—')}")
        if dv.get("divergence"):
            dline += f" \u00b7 divergence {dv['divergence']}"
    age = scan.get("barAge")
    fresh = f"\n\u23f1 data fresh \u2014 last 15m bar closed {age // 60}m ago" \
        if age is not None else ""
    trend_line = ""
    tb = scan.get("bias") or {}
    if tb:
        trend_line = (f"\n\U0001F4C8 TF trends: "
                      f"{tb['bulls']} bull / {tb['bears']} bear / "
                      f"{tb['total']} total")
    _notify(f"\U0001F4E1 {head}\n"
            f"{trend_line}{dline}{fresh}\n\n"
            f"\u26a0 analysis alert \u2014 the 8/8 A+ trade card still "
            f"fires automatically when everything confirms", cat="signal")
    _mtf_mem["lastAlerts"][key] = now
    _mtf_mem["lastAlerts"] = {k: v for k, v in _mtf_mem["lastAlerts"].items()
                              if now - v < 7200}
    _mtf_mem["lastGlobal"] = now


def _mtf_loop():
    time.sleep(150)                       # let all TF data warm up
    while True:
        try:
            scan = _mtf_scan()
            if scan:
                _mtf_mem["lastScan"] = scan
                _mtf_alert(scan)
        except Exception as e:  # noqa: BLE001
            print(f"[mtf] scan failed: {e}", flush=True)
        time.sleep(60)


# ------------------------------------------------- self-upgrade engine
# Research findings now UPDATE the live system by themselves: when a
# challenger variant beats production out-of-sample (n>=5, +0.20R edge)
# for 3 consecutive hourly checks, it is PROMOTED — the live signal gate
# and exit profile switch automatically and an upgrade card posts to the
# DEVELOP topic. Live results are then watched: if the promoted config
# underperforms the old baseline (n>=8 live trades, avgR below baseline
# -0.10R), it rolls back automatically. Max one promotion per 24h.

UPGRADE_EDGE_R = 0.20
UPGRADE_STREAK = 3
UPGRADE_COOLDOWN_S = 86400
ROLLBACK_MIN_N = 8
ROLLBACK_EDGE_R = 0.10

_KNOBS = {   # research variant -> (gate, exit) live knobs it changes
    "no-delta-gate":           ("none", None),
    "continuation-only+delta": ("cont-only", None),
    "live-session+delta":      ("session", None),
    "exit 1:2.5":              (None, (1.0, 2.5)),
    "exit 0.75:1.5":           (None, (0.75, 1.5)),
}


def _upgrade_state(r):
    up = r.get("upgrade")
    if not up:
        up = dict(gate="delta", exit=[1.0, 2.0], history=[],
                  pending=None, prev=None, lastPromoteT=0)
        r["upgrade"] = up
    return up


def _maybe_upgrade(r, res):
    """Promote a challenger that has beaten production out-of-sample for
    UPGRADE_STREAK consecutive hourly checks. Called from the digest."""
    up = _upgrade_state(r)
    po = (res.get("production") or {}).get("oos") or {}
    if (po.get("n") or 0) >= 3 and po.get("avgR") is not None:
        best = None
        for v in res.get("variants") or []:
            vo = v.get("oos") or {}
            if (vo.get("n") or 0) >= 5 and vo.get("avgR") is not None \
                    and vo["avgR"] > po["avgR"] + UPGRADE_EDGE_R:
                if best is None or vo["avgR"] > best[1]["oos"]["avgR"]:
                    best = (v["name"], v)
        if best:
            p = up.get("pending") or {}
            if p.get("name") == best[0]:
                p["streak"] = (p.get("streak") or 0) + 1
            else:
                p = dict(name=best[0], streak=1)
            p["oos"] = dict(best[1]["oos"])
            p["prodOos"] = dict(po)
            up["pending"] = p
            if p["streak"] >= UPGRADE_STREAK and time.time() - \
                    up.get("lastPromoteT", 0) >= UPGRADE_COOLDOWN_S:
                _apply_upgrade(r, up, p)
                return
        else:
            up["pending"] = None
    _maybe_rollback(r)


def _apply_upgrade(r, up, p):
    name = p["name"]
    gate, exitr = _KNOBS.get(name, (None, None))
    prev = dict(gate=up.get("gate") or "delta",
                exit=list(up.get("exit") or [1.0, 2.0]))
    if gate:
        up["gate"] = gate
    if exitr:
        up["exit"] = list(exitr)
    up["prev"] = prev
    up["pending"] = None
    up["lastPromoteT"] = int(time.time())
    o, q = p["oos"], p["prodOos"]
    up["history"] = (up.get("history") or [])[-9:] + [dict(
        ts=up["lastPromoteT"], name=name, gate=up["gate"],
        exit=list(up["exit"]), oos=dict(o), prodOos=dict(q))]
    with STATE_LOCK:
        STATE["research"] = r
        _save_state()
    _notify(
        "⬆️ SYSTEM UPGRADE · auto-applied\n\n"
        f"🔬 research proved: {name}\n"
        f"   OOS {o['n']} trades · {round((o['win'] or 0) * 100)}% · "
        f"{o['avgR']:+.2f}R\n"
        f"   vs production OOS {q['n']} · {round((q['win'] or 0) * 100)}% · "
        f"{q['avgR']:+.2f}R\n\n"
        f"⚙ live config: gate {prev['gate']} → {up['gate']} · exit "
        f"{prev['exit'][0]:g}:{prev['exit'][1]:g} → "
        f"{up['exit'][0]:g}:{up['exit'][1]:g}\n"
        f"🛡 live watch: auto-rollback if live avgR < "
        f"{q['avgR'] - ROLLBACK_EDGE_R:+.2f}R over {ROLLBACK_MIN_N} trades",
        cat="develop")


def _maybe_rollback(r):
    """Watch live results of a promoted config; revert if it disappoints."""
    up = r.get("upgrade") or {}
    prev = up.get("prev")
    if not prev or not up.get("lastPromoteT"):
        return
    prom_t = up["lastPromoteT"]
    rs = [t["r"] for t in (STATE.get("tradeHistory") or [])
          if (t.get("openedAt") or 0) > prom_t
          and t.get("src") in ("A+", "B+") and t.get("status") == "closed"
          and isinstance(t.get("r"), (int, float))]
    if len(rs) < ROLLBACK_MIN_N:
        return
    avg = sum(rs) / len(rs)
    base = None
    for h in reversed(up.get("history") or []):
        if h.get("name") != "rollback":
            base = (h.get("prodOos") or {}).get("avgR")
        break
    if base is None:
        return
    if avg < base - ROLLBACK_EDGE_R:
        gate = prev.get("gate") or "delta"
        exit_p = prev.get("exit") or [1.0, 2.0]
        up["gate"] = gate
        up["exit"] = list(exit_p)
        up["prev"] = None
        up["history"] = (up.get("history") or [])[-9:] + [dict(
            ts=int(time.time()), name="rollback", gate=gate,
            exit=list(exit_p), liveAvgR=round(avg, 2), baseR=base)]
        with STATE_LOCK:
            STATE["research"] = r
            _save_state()
        _notify(
            "⬇️ AUTO-ROLLBACK · upgrade withdrawn\n\n"
            f"📉 live results since upgrade: {len(rs)} trades · "
            f"{avg:+.2f}R avg (baseline {base:+.2f}R)\n"
            f"⚙ config restored: gate {gate} · exit "
            f"{exit_p[0]:g}:{exit_p[1]:g}\n"
            "🔬 research continues — a challenger must re-qualify "
            "out-of-sample to return", cat="develop")


def research_cycle(force=False):
    """Scan + digest in one call (used by tests and the hourly path)."""
    got = _research_scan(force_data=force)
    if not got:
        return None
    r, res = got
    _research_digest(r, res)
    return res


def _research_loop():
    """Always-on research daemon: live status message edited every 30s
    (uptime timer, scan countdown, activity), full research scan every 60s,
    hourly digest to the DEVELOP topic — 24/7 non-stop. Only the
    flock-owning worker runs it."""
    time.sleep(120)
    while True:
        try:
            r = STATE.get("research") or {}
            if time.time() - r.get("lastScanT", 0) >= RESEARCH_SCAN_S:
                got = _research_scan()
                if got:
                    r, res = got
                    if time.time() - r.get("lastDigestT", 0) \
                            >= RESEARCH_DIGEST_S:
                        _research_digest(r, res)
            if _cloud_mem.get("dirty"):
                cloud_save(force=True)
            _mt5_retry_stuck()
            _calendar_share()
            if time.time() - r.get("lastIntelT", 0) >= INTEL_SCAN_S:
                _research_intel()
            if time.time() - (r.get("llm") or {}).get("lastT", 0) \
                    >= LLM_RESEARCH_S:
                _llm_researcher()
            _research_status()
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
        time.sleep(RESEARCH_STATUS_S)


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
        try:
            _ai_trader_resolve()              # score open AI-trader calls
        except Exception:  # noqa: BLE001
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


_desk_mem = dict(lastVersionPost=None, lastDigestPost=None)
DESK_UPDATE_S = 6 * 3600      # member digest cadence


def _announce_version():
    """On boot: if this software version was never announced, tell the
    members in DEVELOP what changed — the system informs the group by
    itself on every deploy."""
    time.sleep(120)                       # let the feeds warm up first
    try:
        if not _desk_speaker():
            return
        if STATE.get("sysVer") == SYSTEM_VERSION:
            return
        lines = [f"\U0001F4E1 SYSTEM UPDATE \u00b7 v{SYSTEM_VERSION}", ""]
        for i, ch in enumerate(
                SYSTEM_CHANGELOG.get(SYSTEM_VERSION) or [], 1):
            lines.append(f"{i}. {ch}")
        lines += ["", "The desk hosts and updates itself \u2014 members "
                  "are informed here automatically on every deploy."]
        _notify("\n".join(lines), cat="develop")
        with STATE_LOCK:
            STATE["sysVer"] = SYSTEM_VERSION
            STATE["sysVerT"] = int(time.time())
            _save_state()
        cloud_save(force=True)
        _desk_mem["lastVersionPost"] = int(time.time())
        print(f"[desk] announced v{SYSTEM_VERSION} to DEVELOP",
              flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[desk] version announce failed: {e}", flush=True)


def _desk_update_digest():
    """One compact card for the members: what the desk is doing."""
    lines = ["\U0001F4E1 DESK UPDATE \u00b7 "
             + time.strftime("%a %H:%M UTC", time.gmtime()), ""]
    res = ((STATE.get("research") or {}).get("last") or {})
    prod = (res.get("production") or {}).get("full") or {}
    if prod.get("n"):
        lines.append(
            f"\U0001F916 Engine record: {prod['n']} trades \u00b7 "
            f"{round((prod.get('win') or 0) * 100)}% win \u00b7 "
            f"{prod.get('avgR', 0):+.2f}R avg")
    ae = STATE.get("autoexec") or {}
    if ae:
        lines.append(
            f"\U0001F9EA Paper engine: {ae.get('totalTrades', 0)} trades "
            f"\u00b7 {(ae.get('totalPnl') or 0):+.2f}$ \u00b7 equity "
            f"{_ae_equity(ae):.2f}$")
    try:
        st = _ai_trader_stats()
        sig = _ai_trader_mem.get("sig") or {}
        if st["n"]:
            lines.append(f"\U0001F9E0 AI trader: {st['n']} calls \u00b7 "
                         f"{st['winPct']}% win \u00b7 {st['avgR']:+.2f}R avg")
        elif sig:
            age = max(1, int((time.time() - sig.get("t", 0)) / 60))
            last = (f"{sig['direction'].upper()} "
                    f"{sig.get('entry', '\u2014')}"
                    if sig.get("direction") != "none" else "stand aside")
            lines.append(f"\U0001F9E0 AI trader: record building \u00b7 "
                         f"latest call {age}m ago: {last}")
    except Exception:  # noqa: BLE001
        pass
    up = (STATE.get("research") or {}).get("upgrade") or {}
    if up.get("pending"):
        p = up["pending"]
        lines.append(f"\U0001F52C Research: challenger {p.get('name')} "
                     f"pending ({p.get('streak', 1)}/3 checks)")
    else:
        ex = up.get("exit") or [1, 2]
        lines.append(f"\U0001F52C Research: {len(res.get('variants') or [])} "
                     f"challengers in test \u00b7 production stable "
                     f"({up.get('gate', 'delta')}-gate, exit "
                     f"{ex[0]:g}:{ex[1]:g})")
    try:
        nx = (fundamentals.snapshot() or {}).get("nextHigh") or {}
        if nx.get("title"):
            lines.append(f"\U0001F4C5 Next high-impact: {nx['title']} "
                         f"in {nx.get('minutesTo', '?')}m")
    except Exception:  # noqa: BLE001
        pass
    live = [a for a in (STATE.get("mt5") or [])
            if a.get("state") == "connected"]
    mode = (f"{len(live)} live MT5 account(s)"
            if live else "paper engine only")
    try:
        tb = (_mtf_mem.get("lastScan") or {}).get("bias") or {}
        if tb:
            lines.append(f"\U0001F4C8 MTF bias: {tb['bulls']} bull / "
                         f"{tb['bears']} bear of {tb['total']} timeframes"
                         + (f" \u00b7 {tb['align']} aligned"
                            if tb.get("align") else ""))
    except Exception:  # noqa: BLE001
        pass
    up_h = int((time.time() - _BOOT_T) / 3600)
    lines.append(f"\u2699\ufe0f System v{SYSTEM_VERSION} \u00b7 uptime "
                 f"{up_h}h \u00b7 loops healthy \u00b7 {mode}")
    return "\n".join(lines)


def _desk_update_loop():
    time.sleep(900)                       # first digest 15 min after boot
    while True:
        try:
            if _desk_speaker():
                _notify(_desk_update_digest(), cat="develop")
                _desk_mem["lastDigestPost"] = int(time.time())
                print("[desk] member digest sent to DEVELOP", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[desk] digest failed: {e}", flush=True)
        time.sleep(DESK_UPDATE_S)


_price_engine_on = {"on": False}


def _price_engine():
    """Background price engine: recompute the displayed spot at ~4Hz so
    every realtime client (SSE + long-poll) reads a hot cache instead of
    each blocking on the slow REST anchor calls (Yahoo / gold-api can
    take seconds). The engine absorbs that latency once; the streams
    stay zero-delay."""
    while True:
        try:
            _tick_spot_raw(0.0)             # recompute + refresh _tick_mem
            _price_engine_on["n"] = _price_engine_on.get("n", 0) + 1
            _price_engine_on["t"] = time.time()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.25)


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
    cloud_load()                        # restore MT5 accounts + engine
    _ae()                               # ensure the auto-trade engine exists
    threading.Thread(target=_background_loop, daemon=True).start()
    threading.Thread(target=_research_loop, daemon=True).start()
    threading.Thread(target=_announce_version, daemon=True).start()
    threading.Thread(target=_desk_update_loop, daemon=True).start()
    threading.Thread(target=_mtf_loop, daemon=True).start()
    if not _price_engine_on["on"]:
        _price_engine_on["on"] = True
        threading.Thread(target=_price_engine, daemon=True).start()
    threading.Thread(target=_tv_spot_loop, daemon=True).start()
    if os.environ.get("TV_PUSHER") == "1":
        threading.Thread(target=_tv_pusher, daemon=True).start()
    if os.environ.get("PUSHER_URL"):
        threading.Thread(target=_pusher_keepalive, daemon=True).start()


# ------------------------------------------------- cross-deploy persistence
# Render wipes the service disk on every deploy, so connected MT5 accounts
# and the auto-trade engine live in a PRIVATE GitHub repo (credentials are
# additionally encrypted before they leave this server). The token comes
# from the GH_STATE_PAT env var — never from code.

STATE_REPO = "EA6455/xauusd-ai-state"
_cloud_mem = {"t": 0.0, "sha": None, "busy": False, "dirty": False}


def _cloud_dirty():
    _cloud_mem["dirty"] = True


def _cloud_req(method, path, body=None, timeout=20):
    tok = os.environ.get("GH_STATE_PAT")
    if not tok:
        return None
    import urllib.request
    req = urllib.request.Request(
        "https://api.github.com" + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"token {tok}",
                 "Accept": "application/vnd.github+json",
                 "User-Agent": "xauusd-ai-desk/1.0"}, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode()
    return json.loads(raw) if raw else {}


def cloud_load():
    """Restore MT5 accounts + auto-trade engine after a deploy wiped the
    local disk."""
    try:
        j = _cloud_req("GET", f"/repos/{STATE_REPO}/contents/state.json")
        if not j or j.get("content") is None:
            return False
        import base64
        blob = json.loads(base64.b64decode(j["content"]))
        _cloud_mem["sha"] = j.get("sha")
        dead = set(blob.get("mt5_deleted") or [])
        with STATE_LOCK:
            loc = {f"{a.get('login')}|{a.get('server')}": a
                   for a in (STATE.get("mt5") or [])}
            for a in blob.get("mt5") or []:
                k = f"{a.get('login')}|{a.get('server')}"
                if k not in dead and k not in loc:
                    loc[k] = a              # remote account this machine lost
            STATE["mt5"] = [a for k, a in loc.items()
                            if k not in dead]
            if blob.get("autoexec") and not STATE.get("autoexec"):
                STATE["autoexec"] = blob["autoexec"]
            STATE["aiTrades"] = _ai_trades_merge(
                blob.get("aiTrades"), STATE.get("aiTrades"))
            if blob.get("sysVer") and not STATE.get("sysVer"):
                STATE["sysVer"] = blob["sysVer"]
            _save_state()
        return True
    except Exception:  # noqa: BLE001  — 404 (nothing saved yet) or offline
        return False


def _cloud_untombstone(key):
    """A fresh Connect overrides an old disconnect tombstone — remove it
    from the shared cloud state too, or the merge in cloud_save would
    keep dropping this account forever."""
    try:
        import base64
        cur = _cloud_req("GET", f"/repos/{STATE_REPO}/contents/"
                                f"state.json")
        if not cur or cur.get("content") is None:
            return
        blob = json.loads(base64.b64decode(cur["content"]))
        dead = list(blob.get("mt5_deleted") or [])
        if key not in dead:
            return                          # nothing to clear
        blob["mt5_deleted"] = [d for d in dead if d != key]
        j = _cloud_req("PUT", f"/repos/{STATE_REPO}/contents/state.json",
                       dict(message=f"untombstone {key}",
                            content=base64.b64encode(
                                json.dumps(blob).encode()).decode(),
                            sha=cur.get("sha")))
        _cloud_mem["sha"] = ((j.get("content") or {}).get("sha")
                             or _cloud_mem.get("sha"))
    except Exception:  # noqa: BLE001  — best effort; local clear already ran
        pass


def cloud_save(force=False):
    """Persist MT5 accounts + engine stats to the private repo so they
    survive the next deploy. Debounced; call with force=True on real
    changes (connect/disconnect/fill/close)."""
    now = time.time()
    if _cloud_mem["busy"]:
        return
    if not force and now - _cloud_mem["t"] < 90:
        return
    _cloud_mem.update(busy=True, t=now, dirty=False)
    try:
        import base64
        # MERGE with the remote copy: a machine whose local state is empty
        # (fresh boot, test run) must never wipe accounts another machine
        # saved. Local entries win (fresher); tombstoned ones are dropped.
        remote = {}
        try:
            cur = _cloud_req("GET", f"/repos/{STATE_REPO}/contents/"
                                    f"state.json")
            if cur and cur.get("content") is not None:
                remote = json.loads(base64.b64decode(cur["content"]))
                _cloud_mem["sha"] = cur.get("sha")
        except Exception:  # noqa: BLE001  — nothing saved yet
            remote = {}
        dead = set(STATE.get("mt5_deleted") or [])
        if remote.get("mt5_deleted"):
            dead |= set(remote["mt5_deleted"])
        merged = {f"{a.get('login')}|{a.get('server')}": a
                  for a in (remote.get("mt5") or [])
                  if f"{a.get('login')}|{a.get('server')}" not in dead}
        for a in STATE.get("mt5") or []:
            k = f"{a.get('login')}|{a.get('server')}"
            if k not in dead:
                merged[k] = a                      # local wins (fresher)
        rae = remote.get("autoexec") or {}
        lae = STATE.get("autoexec") or {}
        ae = lae if (lae.get("totalTrades", 0) >=
                     rae.get("totalTrades", 0)) else rae
        blob = dict(sysVer=SYSTEM_VERSION,
                    mt5=list(merged.values())[:20],
                    autoexec=ae,
                    mt5_deleted=sorted(dead)[-50:],
                    aiTrades=_ai_trades_merge(remote.get("aiTrades"),
                                              STATE.get("aiTrades")))
        body = dict(message="state sync",
                    content=base64.b64encode(
                        json.dumps(blob).encode()).decode())
        if _cloud_mem.get("sha"):
            body["sha"] = _cloud_mem["sha"]
        try:
            j = _cloud_req("PUT", f"/repos/{STATE_REPO}/contents/"
                                  f"state.json", body)
            _cloud_mem["sha"] = ((j.get("content") or {}).get("sha")
                                 or _cloud_mem.get("sha"))
        except Exception:  # noqa: BLE001  — stale sha: refetch and retry
            try:
                g = _cloud_req("GET", f"/repos/{STATE_REPO}/contents/"
                                      f"state.json")
                _cloud_mem["sha"] = g.get("sha")
                body["sha"] = g.get("sha")
                j = _cloud_req("PUT", f"/repos/{STATE_REPO}/contents/"
                                      f"state.json", body)
                _cloud_mem["sha"] = ((j.get("content") or {}).get("sha")
                                     or _cloud_mem.get("sha"))
            except Exception:  # noqa: BLE001
                _cloud_mem["dirty"] = True     # try again on a later tick
    except Exception:  # noqa: BLE001
        _cloud_mem["dirty"] = True
    finally:
        _cloud_mem["busy"] = False


def _calendar_share():
    """Keep the shared cloud calendar fresh: the machine that CAN reach
    the forex-calendar feed pushes it to the private state repo; machines
    the feed blocks (datacenter IP limits) read it via the fallback."""
    try:
        evs = fundamentals.calendar()
        if not evs:
            return
        sha, fetched_at = None, 0
        try:
            j = _cloud_req("GET", f"/repos/{STATE_REPO}/contents/"
                                  f"calendar.json")
            if j and j.get("content") is not None:
                import base64
                sha = j.get("sha")
                try:
                    blob = json.loads(base64.b64decode(j["content"]))
                    fetched_at = blob.get("fetchedAt") or 0
                except Exception:  # noqa: BLE001
                    fetched_at = 0
        except Exception:  # noqa: BLE001  — 404 on first push
            pass
        if time.time() - fetched_at < 1800:
            return                              # cloud copy is fresh enough
        import base64
        body = dict(message="calendar share",
                    content=base64.b64encode(json.dumps(dict(
                        fetchedAt=int(time.time()), events=evs)).encode()
                    ).decode())
        if sha:
            body["sha"] = sha
        try:
            _cloud_req("PUT", f"/repos/{STATE_REPO}/contents/calendar.json",
                       body)
        except Exception:  # noqa: BLE001  — stale sha: refetch and retry
            j = _cloud_req("GET", f"/repos/{STATE_REPO}/contents/"
                                  f"calendar.json")
            body["sha"] = (j or {}).get("sha")
            _cloud_req("PUT", f"/repos/{STATE_REPO}/contents/calendar.json",
                       body)
    except Exception:  # noqa: BLE001
        pass


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
                p, src = tick_spot(max_age=3.0)   # hot cache: price engine
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
            # wake on any tape tick; the 0.5s cap also catches the price
            # engine's refresh — the stream can never sit on a stale price
            wsfeed.wait_for_change(0.5)
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
        p, src = tick_spot(max_age=3.0)          # hot cache: price engine
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


# ------------------------------------------------- MT5 live trading API
@app.route("/api/mt5/accounts")
def api_mt5_accounts():
    ae = STATE.get("autoexec") or {}
    return jsonify(
        accounts=[dict(id=a.get("id"), label=a.get("label"),
                       login=a.get("login"), server=a.get("server"),
                       broker=a.get("broker"), state=a.get("state"),
                       err=a.get("err"), balance=a.get("balance"),
                       equity=a.get("equity"), currency=a.get("currency"),
                       riskPct=a.get("riskPct"), cent=a.get("cent"))
                  for a in _mt5_accounts()],
        paper=dict(balance=ae.get("balance"),
                   equity=_ae_equity(ae) if ae else None,
                   open=len(ae.get("positions") or []),
                   dayPnl=(ae.get("day") or {}).get("pnl"),
                   totalTrades=ae.get("totalTrades"),
                   totalPnl=ae.get("totalPnl")),
        servers=list(EXNESS_SERVERS))


_mt5_srv_mem = {"q": "", "t": 0.0, "servers": []}


@app.route("/api/mt5/servers")
def api_mt5_servers():
    """Live server-name search over MetaApi's broker registry, so people
    always connect with the exact server name from their MT5 app."""
    q = (request.args.get("q") or "").strip()
    if len(q) < 3:
        return jsonify(servers=[])
    now = time.time()
    if _mt5_srv_mem["q"] == q and now - _mt5_srv_mem["t"] < 600:
        return jsonify(servers=_mt5_srv_mem["servers"])
    tok = _mt5_sys_token()
    if not tok:
        return jsonify(servers=list(EXNESS_SERVERS))
    try:
        import urllib.parse
        j = _mt5_req("GET", "/known-mt-servers/5/search?query="
                     + urllib.parse.quote(q), tok, prov=True)
        out = []
        for _broker, servers in (j or {}).items():
            out.extend(servers or [])
        out = sorted(set(out))[:40]
        _mt5_srv_mem.update(q=q, t=now, servers=out)
        return jsonify(servers=out)
    except Exception:  # noqa: BLE001
        return jsonify(servers=list(EXNESS_SERVERS))


@app.route("/api/mt5/connect", methods=["POST"])
def api_mt5_connect():
    j = request.get_json(force=True, silent=True) or {}
    login = str(j.get("login") or "").strip()
    pw = str(j.get("password") or "")
    server = str(j.get("server") or "").strip()
    token = str(j.get("token") or "").strip() or _mt5_sys_token()
    try:
        risk = float(j.get("riskPct") or 1.0)
    except (TypeError, ValueError):
        risk = 1.0
    acc_type = str(j.get("accType") or "real").lower()   # real|demo|cent
    cent = acc_type == "cent"
    label = str(j.get("label") or "").strip()[:30]
    if not (login.isdigit() and pw and server):
        return jsonify(error="MT5 login (number), password and server "
                             "are required"), 400
    if not token:
        return jsonify(error="trading bridge not configured yet — the "
                             "owner must set METAAPI_TOKEN once"), 503
    key = f"{login}|{server}"
    existing = next((a for a in _mt5_accounts()
                     if str(a.get("login")) == login
                     and a.get("server") == server), None)
    if existing and existing.get("state") == "connected":
        return jsonify(error="this account is already connected"), 409
    if existing:
        # pressing Connect again on a failed/stuck account re-queues it:
        # refresh credentials, reset the retry budget, re-run the flow
        # (reusing the bridge record upstream)
        with STATE_LOCK:
            existing.update(pwEnc=_enc(pw), tokenEnc=_enc(token),
                            riskPct=min(max(risk, 0.1), 5.0), cent=cent,
                            label=label or existing.get("label"),
                            state="connecting", err=None,
                            tries=0, lastTryT=0)
            existing.pop("accountId", None)   # re-find / re-use upstream
            STATE["mt5_deleted"] = [d for d in
                                    (STATE.get("mt5_deleted") or [])
                                    if d != key]
            _save_state()
        threading.Thread(target=_mt5_provision_thread, args=(existing,),
                         daemon=True).start()
        threading.Thread(target=_cloud_untombstone, args=(key,),
                         daemon=True).start()
        _cloud_dirty()
        return jsonify(ok=True, id=existing["id"])
    acc = dict(id=_next_id(), login=login, server=server,
               broker="Exness" if "exness" in server.lower() else "MT5",
               label=label or None, pwEnc=_enc(pw), tokenEnc=_enc(token),
               riskPct=min(max(risk, 0.1), 5.0), cent=cent,
               state="connecting", err=None)
    with STATE_LOCK:
        STATE["mt5"] = _mt5_accounts() + [acc]
        # an explicit Connect overrides any old disconnect tombstone for
        # this login+server — otherwise the account could never persist
        # in the shared cloud state again
        STATE["mt5_deleted"] = [d for d in (STATE.get("mt5_deleted") or [])
                                if d != key]
        _save_state()
    threading.Thread(target=_mt5_provision_thread, args=(acc,),
                     daemon=True).start()
    threading.Thread(target=_cloud_untombstone, args=(key,),
                     daemon=True).start()
    _cloud_dirty()
    return jsonify(ok=True, id=acc["id"])


@app.route("/api/mt5/sync", methods=["POST"])
def api_mt5_sync():
    for acc in _mt5_accounts():
        if acc.get("state") == "connected" and acc.get("accountId"):
            try:
                _mt5_sync_acc(acc)
                acc["err"] = None
            except Exception as e:  # noqa: BLE001
                acc["err"] = str(e)[:120]
    with STATE_LOCK:
        STATE["mt5"] = _mt5_accounts()
        _save_state()
    return jsonify(ok=True)


@app.route("/api/mt5/accounts/<int:aid>", methods=["DELETE"])
def api_mt5_delete(aid):
    acc = next((a for a in _mt5_accounts() if a.get("id") == aid), None)
    if not acc:
        return jsonify(error="not found"), 404
    if acc.get("accountId"):
        try:
            _mt5_req("DELETE", f"/users/current/accounts/{acc['accountId']}",
                     _dec(acc["tokenEnc"]), prov=True)
        except Exception:  # noqa: BLE001
            pass
    with STATE_LOCK:
        STATE["mt5"] = [a for a in _mt5_accounts() if a.get("id") != aid]
        _dl = list(STATE.get("mt5_deleted") or [])
        _dl.append(f"{acc.get('login')}|{acc.get('server')}")
        STATE["mt5_deleted"] = _dl[-50:]
        _save_state()
    cloud_save(force=True)
    return jsonify(ok=True)


def _ai_trader_payload(sig=None):
    """Panel payload: latest call + tracked history + honest stats."""
    s = sig if sig is not None else _ai_trader_mem.get("sig")
    if s:
        s = dict(s, age=int(time.time() - s.get("t", 0)))
    return jsonify(sig=s, busy=_ai_trader_mem["busy"],
                   err=_ai_trader_mem.get("lastErr"),
                   trades=(STATE.get("aiTrades") or [])[:8],
                   stats=_ai_trader_stats())


@app.route("/api/ai/trader")
def api_ai_trader_get():
    return _ai_trader_payload()


@app.route("/api/ai/trader", methods=["POST"])
def api_ai_trader_run():
    sig = ai_trader(force=True)
    if sig is None:
        return jsonify(error=_ai_trader_mem.get("lastErr")
                       or "AI trader unavailable"), 503
    return _ai_trader_payload(sig=sig)


@app.route("/api/tv/push", methods=["POST"])
def api_tv_push():
    """Receive the TradingView level from the research machine's pusher
    (different IP, not throttled) and refresh the hard-lock sample."""
    j = request.get_json(force=True, silent=True) or {}
    if not os.environ.get("TV_PUSH_SECRET") \
            or j.get("secret") != os.environ["TV_PUSH_SECRET"]:
        return jsonify(error="forbidden"), 403
    try:
        lvl = float(j.get("price"))
        ts = float(j.get("t") or time.time())
    except (TypeError, ValueError):
        return jsonify(error="bad payload"), 400
    if not lvl or lvl < 100:
        return jsonify(error="bad price"), 400
    now = time.time()
    if ts < _tv_spot_mem["t"]:               # stale push — ignore
        return jsonify(ok=True, stale=True)
    _tv_spot_mem.update(price=lvl, t=now, n=_tv_spot_mem.get("n", 0) + 1,
                        tape_ref=wsfeed.mid(max_age=60)[0],
                        err=None)
    return jsonify(ok=True)


@app.route("/api/feed")
def api_feed():
    e = _price_engine_on
    tv = _tv_spot_mem["price"]
    disp = _tick_mem.get("price")
    return jsonify(feeds=wsfeed.stats(),
                   tv=dict(price=tv, age=round(time.time() - _tv_spot_mem["t"], 1)
                           if _tv_spot_mem["t"] else None,
                           delta=round(disp - tv, 2)
                           if (tv and disp) else None,
                           pollDelay=_tv_spot_mem.get("pollDelay"),
                           err=_tv_spot_mem.get("err")),
                   engine=dict(running=bool(e.get("on")),
                               hz=round(e.get("n", 0) / max(
                                   1e-9, time.time() - _BOOT_T), 2),
                               lastAgo=round(time.time() - e["t"], 1)
                               if e.get("t") else None,
                               tickAge=round(time.time() - _tick_mem["t"], 1)
                               if _tick_mem.get("t") else None))


@app.route("/api/mtf")
def api_mtf():
    s = _mtf_mem.get("lastScan")
    if s:
        s = dict(s, age=int(time.time() - s.get("t", 0)))
    return jsonify(scan=s)


@app.route("/api/desk/updates")
def api_desk_updates():
    return jsonify(version=SYSTEM_VERSION, speaker=_desk_speaker(),
                   announced=STATE.get("sysVer") == SYSTEM_VERSION,
                   lastVersionPost=_desk_mem["lastVersionPost"],
                   lastDigestPost=_desk_mem["lastDigestPost"],
                   changelog=SYSTEM_CHANGELOG.get(SYSTEM_VERSION) or [])


@app.route("/api/trades")
def api_trades():
    """Trade dashboard: everything the autonomous engine ever traded —
    paper + live fills, stats, equity curve, open positions."""
    ae = STATE.get("autoexec") or {}
    closed_new = list(ae.get("closed") or [])          # newest first
    closed_old = list(reversed(closed_new))            # oldest first
    rs = [c.get("r") for c in closed_new
          if isinstance(c.get("r"), (int, float))]
    pnls = [c.get("pnl") for c in closed_new
            if isinstance(c.get("pnl"), (int, float))]
    stats = dict(n=len(closed_new), avgR=None, best=None, worst=None,
                 winRate=None, streak=0, pnl=ae.get("totalPnl", 0.0))
    if rs:
        stats.update(avgR=round(sum(rs) / len(rs), 2),
                     best=round(max(rs), 2), worst=round(min(rs), 2),
                     winRate=round(100 * sum(1 for r in rs if r > 0)
                                   / len(rs), 1))
        for r in rs:                                   # current streak
            sgn = 1 if r > 0 else -1
            if stats["streak"] == 0 or                     (stats["streak"] > 0) == (sgn > 0):
                stats["streak"] += sgn
            else:
                break
    curve = [ae.get("startBalance") or 100.0] +         [c.get("balance") for c in closed_old
         if isinstance(c.get("balance"), (int, float))]
    return jsonify(
        mode=ae.get("mode", "paper"),
        balance=ae.get("balance"), equity=_ae_equity(ae) if ae else None,
        startBalance=ae.get("startBalance", 100.0),
        day=ae.get("day") or {}, stats=stats,
        curve=[round(v, 2) for v in curve],
        open=[dict(src=p.get("src"), dir=p.get("dir"), lots=p.get("lots"),
                   entry=p.get("entry"), sl=p.get("sl"), tp1=p.get("tp1"),
                   tp2=p.get("tp2"), openedAt=p.get("openedAt"),
                   live=p.get("live") or [])
              for p in (ae.get("positions") or [])],
        closed=[dict(src=c.get("src"), dir=c.get("dir"), lots=c.get("lots"),
                     entry=c.get("entry"), r=c.get("r"), pnl=c.get("pnl"),
                     balance=c.get("balance"),
                     openedAt=c.get("openedAt"),
                     closedAt=c.get("closedAt"),
                     result=c.get("result"), live=c.get("live") or [])
                for c in closed_new[:40]],
        accounts=[dict(label=a.get("label") or a.get("login"),
                       login=a.get("login"), server=a.get("server"),
                       state=a.get("state"), balance=a.get("balance"),
                       currency=a.get("currency"), cent=a.get("cent"))
                  for a in _mt5_accounts()])


@app.route("/api/calendar")
def api_calendar():
    """Economic calendar for the dashboard: this week's gold-relevant
    events with impact, forecast and previous."""
    days = min(max(request.args.get("days", 7, type=int) or 7, 1), 10)
    now = time.time()
    wanted = ("USD", "All", "EUR", "GBP", "CNY", "JPY")
    out, nxt = [], None
    for e in fundamentals.calendar():
        dt = e["ts"] - now
        if dt > days * 86400:
            break
        if dt < -6 * 3600:            # keep a little past context out
            continue
        if e.get("country") not in wanted:
            continue
        ev = dict(ts=e["ts"], title=e.get("title"), country=e.get("country"),
                  impact=e.get("impact"), forecast=e.get("forecast"),
                  previous=e.get("previous"),
                  minutesTo=int(dt // 60))
        out.append(ev)
        if nxt is None and dt > 0 and e.get("impact") == "High":
            nxt = ev
    return jsonify(events=out[:80], nextHigh=nxt, now=int(now))


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
    r = STATE.get("research") or {}
    return jsonify(ok=True, tfs=list(data.TFS),
                   telegram=bool(TG_TOKEN and TG_CHAT),
                   topics=_tg_topics(),
                   research=dict(cycles=r.get("cycles", 0),
                                 lastCycleT=r.get("lastCycleT"),
                                 scans=r.get("scans", 0),
                                 lastScanT=r.get("lastScanT"),
                                 statusMsg=(r.get("statusMsg") or {}).get("id"),
                                 intelHeadlines=len(
                                     (r.get("intel") or {}).get("headlines") or []),
                                 lastIntelT=r.get("lastIntelT"),
                                 llm=(r.get("llm") or {}).get("state"),
                                 llmErr=(r.get("llm") or {}).get("err"),
                                 upgrade=dict(
                                     gate=(r.get("upgrade") or {}).get(
                                         "gate", "delta"),
                                     exitR=(r.get("upgrade") or {}).get(
                                         "exit", [1.0, 2.0]),
                                     pending=((r.get("upgrade") or {})
                                              .get("pending") or {}).get("name"),
                                     lastPromoteT=(r.get("upgrade") or {})
                                     .get("lastPromoteT")),
                                 autoexec=dict(
                                     enabled=(STATE.get("autoexec")
                                              or {}).get("enabled", False),
                                     mode=(STATE.get("autoexec")
                                           or {}).get("mode", "paper"),
                                     balance=(STATE.get("autoexec")
                                              or {}).get("balance"),
                                     open=len(((STATE.get("autoexec")
                                                or {}).get("positions")
                                               or [])),
                                     dayTrades=((STATE.get("autoexec")
                                                 or {}).get("day")
                                                or {}).get("trades"),
                                     dayPnl=((STATE.get("autoexec")
                                              or {}).get("day") or {})
                                     .get("pnl"),
                                     totalTrades=(STATE.get("autoexec")
                                                  or {}).get("totalTrades"),
                                     totalPnl=(STATE.get("autoexec")
                                               or {}).get("totalPnl"))))


# Start the realtime feed + background loop at import time so WSGI servers
# (gunicorn in the Dockerfile) get it too — start_background() is idempotent
# and an flock keeps it to ONE loop per machine even with multiple workers.
start_background()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, threaded=True, debug=False)
