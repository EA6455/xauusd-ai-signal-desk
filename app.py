"""XAUUSD · AI Signal Desk — live web app.

Serves a dashboard with the gold price, an AI buy/sell signal per timeframe,
an alert feed (signal flips + custom price levels) and optional Telegram push.

Run:  python app.py   →  http://localhost:7860
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import threading
import time

import numpy as np
from flask import Flask, Response, jsonify, render_template, request

import data
import entries
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
WINDOW = {"15m": 180, "60m": 180, "1d": 240}   # bars shown on the chart

TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "")

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
             liveTrade=None, tradeHistory=[])
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


def _notify(text):
    """Send a Telegram message if TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are set."""
    if not (TG_TOKEN and TG_CHAT):
        return
    try:
        import urllib.request
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            data=json.dumps({"chat_id": TG_CHAT, "text": text}).encode(),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception:  # noqa: BLE001
        pass


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


def maybe_alert(tf, sig_type, price, score, conf):
    """Fire an alert when the composite signal flips to BUY or SELL."""
    if sig_type not in ("BUY", "SELL"):
        with STATE_LOCK:
            if STATE["lastSig"].get(tf) not in (None, "HOLD"):
                STATE["lastSig"][tf] = "HOLD"
                _save_state()
        return
    with STATE_LOCK:
        prev = STATE["lastSig"].get(tf)
        if prev == sig_type:
            return
        STATE["lastSig"][tf] = sig_type
        first = prev is None
        a = dict(id=_next_id(), time=int(time.time()), tf=tf, type=sig_type,
                 price=round(float(price), 2), score=round(float(score), 3),
                 confidence=round(float(conf), 3),
                 msg=f"XAUUSD {tf} {sig_type} @ {price:,.2f} · "
                     f"conf {conf * 100:.0f}% · score {score:+.2f}"
                     + (" · initial stance" if first else " · signal flip"))
        STATE["alerts"].insert(0, a)
        del STATE["alerts"][100:]
        _save_state()
    _notify(("🟢 " if sig_type == "BUY" else "🔴 ") + a["msg"])


def maybe_setup_alert(ent):
    """Fire an alert only on FULL SNR confluence (8/8, A+) retests."""
    if not ent or not ent.get("active") or ent.get("grade") != "A+":
        return
    key = f"{ent['direction']}:{ent['barTime']}"
    with STATE_LOCK:
        if STATE.get("lastSetup") == key:
            return
        STATE["lastSetup"] = key
        typ = "ENTRY_BUY" if ent["direction"] == "LONG" else "ENTRY_SELL"
        a = dict(id=_next_id(), time=int(time.time()), tf="15m", type=typ,
                 price=ent["entry"], score=round(ent["passed"] / 8.0, 2),
                 confidence=round(ent["passed"] / 8.0, 2),
                 msg=(f"SNR {ent['grade']} {ent['direction']} retest @ {ent['entry']:,.2f} · "
                      f"zone {ent['entryZone'][0]:,.1f}–{ent['entryZone'][1]:,.1f} · "
                      f"SL {ent['sl']:,.2f} · TP1 {ent['tp1']:,.2f} · "
                      f"{ent['passed']}/8 SNR checks"))
        STATE["alerts"].insert(0, a)
        del STATE["alerts"][100:]
        _save_state()
    _notify("🎯 " + a["msg"])


def _close_trade(tr, result, r, price):
    tr.update(status="closed", result=result, r=round(r, 2),
              closedAt=int(time.time()), closePrice=round(float(price), 2))
    STATE["tradeHistory"].insert(0, dict(tr))
    del STATE["tradeHistory"][25:]


def ensure_trade(ent):
    """Open a tracked position when an A+ setup fires (one at a time)."""
    if not ent or not ent.get("active"):
        return
    key = f"{ent['direction']}:{ent['barTime']}"
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
        _notify(("✅ " if typ in ("TP1_HIT", "TP2_HIT") else "🛑 " if typ == "SL_HIT" else "⏱ ") + a["msg"])


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
_basis_mem = {"t": 0.0, "value": 0.0, "valid": False}
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


_anchor_mem = {"offset": 0.0, "t": 0.0}
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


def tick_spot(max_age=0.8):
    """Freshest spot XAU/USD for the live tick line.

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
        anchor_target = None
        if spot and spot.get("updatedAt"):
            ts = _feed_epoch(spot["updatedAt"])
            if ts and now - ts <= 120:
                m_then = wsfeed.mid_at(ts)
                if m_then:
                    anchor_target = spot["price"] - m_then
        if anchor_target is None and spot:
            anchor_target = spot["price"] - wm
        if anchor_target is None and q and basis_ok:
            anchor_target = (q - _basis_mem["value"]) - wm
        if anchor_target is not None and abs(anchor_target) <= 8.0:
            _anchor_mem.update(offset=anchor_target, t=now)
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


def build_payload(tf, d):
    raw = d["candles"]

    # ---- spot-align: remove the futures basis so prices match XAU/USD spot
    spot_price, spot_src = spot_reference()
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
    _basis_mem.update(t=time.time(), value=adjust, valid=bool(adjust))

    # live-tick patch: the FORMING candle and the payload price follow the
    # realtime feed (ws book + matched anchor), not the candle source's
    # last close — so /api/data never lags /api/tick and the UI's periodic
    # reload can't drag the displayed price backwards.
    try:
        _tp, _tsrc = tick_spot()
        if _tp and candles:
            _lc = candles[-1]
            _lc["c"] = float(_tp)
            if _tp > _lc["h"]:
                _lc["h"] = float(_tp)
            if _tp < _lc["l"]:
                _lc["l"] = float(_tp)
    except Exception:  # noqa: BLE001
        pass

    ensure_model(tf, candles)
    model = models[tf]

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

    sig = np.zeros(n, int)
    v = ~np.isnan(scores)
    sig[v & (scores >= BUY_TH)] = 1
    sig[v & (scores <= SELL_TH)] = -1

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
    if tf == "15m":
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
            ensure_trade(ent)
            maybe_setup_alert(ent)
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            payload["entry"] = None
            payload["entryStats"] = None
    maybe_alert(tf, sig_type, float(c[-1]), cur_score, conf)
    return payload


# ------------------------------------------------------------------ refresh
class TFState:
    def __init__(self):
        self.payload = None
        self.lock = threading.Lock()


tf_states = {tf: TFState() for tf in data.TFS}


def refresh(tf, force=False):
    if tf not in data.TFS:
        return dict(error=f"unknown timeframe {tf}")
    st = tf_states[tf]
    with st.lock:
        d = data.get_candles(tf, force=force)
        if not d.get("candles"):
            if st.payload is not None:
                st.payload["stale"] = True
                return st.payload
            return dict(error="data source unavailable — retrying", tf=tf)
        if d["changed"] or st.payload is None:
            prev_price = STATE.get("lastPrice")
            st.payload = build_payload(tf, d)
            check_price_alerts(prev_price, st.payload["price"])
            with STATE_LOCK:
                STATE["lastPrice"] = st.payload["price"]
                _save_state()
        with STATE_LOCK:
            if st.payload is not None:
                st.payload["alerts"] = STATE["alerts"][:30]
                st.payload["priceAlerts"] = STATE["priceAlerts"]
                st.payload["note"] = get_note()
                st.payload["tracker"] = dict(live=STATE.get("liveTrade"),
                                             history=STATE.get("tradeHistory", [])[:8])
        return st.payload


_bg_started = False


def _background_loop():
    while True:
        for tf in data.TFS:
            try:
                refresh(tf)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1)
        try:
            update_trade_tracker()
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
    except Exception:  # noqa: BLE001
        pass
    return render_template("index.html", payload=payload)


@app.route("/api/data")
def api_data():
    tf = request.args.get("tf", "60m")
    if tf not in data.TFS:
        return jsonify(error=f"unknown timeframe {tf}"), 400
    force = request.args.get("force") == "1"
    start_background()
    return jsonify(refresh(tf, force=force))


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


@app.route("/api/tick")
def api_tick():
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
        while True:
            try:
                p, src = tick_spot()
            except Exception:  # noqa: BLE001
                p, src = None, None
            now = time.time()
            if p is not None:
                yield ("data: " + json.dumps(
                    dict(price=round(p, 2), source=src, t=int(now))) + "\n\n")
            else:
                yield ": keepalive\n\n"
            if now - started > 3300:      # ~55 min; EventSource auto-reconnects
                break
            time.sleep(0.8)
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
