"""Multi-source realtime gold feed — aggregated like a professional terminal.

Four tick-by-tick websocket streams, no API keys, no REST polling:

  1. OKX   PAXG-USDT  bbo-tbt  (Paxos gold, 1 troy oz token)
  2. OKX   XAUT-USDT  bbo-tbt  (Tether Gold, 1 troy oz token)
  3. Binance PAXGUSDT bookTicker (every best-bid/ask change)
  4. Bybit PAXGUSDT   tickers  (last-price pushes)

All four are arbitaged tightly against spot XAU/USD, so together they
form a dense, always-on spot-gold tape: any single feed can go quiet or
be geo-blocked and the tape keeps moving. The displayed mid is the
MEDIAN of the fresh feeds (stable level — one venue's premium cannot
push the price), and every feed update appends a history sample and
wakes every waiting client instantly (event-driven, zero timers).

Graceful degradation: unreachable feeds simply retry with backoff; if
`websocket-client` is missing entirely, mid() returns None and the app
falls back to its REST quote chain.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque

try:
    import websocket  # websocket-client package
except ImportError:  # pragma: no cover
    websocket = None

OKX_URL = "wss://ws.okx.com:8443/ws/v5/public"
BINANCE_URL = "wss://stream.binance.com:9443/ws/paxgusdt@bookTicker"
BYBIT_URL = "wss://stream.bybit.com/v5/public/spot"

FRESH_S = 10.0        # a feed counts toward the median while younger
STALE_S = 30.0        # beyond this a feed is ignored entirely

_lock = threading.Lock()
_event = threading.Condition()
_feeds = {}           # name -> {bid, ask, t, lastPx, lastT, ticks}
_hist = deque(maxlen=7200)          # (epoch, merged mid) — ~2h of tape
_changes = deque(maxlen=1200)       # epochs of actual mid VALUE changes
_mid = {"v": None, "t": 0.0}        # merged mid at its last update
_started = False


# ------------------------------------------------------------- state core
def _feed(name):
    f = _feeds.get(name)
    if f is None:
        f = dict(bid=None, ask=None, t=0.0, lastPx=None, lastT=0.0,
                 ticks=deque(maxlen=1200))
        _feeds[name] = f
    return f

def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n % 2:
        return xs[n // 2]
    return (xs[n // 2 - 1] + xs[n // 2]) / 2.0


_levels = {}          # feed -> slow EMA of its mid (the venue's "level")


def _merged_locked(now):
    """Consolidated-index price, like a professional terminal:

      base = median of the feeds' SLOW LEVELS (each venue's EMA) — the
             honest cross-market level; no single venue's premium can
             push it, no ping-pong between venues
      dev  = the FRESHEST feed's live deviation from its own level —
             the motion of the market right now

    displayed = base + dev. The result moves at the speed of the
    fastest stream while staying pinned to the consensus level."""
    fresh_m, fresh_t, fresh_name = None, 0.0, None
    levels = []
    for name, f in _feeds.items():
        if not (f["bid"] and f["ask"]):
            continue
        age = now - f["t"]
        if age >= STALE_S:
            continue
        m = (f["bid"] + f["ask"]) / 2.0
        e = _levels.get(name)
        e = m if e is None else e + 0.02 * (m - e)      # slow venue level
        _levels[name] = e
        if age < FRESH_S:
            levels.append(e)
        if f["t"] > fresh_t:
            fresh_m, fresh_t, fresh_name = m, f["t"], name
    if fresh_m is None:
        return None
    if levels:
        base = _median(levels)
    else:
        base = _levels[fresh_name]
    dev = fresh_m - _levels[fresh_name]                 # live motion
    if dev > 3.0:
        dev = 3.0                                       # spike guard
    elif dev < -3.0:
        dev = -3.0
    return base + dev


def _apply(name, bid=None, ask=None, last=None):
    """One tick from one feed: update state, extend the tape, wake all
    waiting clients — instantly, no timer anywhere in the chain."""
    now = time.time()
    with _lock:
        f = _feed(name)
        if bid:
            f["bid"] = float(bid)
        if ask:
            f["ask"] = float(ask)
        if last:
            f["lastPx"] = float(last)
            f["lastT"] = now
        f["t"] = now
        f["ticks"].append(now)
        m = _merged_locked(now)
        if m is not None:
            if _mid["v"] is None or abs(m - _mid["v"]) >= 0.005:
                _changes.append(now)      # the displayed price moved
            _mid.update(v=m, t=now)
            _hist.append((now, m))
    with _event:
        _event.notify_all()


# ------------------------------------------------------------ public API
def hist():
    """Snapshot of the merged tape as [(epoch, mid), ...]."""
    with _lock:
        return list(_hist)


def mid(max_age=30.0):
    """(merged mid, updated_at) or (None, None) if stale/unavailable."""
    with _lock:
        if _mid["v"] and time.time() - _mid["t"] < max_age:
            return _mid["v"], _mid["t"]
    return None, None


def mid_at(ts, tol=45.0):
    """Merged mid closest in time to epoch ts, or None if no sample
    nearby. Used to pair a delayed spot quote with the tape at ITS OWN
    timestamp, so the anchor offset carries no staleness bias."""
    with _lock:
        if not _hist:
            return None
        best = min(_hist, key=lambda x: abs(x[0] - ts))
        if abs(best[0] - ts) <= tol:
            return best[1]
    return None


def last_trade(max_age=300.0):
    with _lock:
        best = None
        for f in _feeds.values():
            if f["lastPx"] and time.time() - f["lastT"] < max_age \
                    and (best is None or f["lastT"] > best[1]):
                best = (f["lastPx"], f["lastT"])
        return best[0] if best else None


def wait_for_change(timeout=0.5):
    """Block until ANY feed ticks (or timeout)."""
    with _event:
        _event.wait(timeout)


def stats():
    """Per-feed health + density (for /api/feed and the desk digest)."""
    now = time.time()
    with _lock:
        out = {}
        total = 0
        for name, f in sorted(_feeds.items()):
            tpm = sum(1 for t in f["ticks"] if now - t < 60)
            total += tpm
            out[name] = dict(
                ticksPerMin=tpm,
                lastAge=round(now - f["t"], 1) if f["t"] else None,
                mid=round((f["bid"] + f["ask"]) / 2.0, 2)
                if f["bid"] and f["ask"] else None)
        out["merged"] = dict(
            ticksPerMin=total,
            changesPerMin=sum(1 for t in _changes if now - t < 60),
            lastAge=round(now - _mid["t"], 1) if _mid["t"] else None,
            mid=_mid["v"] and round(_mid["v"], 2))
        return out


# ------------------------------------------------------------- receivers
def _okx_run():
    """OKX public ws: BOTH gold tokens on one connection — bbo-tbt is
    the tick-by-tick best bid/offer (top of book on every change)."""
    while True:
        try:
            def on_open(ws):
                ws.send(json.dumps({"op": "subscribe", "args": [
                    {"channel": "bbo-tbt", "instId": "PAXG-USDT"},
                    {"channel": "bbo-tbt", "instId": "XAUT-USDT"},
                    {"channel": "trades", "instId": "PAXG-USDT"},
                    {"channel": "trades", "instId": "XAUT-USDT"},
                ]}))

                def pinger():
                    while True:
                        time.sleep(18)
                        try:
                            ws.send("ping")
                        except Exception:  # noqa: BLE001
                            return
                threading.Thread(target=pinger, daemon=True).start()

            def on_message(ws, message):
                if message == "pong":
                    return
                j = json.loads(message)
                arg = j.get("arg") or {}
                inst = arg.get("instId") or ""
                name = "okx-" + inst.split("-")[0].lower()
                if arg.get("channel") == "bbo-tbt" and j.get("data"):
                    d = j["data"][0]
                    bid = (d.get("bids") or [[None]])[0][0]
                    ask = (d.get("asks") or [[None]])[0][0]
                    if bid and ask:
                        _apply(name, bid=bid, ask=ask)
                elif arg.get("channel") == "trades" and j.get("data"):
                    _apply(name, last=j["data"][0]["px"])

            ws = websocket.WebSocketApp(
                OKX_URL, on_open=on_open, on_message=on_message)
            ws.run_forever(ping_interval=15)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)


def _binance_run():
    """Binance bookTicker: the full best bid/ask on EVERY change —
    typically several messages per second. May be geo-blocked in some
    regions; in that case this thread just retries quietly and the
    other feeds carry the tape."""
    while True:
        try:
            def on_open(ws):
                pass                        # raw stream: no subscribe msg

            def on_message(ws, message):
                j = json.loads(message)
                if j.get("b") and j.get("a"):
                    _apply("binance-paxg", bid=j["b"], ask=j["a"])

            ws = websocket.WebSocketApp(
                BINANCE_URL, on_open=on_open, on_message=on_message)
            ws.run_forever(ping_interval=15)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)


def _bybit_run():
    """Bybit spot tickers: last-price pushes ~every 100ms when the
    market moves. Best effort — geo-blocks are fine."""
    while True:
        try:
            def on_open(ws):
                ws.send(json.dumps({"op": "subscribe",
                                    "args": ["tickers.PAXGUSDT"]}))

                def pinger():
                    while True:
                        time.sleep(18)
                        try:
                            ws.send(json.dumps({"op": "ping"}))
                        except Exception:  # noqa: BLE001
                            return
                threading.Thread(target=pinger, daemon=True).start()

            def on_message(ws, message):
                j = json.loads(message)
                if j.get("topic", "").startswith("tickers.") \
                        and (j.get("data") or {}).get("lastPrice"):
                    _apply("bybit-paxg",
                           last=j["data"]["lastPrice"],
                           bid=j["data"].get("bid1Price") or j["data"]["lastPrice"],
                           ask=j["data"].get("ask1Price") or j["data"]["lastPrice"])

            ws = websocket.WebSocketApp(
                BYBIT_URL, on_open=on_open, on_message=on_message)
            ws.run_forever(ping_interval=15)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)


def start():
    """Launch every feed (idempotent). The merged tape starts flowing
    the moment the first stream connects."""
    global _started
    if _started or websocket is None:
        return
    _started = True
    threading.Thread(target=_okx_run, daemon=True).start()
    threading.Thread(target=_binance_run, daemon=True).start()
    threading.Thread(target=_bybit_run, daemon=True).start()
