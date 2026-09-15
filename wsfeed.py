"""Real-time gold feed via OKX public websocket (no API key needed).

Subscribes to the tick-by-tick best-bid/offer (bbo-tbt) and trade prints of
PAXG-USDT — a token backed 1:1 by 1 troy oz of allocated LBMA gold, whose
book is arbitraged tightly against spot XAU/USD. The mid-price is a genuine
real-time spot-gold proxy: sub-second updates, no REST polling, no rate limits.

Graceful degradation: if the `websocket-client` package or the connection is
unavailable, mid() returns None and the app falls back to the REST quote chain.
"""
from __future__ import annotations

import json
import threading
import time

try:
    import websocket  # websocket-client package
except ImportError:  # pragma: no cover
    websocket = None

URL = "wss://ws.okx.com:8443/ws/v5/public"
INST = "PAXG-USDT"

_lock = threading.Lock()
_event = threading.Condition()
_bid = {"v": None, "t": 0.0}
_ask = {"v": None, "t": 0.0}
_last = {"v": None, "t": 0.0}
_started = False


def mid(max_age=30.0):
    """(mid_price, received_at) or (None, None) if stale/unavailable."""
    with _lock:
        if _bid["v"] and _ask["v"] and time.time() - min(_bid["t"], _ask["t"]) < max_age:
            return (_bid["v"] + _ask["v"]) / 2.0, min(_bid["t"], _ask["t"])
    return None, None


def last_trade(max_age=300.0):
    with _lock:
        if _last["v"] and time.time() - _last["t"] < max_age:
            return _last["v"]
    return None


def _on_open(ws):
    ws.send(json.dumps({"op": "subscribe", "args": [
        {"channel": "bbo-tbt", "instId": INST},
        {"channel": "trades", "instId": INST},
    ]}))

    def pinger():
        while True:
            time.sleep(18)
            try:
                ws.send("ping")
            except Exception:  # noqa: BLE001
                return
    threading.Thread(target=pinger, daemon=True).start()


def wait_for_change(timeout=0.5):
    """Block until the exchange book changes (or timeout)."""
    with _event:
        _event.wait(timeout)


def _on_message(ws, message):
    if message == "pong":
        return
    try:
        j = json.loads(message)
        ch = (j.get("arg") or {}).get("channel")
        now = time.time()
        if ch == "bbo-tbt" and j.get("data"):
            d = j["data"][0]
            with _lock:
                _bid.update(v=float(d["bids"][0][0]), t=now)
                _ask.update(v=float(d["asks"][0][0]), t=now)
            with _event:
                _event.notify_all()          # wake every waiting client instantly
        elif ch == "trades" and j.get("data"):
            with _lock:
                _last.update(v=float(j["data"][0]["px"]), t=now)
            with _event:
                _event.notify_all()
    except Exception:  # noqa: BLE001
        pass


def _run():
    backoff = 1
    while True:
        try:
            ws = websocket.WebSocketApp(
                URL, on_open=_on_open, on_message=_on_message)
            ws.run_forever(ping_interval=0)   # we send OKX text-pings ourselves
            backoff = 1
        except Exception:  # noqa: BLE001
            pass
        time.sleep(backoff)
        backoff = min(backoff * 2, 30)


def start():
    """Start the feed thread once per process (safe to call repeatedly)."""
    global _started
    if _started or websocket is None:
        return
    _started = True
    threading.Thread(target=_run, daemon=True).start()
