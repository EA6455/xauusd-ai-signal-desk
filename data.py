"""Market-data layer for XAUUSD (gold).

Primary OHLC source : Yahoo Finance gold futures GC=F (COMEX) — tracks spot XAU/USD closely.
Fallback OHLC       : OKX PAXG-USDT (each PAXG = 1 troy oz of allocated LBMA gold).
Spot reference      : gold-api.com free spot XAU price.

All responses are disk-cached so the app can survive temporary network failures.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

TFS = {
    "15m": dict(interval="15m", range="60d", ttl=12, okx_bar="15m", keep=6000),
    "60m": dict(interval="60m", range="2y", ttl=30, okx_bar="1H", keep=12000),
    "1d": dict(interval="1d", range="10y", ttl=60, okx_bar="1D", keep=4000),
}

# Yahoo health: back off for 90s after repeated failures so we never get
# hard rate-limited — the app falls back to OKX / cache in the meantime.
_yahoo_health = {"fails": 0, "blocked_until": 0.0}


def yahoo_ok():
    return time.time() >= _yahoo_health["blocked_until"]


def note_yahoo_result(ok):
    if ok:
        _yahoo_health["fails"] = 0
        _yahoo_health["blocked_until"] = 0.0
    else:
        _yahoo_health["fails"] += 1
        if _yahoo_health["fails"] >= 3:
            _yahoo_health["blocked_until"] = time.time() + 90
            _yahoo_health["fails"] = 0

YAHOO_SYMBOLS = ["GC=F"]           # COMEX gold front-month futures
OKX_INSTRUMENT = "PAXG-USDT"
UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
}

_mem = {}
_spot_mem = {"fetchedAt": 0.0, "spot": None}


# ----------------------------------------------------------------- helpers
def _http_get(url, timeout=12, headers=None):
    req = urllib.request.Request(url, headers=headers or UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def _http_json(url, timeout=12, headers=None):
    return json.loads(_http_get(url, timeout=timeout, headers=headers))


# ----------------------------------------------------------------- sources
def fetch_yahoo(symbol, interval, rng):
    """Yahoo v8 chart API → list of candles (ascending by time)."""
    if not yahoo_ok():
        raise RuntimeError("yahoo cooling down after failures")
    last_err = None
    for host in ("query1", "query2"):
        try:
            url = (f"https://{host}.finance.yahoo.com/v8/finance/chart/"
                   f"{urllib.parse.quote(symbol)}?interval={interval}&range={rng}")
            j = _http_json(url)
            res = j["chart"]["result"][0]
            ts = res.get("timestamp") or []
            q = res["indicators"]["quote"][0]
            out = []
            for k in range(len(ts)):
                c = q["close"][k]
                if c is None:
                    continue
                o, h, l = q["open"][k], q["high"][k], q["low"][k]
                out.append(dict(t=int(ts[k]),
                                o=float(o) if o else float(c),
                                h=float(h) if h else float(c),
                                l=float(l) if l else float(c),
                                c=float(c)))
            if len(out) > 60:
                note_yahoo_result(True)
                return out, f"Gold futures {symbol} · Yahoo Finance (COMEX)"
        except Exception as e:  # noqa: BLE001
            last_err = e
    note_yahoo_result(False)
    raise RuntimeError(f"Yahoo unavailable for {symbol}: {last_err}")


def fetch_okx(bar, target=1600):
    """OKX public market data → PAXG-USDT candles (ascending by time)."""
    def page(path):
        j = _http_json(f"https://www.okx.com{path}")
        if j.get("code") != "0":
            raise RuntimeError(f"OKX error: {j.get('msg')}")
        return j.get("data") or []

    rows = page(f"/api/v5/market/candles?instId={OKX_INSTRUMENT}&bar={bar}&limit=300")
    while rows and len(rows) < target:
        oldest = rows[-1][0]
        try:
            more = page(f"/api/v5/market/history-candles?instId={OKX_INSTRUMENT}"
                        f"&bar={bar}&limit=100&after={oldest}")
        except Exception:
            break
        if not more:
            break
        rows.extend(more)
        time.sleep(0.06)  # stay well inside OKX rate limits
    if len(rows) < 60:
        raise RuntimeError("OKX returned too little data")
    out = []
    for r in rows:
        # [ts, open, high, low, close, vol, volCcy, volCcyQuote, confirm]
        out.append(dict(t=int(r[0]) // 1000, o=float(r[1]), h=float(r[2]),
                        l=float(r[3]), c=float(r[4])))
    out.sort(key=lambda k: k["t"])
    return out, f"PAXG-USDT (1 oz gold) · OKX"


def fetch_spot():
    """Free spot XAU/USD from gold-api.com (display reference only)."""
    try:
        j = _http_json("https://api.gold-api.com/price/XAU", timeout=8)
        p = float(j["price"])
        return dict(price=p, updatedAt=j.get("updatedAt"), source="gold-api.com spot XAU")
    except Exception:  # noqa: BLE001
        return None


def get_spot(max_age=60):
    now = time.time()
    if now - _spot_mem["fetchedAt"] < max_age:
        return _spot_mem["spot"]
    _spot_mem["spot"] = fetch_spot()
    _spot_mem["fetchedAt"] = now
    return _spot_mem["spot"]


# ----------------------------------------------------------------- cache
def _cache_path(tf):
    return os.path.join(CACHE_DIR, f"{tf}.json")


def get_candles(tf, force=False):
    """Return dict(candles, source, stale, fetchedAt, changed[, error]).

    `changed=True` means a fresh network fetch produced new candles (or a
    failure meant we had to fall back to the disk cache).
    """
    if tf not in TFS:
        raise ValueError(f"unknown timeframe {tf}")
    cfg = TFS[tf]
    st = _mem.get(tf)
    now = time.time()
    if not force and st and now - st["fetchedAt"] < cfg["ttl"]:
        return dict(st, changed=False)

    candles = None
    source = None
    err = None
    for sym in YAHOO_SYMBOLS:
        try:
            candles, source = fetch_yahoo(sym, cfg["interval"], cfg["range"])
            break
        except Exception as e:  # noqa: BLE001
            err = e
    if candles is None:
        try:
            candles, source = fetch_okx(cfg["okx_bar"], target=1600)
        except Exception as e:  # noqa: BLE001
            err = e

    if candles is not None:
        candles = candles[-cfg["keep"]:]
        st = dict(candles=candles, source=source, stale=False,
                  fetchedAt=now, ttl=cfg["ttl"])
        _mem[tf] = st
        try:
            with open(_cache_path(tf), "w") as f:
                json.dump(dict(candles=candles, source=source, fetchedAt=now), f)
        except OSError:
            pass
        return dict(st, changed=True)

    # network failure → serve the disk cache, marked stale
    disk = None
    try:
        with open(_cache_path(tf)) as f:
            disk = json.load(f)
    except Exception:  # noqa: BLE001
        pass
    if disk and disk.get("candles"):
        st = dict(candles=disk["candles"],
                  source=disk.get("source", "?") + " · cached",
                  stale=True, fetchedAt=now, ttl=20)
        _mem[tf] = st
        return dict(st, changed=True)
    return dict(candles=None, source=None, stale=True, fetchedAt=now,
                changed=True, error=str(err) or "no data source available")
