"""Market-data layer for XAUUSD (gold).

Primary OHLC source : Yahoo Finance gold futures GC=F (COMEX) — tracks spot XAU/USD closely.
Fallback OHLC       : OKX PAXG-USDT (each PAXG = 1 troy oz of allocated LBMA gold).
Spot reference      : gold-api.com free spot XAU price.

All responses are disk-cached so the app can survive temporary network failures.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import wsfeed

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

TFS = {
    # seconds — aggregated locally from the realtime tick feed (no API calls)
    "1s":  dict(src="ticks", bar_s=1,    ttl=4,    keep=900),
    "5s":  dict(src="ticks", bar_s=5,    ttl=6,    keep=900),
    "15s": dict(src="ticks", bar_s=15,   ttl=10,   keep=900),
    "30s": dict(src="ticks", bar_s=30,   ttl=15,   keep=720),
    # minutes → months — Yahoo GC=F futures, OKX PAXG fallback
    "1m":  dict(interval="1m",  range="5d",  ttl=30,   okx_bar="1m",  keep=7200),
    "5m":  dict(interval="5m",  range="1mo", ttl=60,   okx_bar="5m",  keep=6000),
    "15m": dict(interval="15m", range="60d", ttl=12,   okx_bar="15m", keep=6000),
    "30m": dict(interval="30m", range="1mo", ttl=90,   okx_bar="30m", keep=4000),
    "60m": dict(interval="60m", range="2y",  ttl=30,   okx_bar="1H",  keep=12000),
    # aggregated from a finer CACHED timeframe — zero extra API calls
    "2h":  dict(src="agg", base="60m", bar_s=7200,  ttl=60,   keep=4000),
    "4h":  dict(src="agg", base="60m", bar_s=14400, ttl=60,   keep=3000),
    "1d":  dict(interval="1d",  range="10y", ttl=60,   okx_bar="1D",  keep=4000),
    "1w":  dict(interval="1wk", range="20y", ttl=900,  okx_bar="1W",  keep=1100),
    "1M":  dict(src="agg", base="1w", bar_s="M", ttl=1200, keep=400),
    "1Y":  dict(src="agg", base="1M", bar_s="Y",   ttl=3600, keep=60),
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


_bg_lock = threading.Lock()
_bg_refreshing = set()


def _month_start(ts):
    d = datetime.fromtimestamp(ts, timezone.utc)
    return int(d.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp())


def _year_start(ts):
    return int(datetime.fromtimestamp(ts, timezone.utc)
               .replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
               .timestamp())


def _bucket(candles, bar_s, keep):
    """Aggregate fine candles into coarser buckets (o=first, h=max, l=min, c=last)."""
    out = []
    for k in candles:
        t = int(k["t"])
        bt = (_year_start(t) if bar_s == "Y" else
              _month_start(t) if bar_s == "M" else t // bar_s * bar_s)
        if out and out[-1]["t"] == bt:
            b = out[-1]
            b["h"] = max(b["h"], k["h"])
            b["l"] = min(b["l"], k["l"])
            b["c"] = k["c"]
        else:
            out.append(dict(t=bt, o=k["o"], h=k["h"], l=k["l"], c=k["c"]))
    return out[-keep:]


def _tick_candles(tf, cfg, force=False):
    """Seconds-timeframe candles aggregated from the live tick feed."""
    now = time.time()
    st = _mem.get(tf)
    if not force and st and now - st["fetchedAt"] < cfg["ttl"]:
        return dict(st, changed=False)
    snaps = wsfeed.hist()
    if len(snaps) >= 10:
        bar_s = cfg["bar_s"]
        buckets = {}
        for ts, px in snaps:
            b = int(ts) // bar_s * bar_s
            cur = buckets.get(b)
            if cur is None:
                buckets[b] = [px, px, px, px]
            else:
                cur[1] = max(cur[1], px)
                cur[2] = min(cur[2], px)
                cur[3] = px
        candles = [dict(t=b, o=round(v[0], 2), h=round(v[1], 2),
                        l=round(v[2], 2), c=round(v[3], 2))
                   for b, v in sorted(buckets.items())][-cfg["keep"]:]
        st = dict(candles=candles, source="Live exchange ticks · OKX PAXG book",
                  stale=False, fetchedAt=now, ttl=cfg["ttl"])
        _mem[tf] = st
        if len(candles) >= 120:            # survive restarts with warm history
            try:
                with open(_cache_path(tf), "w") as f:
                    json.dump(dict(candles=candles, source=st["source"],
                                   fetchedAt=now), f)
            except OSError:
                pass
        return dict(st, changed=True)
    # tick history still warming up — serve the disk cache if we have one
    disk = None
    try:
        with open(_cache_path(tf)) as f:
            disk = json.load(f)
    except Exception:  # noqa: BLE001
        pass
    if disk and disk.get("candles"):
        st = dict(candles=disk["candles"], source=disk.get("source", "?") + " · cached",
                  stale=True, fetchedAt=now, ttl=10)
        _mem[tf] = st
        return dict(st, changed=True)
    return dict(candles=None, source=None, stale=True, fetchedAt=now, changed=True,
                error="tick history warming up — try again in a minute")


def _agg_candles(tf, cfg, force=False):
    """Coarser timeframe aggregated from a finer CACHED timeframe."""
    now = time.time()
    st = _mem.get(tf)
    if not force and st and now - st["fetchedAt"] < cfg["ttl"]:
        return dict(st, changed=False)
    base = get_candles(cfg["base"])
    src = base.get("candles")
    if src:
        candles = _bucket(src, cfg["bar_s"], cfg["keep"])
        st = dict(candles=candles,
                  source=(base.get("source") or "?") + f" · aggregated to {tf}",
                  stale=bool(base.get("stale")), fetchedAt=now, ttl=cfg["ttl"])
        _mem[tf] = st
        return dict(st, changed=True)
    if st:
        return dict(st, changed=False)
    return dict(candles=None, source=None, stale=True, fetchedAt=now, changed=True,
                error=base.get("error") or "base timeframe unavailable")


def get_candles(tf, force=False):
    """Return dict(candles, source, stale, fetchedAt, changed[, error]).

    `changed=True` means a fresh network fetch produced new candles (or a
    failure meant we had to fall back to the disk cache).
    """
    if tf not in TFS:
        raise ValueError(f"unknown timeframe {tf}")
    cfg = TFS[tf]
    if cfg.get("src") == "ticks":
        return _tick_candles(tf, cfg, force)
    if cfg.get("src") == "agg":
        return _agg_candles(tf, cfg, force)
    st = _mem.get(tf)
    now = time.time()
    if not force and st and now - st["fetchedAt"] < cfg["ttl"]:
        return dict(st, changed=False)

    # stale-while-revalidate: a recently-expired cache is served INSTANTLY
    # while a background thread refreshes it — /api/data never blocks on a
    # slow upstream (Yahoo can take 1-3s). One refresh per tf at a time.
    if not force and st and now - st["fetchedAt"] < cfg["ttl"] * 5:
        with _bg_lock:
            if tf not in _bg_refreshing:
                _bg_refreshing.add(tf)

                def _bg(t=tf):
                    try:
                        get_candles(t, force=True)
                    finally:
                        with _bg_lock:
                            _bg_refreshing.discard(t)

                threading.Thread(target=_bg, daemon=True).start()
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
