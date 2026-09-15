"""Fundamental layer for the XAUUSD desk.

Gold is driven by macro as much as by charts. This module adds:
  • Economic calendar  — ForexFactory's free weekly JSON (CPI, FOMC, NFP, ...)
  • Macro backdrop     — dollar index (DXY) and US 10Y yield from Yahoo
  • News               — WatcherGuru's public Telegram feed + Google News RSS,
                         filtered to what actually moves gold

Everything is cached and failure-tolerant: any dead source just returns an
empty result, the app keeps working.
"""
from __future__ import annotations

import html as _html
import json
import re
import time
import urllib.request
from datetime import datetime, timezone

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126"}

CAL_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
CAL_NEXT_URL = "https://nfs.faireconomy.media/ff_calendar_nextweek.json"
WATCHER_URL = "https://t.me/s/WatcherGuru"
GNEWS_URL = ("https://news.google.com/rss/search?q="
             "gold+OR+XAUUSD+OR+%22federal+reserve%22+when:1d&hl=en-US&gl=US&ceid=US:en")

_cal_mem = {"t": 0.0, "events": []}
_macro_mem = {"t": 0.0, "macro": None}
_watcher_mem = {"t": 0.0, "items": []}
_gnews_mem = {"t": 0.0, "items": []}

# keywords that mean "this moves gold"
HOT_KEYS = ("gold", "xau", "bullion", "fed", "fomc", "powell", "cpi", "inflation",
            "pce", "nfp", "payroll", "jobless", "unemployment", "jobs report",
            "dollar", "dxy", "treasury", "yields", "yield", "rate cut", "rate hike",
            "interest rate", "federal reserve", "tariff", "tariffs", "trade war",
            "sanction", "sanctions", "recession", "world war", "nuclear",
            "geopolit", "central bank", "imf", "war", "russia", "ukraine",
            "israel", "iran", "middle east", "missile", "troops", "invasion",
            "ceasefire", "offensive", "safe haven", "haven demand")
# crypto noise that WatcherGuru posts in volume and does NOT move gold
COLD_KEYS = ("bitcoin", "btc", "ethereum", "eth", "solana", "crypto", "altcoin",
             "memecoin", "nft", "exchange outflow", "etf inflow", "etf outflow",
             "whale ", "binance", "coinbase", "bybit", "okx listing")


def _get(url, timeout=10):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def _strip_tags(s):
    s = re.sub(r"<[^>]+>", " ", s)
    s = _html.unescape(s)
    return re.sub(r"\s+", " ", s).strip()


# ------------------------------------------------------------- calendar
FAIL_BACKOFF = 600                 # retry a dead source at most every 10 min


def calendar(max_age=6 * 3600):
    """This + next week's events as dicts (ts, title, country, impact,
    forecast, previous). Weekly data -> cached 6 hours; failures back off."""
    now = time.time()
    if now - _cal_mem["t"] < max_age and _cal_mem["events"]:
        return _cal_mem["events"]
    if now - _cal_mem.get("failT", 0) < FAIL_BACKOFF:
        return _cal_mem["events"]          # source recently failed — back off
    events = []
    for url in (CAL_URL, CAL_NEXT_URL):
        try:
            for e in json.loads(_get(url, timeout=12)):
                try:
                    ts = datetime.fromisoformat(e["date"]).timestamp()
                except Exception:  # noqa: BLE001
                    continue
                events.append(dict(
                    ts=int(ts), title=str(e.get("title", "")).strip(),
                    country=str(e.get("country", "")).strip(),
                    impact=str(e.get("impact", "")).strip(),
                    forecast=e.get("forecast") or "", previous=e.get("previous") or ""))
        except Exception:  # noqa: BLE001
            continue
    events.sort(key=lambda e: e["ts"])
    if events:
        _cal_mem.update(t=now, events=events)
    else:
        _cal_mem["failT"] = now
    return events


def upcoming(hours=48, countries=("USD", "ALL"), impacts=("High", "Medium")):
    """Next calendar events within `hours`, gold-relevant."""
    now = time.time()
    out = []
    for e in calendar():
        dt = e["ts"] - now
        if 0 <= dt <= hours * 3600 and e["country"] in countries and e["impact"] in impacts:
            out.append(dict(e, minutesTo=int(dt // 60)))
        if dt > hours * 3600:
            break
    return out[:8]


def next_high_impact(countries=("USD", "ALL")):
    for e in calendar():
        if e["ts"] > time.time() and e["country"] in countries and e["impact"] == "High":
            return dict(e, minutesTo=int((e["ts"] - time.time()) // 60))
    return None


# ------------------------------------------------------------- macro
def macro_snapshot(max_age=600):
    """Dollar index + US 10Y yield (level and daily change). Cached 10 min."""
    now = time.time()
    if now - _macro_mem["t"] < max_age and _macro_mem["macro"]:
        return _macro_mem["macro"]
    out = {}
    for key, sym in (("dxy", "DX-Y.NYB"), ("us10y", "%5ETNX")):
        try:
            j = json.loads(_get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d&range=5d",
                timeout=10))
            m = j["chart"]["result"][0]["meta"]
            px = float(m["regularMarketPrice"])
            prev = float(m.get("chartPreviousClose") or m.get("previousClose") or px)
            out[key] = dict(price=round(px, 2), chg=round(px - prev, 2),
                            chgPct=round((px - prev) / prev * 100, 2) if prev else 0.0)
        except Exception:  # noqa: BLE001
            continue
    _macro_mem.update(t=now, macro=out or None)
    return out or None


# ------------------------------------------------------------- news
def watcher_headlines(max_age=180, limit=25):
    """Latest WatcherGuru posts as (ts, text) — public web preview, no login."""
    now = time.time()
    if now - _watcher_mem["t"] < max_age and _watcher_mem["items"]:
        return _watcher_mem["items"]
    if now - _watcher_mem.get("failT", 0) < FAIL_BACKOFF:
        return _watcher_mem["items"]
    items = []
    try:
        page = _get(WATCHER_URL, timeout=12)
        blocks = re.findall(
            r'<time[^>]+datetime="([^"]+)"[^>]*>.*?'
            r'tgme_widget_message_text[^>]*>(.*?)</div>', page, re.S)
        for iso, body in blocks[:limit]:
            try:
                ts = datetime.fromisoformat(iso).timestamp()
            except Exception:  # noqa: BLE001
                ts = now
            txt = _strip_tags(body)[:280]
            if txt:
                items.append((int(ts), txt))
    except Exception:  # noqa: BLE001
        pass
    if items:
        _watcher_mem.update(t=now, items=items)
    else:
        _watcher_mem["failT"] = now
    return items


def google_news(max_age=900, limit=12):
    """Gold-relevant headlines from Google News RSS as (ts, title, source)."""
    now = time.time()
    if now - _gnews_mem["t"] < max_age and _gnews_mem["items"]:
        return _gnews_mem["items"]
    if now - _gnews_mem.get("failT", 0) < FAIL_BACKOFF:
        return _gnews_mem["items"]
    items = []
    try:
        rss = _get(GNEWS_URL, timeout=12)
        for m in re.finditer(r"<item>(.*?)</item>", rss, re.S):
            blk = m.group(1)
            t = re.search(r"<title>(.*?)</title>", blk, re.S)
            d = re.search(r"<pubDate>(.*?)</pubDate>", blk, re.S)
            if not t:
                continue
            title = _strip_tags(t.group(1))
            src = title.rsplit(" - ", 1)
            try:
                ts = datetime.strptime(d.group(1).strip(),
                                       "%a, %d %b %Y %H:%M:%S %z").timestamp()
            except Exception:  # noqa: BLE001
                ts = now
            items.append((int(ts), title, src[1] if len(src) == 2 else "news"))
            if len(items) >= limit:
                break
        items.sort(key=lambda x: -x[0])
    except Exception:  # noqa: BLE001
        pass
    if items:
        _gnews_mem.update(t=now, items=items)
    else:
        _gnews_mem["failT"] = now
    return items


_HOT_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in HOT_KEYS) + r")\b")


def gold_relevant(text):
    """True if a headline plausibly moves XAU/USD (hot keywords, not noise)."""
    t = text.lower()
    if any(k in t for k in COLD_KEYS) and not any(
            k in t for k in ("gold", "fed", "fomc", "cpi", "inflation")):
        return False
    # word boundaries both sides: 'war' must not hit 'warehouse', etc.
    return bool(_HOT_RE.search(t))


def snapshot():
    """Everything the web UI's Fundamentals & News card needs."""
    ev = upcoming(hours=48)
    hi = [e for e in ev if e["impact"] == "High"] or ev[:3]
    news = [(ts, txt, "WatcherGuru") for ts, txt in watcher_headlines(limit=15)
            if gold_relevant(txt)][:4]
    news += [(ts, ttl, src) for ts, ttl, src in google_news(limit=10)
             if gold_relevant(ttl)][:4]
    news.sort(key=lambda x: -x[0])
    return dict(
        macro=macro_snapshot(),
        events=hi[:5],
        nextHigh=next_high_impact(),
        news=news[:8],
        asOf=int(time.time()))
