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
GNEWS_URL2 = ("https://news.google.com/rss/search?q="
              "gold+price+forecast+OR+outlook+OR+analysis+when:2d"
              "&hl=en-US&gl=US&ceid=US:en")

# every online source the AI reads on its own, 24/7
WIDE_FEEDS = [
    ("investing", "https://www.investing.com/rss/news_11.rss"),
    ("yahoo", "https://feeds.finance.yahoo.com/rss/2.0/headline?"
              "s=GC=F&region=US&lang=en-US"),
    ("marketwatch", "https://feeds.content.dowjones.io/public/rss/"
                    "mw_topstories"),
]

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


def _calendar_cloud():
    """Calendar shared via the private state repo: whichever machine can
    reach the feed pushes calendar.json; blocked machines read it."""
    import os
    import base64
    tok = os.environ.get("GH_STATE_PAT")
    if not tok:
        return []
    try:
        req = urllib.request.Request(
            "https://api.github.com/repos/EA6455/xauusd-ai-state/contents/"
            "calendar.json",
            headers={"Authorization": f"token {tok}",
                     "Accept": "application/vnd.github+json",
                     "User-Agent": UA.get("User-Agent", "xauusd-ai")})
        with urllib.request.urlopen(req, timeout=12) as r:
            j = json.loads(r.read().decode())
        blob = json.loads(base64.b64decode(j["content"]))
        if time.time() - (blob.get("fetchedAt") or 0) > 6 * 3600 + 1800:
            return []
        return blob.get("events") or []
    except Exception:  # noqa: BLE001
        return []


def calendar(max_age=6 * 3600):
    """This + next week's events as dicts (ts, title, country, impact,
    forecast, previous). Weekly data -> cached 6 hours; failures back off
    and fall back to the shared cloud copy."""
    now = time.time()
    if now - _cal_mem["t"] < max_age and _cal_mem["events"]:
        return _cal_mem["events"]
    if now - _cal_mem.get("failT", 0) < FAIL_BACKOFF:
        if _cal_mem["events"]:
            return _cal_mem["events"]      # source recently failed — back off
        ev = _calendar_cloud()             # blocked here? use the shared copy
        if ev:
            _cal_mem.update(t=now, events=ev)
        return _cal_mem["events"]
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
    if not events:
        events = _calendar_cloud()
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


_wide_mem = {"t": 0.0, "items": [], "fail": {}}


def _rss_items(rss, limit=12):
    """Parse an RSS body into (ts, title, source-unknown) tuples."""
    out = []
    now = time.time()
    for m in re.finditer(r"<item>(.*?)</item>", rss, re.S):
        blk = m.group(1)
        t = re.search(r"<title>(.*?)</title>", blk, re.S)
        d = re.search(r"<pubDate>(.*?)</pubDate>", blk, re.S)
        if not t:
            continue
        title = _strip_tags(t.group(1))
        try:
            ts = datetime.strptime(d.group(1).strip(),
                                   "%a, %d %b %Y %H:%M:%S %z").timestamp()
        except Exception:  # noqa: BLE001
            ts = now
        out.append((int(ts), title))
        if len(out) >= limit:
            break
    return out


def wide_news(max_age=900, limit=24):
    """ALL online sources merged: Google News (two queries), Investing.com
    commodities, Yahoo gold-futures wire, MarketWatch. Deduped across
    feeds, newest first. Per-feed failure backoff so one dead site never
    stalls the loop. Returns (ts, title, source) tuples."""
    now = time.time()
    if now - _wide_mem["t"] < max_age and _wide_mem["items"]:
        return _wide_mem["items"]
    items = []
    for q in (GNEWS_URL, GNEWS_URL2):
        try:
            for ts, title in _rss_items(_get(q, timeout=12), limit=12):
                items.append((ts, title, "google"))
        except Exception:  # noqa: BLE001
            pass
    for name, url in WIDE_FEEDS:
        if now - _wide_mem["fail"].get(name, 0) < FAIL_BACKOFF:
            continue
        try:
            for ts, title in _rss_items(_get(url, timeout=12), limit=12):
                items.append((ts, title, name))
        except Exception:  # noqa: BLE001
            _wide_mem["fail"][name] = now
    seen, deduped = set(), []
    for ts, title, src in items:
        key = re.sub(r"\s+", " ", title.lower()).strip()[:60]
        if key in seen:
            continue
        seen.add(key)
        deduped.append((ts, title, src))
    deduped.sort(key=lambda x: -x[0])
    # round-robin across sources so every live feed is represented,
    # not just the fastest publisher
    by_src = {}
    for it in deduped:
        by_src.setdefault(it[2], []).append(it)
    merged = []
    while len(merged) < limit:
        took = False
        for src in sorted(by_src):
            if by_src[src]:
                merged.append(by_src[src].pop(0))
                took = True
                if len(merged) >= limit:
                    break
        if not took:
            break
    merged.sort(key=lambda x: -x[0])
    _wide_mem.update(t=now, items=merged)
    return _wide_mem["items"]


_wmacro_mem = {"t": 0.0, "macro": None}


def wide_macro(max_age=900):
    """Extended macro panel: gold + silver (and the gold/silver ratio),
    S&P 500 and oil — risk sentiment alongside the dollar/yield backdrop.
    Cached 15 min."""
    now = time.time()
    if now - _wmacro_mem["t"] < max_age and _wmacro_mem["macro"]:
        return _wmacro_mem["macro"]
    out = {}
    for key, sym in (("gc", "GC=F"), ("si", "SI=F"),
                     ("spx", "%5EGSPC"), ("cl", "CL=F")):
        try:
            j = json.loads(_get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
                "?interval=1d&range=5d", timeout=10))
            m = j["chart"]["result"][0]["meta"]
            px = float(m["regularMarketPrice"])
            prev = float(m.get("chartPreviousClose")
                         or m.get("previousClose") or px)
            out[key] = dict(price=round(px, 2),
                            chgPct=round((px - prev) / prev * 100, 2)
                            if prev else 0.0)
        except Exception:  # noqa: BLE001
            continue
    if out.get("gc") and out.get("si") and out["si"]["price"]:
        out["ratio"] = round(out["gc"]["price"] / out["si"]["price"], 1)
    _wmacro_mem.update(t=now, macro=out or None)
    return out or None


_HOT_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in HOT_KEYS) + r")\b")


def gold_relevant(text):
    """True if a headline plausibly moves XAU/USD (hot keywords, not noise)."""
    t = text.lower()
    if any(k in t for k in COLD_KEYS) and not any(
            k in t for k in ("gold", "fed", "fomc", "cpi", "inflation")):
        return False
    # word boundaries both sides: 'war' must not hit 'warehouse', etc.
    return bool(_HOT_RE.search(t))


# ---------------------------------------------------- headline direction
# Desk-style read of a headline for XAU/USD. Gold trades off real yields,
# the dollar and risk appetite: dovish Fed / weak data / falling yields /
# weak dollar / geopolitical escalation = bullish; hawkish Fed / hot data /
# rising yields / strong dollar / risk-on = bearish. Rules are weighted and
# when both sides fire the read is honestly MIXED. Deterministic on purpose
# — same headline, same verdict, no hallucination.

_BULL_RULES = [
    # lookbehinds stop 'rate cuts' firing inside bearish phrases like
    # 'fewer rate cuts' / 'no rate cuts' / 'delayed rate cuts'
    (r"(?:(?<!fewer )(?<!no )(?<!delay )(?<!delayed )(?<!postpone )"
     r"\brate[- ]cuts?\b|cuts? (?:key |benchmark )?(?:interest )?rate\b|"
     r"cuts? (?:interest )?rates|lower(?:s|ing)? (?:interest )?rates?|"
     r"reduc(?:e|es|ing) (?:interest )?rates?|votes? to cut|"
     r"rate[- ]cut cycle|easing cycle|dovish|\bpivot\b)", 3,
     "rate-cut bets rise → yields & dollar headwind lifts gold"),
    (r"inflation (?:eases|cools|falls|slows|drops|retreats)|"
     r"cooler[- ]than[- ](?:expected|forecast)|cpi.*?(?:cooler|slower|"
     r"below (?:the )?forecast)|pce.*?(?:cooler|slower|below)", 3,
     "cooling inflation → more Fed cuts priced"),
    (r"jobless claims (?:rise|surge|jump|climb)|unemployment (?:rises|"
     r"jumps|climbs)|payrolls? (?:fall|drop|decline|miss)|nfp (?:miss|"
     r"disappoint)|weaker[- ]than[- ](?:expected|forecast) (?:jobs|"
     r"payrolls|employment|claims)", 2, "weak labor market → cut bets rise"),
    (r"recession (?:fears?|warning|risk)|economic slowdown|contraction", 2,
     "recession fears → safe-haven bid"),
    (r"invad|missile|airstrike|air strike|nuclear|escalat|offensive|"
     r"troops|declares? war|enters? the war|attacks? (?:on|near)|"
     r"strikes? (?:on|near)", 2, "geopolitical escalation → haven demand"),
    (r"sanction", 1, "sanctions tension → mild haven bid"),
    (r"(?:central banks?|official sector)[^.]{0,30}"
     r"(?:buy|buying|add|adding|purchas)|"
     r"(?:buying|buys|purchas\w*) gold|gold buying|bullion (?:buying|"
     r"demand)|adds? gold|gold purchases", 2,
     "official-sector gold demand"),
    (r"gold (?:rises|rallies|surges|jumps|climbs|gains|soars|hits? (?:a |"
     r"another )?record|extends? (?:gains|rally))|record (?:high|run) "
     r"(?:for gold|in gold)|gold price[^.]*?(?:up|rally|rise)", 3,
     "gold momentum is already up"),
    (r"dollar (?:weakens?|slides?|falls|drops|sinks|softens?)|"
     r"dxy (?:drops?|falls|slides|sinks)", 2, "weaker dollar lifts gold"),
    (r"yields? (?:fall|falls|drop|drops|slide|slides|ease|eases)", 2,
     "falling yields lift gold"),
    (r"de[- ]dollarisation|de[- ]dollarization|dollar loses (?:its )?"
     r"reserve", 2, "de-dollarization bid"),
    (r"gold[- ]backed|gold etf (?:inflow|inflows)|etf holdings (?:rise|"
     r"climb|jump|grow)", 2, "ETF money flowing into gold"),
    (r"tariff|trade war", 1, "trade-war premium (mildly supportive)"),
]
_BEAR_RULES = [
    (r"rate hikes?|hikes? (?:key |benchmark )?(?:interest )?rate\b|"
     r"hikes (?:interest )?rates|rais(?:e|es|ing) (?:interest )?rates?|"
     r"lifts? (?:its )?rate|votes? to hike|votes? to raise|hawkish|"
     r"higher for longer|"
     r"no rush to cut|push(?:es|ing)? back (?:on )?cut|patient on "
     r"(?:rate )?cuts|fewer (?:rate )?cuts|no (?:rate )?cuts|"
     r"delay(?:s|ed|ing)? (?:rate )?cuts?|cuts? (?:off the table|"
     r"unlikely|ruled out)", 3,
     "hawkish repricing → yields & dollar up"),
    (r"hot(?:ter)?[- ]than[- ](?:expected|forecast)|inflation "
     r"(?:accelerat|heats? up|picks up|rises faster|surges)|cpi (?:beats?|"
     r"jumps|surges|hot)|reaccelerat", 3, "hot inflation → fewer cuts priced"),
    (r"payrolls? (?:beat|beats|surge|jump|rise|climb|soar)|nfp (?:beats?|"
     r"strong)|strong (?:jobs|employment|labor market)|jobless claims "
     r"(?:drop|fall|plunge)|unemployment (?:falls|drops)", 2,
     "strong labor market → fewer cuts priced"),
    (r"dollar (?:strengthens?|rallies|rises|firms|jumps|gains|"
     r"rebounds?)|dxy (?:rises|jumps|rallies|climbs)", 2,
     "stronger dollar weighs on gold"),
    (r"yields? (?:rise|rises|surge|surges|climb|climbs|jump|jumps|spike|"
     r"rebound)", 2, "rising yields weigh on gold"),
    (r"gold (?:falls|drops|slides|plunges|tumbles|sinks|dips|retreats|"
     r"under pressure|extends? (?:decline|losses)|set for[^.]*?loss|"
     r"eases? (?:from|off))", 3, "gold momentum is already down"),
    (r"profit[- ]taking", 2, "profit-taking flow"),
    (r"risk[- ]on rally|stocks (?:rally|surge|jump|climb)|trade deal "
     r"(?:reached|struck|agreed)|tariffs? (?:lifted|cut|removed|eased|"
     r"paused)|de[- ]escalat|cease[- ]?fire|peace (?:deal|plan|talks)|"
     r"truce", 2, "risk appetite returns → haven bid fades"),
    (r"gold etf (?:outflow|outflows)|etf holdings (?:fall|drop|shrink)", 2,
     "ETF money flowing out of gold"),
]

_BULL_RE = [(re.compile(p, re.I), w, why) for p, w, why in _BULL_RULES]
_BEAR_RE = [(re.compile(p, re.I), w, why) for p, w, why in _BEAR_RULES]


def gold_bias(text):
    """Direction a headline implies for gold.

    Returns dict(bias='BULLISH'|'BEARISH'|'MIXED', score=int, why=[reasons]).
    |score| >= 2 (one solid rule or two weak ones) is needed for a directional
    read — anything less is honestly MIXED."""
    bull = [why for rx, _w, why in _BULL_RE if rx.search(text)]
    bear = [why for rx, _w, why in _BEAR_RE if rx.search(text)]
    score = 0
    for rx, w, _why in _BULL_RE:
        if rx.search(text):
            score += w
    for rx, w, _why in _BEAR_RE:
        if rx.search(text):
            score -= w
    if bull and bear:
        bias = "MIXED"                       # both sides fired — say so
    elif score >= 2:
        bias = "BULLISH"
    elif score <= -2:
        bias = "BEARISH"
    else:
        bias = "MIXED"
    return dict(bias=bias, score=score,
                why=(bull + bear) if (bull or bear) else
                ["no clear directional trigger in the headline"])


def event_playbook(title):
    """How gold typically reacts to a calendar event's actual vs forecast."""
    t = title.lower()
    if "cpi" in t or "inflation" in t or "pce" in t:
        return ("• Cooler than forecast → 🟢 gold rallies (more cuts priced)\n"
                "• Hotter than forecast → 🔴 gold drops (fewer cuts priced)")
    if ("fomc" in t or "fed" in t or "rate decision" in t
            or "interest rate" in t or "powell" in t or "federal" in t):
        return ("• Cut / dovish tone → 🟢 gold rallies\n"
                "• Hold + hawkish tone → 🔴 gold drops")
    if ("nfp" in t or "non-farm" in t or "nonfarm" in t
            or "unemployment" in t or "jobless" in t or "employment" in t
            or "payroll" in t):
        return ("• Weak jobs (miss) → 🟢 gold rallies (cut bets rise)\n"
                "• Strong jobs (beat) → 🔴 gold drops")
    if "gdp" in t:
        return ("• Weaker than forecast → 🟢 gold rallies\n"
                "• Stronger than forecast → 🔴 gold drops")
    if ("retail" in t or "pmi" in t or "consumer" in t or "sentiment" in t
            or "durable goods" in t or "housing" in t or "ism" in t
            or "orders" in t):
        return ("• Weaker than forecast → 🟢 gold rallies\n"
                "• Stronger than forecast → 🔴 gold drops")
    return "• Direction depends on the actual number vs forecast"



def snapshot():
    """Everything the web UI's Fundamentals & News card needs."""
    ev = upcoming(hours=48)
    hi = [e for e in ev if e["impact"] == "High"] or ev[:3]
    news = [(ts, txt, "WatcherGuru", gold_bias(txt)["bias"])
            for ts, txt in watcher_headlines(limit=15)
            if gold_relevant(txt)][:4]
    news += [(ts, ttl, src, gold_bias(ttl)["bias"])
             for ts, ttl, src in google_news(limit=10)
             if gold_relevant(ttl)][:4]
    news.sort(key=lambda x: -x[0])
    return dict(
        macro=macro_snapshot(),
        events=hi[:5],
        nextHigh=next_high_impact(),
        news=news[:8],
        asOf=int(time.time()))
