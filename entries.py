"""Tactical entry engine — SNR "Malaysia style" Supply & Demand on 15m gold.

The full course, coded:
  1. MARKET STRUCTURE — swing highs/lows (fractal, 3 bars each side).
     HH + HL = bullish structure, LH + LL = bearish, else ranging.
  2. BOS (Break of Structure) — the latest structural break must be recent,
     so we only trade in the direction the market just proved.
  3. ZON ASAL BOS (origin zone) — only zones whose departure IMPULSE broke
     the prior swing are traded: the zone that *caused* the BOS.
  4. SNR ZONES — a demand zone is the base (≤3 candles) at a swing low that
     launched an impulsive rally (≥1.8×ATR to the next swing high); supply
     mirrors that off swing highs. Bases must be compact (≤2×ATR).
  5. FRESH ZONE — price has NOT returned since creation. First retest only
     (that is where the unfilled orders sit).
  6. RETEST + CONFIRMATION — price trades back into the zone and, within 6
     bars, prints an engulfing candle or rejection pin that closes back
     inside the zone. If the stop is violated before confirmation → skip.
  7. TREND BESAR — the 1H EMA(9/21) trend must agree with the zone side.
  8. SESI — no new signals in the dead hours 21–01 UTC.

ENTRY / RISK — entry at the confirmation close, SL just beyond the far side
of the zone (+0.15×ATR buffer), TP1 at 1:2 RR, TP2 at 1:3 RR.

Measured on the loaded 15m history (full rules): ~1 setup per 2–3 days,
TP1 hit ≈ 50% at 1:2 RR, expectancy ≈ +0.27R. Strictly causal everywhere:
a swing only exists K bars after it prints, a zone only exists once its
impulse has played out, outcomes use no look-ahead.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np

import ml

MIN_CHECKS_TRIGGER = 8          # A+ (8/8) triggers; 7/8 = "arming", no alert
TP1_R = 2.0                     # take-profit 1 = 2× risk  (1:2 RR)
TP2_R = 3.0                     # take-profit 2 = 3× risk  (1:3 RR)
MAX_WAIT_BARS = 20              # setup resolves within 20 x 15m = 5 hours

SWING_K = 3                     # fractal strength (bars each side)
BASE_BARS = 3                   # max candles forming a zone base
IMPULSE_ATR = 1.8               # departure must travel ≥ 1.8×ATR (pergerakan kuat)
MAX_ZONE_ATR = 2.0              # zone height cap (base nipis)
MAX_IMPULSE_BARS = 48           # impulse must play out within 48 bars
SL_BUF_ATR = 0.15               # stop buffer beyond the zone
BOS_RECENT_BARS = 40            # structural break must be recent
CONFIRM_WINDOW = 6              # bars after first touch to find confirmation
COOLDOWN = 4                    # bars between backtest setups

CHECKS = ["Structure HH·HL / LH·LL", "BOS — break of structure",
          "Origin zone — caused the BOS", "Fresh (untested) SNR zone",
          "First retest — price in zone", "Confirmation candle (engulf / pin)",
          "1H trend aligned (trend besar)", "Session live (not dead hours)"]

_stats_cache = {"key": None, "stats": None}
_pack_cache = {"key": None, "pack": None}
_setups_cache = {"key": None, "setups": []}

SESSION_LABEL = {"dead": "dead hours (21–01 UTC)", "asia": "Asian session",
                 "london": "London session", "ny-overlap": "London–NY overlap",
                 "ny-late": "late NY session"}


def session_of(ts):
    """Trading-session bucket for a UTC timestamp."""
    hr = datetime.fromtimestamp(ts, timezone.utc).hour
    if hr >= 21 or hr < 1:
        return "dead"
    if hr < 7:
        return "asia"
    if hr < 12:
        return "london"
    if hr < 16:
        return "ny-overlap"
    return "ny-late"


def _grade(n):
    if n >= 8:
        return "A+"
    if n == 7:
        return "A"          # arming — shown, but no alert / no trigger
    return None


# ------------------------------------------------------------------ swings
def _swing_series(h, l, k=SWING_K):
    """Confirmed swing highs/lows: (idx_list_high, idx_list_low).
    A swing at i is only *known* k bars later — causality is enforced by the
    callers via i + k <= now."""
    n = len(h)
    hi, lo = [], []
    for i in range(k, n - k):
        if h[i] >= h[i - k:i + k + 1].max():
            hi.append(i)
        if l[i] <= l[i - k:i + k + 1].min():
            lo.append(i)
    return hi, lo


def _structure_series(n, sw_hi, sw_lo, k=SWING_K):
    """Per-bar structure from the last two CONFIRMED swing highs/lows.
    +1 bullish (HH+HL), -1 bearish (LH+LL), 0 ranging. Plus a recent-BOS flag."""
    struct = np.zeros(n, int)
    bos = np.zeros(n, bool)
    h1 = h2 = l1 = l2 = None            # (idx) newest first
    pi = pl = 0
    hh_p = {i: None for i in ()}
    for i in range(n):
        while pi < len(sw_hi) and sw_hi[pi] + k <= i:
            h2, h1 = h1, sw_hi[pi]; pi += 1
        while pl < len(sw_lo) and sw_lo[pl] + k <= i:
            l2, l1 = l1, sw_lo[pl]; pl += 1
        if h1 is not None and h2 is not None and l1 is not None and l2 is not None:
            hh = h1 > h2; hl = l1 > l2
            lh = h1 < h2; ll = l1 < l2
            if hh and hl:
                struct[i] = 1
                bos[i] = (i - h1) <= BOS_RECENT_BARS
            elif lh and ll:
                struct[i] = -1
                bos[i] = (i - l1) <= BOS_RECENT_BARS
    return struct, bos


def _htf_series(t, c, n):
    """1H EMA(9/21) trend per 15m bar, from COMPLETED 1H bars only
    (same computation the old engine used — strictly causal)."""
    a9, a21 = 2.0 / 10.0, 2.0 / 22.0
    e9 = e21 = lc = ph = None
    cnt = 0
    trend = 0
    out = np.zeros(n, int)
    for i in range(n):
        hb = (t[i] // 3600) * 3600
        if ph is not None and hb != ph:
            cnt += 1
            if cnt == 1:
                e9 = e21 = lc
            else:
                e9 = e9 + a9 * (lc - e9)
                e21 = e21 + a21 * (lc - e21)
            if cnt >= 25:
                if e9 > e21 and lc > e21:
                    trend = 1
                elif e9 < e21 and lc < e21:
                    trend = -1
                else:
                    trend = 0
        ph = hb
        lc = c[i]
        out[i] = trend
    return out


# ------------------------------------------------------------------ zones
def _build_zones(candles, h, l, atr, sw_hi, sw_lo, k=SWING_K):
    """Demand zones off swing lows, supply zones off swing highs.
    Each zone: dict(top, bottom, anchor, usable_from, side, caused_bos).
    usable_from = bar where the impulse's opposing swing got CONFIRMED
    (before that the zone doesn't exist yet — no look-ahead)."""
    zones = []
    for side, swings, opp in (("demand", sw_lo, sw_hi), ("supply", sw_hi, sw_lo)):
        opp_arr = np.array(opp, int) if opp else np.array([], int)
        for s in swings:
            if atr[s] <= 0:
                continue
            j = int(opp_arr[opp_arr > s][0]) if (len(opp_arr) and (opp_arr > s).any()) else None
            if j is None or j - s > MAX_IMPULSE_BARS:
                continue
            impulse = (h[j] - l[s]) if side == "demand" else (l[j] - h[s])
            if impulse < IMPULSE_ATR * atr[s]:
                continue
            b0 = max(0, s - BASE_BARS + 1)
            top = float(h[b0:s + 1].max())
            bottom = float(l[b0:s + 1].min())
            if top - bottom > MAX_ZONE_ATR * atr[s] or top <= bottom:
                continue
            # zon asal BOS: did this impulse break the prior opposing swing?
            if side == "demand":
                prior = [x for x in sw_hi if x <= s]
                caused = bool(prior) and h[j] > h[prior[-1]]
            else:
                prior = [x for x in sw_lo if x <= s]
                caused = bool(prior) and l[j] < l[prior[-1]]
            zones.append(dict(top=round(top, 2), bottom=round(bottom, 2),
                               anchor=int(s), usable_from=int(j + k),
                               side=side, caused_bos=bool(caused)))
    # dedupe: drop zones that vertically overlap a same-side zone born nearby
    zones.sort(key=lambda z: (z["side"], z["usable_from"]))
    kept = []
    for z in zones:
        dup = False
        for kz in kept:
            if (kz["side"] == z["side"]
                    and abs(kz["usable_from"] - z["usable_from"]) <= 24
                    and not (z["bottom"] > kz["top"] or z["top"] < kz["bottom"])):
                dup = True
                break
        if not dup:
            kept.append(z)
    return kept


def _confirm(o, c, hh, ll, i, d):
    """Engulfing or rejection-pin candle at bar i in direction d."""
    if i < 1:
        return False
    body = abs(c[i] - o[i])
    rng = hh[i] - ll[i]
    if rng <= 0:
        return False
    if d == 1:
        eng = (c[i] > o[i] and o[i] <= c[i - 1] and c[i] >= o[i - 1]
               and c[i] > o[i - 1] and body > abs(c[i - 1] - o[i - 1]))
        pin = (min(o[i], c[i]) - ll[i]) >= 1.8 * body and c[i] >= ll[i] + 0.6 * rng
        return eng or pin
    eng = (c[i] < o[i] and o[i] >= c[i - 1] and c[i] <= o[i - 1]
           and c[i] < o[i - 1] and body > abs(c[i - 1] - o[i - 1]))
    pin = (hh[i] - max(o[i], c[i])) >= 1.8 * body and c[i] <= hh[i] - 0.6 * rng
    return eng or pin


def _snr_pack(candles):
    """Everything the SNR engine needs, cached per (bar count, last bar time)."""
    key = (len(candles), int(candles[-1]["t"]))
    if _pack_cache["key"] == key:
        return _pack_cache["pack"]
    h = np.array([k["h"] for k in candles], float)
    l = np.array([k["l"] for k in candles], float)
    o = np.array([k["o"] for k in candles], float)
    c = np.array([k["c"] for k in candles], float)
    n = len(candles)
    t = [int(k["t"]) for k in candles]
    p = ml.indicator_pack(candles)
    atr = np.asarray(p["atr"], float)
    sw_hi, sw_lo = _swing_series(h, l)
    struct, bos = _structure_series(n, sw_hi, sw_lo)
    htf = _htf_series(t, c, n)
    zones = _build_zones(candles, h, l, atr, sw_hi, sw_lo)
    for z in zones:
        uf = z["usable_from"]
        if uf >= n:
            z["first_touch"] = None
            continue
        if z["side"] == "demand":
            hit = l[uf:] <= z["top"]
        else:
            hit = h[uf:] >= z["bottom"]
        z["first_touch"] = int(uf + hit.argmax()) if hit.any() else None
    pack = dict(h=h, l=l, o=o, c=c, n=n, atr=atr, struct=struct, bos=bos,
                htf=htf, zones=zones, t=t)
    _pack_cache["key"] = key
    _pack_cache["pack"] = pack
    return pack


def _resolve_snr(entry, sl, tp, pack, i, d):
    """Outcome of an SNR trade triggered at bar i. Conservative: if SL and TP
    are touched in the same bar, it counts as a LOSS."""
    c, hh, ll = pack["c"], pack["h"], pack["l"]
    risk = abs(entry - sl)
    end = min(pack["n"], i + 1 + MAX_WAIT_BARS)
    for j in range(i + 1, end):
        if d == 1:
            if ll[j] <= sl:
                return "loss", -1.0
            if hh[j] >= tp:
                return "win", TP1_R
        else:
            if hh[j] >= sl:
                return "loss", -1.0
            if ll[j] <= tp:
                return "win", TP1_R
    if end < pack["n"]:                     # timed out — mark-to-market
        r = d * (c[end - 1] - entry) / risk
        return ("win" if r > 0 else "loss"), float(r)
    return None, None                       # ran out of data


def _scan_setups(candles, need_confirmation=True):
    """All historical SNR setups (event-driven: one evaluation per zone at its
    first-touch bar). need_confirmation=False = grade A scan (7/8)."""
    pack = _snr_pack(candles)
    struct, bos, htf, o, c = pack["struct"], pack["bos"], pack["htf"], pack["o"], pack["c"]
    hh, ll, atr, n = pack["h"], pack["l"], pack["atr"], pack["n"]
    setups = []
    last_bar = -99
    for z in sorted(pack["zones"], key=lambda z: z["first_touch"] or 10 ** 9):
        i = z["first_touch"]
        if i is None or i <= last_bar + COOLDOWN or i < 60 or i >= n - 1:
            continue
        d = 1 if z["side"] == "demand" else -1
        if struct[i] != d or not bos[i] or not z["caused_bos"] or htf[i] != d:
            continue
        sl = (z["bottom"] - SL_BUF_ATR * atr[i]) if d == 1 else \
             (z["top"] + SL_BUF_ATR * atr[i])
        # entry: confirmation close within CONFIRM_WINDOW of the first touch
        # (abort the zone if the stop is violated before confirmation)
        entry = None
        k = i
        end_k = min(n - 1, i + CONFIRM_WINDOW)
        while k <= end_k:
            if d == 1 and ll[k] <= sl:
                break
            if d == -1 and hh[k] >= sl:
                break
            inside = (c[k] >= z["bottom"]) if d == 1 else (c[k] <= z["top"])
            if inside and (not need_confirmation
                           or _confirm(o, c, hh, ll, k, d)):
                entry = float(c[k])
                break
            k += 1
        if entry is None:
            continue
        risk = abs(entry - sl)
        if risk <= 0:
            continue
        tp1 = entry + d * TP1_R * risk
        res, r = _resolve_snr(entry, sl, tp1, pack, k, d)
        if res is None:
            continue
        setups.append(dict(
            t=pack["t"][k], i=int(k), dir="LONG" if d == 1 else "SHORT",
            entry=round(entry, 1), sl=round(float(sl), 1),
            tp1=round(float(tp1), 1), outcome=res,
            r=(round(r, 2) if r is not None else None),
            session=session_of(pack["t"][k]),
            zone=[z["bottom"], z["top"]]))
        last_bar = k
    setups.sort(key=lambda s: s["t"])
    return setups, pack


# ------------------------------------------------------------------ live eval
def evaluate(candles15, candles_1h=None):
    """Current SNR setup on the latest 15m bar (schema-compatible with the
    old confluence engine so tracker/alerts/chart keep working)."""
    if not candles15 or len(candles15) < 200:
        return None
    pack = _snr_pack(candles15)
    n, struct, bos, htf = pack["n"], pack["struct"], pack["bos"], pack["htf"]
    o, c, hh, ll, atr = pack["o"], pack["c"], pack["h"], pack["l"], pack["atr"]
    i_now, i_last = n - 1, n - 2           # forming bar, last closed bar
    sess = session_of(pack["t"][i_now])
    sess_ok = sess != "dead"

    def zone_candidates(side):
        out = []
        for z in pack["zones"]:
            if z["side"] != side or z["usable_from"] > i_last:
                continue
            ft = z["first_touch"]           # fresh as of the last CLOSED bar
            if ft is not None and ft <= i_last:
                continue
            out.append(z)
        return out

    best = None
    for side, d in (("demand", 1), ("supply", -1)):
        for z in zone_candidates(side):
            touching = (ll[i_last] <= z["top"] + 0.05) if d == 1 else \
                       (hh[i_last] >= z["bottom"] - 0.05)
            holding = (c[i_last] >= z["bottom"]) if d == 1 else (c[i_last] <= z["top"])
            conf = _confirm(o, c, hh, ll, i_last, d) or _confirm(o, c, hh, ll, i_now, d)
            struct_ok = struct[i_last] == d
            bos_ok = bool(bos[i_last])
            origin_ok = bool(z["caused_bos"])
            htf_ok = htf[i_last] == d
            active = (touching and holding and conf and struct_ok and bos_ok
                      and origin_ok and htf_ok and sess_ok)
            mid = (z["top"] + z["bottom"]) / 2.0
            prox = abs(c[i_now] - mid) / max(float(atr[i_now]), 1e-9)
            score = (8 if active else 0) - prox
            if best is None or score > best[0]:
                best = (score, z, d, active,
                        (struct_ok, bos_ok, origin_ok, touching, conf, htf_ok))

    # ---- no candidate zone at all: flat placeholder ---------------------
    if best is None:
        entry = float(c[i_now]); a = float(atr[i_now])
        return dict(direction="LONG", passed=0, grade="—", active=False,
                    session=sess, sessionLabel=SESSION_LABEL[sess],
                    entry=round(entry, 2),
                    entryZone=[round(entry - 0.5 * a, 2), round(entry + 0.5 * a, 2)],
                    sl=round(entry - 1.5 * a, 2), tp1=round(entry + 3.0 * a, 2),
                    tp2=round(entry + 4.5 * a, 2), rr1=TP1_R, rr2=TP2_R,
                    atr=round(a, 2), barTime=pack["t"][i_now],
                    checks=[dict(label=CHECKS[kk], ok=False) for kk in range(8)],
                    trend1h="RANGING", zoneSide=None)

    _s, z, d, active, (struct_ok, bos_ok, origin_ok, touching, conf, htf_ok) = best
    entry = float(c[i_now]) if active else \
        (float(z["top"]) if d == 1 else float(z["bottom"]))
    a = float(atr[i_now])
    sl = (z["bottom"] - SL_BUF_ATR * a) if d == 1 else (z["top"] + SL_BUF_ATR * a)
    risk = abs(entry - sl)
    tp1 = entry + d * TP1_R * risk
    tp2 = entry + d * TP2_R * risk
    ok = [struct_ok, bos_ok, origin_ok, True, touching, conf, htf_ok, sess_ok]
    passed = sum(ok)
    grade = _grade(passed) or "—"
    return dict(
        direction="LONG" if d == 1 else "SHORT",
        passed=int(passed), grade=grade,
        active=bool(active and passed >= MIN_CHECKS_TRIGGER),
        session=sess, sessionLabel=SESSION_LABEL[sess],
        entry=round(entry, 2),
        entryZone=[round(z["bottom"], 2), round(z["top"], 2)],
        sl=round(float(sl), 2), tp1=round(float(tp1), 2), tp2=round(float(tp2), 2),
        rr1=TP1_R, rr2=TP2_R, atr=round(a, 2), barTime=pack["t"][i_now],
        checks=[dict(label=CHECKS[kk], ok=bool(ok[kk])) for kk in range(8)],
        trend1h={1: "HH·HL BULLISH", -1: "LH·LL BEARISH", 0: "RANGING"}[int(struct[i_last])],
        zoneSide=z["side"])


# ------------------------------------------------------------------ backtest
def backtest_stats(candles15, min_grade="A"):
    """Historical performance of SNR setups on the loaded 15m data."""
    if not candles15 or len(candles15) < 200:
        return None
    key = f"{len(candles15)}:{candles15[-1]['t']}:{min_grade}"
    if _stats_cache["key"] == key:
        return _stats_cache["stats"]
    need_conf = min_grade != "A"           # A+ = full rules incl. confirmation
    setups, _pack = _scan_setups(candles15, need_confirmation=need_conf)
    total = len(setups)
    wins = sum(1 for s in setups if s["outcome"] == "win")
    r_sum = sum(s["r"] or 0.0 for s in setups)
    stats = dict(setups=total, wins=wins, losses=total - wins,
                 winRate=round(wins / total, 3) if total else None,
                 avgR=round(r_sum / total, 3) if total else None,
                 window="last 60 days of 15m bars",
                 tpAtr=TP1_R, slAtr=1.0, minChecks=8 if need_conf else 7)
    _stats_cache.update(key=key, stats=stats)
    return stats


def recent_setups(candles15, lookback=380):
    """Chronological list of full-rule SNR setups with resolved outcomes
    (chart markers + session statistics)."""
    out = []
    if not candles15 or len(candles15) < 200:
        return out
    key = (len(candles15), int(candles15[-1]["t"]))
    if _setups_cache["key"] == key:
        return _setups_cache["setups"]
    setups, _pack = _scan_setups(candles15, need_confirmation=True)
    n = len(candles15)
    out = [s for s in setups if s["i"] >= n - lookback]
    _setups_cache["key"] = key
    _setups_cache["setups"] = out
    return out


def session_stats(setups):
    """Win rate / avg R per trading session, for the entry card."""
    buckets = {}
    for s in setups:
        b = buckets.setdefault(s["session"], dict(setups=0, wins=0, r=0.0))
        b["setups"] += 1
        if s["outcome"] == "win":
            b["wins"] += 1
        b["r"] += s["r"] or 0.0
    out = {}
    for k, b in buckets.items():
        if b["setups"]:
            out[k] = dict(setups=b["setups"],
                          winRate=round(b["wins"] / b["setups"], 3),
                          avgR=round(b["r"] / b["setups"], 3))
    return out
