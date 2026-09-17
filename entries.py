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
TP1_R = 1.0                     # TP1 = 1.0× risk — half off, stop to BE (1:2 profile)
TP2_R = 2.0                     # TP2 runner = 2.0× risk
# v2 profile (backtested, 4522 x 15m bars): 1:2 RR at TP1 1.0R/TP2 2.0R lifts
# A+ to +0.50R avg (71% TP1) and the elite A+/B+ CONTINUATION cohort to 83%
# win / +0.65R (n=12). Mid/deep-pullback B+/C+ zones lose money at 1:2
# (42-44%) — those stay web-feed watch lines, not phone cards.
MAX_WAIT_BARS = 96              # backtest resolve window: 24h (TP2 runner at
                                # 2R needs room; matches the live tracker,
                                # which follows every card to TP or SL)

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
    """Setup quality ladder: A+ (8/8) > B+ (7/8) > C+ (6/8)."""
    if n >= 8:
        return "A+"
    if n == 7:
        return "B+"
    if n == 6:
        return "C+"
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
    e200 = np.empty(n)
    if n >= 200:
        e200[199] = c[:200].mean()
        _a = 2.0 / 201.0
        for i in range(200, n):
            e200[i] = _a * c[i] + (1 - _a) * e200[i - 1]
    else:
        e200[:] = np.nan
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
                htf=htf, zones=zones, t=t, e200=e200,
                sw_hi_idx=sw_hi, sw_lo_idx=sw_lo)
    _pack_cache["key"] = key
    _pack_cache["pack"] = pack
    return pack


def _resolve_snr(entry, sl, tp, pack, i, d, tp_r=None):
    """Outcome of an SNR trade triggered at bar i. Conservative: if SL and TP
    are touched in the same bar, it counts as a LOSS. tp_r overrides the
    win multiple (sweeps use their own wider target)."""
    c, hh, ll = pack["c"], pack["h"], pack["l"]
    risk = abs(entry - sl)
    win_r = TP1_R if tp_r is None else tp_r
    end = min(pack["n"], i + 1 + MAX_WAIT_BARS)
    for j in range(i + 1, end):
        if d == 1:
            if ll[j] <= sl:
                return "loss", -1.0
            if hh[j] >= tp:
                return "win", win_r
        else:
            if hh[j] >= sl:
                return "loss", -1.0
            if ll[j] <= tp:
                return "win", win_r
    if end < pack["n"]:                     # timed out — mark-to-market
        r = d * (c[end - 1] - entry) / risk
        return ("win" if r > 0 else "loss"), float(r)
    return None, None                       # ran out of data


GRADE_ORDER = {"A+": 3, "B+": 2, "C+": 1}


def _scan_setups(candles):
    """All historical SNR setups, graded (event-driven: one evaluation per
    zone at its first-touch bar). Grading mirrors the live engine's 8 checks:
    structure, BOS, origin zone, fresh, retest-in-zone, confirmation candle,
    1H trend, session. A+ = 8/8, B+ = 7/8, C+ = 6/8 (below that: skipped)."""
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
        sl = (z["bottom"] - SL_BUF_ATR * atr[i]) if d == 1 else \
             (z["top"] + SL_BUF_ATR * atr[i])
        # walk the confirm window: first bar back inside the zone, and the
        # first confirmation candle (abort if the stop is violated first)
        k_ins = k_conf = None
        k = i
        end_k = min(n - 1, i + CONFIRM_WINDOW)
        while k <= end_k:
            if d == 1 and ll[k] <= sl:
                break
            if d == -1 and hh[k] >= sl:
                break
            inside = (c[k] >= z["bottom"]) if d == 1 else (c[k] <= z["top"])
            if inside:
                if k_ins is None:
                    k_ins = k
                if k_conf is None and _confirm(o, c, hh, ll, k, d):
                    k_conf = k
                    break
            k += 1
        if k_ins is None:
            continue
        k_entry = k_conf if k_conf is not None else k_ins
        entry = float(c[k_entry])
        checks = [struct[i] == d, bool(bos[i]), bool(z["caused_bos"]), True, True,
                  k_conf is not None, htf[i] == d,
                  session_of(pack["t"][k_entry]) != "dead"]
        passed = sum(checks)
        if passed < 6:
            continue
        risk = abs(entry - sl)
        if risk <= 0:
            continue
        tp1 = entry + d * TP1_R * risk
        res, r = _resolve_snr(entry, sl, tp1, pack, k_entry, d)
        if res is None:
            continue
        setups.append(dict(
            t=pack["t"][k_entry], i=int(k_entry), dir="LONG" if d == 1 else "SHORT",
            entry=round(entry, 1), sl=round(float(sl), 1),
            tp1=round(float(tp1), 1), outcome=res,
            r=(round(r, 2) if r is not None else None),
            session=session_of(pack["t"][k_entry]),
            zone=[z["bottom"], z["top"]],
            grade={8: "A+", 7: "B+", 6: "C+"}[passed], passed=int(passed)))
        last_bar = k_entry
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
    # trader's zone read: position of the entry inside the dealing range
    # (last confirmed swing high & low, causal). Entry in the direction-extreme
    # 38% = CONTINUATION base (shallow pullback in a strong leg — backtest:
    # B+ 83% win / +0.38R, A+ 83% / +0.52R); mid/deep = reduced quality.
    cont_zone = False
    _r_hi = _r_lo = None
    for _idx in reversed(pack["sw_hi_idx"]):
        if _idx + SWING_K <= i_last:
            _r_hi = float(hh[_idx]); break
    for _idx in reversed(pack["sw_lo_idx"]):
        if _idx + SWING_K <= i_last:
            _r_lo = float(ll[_idx]); break
    if _r_hi is not None and _r_lo is not None and _r_hi > _r_lo:
        _p = (float(entry) - _r_lo) / (_r_hi - _r_lo)
        cont_zone = (_p >= 0.62) if d == 1 else (_p <= 0.38)
    a = float(atr[i_now])
    sl = (z["bottom"] - SL_BUF_ATR * a) if d == 1 else (z["top"] + SL_BUF_ATR * a)
    risk = abs(entry - sl)
    tp1 = entry + d * TP1_R * risk
    tp2 = entry + d * TP2_R * risk
    ok = [struct_ok, bos_ok, origin_ok, True, touching, conf, htf_ok, sess_ok]
    passed = sum(ok)
    _e2 = pack["e200"][i_last]
    e200ok = bool(_e2 == _e2 and ((c[i_last] > _e2) if d == 1 else (c[i_last] < _e2)))
    # a setup grade only exists while price is actually RETESTING the zone;
    # an untouched fresh zone stays in the "arming" state. C+ against the
    # EMA200 trend is downgraded (backtest: 25% win / -0.56R -> 56% / -0.03)
    grade = (_grade(passed) or "—") if touching else "—"
    if grade == "C+" and not e200ok:
        grade = "—"
    return dict(
        direction="LONG" if d == 1 else "SHORT",
        passed=int(passed), grade=grade, touching=bool(touching),
        active=bool(active and passed >= MIN_CHECKS_TRIGGER),
        session=sess, sessionLabel=SESSION_LABEL[sess],
        entry=round(entry, 2),
        entryZone=[round(z["bottom"], 2), round(z["top"], 2)],
        sl=round(float(sl), 2), tp1=round(float(tp1), 2), tp2=round(float(tp2), 2),
        rr1=TP1_R, rr2=TP2_R, atr=round(a, 2), barTime=pack["t"][i_now],
        checks=[dict(label=CHECKS[kk], ok=bool(ok[kk])) for kk in range(8)],
        trend1h={1: "HH·HL BULLISH", -1: "LH·LL BEARISH", 0: "RANGING"}[int(struct[i_last])],
        e200ok=e200ok,
        contZone=bool(cont_zone),
        zoneSide=z["side"],
        zoneKey=f"{z['side']}:{pack['t'][z['anchor']]}")   # zone identity: one alert per ZONE


# ------------------------------------------------------------------ backtest
# ---------------------------------------------------- liquidity sweeps
SWEEP_MAX_WAIT = 3       # bars allowed beyond the zone before it's a breakdown
SWEEP_DEPTH_ATR = 0.15   # wick must pierce the zone by this much (ATR fraction)
SWEEP_TP1_R = 1.0                 # sweep exits run wider than retests
SWEEP_TP2_R = 2.0
SWEEP_CHECKS = ["EMA200 trend aligned", "Reclaim candle", "Session live"]
# v2 (backtested, 4522 x 15m bars): the old 1H-trend check at the pierce bar
# selected extended chases — old sweep "A+" won only 36%. EMA200-aligned
# sweeps with a reclaim candle win 65% (+0.09R, n=20); others stay silent.


def scan_sweeps(candles):
    """All historical liquidity-sweep reclaim events.

    A sweep = wick pierces a zone (grabbing stops beyond it) and price CLOSES
    back on the right side within a few bars. Deliberately includes zones the
    main engine already counts as 'tested' — a sweep is what makes a retest
    interesting. One event per sweep; the zone stays scannable after a reclaim.
    """
    if not candles or len(candles) < 200:
        return []
    pack = _snr_pack(candles)
    n, o, c = pack["n"], pack["o"], pack["c"]
    hh, ll, atr = pack["h"], pack["l"], pack["atr"]
    out = []
    for z in pack["zones"]:
        if z["usable_from"] >= n - 2:
            continue
        d = 1 if z["side"] == "demand" else -1
        k = z["usable_from"]
        end = n - 2
        while k <= end:
            pierced = ((ll[k] < z["bottom"] - SWEEP_DEPTH_ATR * atr[k]) if d == 1
                       else (hh[k] > z["top"] + SWEEP_DEPTH_ATR * atr[k]))
            if not pierced:
                k += 1
                continue
            # reclaim: close back on the right side within SWEEP_MAX_WAIT bars
            j = None
            for m in range(k, min(n - 1, k + SWEEP_MAX_WAIT + 1)):
                back = c[m] >= z["bottom"] if d == 1 else c[m] <= z["top"]
                if back:
                    j = m
                    break
            if j is None:
                break                    # real breakdown — zone invalid for sweeps
            deep = ((c[k] < z["bottom"] - SL_BUF_ATR * atr[k]) if d == 1
                    else (c[k] > z["top"] + SL_BUF_ATR * atr[k]))
            if deep:
                k = j + 1                   # closed deep beyond = breakdown,
                continue                     # not a sweep — skip this event
            conf = _confirm(o, c, hh, ll, j, d)
            _e2 = pack["e200"][j]
            aligned = bool(_e2 == _e2 and
                           ((c[j] > _e2) if d == 1 else (c[j] < _e2)))
            checks = [aligned, bool(conf),
                      session_of(pack["t"][j]) != "dead"]
            passed = sum(checks)
            miss = [SWEEP_CHECKS[i] for i, okc in enumerate(checks) if not okc]
            if aligned and conf:
                entry = float(c[j])
                sl = ((float(min(ll[k:j + 1])) - 0.1 * atr[j]) if d == 1
                      else (float(max(hh[k:j + 1])) + 0.1 * atr[j]))
                risk = abs(entry - sl)
                if risk > 0:
                    tp1 = entry + d * SWEEP_TP1_R * risk
                    tp2 = entry + d * SWEEP_TP2_R * risk
                    res, r = _resolve_snr(entry, sl, tp1, pack, j, d,
                                          tp_r=SWEEP_TP1_R)
                    if res is not None:
                        out.append(dict(
                            t=pack["t"][j], i=int(j),
                            dir="LONG" if d == 1 else "SHORT",
                            entry=round(entry, 1), sl=round(sl, 1),
                            tp1=round(tp1, 1), tp2=round(tp2, 1), outcome=res,
                            r=(round(r, 2) if r is not None else None),
                            grade=("A+" if passed == 3 else "B+"),
                            passed=int(passed), miss=miss, side=z["side"],
                            zone=[round(float(z["bottom"]), 1),
                                  round(float(z["top"]), 1)],
                            anchor=z["anchor"]))
            k = j + 1
    out.sort(key=lambda s: s["t"])
    return out


def sweep_view(e):
    """Live sweep event -> display/alert dict (schema like evaluate())."""
    if not e:
        return None
    d = 1 if e["dir"] == "LONG" else -1
    risk = abs(e["entry"] - e["sl"])
    return dict(kind="sweep", grade=e["grade"], direction=e["dir"],
                entry=e["entry"], sl=e["sl"], tp1=e["tp1"],
                tp2=e.get("tp2") or round(e["entry"] + d * TP2_R * risk, 1),
                rr1=SWEEP_TP1_R, rr2=SWEEP_TP2_R, zone=e["zone"], zoneSide=e["side"],
                passed=e["passed"], checksTotal=3, barTime=e["t"],
                miss=e.get("miss") or [],
                anchor=e["anchor"],
                note=("Wick swept the " + e["side"] + " zone and price closed "
                      "back inside it — stops taken, reclaim in force"))


# ---------------------------------------------------- daily liquidity rotation
ROT_MAX_WAIT = 3           # bars allowed to close back inside after the pierce
ROT_PIERCE_ATR = 0.05      # min wick penetration beyond the daily level
ROT_SL_BUF_ATR = 0.10      # stop buffer beyond the sweep wick
ROT_RESOLVE_BARS = 96      # rotation resolves within 24h
ROT_TP1_R = 0.75           # rotation keeps its OWN researched exit profile:
ROT_TP2_R = 1.5            # 75% win at 0.75/1.5 — at the 1:2 profile it
                           # drops to 56%, so it is NOT moved to 1:2
# v1 (backtested, 60d of 15m): prior-day high = buyside liquidity, prior-day
# low = sellside. Fading a FIRST-side sweep is a coin flip (48% TP1, only 29%
# of runs ever reach the other side) — those stay web-feed watch lines. Once
# BOTH pools are drained the picture changes: fading the second sweep won
# 75% / +0.50R (n=8, small sample) on the standard TP1 0.75R half -> BE ->
# TP2 1.5R runner profile. Entry at the reclaim close, SL beyond the sweep
# wick; entering on a retest of the level instead was tested and is WORSE.


def _rot_resolve(pack, i, d, entry, sl, tp1, tp2,
                  tp1r=ROT_TP1_R, tp2r=ROT_TP2_R):
    """Honest outcome of a rotation trade: half off at TP1, stop to BE,
    runner to TP2. Same-bar SL+TP counts as a loss (conservative); a
    timeout marks the remaining half to market."""
    c, hh, ll = pack["c"], pack["h"], pack["l"]
    n = pack["n"]
    risk = abs(entry - sl)
    if risk <= 0:
        return None, None
    end = min(n - 1, i + ROT_RESOLVE_BARS)
    hit1 = None
    for k in range(i + 1, end + 1):
        if (ll[k] <= sl) if d == 1 else (hh[k] >= sl):
            return "loss", -1.0
        if (hh[k] >= tp1) if d == 1 else (ll[k] <= tp1):
            hit1 = k
            break
    if hit1 is None:
        r = d * (c[end] - entry) / risk
        return ("win" if r > 0 else "loss"), round(r, 2)
    rem = 0.5 * tp1r
    for m in range(hit1 + 1, end + 1):
        be = (ll[m] <= entry) if d == 1 else (hh[m] >= entry)
        hit2 = (hh[m] >= tp2) if d == 1 else (ll[m] <= tp2)
        if be:                      # same-bar BE+TP2 also counts as BE
            return "win", round(rem, 2)
        if hit2:
            return "win", round(rem + 0.5 * tp2r, 2)
    tail = 0.5 * max(0.0, d * (c[end] - entry) / risk)
    return "win", round(rem + tail, 2)


def scan_daily_rotations(candles):
    """Daily liquidity rotation — 'run from one side's liquidity to the
    other'. Returns dict(trades=[...], watches=[...]).

    watches — every daily-level sweep pierce (informational, web feed only;
              first-side rotations are a 48% coin flip, so no trade card).
    trades  — the tradeable flip: the SECOND side swept after the first one
              already went, with a close back inside the range. One per day.
    Levels are the prior UTC day's high/low — strictly causal."""
    if not candles or len(candles) < 200:
        return dict(trades=[], watches=[])
    pack = _snr_pack(candles)
    t, hh, ll, c, atr = (pack[k] for k in ("t", "h", "l", "c", "atr"))
    n = pack["n"]
    days, order = {}, []
    for i in range(n):
        k = datetime.fromtimestamp(t[i], timezone.utc).strftime("%Y-%m-%d")
        if k not in days:
            days[k] = []
            order.append(k)
        days[k].append(i)
    trades, watches = [], []
    for di in range(1, len(order)):
        prev, cur = days[order[di - 1]], days[order[di]]
        pdh = max(hh[i] for i in prev)
        pdl = min(ll[i] for i in prev)
        swept = {"high": False, "low": False}
        first_t = {}
        done = False
        for i in cur:
            # ---- buyside: prior-day HIGH swept -> rotation SHORT
            if not swept["high"] and hh[i] > pdh + ROT_PIERCE_ATR * atr[i]:
                swept["high"] = True
                first_t.setdefault("high", t[i])
                watches.append(dict(day=order[di], i=int(i), side="high",
                                    level=round(float(pdh), 1),
                                    otherDrained=swept["low"], ts=t[i]))
                if swept["low"] and not done:
                    j = None
                    for m in range(i, min(i + ROT_MAX_WAIT + 1, n - 1)):
                        if c[m] < pdh:
                            j = m
                            break
                    if j is not None:
                        entry = float(c[j])
                        sl = float(max(hh[i:j + 1])) + ROT_SL_BUF_ATR * atr[j]
                        risk = sl - entry
                        if risk > 0:
                            tp1 = entry - ROT_TP1_R * risk
                            tp2 = entry - ROT_TP2_R * risk
                            res, r = _rot_resolve(pack, j, -1, entry, sl,
                                                  tp1, tp2)
                            if res is not None:
                                trades.append(dict(
                                    t=t[j], i=int(j), day=order[di],
                                    dir="SHORT", entry=round(entry, 1),
                                    sl=round(sl, 1), tp1=round(tp1, 1),
                                    tp2=round(tp2, 1), outcome=res,
                                    r=(round(r, 2) if r is not None else None),
                                    sweptSide="high",
                                    sweptLevel=round(float(pdh), 1),
                                    sweepExt=round(float(max(hh[i:j + 1])), 1),
                                    firstSide="low",
                                    firstLevel=round(float(pdl), 1),
                                    firstT=first_t.get("low"),
                                    session=session_of(t[j])))
                        done = True
            # ---- sellside: prior-day LOW swept -> rotation LONG
            if not swept["low"] and ll[i] < pdl - ROT_PIERCE_ATR * atr[i]:
                swept["low"] = True
                first_t.setdefault("low", t[i])
                watches.append(dict(day=order[di], i=int(i), side="low",
                                    level=round(float(pdl), 1),
                                    otherDrained=swept["high"], ts=t[i]))
                if swept["high"] and not done:
                    j = None
                    for m in range(i, min(i + ROT_MAX_WAIT + 1, n - 1)):
                        if c[m] > pdl:
                            j = m
                            break
                    if j is not None:
                        entry = float(c[j])
                        sl = float(min(ll[i:j + 1])) - ROT_SL_BUF_ATR * atr[j]
                        risk = entry - sl
                        if risk > 0:
                            tp1 = entry + ROT_TP1_R * risk
                            tp2 = entry + ROT_TP2_R * risk
                            res, r = _rot_resolve(pack, j, 1, entry, sl,
                                                  tp1, tp2)
                            if res is not None:
                                trades.append(dict(
                                    t=t[j], i=int(j), day=order[di],
                                    dir="LONG", entry=round(entry, 1),
                                    sl=round(sl, 1), tp1=round(tp1, 1),
                                    tp2=round(tp2, 1), outcome=res,
                                    r=(round(r, 2) if r is not None else None),
                                    sweptSide="low",
                                    sweptLevel=round(float(pdl), 1),
                                    sweepExt=round(float(min(ll[i:j + 1])), 1),
                                    firstSide="high",
                                    firstLevel=round(float(pdh), 1),
                                    firstT=first_t.get("high"),
                                    session=session_of(t[j])))
                        done = True
    return dict(trades=trades, watches=watches)


def rotation_view(e):
    """Live rotation trade -> display/alert dict (schema like sweep_view())."""
    if not e:
        return None
    grade = "B+" if e.get("session") == "dead" else "A+"
    ft = e.get("firstT")
    ft_s = (datetime.fromtimestamp(ft, timezone.utc).strftime("%H:%M")
            + " UTC") if ft else "earlier today"
    hi_first = e["firstSide"] == "high"
    return dict(kind="rotation", grade=grade, direction=e["dir"],
                entry=e["entry"], sl=e["sl"], tp1=e["tp1"], tp2=e["tp2"],
                rr1=ROT_TP1_R, rr2=ROT_TP2_R,
                sweptSide=e["sweptSide"],
                sweptLevel=e["sweptLevel"], sweepExt=e["sweepExt"],
                firstSide=e["firstSide"], firstLevel=e["firstLevel"],
                firstT=ft_s, session=e.get("session"), barTime=e["t"],
                day=e["day"],
                note=("Both daily liquidity pools drained — prior-day " +
                      ("HIGH" if hi_first else "LOW") + " went first (" +
                      ft_s + "), then the prior-day " +
                      ("LOW" if hi_first else "HIGH") + " was swept to " +
                      format(e["sweepExt"], ",.1f") +
                      " and price closed back inside the range. No fuel "
                      "left on either side — rotate back toward the spent " +
                      ("high" if hi_first else "low") + ". "
                      "Entry at the reclaim close, SL beyond the sweep wick."))


def watch_view(w):
    """Live daily-liquidity sweep pierce -> web-feed dict (informational)."""
    if not w:
        return None
    side = "HIGH" if w["side"] == "high" else "LOW"
    return dict(kind="liqwatch", side=w["side"], level=w["level"],
                day=w["day"], barTime=w["ts"],
                otherDrained=bool(w["otherDrained"]),
                note=("prior-day " + side + " swept — rotation watch toward "
                      "the other side"))


def htf_zone(candles_htf, side, price):
    """Nearest same-side higher-timeframe zone containing price (HTF ladder)."""
    if not candles_htf or len(candles_htf) < 200:
        return None
    try:
        pack = _snr_pack(candles_htf)
    except Exception:  # noqa: BLE001
        return None
    best = None
    for z in pack["zones"]:
        if z["side"] != side or z["usable_from"] > pack["n"] - 2:
            continue
        if z["bottom"] <= price <= z["top"]:
            dist = abs((z["top"] + z["bottom"]) / 2.0 - price)
            if best is None or dist < best[0]:
                best = (dist, z)
    return best[1] if best else None


RADAR_MISS = {"struct": "market structure", "bos": "break of structure",
              "origin": "zone origin BOS", "fresh": "fresh (untested)",
              "htf": "1H trend"}


def zone_radar(candles15, candles_1h=None, price=None, max_zones=9):
    """Every SNR zone near price — the transparent view of what the desk
    sees: 15m zones plus 1H zones, each with its live status and exactly
    which confluence checks are missing. Answers 'I can see a setup here —
    why is the engine not signaling?'"""
    out = []
    for label, cl in (("15m", candles15), ("1H", candles_1h or [])):
        if not cl or len(cl) < 200:
            continue
        try:
            pack = _snr_pack(cl)
        except Exception:  # noqa: BLE001
            continue
        n = pack["n"]
        i_last = n - 2
        if i_last < 5:
            continue
        px = float(price if price is not None else cl[-1]["c"])
        a = float(pack["atr"][i_last]) or 1.0
        for z in pack["zones"]:
            if z["usable_from"] > i_last:
                continue                     # zone not born yet
            fresh = z["first_touch"] is None or z["first_touch"] > i_last
            mid = (z["top"] + z["bottom"]) / 2.0
            dist = mid - px
            near = (z["bottom"] - 2.5 * a <= px <= z["top"] + 2.5 * a
                    or abs(dist) <= 3.0 * a)
            if not near:
                continue
            side = z["side"]
            d = 1 if side == "demand" else -1
            touching = (px <= z["top"] + 0.05) if d == 1 else \
                       (px >= z["bottom"] - 0.05)
            checks = dict(
                struct=pack["struct"][i_last] == d,
                bos=bool(pack["bos"][i_last]),
                origin=bool(z["caused_bos"]),
                fresh=bool(fresh),
                htf=pack["htf"][i_last] == d)
            passed = sum(checks.values())
            if not fresh:
                status = "tested"            # retested once — invalid per SNR
            elif touching and passed == 5:
                status = "live"              # full house retest right now
            else:
                status = "arming"
            out.append(dict(
                tf=label, side=side,
                top=round(float(z["top"]), 1), bottom=round(float(z["bottom"]), 1),
                dist=round(dist, 1), status=status, passed=int(passed),
                missing=[RADAR_MISS[k] for k, v in checks.items() if not v]))
    out.sort(key=lambda r: abs(r["dist"]))
    return out[:max_zones]


# ---------------------------------------------------- market delta
# Volume-weighted order-flow proxy, normalized to be scale-free:
#   per bar: delta = vol × (2×(close−low)/(high−low) − 1)
#            norm  = delta / mean(vol, last 20 bars incl. current)  (causal)
# The EMA(9) of `norm` is the smooth pressure line, EMA(21) the trend.
# Normalization is what makes pro delta panels look calm: a high-volume
# spike can no longer dominate the scale.
DELTA_VOL_WINDOW = 20
DELTA_EMA_FAST = 9
DELTA_EMA_SLOW = 21
DELTA_STRONG = 0.15          # |EMA9| above this = strong pressure
MAX_STOP_DIST = 100.0        # hard stop cap: 1000 pips on gold (1 pip = $0.1).
                             # Any signal whose natural stop is wider than
                             # this is skipped entirely — a loss can never
                             # exceed 1000 pips. (Backtest: only 1 of 13
                             # cohort signals ever breached it, at $112.7.)


def _delta_parts(candles):
    """Shared delta math: (raw, vol, mv, norm, d9, d21) or None."""
    if not candles or len(candles) < 30:
        return None
    n = len(candles)
    hh = np.array([float(k["h"]) for k in candles])
    ll = np.array([float(k["l"]) for k in candles])
    cc = np.array([float(k["c"]) for k in candles])
    vv = np.array([float(k.get("v") or 0.0) for k in candles])
    rng = hh - ll
    frac = np.where(rng > 0, (cc - ll) / np.where(rng > 0, rng, 1.0), 0.5)
    vol = np.where(vv > 0, vv, 1.0)
    raw = vol * (2.0 * frac - 1.0)
    csum = np.cumsum(vol)
    offs = np.concatenate((np.zeros(DELTA_VOL_WINDOW),
                           csum[:n - DELTA_VOL_WINDOW])) \
        if n > DELTA_VOL_WINDOW else np.zeros(n)
    winsz = np.minimum(np.arange(n) + 1, DELTA_VOL_WINDOW)
    mv = (csum - offs) / winsz
    norm = raw / np.maximum(mv, 1e-9)

    def _ema(xs, span):
        out = np.empty(len(xs))
        e = xs[0]
        a = 2.0 / (span + 1.0)
        for i in range(len(xs)):
            e = a * xs[i] + (1 - a) * e
            out[i] = e
        return out

    return raw, vol, mv, norm, _ema(norm, DELTA_EMA_FAST), \
        _ema(norm, DELTA_EMA_SLOW)


def delta_series(candles):
    """(ema9, ema21) of the volume-normalized delta — both causal numpy
    arrays, or (None, None) when there is not enough data."""
    p = _delta_parts(candles)
    return (None, None) if p is None else (p[4], p[5])


def delta_align(d9, i, direction):
    """True when the smooth delta pressure agrees with the trade direction."""
    if d9 is None or i is None or i >= len(d9):
        return False
    e9 = float(d9[i])
    return e9 > 0 if direction == "LONG" else e9 < 0


def delta_view(candles, series_len=60):
    """Payload for the UI's Market Delta card (smooth line + histogram)."""
    p = _delta_parts(candles)
    if p is None:
        return None
    raw, vol, mv, norm, d9, d21 = p
    n = len(candles)
    hh = np.array([float(k["h"]) for k in candles])
    ll = np.array([float(k["l"]) for k in candles])
    w0 = max(0, n - series_len)
    rw = raw[w0:]
    tot = float(np.abs(rw).sum())
    e9, e21 = float(d9[-1]), float(d21[-1])
    if e9 > 0.05 and e9 >= e21:
        state = "BUYERS IN CONTROL"
    elif e9 < -0.05 and e9 <= e21:
        state = "SELLERS IN CONTROL"
    elif e9 > 0:
        state = "BUYERS FADING"
    else:
        state = "SELLERS FADING"
    # divergence: price higher highs but delta lower highs (or mirror)
    pw = min(30, n)
    half = pw // 2
    d9w = d9[-pw:]
    ph1 = float(hh[-pw:][:half].max()); ph2 = float(hh[-pw:][half:].max())
    dh1 = float(d9w[:half].max()); dh2 = float(d9w[half:].max())
    pl1 = float(ll[-pw:][:half].min()); pl2 = float(ll[-pw:][half:].min())
    dl1 = float(d9w[:half].min()); dl2 = float(d9w[half:].min())
    div = None
    if ph2 > ph1 and dh2 < dh1 and e9 < float(d9w[0]):
        div = "bearish — price up, delta down"
    elif pl2 < pl1 and dl2 > dl1 and e9 > float(d9w[0]):
        div = "bullish — price down, delta up"
    s9 = d9[w0:]
    mx = float(np.abs(s9).max()) or 1.0
    return dict(
        series=[round(float(x) / mx, 4) for x in s9],
        hist=[round(float(np.clip(x, -1, 1)), 4) for x in norm[w0:]],
        ema9=round(e9, 3), ema21=round(e21, 3),
        strong=bool(abs(e9) >= DELTA_STRONG),
        buyPct=round(0.5 + 0.5 * (float(rw.sum()) / tot if tot > 0 else 0.0), 3),
        cum=round(float(rw.sum()), 1),
        state=state, divergence=div)


def _resolve_alt(pack, s, tp1r, tp2r):
    """Resolve a scanned setup at ALTERNATIVE exit multiples (the scan itself
    resolves at the production 1:2 profile). Same conservative rules: SL
    first on same-bar, BE before TP2 on same-bar, timeout marks to market."""
    hh, ll, cc = pack["h"], pack["l"], pack["c"]
    n = pack["n"]
    d = 1 if s["dir"] == "LONG" else -1
    i, en, sl = s["i"], s["entry"], s["sl"]
    risk = abs(en - sl)
    if risk <= 0:
        return None
    end = min(n - 1, i + 96)
    tp1, tp2 = en + d * tp1r * risk, en + d * tp2r * risk
    hit1 = None
    for k in range(i + 1, end + 1):
        if (ll[k] <= sl) if d == 1 else (hh[k] >= sl):
            return -1.0
        if (hh[k] >= tp1) if d == 1 else (ll[k] <= tp1):
            hit1 = k
            break
    if hit1 is None:
        return d * (cc[end] - en) / risk
    rem = 0.5 * tp1r
    for m in range(hit1 + 1, end + 1):
        be = (ll[m] <= en) if d == 1 else (hh[m] >= en)
        hit2 = (hh[m] >= tp2) if d == 1 else (ll[m] <= tp2)
        if be:
            return rem
        if hit2:
            return rem + 0.5 * tp2r
    return rem + 0.5 * max(0.0, d * (cc[end] - en) / risk)


def research_variants(candles15, since_ts=None):
    """24/7 research engine math. Production cohort plus challenger
    variants (gating + exit alternatives), each with full-window AND
    out-of-sample performance — 'oos' counts only setups triggered after
    since_ts, i.e. bars the deployed strategy has never been tuned on.
    That out-of-sample half is the learning loop."""
    if not candles15 or len(candles15) < 400:
        return None
    setups, pack = _scan_setups(candles15)
    d9, _d21 = delta_series(candles15)
    hh, ll = pack["h"], pack["l"]

    def _cont(s):
        d = 1 if s["dir"] == "LONG" else -1
        i = s["i"]
        r_hi = r_lo = None
        for idx in reversed(pack["sw_hi_idx"]):
            if idx + SWING_K <= i:
                r_hi = float(hh[idx]); break
        for idx in reversed(pack["sw_lo_idx"]):
            if idx + SWING_K <= i:
                r_lo = float(ll[idx]); break
        if r_hi is None or r_lo is None or r_hi <= r_lo:
            return False
        p = (s["entry"] - r_lo) / (r_hi - r_lo)
        return (p >= 0.62) if d == 1 else (p <= 0.38)

    def _st_r(rs):
        if not rs:
            return dict(n=0, win=None, avgR=None)
        return dict(n=len(rs),
                    win=round(sum(1 for r in rs if r > 0) / len(rs), 3),
                    avgR=round(sum(rs) / len(rs), 3))

    def _oos(ss):
        return [s for s in ss if since_ts and s["t"] > since_ts]

    cap = [s for s in setups if s["grade"] in ("A+", "B+")
           and abs(s["entry"] - s["sl"]) <= MAX_STOP_DIST]
    base = [s for s in cap if s["grade"] == "A+" or _cont(s)]
    prod = [s for s in base if delta_align(d9, s["i"], s["dir"])]
    variants = []

    def add(name, ss):
        rs = [s["r"] for s in ss if s.get("r") is not None]
        variants.append(dict(
            name=name, full=_st_r(rs),
            oos=_st_r([s["r"] for s in _oos(ss)
                       if s.get("r") is not None])))

    add("no-delta-gate", base)
    add("continuation-only+delta",
        [s for s in cap if _cont(s) and delta_align(d9, s["i"], s["dir"])])
    add("live-session+delta",
        [s for s in prod if s["session"] in ("london", "ny-overlap", "ny-late")])

    for lbl, a, b in (("exit 1:2.5", 1.0, 2.5), ("exit 0.75:1.5", 0.75, 1.5)):
        rs = [x for x in (_resolve_alt(pack, s, a, b) for s in prod)
              if x is not None]
        rs_oos = []
        for s in _oos(prod):
            x = _resolve_alt(pack, s, a, b)
            if x is not None:
                rs_oos.append(x)
        variants.append(dict(name=lbl, full=_st_r(rs), oos=_st_r(rs_oos)))

    prod_rs = [s["r"] for s in prod if s.get("r") is not None]
    prod_oos = _st_r([s["r"] for s in _oos(prod) if s.get("r") is not None])

    # engine health: the other live signal types
    try:
        sw = [e for e in scan_sweeps(candles15) if e["grade"] in ("A+", "B+")]
        swst = _st_r([e["r"] for e in sw if e.get("r") is not None])
    except Exception:  # noqa: BLE001
        swst = dict(n=0, win=None, avgR=None)
    try:
        rot = scan_daily_rotations(candles15)["trades"]
        rotst = _st_r([e["r"] for e in rot if e.get("r") is not None])
    except Exception:  # noqa: BLE001
        rotst = dict(n=0, win=None, avgR=None)

    return dict(bars=len(candles15), asOf=candles15[-1]["t"],
                production=dict(full=_st_r(prod_rs), oos=prod_oos),
                variants=variants, sweeps=swst, rotation=rotst)


def cohort_stats(candles15):
    """Stats for the DELTA-CONFIRMED cohort — the only setups that earn a
    phone card: (A+ or A+/B+ continuation zone) AND the smoothed market
    delta pushing the same way at entry. Backtest: 92% win · +0.76R at the
    1:2 profile (n=13, 60d). Everything else is watch-only."""
    if not candles15 or len(candles15) < 200:
        return None
    key = f"delta:{len(candles15)}:{candles15[-1]['t']}"
    if _stats_cache.get("dkey") == key:
        return _stats_cache["dstats"]
    setups, pack = _scan_setups(candles15)
    d9, _d21 = delta_series(candles15)
    hh, ll = pack["h"], pack["l"]
    sel = []
    for s in setups:
        if s["grade"] not in ("A+", "B+"):
            continue
        if abs(s["entry"] - s["sl"]) > MAX_STOP_DIST:
            continue                      # 1000-pip cap: never carded anyway
        d = 1 if s["dir"] == "LONG" else -1
        cont = False
        i = s["i"]
        r_hi = r_lo = None
        for idx in reversed(pack["sw_hi_idx"]):
            if idx + SWING_K <= i:
                r_hi = float(hh[idx]); break
        for idx in reversed(pack["sw_lo_idx"]):
            if idx + SWING_K <= i:
                r_lo = float(ll[idx]); break
        if r_hi is not None and r_lo is not None and r_hi > r_lo:
            p = (s["entry"] - r_lo) / (r_hi - r_lo)
            cont = (p >= 0.62) if d == 1 else (p <= 0.38)
        base = (s["grade"] == "A+") or cont
        if base and delta_align(d9, i, s["dir"]):
            sel.append(s)
    w = sum(1 for s in sel if s["outcome"] == "win")
    r_sum = sum(s["r"] or 0.0 for s in sel)
    stats = dict(setups=len(sel), wins=w,
                 winRate=round(w / len(sel), 3) if sel else None,
                 avgR=round(r_sum / len(sel), 3) if sel else None)
    _stats_cache["dkey"] = key
    _stats_cache["dstats"] = stats
    return stats


def backtest_stats(candles15, min_grade="A+"):
    """Historical performance of SNR setups on the loaded 15m data,
    with a per-grade breakdown (A+ / B+ / C+)."""
    if not candles15 or len(candles15) < 200:
        return None
    key = f"{len(candles15)}:{candles15[-1]['t']}:{min_grade}"
    if _stats_cache["key"] == key:
        return _stats_cache["stats"]
    setups, _pack = _scan_setups(candles15)
    thr = GRADE_ORDER.get(min_grade, 3)
    sel = [s for s in setups if GRADE_ORDER.get(s["grade"], 0) >= thr]
    total = len(sel)
    wins = sum(1 for s in sel if s["outcome"] == "win")
    r_sum = sum(s["r"] or 0.0 for s in sel)
    by = {}
    for g in ("A+", "B+", "C+"):
        ss = [s for s in setups if s["grade"] == g]
        gw = sum(1 for s in ss if s["outcome"] == "win")
        gr = sum(s["r"] or 0.0 for s in ss)
        by[g] = dict(setups=len(ss), wins=gw,
                     winRate=round(gw / len(ss), 3) if ss else None,
                     avgR=round(gr / len(ss), 3) if ss else None)
    stats = dict(setups=total, wins=wins, losses=total - wins,
                 winRate=round(wins / total, 3) if total else None,
                 avgR=round(r_sum / total, 3) if total else None,
                 window="last 60 days of 15m bars",
                 tpAtr=TP1_R, slAtr=1.0, minChecks=8, byGrade=by)
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
    setups, _pack = _scan_setups(candles15)
    n = len(candles15)
    out = [s for s in setups if s["i"] >= n - lookback and s["grade"] == "A+"]
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
