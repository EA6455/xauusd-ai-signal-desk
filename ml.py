"""AI layer: technical indicators, feature engineering and a pure-numpy
L2-regularized, class-balanced logistic-regression direction model.

No sklearn/heavy deps — trains in a couple of seconds on a laptop.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np

FEATURES = ["ret1", "ret3", "ret5", "ret10", "rsi14", "macd_hist",
            "ema_spread", "bb_z", "atr_pct", "vol10"]
HORIZON = 5        # bars ahead used for the training target
MIN_HISTORY = 30   # bars of indicator warm-up before a feature row exists


# ------------------------------------------------------------- indicators
def ema(a, n):
    a = np.asarray(a, float)
    out = np.empty_like(a)
    alpha = 2.0 / (n + 1.0)
    out[0] = a[0]
    for i in range(1, len(a)):
        out[i] = alpha * a[i] + (1.0 - alpha) * out[i - 1]
    return out


def rolling_mean(a, n):
    a = np.asarray(a, float)
    out = np.full(len(a), np.nan)
    if len(a) >= n:
        c = np.cumsum(np.insert(a, 0, 0.0))
        out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def rolling_std(a, n):
    a = np.asarray(a, float)
    out = np.full(len(a), np.nan)
    for i in range(n - 1, len(a)):
        out[i] = a[i - n + 1:i + 1].std()
    return out


def rsi(c, n=14):
    """Wilder-smoothed RSI. Causal: row i uses data up to and including c[i]."""
    c = np.asarray(c, float)
    out = np.full(len(c), 50.0)
    if len(c) < n + 2:
        return out
    d = np.diff(c)
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    au = np.zeros(len(d))
    ad = np.zeros(len(d))
    au[n - 1] = up[:n].mean()
    ad[n - 1] = dn[:n].mean()
    for i in range(n, len(d)):
        au[i] = (au[i - 1] * (n - 1) + up[i]) / n
        ad[i] = (ad[i - 1] * (n - 1) + dn[i]) / n
    rs = au / np.where(ad == 0, 1e-12, ad)
    val = 100.0 - 100.0 / (1.0 + rs)
    out[n:] = val[n - 1:]
    return out


def atr(h, l, c, n=14):
    h = np.asarray(h, float)
    l = np.asarray(l, float)
    c = np.asarray(c, float)
    out = np.zeros(len(c))
    if len(c) < n + 2:
        out[:] = float((h - l).mean()) if len(c) else 0.0
        return out
    prev = np.roll(c, 1)
    prev[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    out[n - 1] = tr[1:n].mean()
    for i in range(n, len(c)):
        out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    out[:n - 1] = out[n - 1]
    return out


def indicator_pack(candles):
    c = np.array([k["c"] for k in candles], float)
    h = np.array([k["h"] for k in candles], float)
    l = np.array([k["l"] for k in candles], float)
    e9, e21 = ema(c, 9), ema(c, 21)
    macd_line = ema(c, 12) - ema(c, 26)
    macd_sig = ema(macd_line, 9)
    return dict(c=c, h=h, l=l,
                e9=e9, e21=e21,
                macd=macd_line, macd_sig=macd_sig, macd_hist=macd_line - macd_sig,
                s20=rolling_mean(c, 20), sd20=rolling_std(c, 20),
                rsi=rsi(c, 14), atr=atr(h, l, c, 14))


# ------------------------------------------------------------- features
def build_features(candles, pack=None):
    """Feature matrix for bars [MIN_HISTORY, n) — strictly causal."""
    p = pack if pack is not None else indicator_pack(candles)
    c = p["c"]
    n = len(c)
    rows, idxs = [], []
    for i in range(MIN_HISTORY, n):
        w = c[i - 10:i + 1]
        rets = np.diff(w) / w[:-1]
        if np.isfinite(p["s20"][i]) and np.isfinite(p["sd20"][i]):
            z = (c[i] - p["s20"][i]) / (p["sd20"][i] + 1e-9)
        else:
            z = 0.0
        rows.append([
            c[i] / c[i - 1] - 1.0,
            c[i] / c[i - 3] - 1.0,
            c[i] / c[i - 5] - 1.0,
            c[i] / c[i - 10] - 1.0,
            (p["rsi"][i] - 50.0) / 50.0,
            p["macd_hist"][i] / c[i] * 100.0,
            (p["e9"][i] - p["e21"][i]) / c[i] * 100.0,
            z,
            p["atr"][i] / c[i] * 100.0,
            float(rets.std()) * 100.0,
        ])
        idxs.append(i)
    return np.asarray(rows, float), np.asarray(idxs, int), p


# ------------------------------------------------------------- model
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class Model:
    def __init__(self, d=None):
        if d is None:
            d = dict(name="Logistic Regression (L2, class-balanced)",
                     features=FEATURES,
                     w=[0.0] * len(FEATURES), b=0.0,
                     mu=[0.0] * len(FEATURES), sd=[1.0] * len(FEATURES),
                     n=0, horizon=HORIZON,
                     train_acc=None, test_acc=None, test_bal_acc=None,
                     trained_at=None)
        self.d = d

    @property
    def available(self):
        return bool(self.d.get("trained_at")) and self.d.get("n", 0) > 0

    def predict_proba(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        if X.size == 0:
            return np.zeros(0)
        w = np.asarray(self.d["w"], float)
        mu = np.asarray(self.d["mu"], float)
        sd = np.asarray(self.d["sd"], float)
        Z = (X - mu) / sd
        return _sigmoid(Z @ w + self.d["b"])

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.d, f)

    @staticmethod
    def load(path):
        with open(path) as f:
            return Model(json.load(f))


def _fit(X, y, l2=1e-3, iters=6000, lr=0.5):
    n, d = X.shape
    mu = X.mean(0)
    sd = X.std(0) + 1e-9
    Z = (X - mu) / sd
    npos = max(int(y.sum()), 1)
    nneg = max(int(n - y.sum()), 1)
    sw = np.where(y == 1, 0.5 * n / npos, 0.5 * n / nneg)
    w = np.zeros(d)
    b = 0.0
    for _ in range(iters):
        p = _sigmoid(Z @ w + b)
        g = (p - y) * sw
        w -= lr * (Z.T @ g / n + l2 * w)
        b -= lr * float(g.mean())
    return w, b, mu, sd


def train_from_candles(candles, test_frac=0.2):
    """Train on the first (1-test_frac) of history, validate on the most
    recent chunk — time-ordered split, no shuffling (no look-ahead leakage)."""
    X, idxs, _ = build_features(candles)
    c = np.array([k["c"] for k in candles], float)
    mask = idxs + HORIZON < len(c)
    X, idxs = X[mask], idxs[mask]
    y = (c[idxs + HORIZON] > c[idxs]).astype(float)
    split = max(1, int(len(y) * (1 - test_frac)))

    w, b, mu, sd = _fit(X[:split], y[:split])
    m = Model(dict(name="Logistic Regression (L2, class-balanced)",
                   features=FEATURES,
                   w=[float(v) for v in w], b=float(b),
                   mu=[float(v) for v in mu], sd=[float(v) for v in sd],
                   n=int(len(y)), horizon=HORIZON,
                   train_acc=None, test_acc=None, test_bal_acc=None,
                   trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds")))

    p_tr = m.predict_proba(X[:split])
    m.d["train_acc"] = float(((p_tr >= 0.5) == (y[:split] == 1)).mean())
    y_te = y[split:]
    if len(y_te):
        acc = (m.predict_proba(X[split:]) >= 0.5) == (y_te == 1)
        m.d["test_acc"] = float(acc.mean())
        pos = y_te == 1
        if pos.any() and (~pos).any():
            m.d["test_bal_acc"] = float((acc[pos].mean() + acc[~pos].mean()) / 2)
        else:
            m.d["test_bal_acc"] = m.d["test_acc"]
    return m
