# 🪙 XAUUSD · AI Signal Desk

A self-hosted web app that tracks gold (XAU/USD), computes an **AI buy/sell signal**
for three timeframes (15m / 1H / 1D), and fires alerts when the stance flips or a
price level is crossed.

## Features

- **Live chart** — gold price, EMA 9/21, and every historical BUY/SELL signal marked on the chart
- **AI signal** — composite score blending a trained logistic-regression model (45%)
  with a classic technical-rule ensemble (55%): EMA trend, SMA-20, MACD, RSI, Bollinger
- **Alert feed** — fires on every signal flip + custom price levels; optional sound
- **Telegram push** — set two env vars and alerts go straight to your phone
- **Honest stats** — model test accuracy and historical signal hit rate shown on the dashboard

## Quick start

```bash
pip install -r requirements.txt
python train.py      # fetch history & train the 3 models (~10s, cached in models/)
python app.py        # → http://localhost:7860
```

The app auto-refreshes every ~30 s via a background thread and re-checks all alerts.
It trains missing models on first run if you skip `train.py`.

## How the signal works

| Component | Weight | Details |
|---|---|---|
| ML model | 45% | L2-regularized, class-balanced logistic regression. 10 features per bar (multi-horizon returns, RSI, MACD histogram, EMA spread, Bollinger z-score, ATR%, realized vol). Trained to predict the direction of the **next 5 bars**. Time-ordered 80/20 split — no look-ahead leakage. |
| Rule engine | 55% | EMA 9/21 trend (golden/death cross), price vs SMA-20, MACD momentum, RSI <30/>70, Bollinger mean-reversion |

Composite score ∈ [−1, +1]: **≥ +0.20 → BUY**, **≤ −0.20 → SELL**, else **HOLD**.
Confidence = 50% + |score| × 45%.

**Realistic expectations:** short-horizon gold direction is close to a coin flip
(test accuracies: ~50% intraday, ~55% daily in backtests). This is an educational
tool — not a trading system with an edge you can monetize.

## Data sources (automatic failover)

1. **Yahoo Finance** `GC=F` — COMEX gold front-month futures, close proxy for spot XAU/USD (15m: 60 days, 1H: 2 years, 1D: 10 years)
2. **OKX** `PAXG-USDT` — tokenized 1-oz allocated LBMA gold (fallback)
3. **gold-api.com** — spot XAU reference shown next to the chart price

All data is disk-cached; if the network drops the app serves the cache and marks it
"cached" on screen.

## Telegram alerts (optional)

1. Create a bot with [@BotFather](https://t.me/BotFather) → get the token
2. Message your bot once, then get your chat id from `https://api.telegram.org/bot<TOKEN>/getUpdates`
3. Run the app with:

```bash
TELEGRAM_BOT_TOKEN=123:abc TELEGRAM_CHAT_ID=456 python app.py
```

## Hosting 24/7

**Docker (any VPS — Hetzner/DigitalOcean/etc.):**

```bash
docker build -t xauusd-ai .
docker run -d -p 7860:7860 --restart unless-stopped --name xauusd-ai xauusd-ai
```

**systemd (bare metal):** `ExecStart=/usr/bin/python3 /opt/xauusd-ai/app.py` with `Restart=always`.

**Free-tier PaaS:** deploy to Render/Railway/Fly.io — needs outbound internet and a
port env override (change `port=7860` or read `PORT`). Wake-up lag on free tiers
makes alerting less timely than a cheap VPS.

## Files

```
app.py               Flask server: signal engine, alerts, REST API
ml.py                indicators, feature engineering, logistic-regression model
data.py              Yahoo / OKX / gold-api fetchers + disk cache
train.py             one-shot model training per timeframe
templates/index.html dashboard (self-contained, no CDN)
models/*.json        trained model weights
state.json           persisted alerts + price levels (survives restarts)
cache/*.json         market-data cache
```

## ⚠️ Disclaimer

Educational software. **Not financial advice.** Gold is volatile and leveraged
trading can lose your entire capital. Model accuracies near 50–55% mean you should
not treat these signals as trade recommendations.
