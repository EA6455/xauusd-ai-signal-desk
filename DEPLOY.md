# 🚀 Deploying the XAUUSD AI Signal Desk

This app runs fine anywhere Python or Docker runs. Pick one of the options below.

**Why deploy at all?** In the dev sandbox the app sleeps when idle. On a real host
it stays reachable — and on Render's free plan it auto-wakes within ~1 minute
of the first visit, no manual nudge needed.

---

## Option 1 · Render.com (recommended, free, ~5 minutes)

1. **Put this folder on GitHub**
   - Create a free account at github.com, then a new **private** repository
     (e.g. `xauusd-desk`).
   - Upload the contents of this folder (Drag-and-drop won't work for folders
     with subfolders via the web UI — easiest is:
     ```bash
     git init && git add -A && git commit -m "xauusd desk"
     git remote add origin https://github.com/YOURNAME/xauusd-desk.git
     git push -u origin main
     ```
     Or use GitHub Desktop.)
2. **Deploy the blueprint**
   - Create a free account at render.com (sign in with GitHub).
   - Dashboard → **New → Blueprint** → select the repo. Render reads
     `render.yaml` and pre-fills everything (Docker, free plan, health check).
   - Click **Apply**. First build takes ~3–5 minutes.
3. **Open your URL** — `https://xauusd-ai-signal-desk.onrender.com` (rename it in
   Settings → Custom Domains if you like).
4. *(Optional)* Telegram alerts: in the Render dashboard add environment
   variables `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.

**Free-plan notes:** the service sleeps after ~15 min without visitors and wakes
on the next request (~30–60 s, first load is slow, then fast). Data caches,
alerts and trade history live on the container's ephemeral disk — they reset if
Render restarts the container (the app rebuilds caches automatically in ~30 s).

---

## Option 2 · Any VPS / home server with Docker

```bash
# copy this folder to the server, then:
docker build -t xauusd-desk .
docker run -d --name xauusd-desk --restart unless-stopped \
  -p 7860:7860 -v $(pwd)/state.json:/app/state.json xauusd-desk
# app: http://YOUR_SERVER_IP:7860
```

Put a reverse proxy with HTTPS in front (Caddy is one line:
`yourdomain.com { reverse_proxy localhost:7860 }`).

Without Docker (plain Python 3.11+):

```bash
pip install -r requirements.txt
python train.py        # trains the 3 models (~10 s, cached in models/)
./run.sh               # dev server with auto-restart on port 7860
```

---

## Option 3 · Keep your current ngrok domain

You already own `https://spinout-handbag-embellish.ngrok-free.dev`.
Once the app runs on Render (Option 1), you can point that same domain at it
instead of the sandbox:

```bash
ngrok http --url=spinout-handbag-embellish.ngrok-free.dev https://xauusd-ai-signal-desk.onrender.com
```

…but the ngrok process must keep running somewhere, so this only makes sense on
a VPS. On Render, the `onrender.com` URL is simpler.

---

## What to check after deploying

| Check | URL |
|---|---|
| Health | `https://YOUR-URL/api/health` → `{"ok":true,...}` |
| Live tick | `https://YOUR-URL/api/tick` → price moves every few seconds |
| Full data | `https://YOUR-URL/api/data?tf=15m` → JSON with candles, entry, tracker |

The dashboard self-reloads when a new version is deployed (version fingerprint).

**Files that must NOT be committed:** `ngrok_token`, `ngrok_domain`,
`serveo_key*`, `state.json` (personal alerts), `cache/`, `bg.lock` — they are
listed in `.gitignore` / `.dockerignore` in this package.
