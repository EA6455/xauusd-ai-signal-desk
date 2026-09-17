"""External AI analysts — OpenAI / Anthropic / Google Gemini / Groq.

Each configured provider gets one seat at the AI Analyst Desk: it reads the
same market snapshot the local models saw and replies with a verdict, a
confidence and a one-line trader reason.

Keys live in the environment (never in the repo):
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, GROQ_API_KEY
Models can be overridden via LLM_MODEL_OPENAI / _ANTHROPIC / _GEMINI / _GROQ.
No key -> that seat simply stays absent; nothing errors, nothing costs.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.request

PROMPT_SYS = (
    "You are a professional XAUUSD (gold) desk analyst. Assess the market "
    "snapshot below and reply with ONLY compact JSON, no markdown fences: "
    '{"verdict":"bullish|bearish|neutral","confidence":0.0-1.0,'
    '"note":"max 40 words, written like a human trader"}'
)

PROVIDERS = {
    "openai": dict(name="GPT-4o mini", icon="🤖", key_env="OPENAI_API_KEY",
                   model_env="LLM_MODEL_OPENAI", model="gpt-4o-mini"),
    "anthropic": dict(name="Claude Haiku", icon="🧭", key_env="ANTHROPIC_API_KEY",
                      model_env="LLM_MODEL_ANTHROPIC", model="claude-3-5-haiku-latest"),
    "gemini": dict(name="Gemini Flash", icon="✨", key_env="GEMINI_API_KEY",
                   model_env="LLM_MODEL_GEMINI", model="gemini-2.5-flash"),
    "groq": dict(name="Llama 70B", icon="🦙", key_env="GROQ_API_KEY",
                 model_env="LLM_MODEL_GROQ", model="llama-3.3-70b-versatile"),
}

REFRESH_S = 900.0      # each seat re-analyzes every 15 minutes (24/7)
BACKOFF_S = 1800.0     # after a failure, that seat waits 30 minutes

_mem = {k: dict(t=0.0, data=None, next_try=0.0) for k in PROVIDERS}
_lock = threading.Lock()


def _post(url, headers, body, timeout=25):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _call_openai(key, model, user_txt, sys_txt=None):
    j = _post("https://api.openai.com/v1/chat/completions",
              {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
              dict(model=model, max_tokens=220, temperature=0.3,
                   messages=[dict(role="system",
                               content=sys_txt or PROMPT_SYS),
                             dict(role="user", content=user_txt)]))
    return j["choices"][0]["message"]["content"]


def _call_groq(key, model, user_txt, sys_txt=None):
    # Groq speaks the OpenAI schema
    j = _post("https://api.groq.com/openai/v1/chat/completions",
              {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
              dict(model=model, max_tokens=220, temperature=0.3,
                   messages=[dict(role="system",
                               content=sys_txt or PROMPT_SYS),
                             dict(role="user", content=user_txt)]))
    return j["choices"][0]["message"]["content"]


def _call_anthropic(key, model, user_txt, sys_txt=None):
    j = _post("https://api.anthropic.com/v1/messages",
              {"x-api-key": key, "anthropic-version": "2023-06-01",
               "Content-Type": "application/json"},
              dict(model=model, max_tokens=220,
                   system=sys_txt or PROMPT_SYS,
                   messages=[dict(role="user", content=user_txt)]))
    return j["content"][0]["text"]


def _call_gemini(key, model, user_txt, sys_txt=None):
    j = _post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        {"Content-Type": "application/json", "x-goog-api-key": key},
        dict(system_instruction=dict(parts=[dict(text=sys_txt or PROMPT_SYS)]),
             contents=[dict(parts=[dict(text=user_txt)])],
             generationConfig=dict(maxOutputTokens=220, temperature=0.3)))
    return j["candidates"][0]["content"]["parts"][0]["text"]


_CALLS = {"openai": _call_openai, "anthropic": _call_anthropic,
          "gemini": _call_gemini, "groq": _call_groq}


def _parse(txt):
    if not txt:
        return None
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        j = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None
    v = str(j.get("verdict", "")).lower()
    if v not in ("bullish", "bearish", "neutral"):
        v = ("bullish" if "bull" in v else
             "bearish" if "bear" in v else "neutral")
    try:
        conf = float(j.get("confidence", 0.5))
    except Exception:  # noqa: BLE001
        conf = 0.5
    note = str(j.get("note") or "").strip()[:240]
    return dict(verdict=v, conf=round(max(0.05, min(0.95, conf)), 2), note=note)


def configured():
    """Provider keys present in the environment, canonical order."""
    return [k for k in ("openai", "anthropic", "gemini", "groq")
            if os.environ.get(PROVIDERS[k]["key_env"])]


def due():
    """True if any configured seat is due for a fresh analysis."""
    now = time.time()
    with _lock:
        return any(now - _mem[k]["t"] >= REFRESH_S and now >= _mem[k]["next_try"]
                   for k in configured())


def refresh(user_txt):
    """Analyze with every configured provider (parallel, one flight each)."""
    for k in configured():
        threading.Thread(target=_one, args=(k, user_txt), daemon=True).start()


def _one(k, user_txt):
    p = PROVIDERS[k]
    key = os.environ.get(p["key_env"])
    model = os.environ.get(p["model_env"]) or p["model"]
    now = time.time()
    with _lock:
        m = _mem[k]
        if now - m["t"] < REFRESH_S or now < m["next_try"]:
            return
        m["t"] = now                      # claim the slot — one flight at a time
    try:
        txt = _CALLS[k](key, model, user_txt)
        data = _parse(txt)
        if not data:
            raise ValueError("unparseable reply")
        data.update(model=model, t=int(now))
        with _lock:
            _mem[k]["data"] = data
            _mem[k]["next_try"] = 0.0
    except Exception as e:  # noqa: BLE001
        with _lock:
            _mem[k]["next_try"] = now + BACKOFF_S
        print(f"[llm_desk] {k} failed: {e}", flush=True)


def snapshot():
    """Current external seats for the web payload (instant, cached)."""
    seats = []
    for k in configured():
        with _lock:
            d = _mem[k]["data"]
        if d:
            seats.append(dict(key=k, name=PROVIDERS[k]["name"], icon=PROVIDERS[k]["icon"],
                              model=d["model"], verdict=d["verdict"], conf=d["conf"],
                              note=d["note"], age=int(time.time() - d["t"])))
    return dict(seats=seats, n=len(configured()), asOf=int(time.time()))
