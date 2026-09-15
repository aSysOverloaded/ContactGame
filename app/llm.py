"""Provider-switchable LLM calls for CONTROL.

LLM_PROVIDER picks the backend:
  openrouter - OpenRouter `:free` models (OpenAI-compatible API, default)
  gemini     - Google AI Studio free tier (same OpenAI-compatible code path)
  anthropic  - Claude via the official SDK (new-account trial credit)
  fake       - canned replies; spends nothing, use it while building UI
"""

import asyncio
import json
import logging
import os
import re
from datetime import date

import httpx

log = logging.getLogger("control.llm")


class LLMUnavailable(Exception):
    """Quota exhausted, rate limited, misconfigured, or network down.

    Callers catch this and fall back (word list / no guess).
    """


# Sensible defaults so a host with their own key doesn't have to pick a model.
DEFAULT_MODELS = {
    "openrouter": "nex-agi/nex-n2.5-mini:free,google/gemma-4-31b-it:free,google/gemma-4-26b-a4b-it:free",
    "gemini": "gemini-3.5-flash",
}

OPENAI_COMPATIBLE = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "model_env": "OPENROUTER_MODEL",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "key_env": "GEMINI_API_KEY",
        "model_env": "GEMINI_MODEL",
    },
}


def provider() -> str:
    return os.getenv("LLM_PROVIDER", "openrouter").strip().lower()


# --- call accounting and caps on the shared key -----------------------------
# A room that brings its own key spends its own free quota and is never capped.
# Rooms on the server's shared key share one small daily budget between them.

def _limit(var: str, default: int) -> int:
    try:
        return int(os.getenv(var, default))
    except ValueError:
        return default


_usage = {"day": date.today().isoformat(), "calls": 0, "failures": 0, "shared": 0, "rooms": {}}


def _roll_day() -> None:
    today = date.today().isoformat()
    if _usage["day"] != today:
        _usage.update(day=today, calls=0, failures=0, shared=0, rooms={})


def check_budget(room: str | None, own_key: bool) -> None:
    """Raises LLMUnavailable when the shared key's budget for today is used up."""
    if own_key:
        return
    _roll_day()
    if _usage["shared"] >= _limit("SHARED_DAILY_LIMIT", 45):
        raise LLMUnavailable("shared daily quota used up - add your own key to keep CONTROL sharp")
    if room and _usage["rooms"].get(room, 0) >= _limit("ROOM_DAILY_LIMIT", 15):
        raise LLMUnavailable("this room used its share of the shared quota - add your own key")


def usage() -> dict:
    _roll_day()
    return {"provider": provider(), "day": _usage["day"], "calls": _usage["calls"],
            "failures": _usage["failures"], "shared": _usage["shared"],
            "sharedLimit": _limit("SHARED_DAILY_LIMIT", 45),
            "roomLimit": _limit("ROOM_DAILY_LIMIT", 15), "rooms": len(_usage["rooms"])}


def _count(ok: bool, room: str | None, own_key: bool) -> None:
    _roll_day()
    _usage["calls"] += 1
    if not ok:
        _usage["failures"] += 1
    if not own_key:
        _usage["shared"] += 1
        if room:
            _usage["rooms"][room] = _usage["rooms"].get(room, 0) + 1


# --- public entry point -----------------------------------------------------

async def complete(system: str, user: str, *, max_tokens: int = 1000, tag: str = "",
                   key: str | None = None, room: str | None = None) -> str:
    """One system+user turn, returns the text reply. Raises LLMUnavailable.

    `key` is a room's own provider key; without it the server's shared key is used,
    subject to the daily caps above.
    """
    name = provider()
    own_key = bool(key)
    # Refused before we spend anything, so it must not count against the budget itself.
    try:
        check_budget(room, own_key)
    except LLMUnavailable as e:
        log.warning("llm %-10s %-8s REFUSED (%s) | room=%s", name, tag, e, room)
        raise
    try:
        if name == "fake":
            text = await _fake(system, user)
        elif name in OPENAI_COMPATIBLE:
            text = await _openai_compatible(name, system, user, max_tokens, key)
        elif name == "anthropic":
            text = await _anthropic(system, user, max_tokens)
        else:
            raise LLMUnavailable(f"unknown LLM_PROVIDER {name!r}")
    except LLMUnavailable as e:
        _count(False, room, own_key)
        log.warning("llm %-10s %-8s FAILED (%s) | room=%s own_key=%s today=%s",
                    name, tag, e, room, own_key, _usage["calls"])
        raise
    _count(True, room, own_key)
    log.info("llm %-10s %-8s ok | room=%s own_key=%s today=%s shared=%s",
             name, tag, room, own_key, _usage["calls"], _usage["shared"])
    return text


def parse_json(text: str):
    """Pull the first JSON object out of a reply (free models often wrap it in prose/fences)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in reply")
    return json.loads(m.group(0))


# --- backends ---------------------------------------------------------------

_http: httpx.AsyncClient | None = None


async def _openai_compatible(name: str, system: str, user: str, max_tokens: int,
                             key: str | None = None) -> str:
    global _http
    cfg = OPENAI_COMPATIBLE[name]
    key = key or os.getenv(cfg["key_env"])
    model = os.getenv(cfg["model_env"], DEFAULT_MODELS.get(name, ""))
    if not key:
        raise LLMUnavailable("no API key - the room host can add their own free key")
    if not model:
        raise LLMUnavailable(f"set {cfg['model_env']}")
    if _http is None:
        _http = httpx.AsyncClient(timeout=20.0)
    # A comma-separated list means "try these in order" (OpenRouter fallback routing);
    # free models are often rate-limited upstream, so backups keep the game going.
    models = [m.strip() for m in model.split(",") if m.strip()]
    body = {
        "model": models[0],
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if name == "openrouter" and len(models) > 1:
        body["models"] = models
    try:
        r = await _http.post(
            f"{cfg['base_url']}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=body,
        )
    except httpx.HTTPError as e:
        raise LLMUnavailable(f"network: {e.__class__.__name__}") from e
    if r.status_code == 429:
        raise LLMUnavailable("rate limited / daily free quota used up")
    if r.status_code >= 400:
        raise LLMUnavailable(f"HTTP {r.status_code}: {r.text[:200]}")
    try:
        text = r.json()["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, ValueError) as e:
        raise LLMUnavailable(f"unexpected response: {r.text[:200]}") from e
    return text.strip()


_claude = None


async def _anthropic(system: str, user: str, max_tokens: int) -> str:
    global _claude
    import anthropic  # imported lazily so the free providers don't need it configured

    if _claude is None:
        _claude = anthropic.AsyncAnthropic(timeout=20.0, max_retries=1)
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")  # what contact.html used
    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    if model.startswith(("claude-opus-5", "claude-fable")):
        # These think by default; keep it short - CONTROL's guess races a 2.5s window.
        kwargs["output_config"] = {"effort": "low"}
    try:
        if model.startswith(("claude-opus-5", "claude-fable")):
            # Server-side refusal fallback, on by default for these models.
            resp = await _claude.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
            )
        else:
            resp = await _claude.messages.create(**kwargs)
    except anthropic.AuthenticationError as e:
        raise LLMUnavailable("invalid ANTHROPIC_API_KEY") from e
    except anthropic.RateLimitError as e:
        raise LLMUnavailable("rate limited") from e
    except anthropic.APIStatusError as e:
        raise LLMUnavailable(f"HTTP {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise LLMUnavailable("network") from e
    if resp.stop_reason == "refusal":
        raise LLMUnavailable("refused")
    return "".join(b.text for b in resp.content if b.type == "text").strip()


# --- fake -------------------------------------------------------------------

async def _fake(system: str, user: str) -> str:
    await asyncio.sleep(0.05)
    if system.startswith("You pick secret words"):
        word = "RIVER" if "4 to 6" in user else "LABYRINTH" if "8 to 12" in user else "LANTERN"
        word = word if word not in user else "PENCIL"
        return json.dumps({"word": word, "flavor": "Fake flavor: a word from the test bench."})
    if system.startswith("You are CONTROL"):
        # Deterministic hooks for tests: a clue containing "fake:xyz" makes CONTROL guess xyz;
        # "fakeplain:xyz" answers in bare text, like free models sometimes do.
        m = re.search(r"fakeplain:(\w+)", user)
        if m:
            return m.group(1)
        m = re.search(r"fake:(\w+)", user)
        return json.dumps({"guess": m.group(1).upper() if m else "STATIC",
                           "reasoning": "Signal's clear. I've heard that one before."})
    return "{}"
