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


# --- call accounting (the feasibility number: calls per game) ---------------

_usage = {"day": date.today().isoformat(), "calls": 0, "failures": 0}


def _count(ok: bool) -> None:
    today = date.today().isoformat()
    if _usage["day"] != today:
        _usage.update(day=today, calls=0, failures=0)
    _usage["calls"] += 1
    if not ok:
        _usage["failures"] += 1


def usage() -> dict:
    return {"provider": provider(), **_usage}


# --- public entry point -----------------------------------------------------

async def complete(system: str, user: str, *, max_tokens: int = 1000, tag: str = "") -> str:
    """One system+user turn, returns the text reply. Raises LLMUnavailable."""
    name = provider()
    try:
        if name == "fake":
            text = await _fake(system, user)
        elif name in OPENAI_COMPATIBLE:
            text = await _openai_compatible(name, system, user, max_tokens)
        elif name == "anthropic":
            text = await _anthropic(system, user, max_tokens)
        else:
            raise LLMUnavailable(f"unknown LLM_PROVIDER {name!r}")
    except LLMUnavailable as e:
        _count(False)
        log.warning("llm %-10s %-8s FAILED (%s) | today=%s", name, tag, e, _usage["calls"])
        raise
    _count(True)
    log.info("llm %-10s %-8s ok | today=%s", name, tag, _usage["calls"])
    return text


def parse_json(text: str):
    """Pull the first JSON object out of a reply (free models often wrap it in prose/fences)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in reply")
    return json.loads(m.group(0))


# --- backends ---------------------------------------------------------------

_http: httpx.AsyncClient | None = None


async def _openai_compatible(name: str, system: str, user: str, max_tokens: int) -> str:
    global _http
    cfg = OPENAI_COMPATIBLE[name]
    key = os.getenv(cfg["key_env"])
    model = os.getenv(cfg["model_env"])
    if not key or not model:
        raise LLMUnavailable(f"set {cfg['key_env']} and {cfg['model_env']}")
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
