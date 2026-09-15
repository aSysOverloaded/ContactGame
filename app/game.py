"""CONTROL's brain, ported from contact.html: secret-word pick, clue guessing and
the match rules. Prompts are verbatim from the original.

Group memory is a dictionary of local references the players save themselves
after CONTROL misses a clue (see db.py) - there is no automatic learning pass.
"""

import asyncio
import logging
import random
import re
from pathlib import Path

from pydantic import BaseModel, ValidationError

from . import llm

log = logging.getLogger("control.game")

GUESS_TIMEOUT = 2.5  # seconds CONTROL gets at reveal time before it's treated as no guess

DIFF_TEXT = {
    "easy": "Pick a short, very common everyday word, 4 to 6 letters.",
    "medium": "Pick a moderately common word, 6 to 8 letters.",
    "hard": "Pick a trickier, less common word, 8 to 12 letters.",
}
FALLBACK_LENGTHS = {"easy": (4, 6), "medium": (6, 8), "hard": (8, 12)}
FALLBACK_FLAVOR = "A plain old reliable word, sent over a backup line."

PICK_SYSTEM = 'You pick secret words for the party game Contact, where guessers reveal a hidden word one letter at a time. Respond ONLY with strict JSON, no markdown fences, no commentary: {"word": "UPPERCASEWORD", "flavor": "one playful sentence about the word, to be revealed only after the round ends"}. The word must be a single common English dictionary word, letters only, no proper nouns, no hyphens or spaces.'

GUESS_SYSTEM = 'You are CONTROL, the sharp, dry-witted Wordmaster in the party game Contact. Guessers give clues describing a word that starts with known letters, and you race to guess it before two Guessers can silently agree on it. Respond ONLY with strict JSON, no markdown fences, no commentary: {"guess": "YOURGUESSWORD", "reasoning": "one short, dry, in-character line as CONTROL, max 20 words"}.'


class PickReply(BaseModel):
    word: str = ""
    flavor: str = ""


class GuessReply(BaseModel):
    guess: str | None = None
    reasoning: str = ""


# --- matching (same rules as wordsMatch in contact.html) ---------------------------

def normalize_strict(w: str) -> str:
    return re.sub(r"[^a-z]", "", (w or "").strip().lower())


def normalize_loose(w: str) -> str:
    n = normalize_strict(w)
    if len(n) > 3 and n.endswith("es"):
        return n[:-2]
    if len(n) > 3 and n.endswith("s"):
        return n[:-1]
    return n


def words_match(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    if normalize_strict(a) == normalize_strict(b):
        return True
    return normalize_loose(a) == normalize_loose(b)


def outcome_of(giver_word: str, caller_word: str, control_guess: str | None) -> str:
    """'contact' | 'blocked' | 'miss' - same order as resolveReveal in contact.html."""
    if not words_match(giver_word, caller_word):
        return "miss"
    if control_guess and words_match(control_guess, giver_word):
        return "blocked"
    return "contact"


# --- secret word ------------------------------------------------------------------

_WORDS: list[str] | None = None


def _fallback(difficulty: str, used: list[str]) -> str:
    global _WORDS
    if _WORDS is None:
        _WORDS = [w.upper() for w in Path(__file__).with_name("words.txt").read_text().split()]
    lo, hi = FALLBACK_LENGTHS.get(difficulty, (6, 8))
    sized = [w for w in _WORDS if lo <= len(w) <= hi]
    pool = [w for w in sized if w not in used] or sized
    return random.choice(pool)


async def pick_secret_word(difficulty: str, used: list[str], **creds) -> tuple[str, str, str]:
    """Returns (WORD, flavor, source). The local dictionary is deliberately never sent here:
    the secret must stay a plain, universal English word."""
    user = (f"{DIFF_TEXT.get(difficulty, DIFF_TEXT['medium'])} Do not reuse any of these already-used words: "
            f"{', '.join(used) or 'none yet'}.")
    try:
        reply = PickReply.model_validate(
            llm.parse_json(await llm.complete(PICK_SYSTEM, user, tag="pick", **creds)))
        w = re.sub(r"[^A-Z]", "", reply.word.upper())
        if 3 <= len(w) <= 15 and w not in used:
            return w, reply.flavor, "control"
        log.info("pick rejected %r, using backup word", reply.word)
    except (llm.LLMUnavailable, ValueError, ValidationError) as e:
        log.info("pick fell back to word list: %s", e)
    return _fallback(difficulty, used), FALLBACK_FLAVOR, "wordlist"


# --- CONTROL's guess --------------------------------------------------------------

def relevant_terms(dictionary: list[dict], prefix: str) -> list[dict]:
    """Only dictionary entries whose spelling fits the revealed prefix (keeps the prompt small)."""
    p = prefix.upper()
    return [t for t in dictionary if t["term"].upper().startswith(p)]


async def wordmaster_guess(prefix: str, clue: str, dictionary: list[dict], **creds) -> dict | None:
    """{'guess', 'reasoning'} or None if CONTROL couldn't get a read (quota, unreadable reply)."""
    terms = relevant_terms(dictionary, prefix)
    glossary = ""
    if terms:
        glossary = (" This group also uses these local/slang terms starting with the same letters — "
                    "if the clue clearly points at one, answer with it spelled exactly as shown here "
                    "instead of a generic English word: "
                    + "; ".join(f'"{t["term"]}" = {t["meaning"]}' for t in terms) + ".")
    user = (f'The word must start with: "{prefix}". The clue just given was: "{clue}".'
            + glossary + " Guess the single word being described.")
    try:
        text = await llm.complete(GUESS_SYSTEM, user, tag="guess", **creds)
    except llm.LLMUnavailable as e:
        log.info("guess unavailable: %s", e)
        return None
    try:
        return GuessReply.model_validate(llm.parse_json(text)).model_dump()
    except (ValueError, ValidationError):
        pass
    # Free models sometimes ignore the JSON instruction; salvage the guess if we can.
    m = re.search(r'guess"?\s*[:=]\s*"?([A-Za-z]+)', text, re.I)
    words = re.findall(r"[A-Za-z]+", text)
    if m:
        guess = m.group(1)
    elif len(words) == 1 or (words and words[0].upper().startswith(prefix.upper()) and len(words) <= 3):
        guess = words[0]
    else:
        log.warning("guess unreadable, raw reply: %r", text[:300])
        return None
    log.info("guess salvaged from non-JSON reply: %r", text[:120])
    return {"guess": guess.upper(), "reasoning": ""}


async def await_guess(task: asyncio.Task | None) -> dict | None:
    """Wait for CONTROL's background guess at reveal time; give up after GUESS_TIMEOUT."""
    if task is None:
        return None
    try:
        return await asyncio.wait_for(asyncio.shield(task), GUESS_TIMEOUT)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        return None
