"""SQLite store for each group's local dictionary - replaces window.storage.

Every "Save as local reference" is kept as its own row (term + the clue that
described it + what CONTROL wrongly guessed), so the table grows into a corpus
of real localized clue -> word examples. CONTROL's prompt gets a compact view:
one entry per term with its most recent meanings.

Prototype identity model: a group is keyed by the name typed when a room is
created. No login; anyone who types the same name shares the dictionary.
"""

import os
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(os.getenv("CONTACT_DB", Path(__file__).resolve().parent.parent / "contact.db"))
MEANINGS_PER_TERM = 3


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS local_references (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        group_name TEXT NOT NULL,
        term TEXT NOT NULL,
        meaning TEXT NOT NULL,
        control_guess TEXT,
        saved_by TEXT,
        created_at REAL NOT NULL)""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_refs_group ON local_references (group_name, term)")
    return conn


def group_key(name: str) -> str:
    return " ".join(name.lower().split())


def add_reference(group: str, term: str, meaning: str, control_guess: str | None, saved_by: str) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO local_references (group_name, term, meaning, control_guess, saved_by, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (group_key(group), term, meaning, control_guess, saved_by, time.time()),
        )


def load_dictionary(group: str) -> list[dict]:
    """[{term, meaning, examples}] - newest term first; meaning joins the latest distinct clues."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT term, meaning FROM local_references WHERE group_name = ? ORDER BY created_at DESC",
            (group_key(group),),
        ).fetchall()
    entries: dict[str, dict] = {}
    for term, meaning in rows:
        e = entries.setdefault(term.lower(), {"term": term, "meanings": [], "examples": 0})
        e["examples"] += 1
        if meaning.lower() not in (m.lower() for m in e["meanings"]) and len(e["meanings"]) < MEANINGS_PER_TERM:
            e["meanings"].append(meaning)
    return [{"term": e["term"], "meaning": " / ".join(e["meanings"]), "examples": e["examples"]}
            for e in entries.values()]


def delete_term(group: str, term: str) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM local_references WHERE group_name = ? AND lower(term) = lower(?)",
                     (group_key(group), term))
