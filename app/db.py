"""Storage for each group's local dictionary.

Two backends, chosen by the DATABASE_URL env var:
  set   -> Postgres (Neon free tier) - used in production, survives restarts
  unset -> SQLite file - zero setup for local dev and tests

Every "Save as local reference" is one row: the term, the clue that described it,
what CONTROL wrongly guessed, who saved it, and **who was in the room at the time**
(witnesses). The witness list is what will later let memory follow the players
instead of a typed group name; nothing reads it yet.
"""

import json
import os
import sqlite3
import time
from pathlib import Path

MEANINGS_PER_TERM = 3

SQLITE_PATH = Path(os.getenv("CONTACT_DB", Path(__file__).resolve().parent.parent / "contact.db"))
_pool = None


def database_url() -> str:
    url = os.getenv("DATABASE_URL", "")
    # Neon and Render hand out postgres:// URLs; asyncpg wants postgresql://
    return url.replace("postgres://", "postgresql://", 1) if url.startswith("postgres://") else url


def group_key(name: str) -> str:
    return " ".join(name.lower().split())


# --- schema -----------------------------------------------------------------------

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS local_references (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    group_name TEXT NOT NULL,
    term TEXT NOT NULL,
    meaning TEXT NOT NULL,
    control_guess TEXT,
    saved_by TEXT,
    saved_by_id TEXT,
    witnesses TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_refs_group ON local_references (group_name, term);
"""

PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS local_references (
    id BIGSERIAL PRIMARY KEY,
    group_name TEXT NOT NULL,
    term TEXT NOT NULL,
    meaning TEXT NOT NULL,
    control_guess TEXT,
    saved_by TEXT,
    saved_by_id TEXT,
    witnesses JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at DOUBLE PRECISION NOT NULL);
CREATE INDEX IF NOT EXISTS idx_refs_group ON local_references (group_name, term);
"""


async def connect() -> None:
    """Called once at startup."""
    global _pool
    if database_url():
        import asyncpg
        _pool = await asyncpg.create_pool(database_url(), min_size=1, max_size=4)
        async with _pool.acquire() as conn:
            await conn.execute(PG_SCHEMA)
    else:
        with _sqlite() as conn:
            conn.executescript(SQLITE_SCHEMA)


async def disconnect() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def _sqlite() -> sqlite3.Connection:
    conn = sqlite3.connect(SQLITE_PATH)
    conn.executescript(SQLITE_SCHEMA)
    return conn


# --- reads and writes -------------------------------------------------------------

async def add_reference(group: str, term: str, meaning: str, control_guess: str | None,
                        saved_by: str, saved_by_id: str, witnesses: list[dict]) -> None:
    row = (group_key(group), term, meaning, control_guess, saved_by, saved_by_id,
           json.dumps(witnesses), time.time())
    if _pool is not None:
        async with _pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO local_references (group_name, term, meaning, control_guess, saved_by,"
                " saved_by_id, witnesses, created_at) VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb,$8)", *row)
    else:
        with _sqlite() as conn:
            conn.execute(
                "INSERT INTO local_references (group_name, term, meaning, control_guess, saved_by,"
                " saved_by_id, witnesses, created_at) VALUES (?,?,?,?,?,?,?,?)", row)


async def load_dictionary(group: str) -> list[dict]:
    """[{term, meaning, examples}] - newest term first; meaning joins the latest distinct clues."""
    sql_pg = ("SELECT term, meaning FROM local_references WHERE group_name = $1 "
              "ORDER BY created_at DESC LIMIT 500")
    if _pool is not None:
        async with _pool.acquire() as conn:
            rows = [(r["term"], r["meaning"]) for r in await conn.fetch(sql_pg, group_key(group))]
    else:
        with _sqlite() as conn:
            rows = conn.execute(sql_pg.replace("$1", "?"), (group_key(group),)).fetchall()

    entries: dict[str, dict] = {}
    for term, meaning in rows:
        e = entries.setdefault(term.lower(), {"term": term, "meanings": [], "examples": 0})
        e["examples"] += 1
        if meaning.lower() not in (m.lower() for m in e["meanings"]) and len(e["meanings"]) < MEANINGS_PER_TERM:
            e["meanings"].append(meaning)
    return [{"term": e["term"], "meaning": " / ".join(e["meanings"]), "examples": e["examples"]}
            for e in entries.values()]


async def delete_term(group: str, term: str) -> None:
    if _pool is not None:
        async with _pool.acquire() as conn:
            await conn.execute("DELETE FROM local_references WHERE group_name = $1 AND lower(term) = lower($2)",
                               group_key(group), term)
    else:
        with _sqlite() as conn:
            conn.execute("DELETE FROM local_references WHERE group_name = ? AND lower(term) = lower(?)",
                         (group_key(group), term))
