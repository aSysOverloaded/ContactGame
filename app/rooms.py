"""Rooms: the server is the single source of truth for game state.

Phases (same as contact.html):
  setup -> idle -> clue-entry -> clue-shown -> contact-entry -> countdown -> result -> idle | won

Each Room.a_* method is the server-side twin of an action in the original
contact.html (named in its docstring). After every change the room pushes a
per-player *sanitized* view to each socket: the secret word, the clue-giver's
word and the caller's word never leave the server until the reveal.
"""

import asyncio
import logging
import os
import random
import secrets
import time
from dataclasses import dataclass, field

from fastapi import WebSocket

from . import db, game

log = logging.getLogger("control.rooms")

COUNTDOWN_SECONDS = float(os.getenv("COUNTDOWN_SECONDS", "2.4"))  # 5 steps x 480ms, as before
EMPTY_ROOM_TTL = 15 * 60
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # no I/O - easy to read aloud


class ActionError(Exception):
    pass


@dataclass
class Player:
    id: str
    name: str
    sockets: int = 0


@dataclass
class Room:
    code: str
    group: str
    host_id: str
    dictionary: list[dict]       # this group's local references, see db.load_dictionary
    difficulty: str = "medium"
    phase: str = "setup"
    players: dict[str, Player] = field(default_factory=dict)
    sockets: dict[WebSocket, str] = field(default_factory=dict)

    secret: str = ""
    flavor: str = ""
    revealed: str = ""
    used_words: list[str] = field(default_factory=list)
    round: int = 1
    log: list[dict] = field(default_factory=list)          # newest first, reset each round
    guesses: list[dict] = field(default_factory=list)      # every word revealed this round, in order
    scoreboard: dict[str, int] = field(default_factory=dict)
    loading: bool = False
    loading_label: str = ""

    clue: dict | None = None     # {text, giverWord*, giverName, giverId, callerWord*, callerName, callerId}
    guess_task: asyncio.Task | None = None
    last_result: dict | None = None
    winner: str | None = None
    countdown_ends: float = 0.0
    empty_since: float | None = None

    # --- views ----------------------------------------------------------

    def view(self, pid: str) -> dict:
        c = self.clue or {}
        return {
            "type": "state",
            "you": pid,
            "code": self.code,
            "group": self.group,
            "host": self.host_id,
            "players": [{"id": p.id, "name": p.name, "online": p.sockets > 0}
                        for p in self.players.values()],
            "phase": self.phase,
            "difficulty": self.difficulty,
            "round": self.round,
            "loading": self.loading,
            "loadingLabel": self.loading_label,
            "dictionary": self.dictionary,
            "revealed": self.revealed,
            "guesses": self.guesses,
            "secretWord": self.secret if self.phase == "won" else "",
            "flavor": self.flavor if self.phase == "won" else "",
            # Log entries keep only what the original UI displayed (tag, clue, note).
            "log": [{"type": e["type"], "clue": e["clue"], "note": e["note"]} for e in self.log],
            "scoreboard": self.scoreboard,
            "currentClue": {
                "text": c.get("text", ""), "giverName": c.get("giverName", ""),
                "giverId": c.get("giverId"), "callerName": c.get("callerName", ""),
                "callerId": c.get("callerId"),
            } if self.clue else None,
            "lastResult": self.last_result if self.phase == "result" else None,
            "winner": self.winner if self.phase == "won" else None,
            "countdownMs": max(0, int((self.countdown_ends - time.time()) * 1000)),
        }

    async def broadcast(self) -> None:
        for ws, pid in list(self.sockets.items()):
            try:
                await ws.send_json(self.view(pid))
            except Exception:
                self.sockets.pop(ws, None)

    # --- dispatch -------------------------------------------------------

    async def handle(self, pid: str, msg: dict) -> dict | None:
        """Runs one action and broadcasts. May return a private reply for the sender only."""
        fn = getattr(self, "a_" + str(msg.get("type", "")).replace("-", "_"), None)
        if fn is None:
            raise ActionError(f"unknown action {msg.get('type')!r}")
        if self.loading:
            raise ActionError("Channel is still opening, one moment.")
        reply = await fn(pid, msg)
        await self.broadcast()
        return reply

    def need(self, *phases: str) -> None:
        if self.phase not in phases:
            raise ActionError(f"Not available right now ({self.phase}).")

    def need_host(self, pid: str) -> None:
        if pid != self.host_id:
            raise ActionError("Only the host can do that.")

    def player_name(self, pid: str, typed: str = "") -> str:
        """Typed name wins (pass-the-phone), else the device's player name."""
        typed = str(typed or "").strip()[:24]
        return typed or (self.players[pid].name if pid in self.players else "")

    # --- setup ----------------------------------------------------------

    async def a_set_difficulty(self, pid, msg):
        """setDifficulty"""
        self.need("setup"); self.need_host(pid)
        if msg.get("difficulty") not in game.DIFF_TEXT:
            raise ActionError("Difficulty must be easy, medium or hard.")
        self.difficulty = msg["difficulty"]

    async def a_remove_term(self, pid, msg):
        """removeLocalTerm - deletes every saved example of that term for this group."""
        self.need_host(pid)
        term = str(msg.get("term", ""))
        if not any(t["term"].lower() == term.lower() for t in self.dictionary):
            raise ActionError("No such entry.")
        db.delete_term(self.group, term)
        self.dictionary = db.load_dictionary(self.group)

    async def a_begin_game(self, pid, msg):
        """beginGame"""
        self.need("setup"); self.need_host(pid)
        await self._pick("Establishing channel…")
        self.phase = "idle"

    async def _pick(self, label: str) -> None:
        self.loading, self.loading_label = True, label
        await self.broadcast()
        try:
            word, flavor, source = await game.pick_secret_word(self.difficulty, self.used_words)
        finally:
            self.loading = False
        log.info("room %s round %d word from %s", self.code, self.round, source)
        self.secret, self.flavor = word, flavor
        self.used_words.append(word)
        self.revealed = word[0]

    # --- clue -----------------------------------------------------------

    async def a_open_clue(self, pid, msg):
        """openClueForm"""
        self.need("idle")
        self.clue = {"giverId": pid, "giverName": self.player_name(pid)}
        self.phase = "clue-entry"

    async def a_back_to_idle(self, pid, msg):
        """backToIdle (Cancel on the clue form)"""
        self.need("clue-entry")
        if pid not in (self.clue["giverId"], self.host_id):
            raise ActionError("Someone else is writing a clue.")
        self.clue = None
        self.phase = "idle"

    async def a_submit_clue(self, pid, msg):
        """submitClue - CONTROL starts guessing in the background right away."""
        self.need("clue-entry")
        if pid != self.clue["giverId"]:
            raise ActionError("Someone else is writing a clue.")
        text = str(msg.get("text", "")).strip()[:300]
        word = str(msg.get("word", "")).strip()[:40]
        if not text or not word:
            raise ActionError("Need both a clue and the word you're describing.")
        if not game.normalize_strict(word).startswith(self.revealed.lower()):
            raise ActionError(f"Your word has to start with {self.revealed}.")
        self.clue.update(text=text, giverWord=word, giverName=self.player_name(pid, msg.get("name")))
        self.guess_task = asyncio.create_task(game.wordmaster_guess(self.revealed, text, self.dictionary))
        self.phase = "clue-shown"

    async def a_no_contact(self, pid, msg):
        """noContactCalled ("Nobody got it")"""
        self.need("clue-shown")
        c = self.clue
        self.log.insert(0, {"type": "miss", "clue": c["text"], "note": "No one called it.",
                            "giverWord": c["giverWord"], "callerWord": None})
        self._drop_clue()
        self.phase = "idle"

    def _drop_clue(self) -> None:
        if self.guess_task and not self.guess_task.done():
            self.guess_task.cancel()
        self.clue, self.guess_task = None, None

    # --- contact --------------------------------------------------------

    async def a_open_contact(self, pid, msg):
        """openContactForm"""
        self.need("clue-shown")
        if pid == self.clue["giverId"] and sum(p.sockets > 0 for p in self.players.values()) > 1:
            raise ActionError("You gave this clue, so someone else has to call Contact.")
        self.clue.update(callerId=pid, callerName=self.player_name(pid))
        self.phase = "contact-entry"

    async def a_back_to_clue(self, pid, msg):
        """'Back' on the contact form"""
        self.need("contact-entry")
        if pid not in (self.clue["callerId"], self.host_id):
            raise ActionError("Someone else is calling Contact.")
        self.clue.update(callerId=None, callerName="")
        self.phase = "clue-shown"

    async def a_submit_contact(self, pid, msg):
        """submitContactGuess -> countdown -> resolveReveal"""
        self.need("contact-entry")
        if pid != self.clue["callerId"]:
            raise ActionError("Someone else is calling Contact.")
        word = str(msg.get("word", "")).strip()[:40]
        if not word:
            raise ActionError("Type the word you think it is.")
        self.clue.update(callerWord=word, callerName=self.player_name(pid, msg.get("name")))
        self.phase = "countdown"
        self.countdown_ends = time.time() + COUNTDOWN_SECONDS
        asyncio.create_task(self._resolve_after_countdown())

    async def _resolve_after_countdown(self) -> None:
        await asyncio.sleep(COUNTDOWN_SECONDS)
        if self.phase != "countdown":
            return
        c = self.clue
        wm = await game.await_guess(self.guess_task)
        wm_guess = (wm or {}).get("guess") or None
        outcome = game.outcome_of(c["giverWord"], c["callerWord"], wm_guess)
        giver, caller = c["giverName"], c["callerName"]

        # If either player's word IS the secret, the round ends here - even on a block or a miss.
        secret_hit = [name or "Someone" for name, word in ((giver, c["giverWord"]), (caller, c["callerWord"]))
                      if game.words_match(word, self.secret)]
        secret_hit = list(dict.fromkeys(secret_hit))  # same person could be both on one phone

        if secret_hit:
            note = f'"{self.secret}" came up — that\'s the secret word. Round over.'
        elif outcome == "miss":
            note = f'{caller or "Guess"}: "{c["callerWord"]}" didn\'t match "{c["giverWord"]}".'
        elif outcome == "blocked":
            note = f'CONTROL called "{wm_guess}" first.'
        else:
            self.revealed = self.secret[: len(self.revealed) + 1]
            if caller:
                self.scoreboard[caller] = self.scoreboard.get(caller, 0) + 1
            note = f'"{c["giverWord"]}" confirmed. Next letter released.'
        self.log.insert(0, {"type": outcome, "clue": c["text"], "note": note,
                            "giverWord": c["giverWord"], "callerWord": c["callerWord"]})
        self.guesses.append({"word": c["giverWord"].upper(), "by": giver or "Clue-giver", "kind": "giver"})
        self.guesses.append({"word": c["callerWord"].upper(), "by": caller or "Caller", "kind": "caller"})
        if wm_guess:
            self.guesses.append({"word": wm_guess.upper(), "by": "CONTROL", "kind": "control"})

        self.last_result = {
            "outcome": outcome,
            "clue": c["text"],
            "giverWord": c["giverWord"],
            "callerWord": c["callerWord"],
            "callerName": caller,
            "controlGuess": wm_guess,
            "controlLine": (wm or {}).get("reasoning")
                           or (f'Went with "{wm_guess}."' if wm_guess else "Came up empty on that one."),
            "secretHit": {"word": self.secret, "names": secret_hit} if secret_hit else None,
            # CONTROL missed the clue-giver's word -> players can teach it this reference.
            "canSave": not game.words_match(wm_guess, c["giverWord"]),
            "saved": False,
        }
        self.clue, self.guess_task = None, None
        self.phase = "result"
        await self.broadcast()

    async def a_save_reference(self, pid, msg):
        """'Save as local reference' - after CONTROL failed to read a clue."""
        self.need("result")
        r = self.last_result
        if not r or not r["canSave"]:
            raise ActionError("CONTROL got that one, nothing to teach it.")
        if r["saved"]:
            raise ActionError("Already saved.")
        term = str(msg.get("term") or r["giverWord"]).strip()[:40]
        meaning = str(msg.get("meaning") or r["clue"]).strip()[:300]
        if not game.normalize_strict(term) or not meaning:
            raise ActionError("Need the word and what it means.")
        db.add_reference(self.group, term, meaning, r["controlGuess"], self.player_name(pid))
        self.dictionary = db.load_dictionary(self.group)
        r["saved"] = True
        log.info("room %s saved reference %r = %r", self.code, term, meaning)

    async def a_continue(self, pid, msg):
        """continueAfterResult"""
        self.need("result")
        r, self.last_result = self.last_result, None
        if r and r["secretHit"]:
            self._win(r["secretHit"]["names"])
        elif len(self.revealed) == len(self.secret) and r and r["outcome"] == "contact":
            self._win([r["callerName"]] if r["callerName"] else [])
        else:
            self.phase = "idle"

    def _win(self, names: list[str]) -> None:
        for n in names:
            self.scoreboard[n] = self.scoreboard.get(n, 0) + 2
        self.winner = " & ".join(names) or None
        self.phase = "won"

    # --- whole word / round end -------------------------------------------

    async def a_full_guess(self, pid, msg):
        """submitFullGuess - a miss is only told to the guesser, as on the original screen."""
        self.need("idle")
        word = str(msg.get("word", "")).strip()
        if not word:
            raise ActionError("Type your best shot.")
        if not game.words_match(word, self.secret):
            return {"type": "fullGuessMiss", "message": "Not it. The channel stays open."}
        name = self.player_name(pid, msg.get("name"))
        self._win([name] if name else [])

    async def a_forfeit(self, pid, msg):
        """forfeitRound - host only, so one player can't wipe a round for everyone."""
        self.need("idle"); self.need_host(pid)
        await self._new_round()

    async def a_new_round(self, pid, msg):
        """startNewRound ("Start next transmission")"""
        self.need("won")
        await self._new_round()

    async def _new_round(self) -> None:
        self.round += 1
        self._drop_clue()
        self.last_result, self.winner = None, None
        self.phase = "idle"
        self.log, self.guesses = [], []
        await self._pick("Cutting to a new frequency…")


# --- registry -----------------------------------------------------------------

ROOMS: dict[str, Room] = {}


def _new_code() -> str:
    while True:
        code = "".join(random.choice(CODE_ALPHABET) for _ in range(4))
        if code not in ROOMS:
            return code


def create_room(group: str, host_name: str) -> tuple[Room, str]:
    group = group.strip()[:60] or "friends"
    pid = secrets.token_urlsafe(8)
    room = Room(code=_new_code(), group=group, host_id=pid, dictionary=db.load_dictionary(group))
    room.players[pid] = Player(pid, host_name.strip()[:24] or "Host")
    ROOMS[room.code] = room
    log.info("room %s created for group %r", room.code, group)
    return room, pid


def join_room(code: str, name: str, pid: str | None) -> tuple[Room, str]:
    room = ROOMS.get(code.strip().upper())
    if room is None:
        raise ActionError("No room with that code.")
    if pid and pid in room.players:           # reconnect
        return room, pid
    pid = secrets.token_urlsafe(8)
    room.players[pid] = Player(pid, name.strip()[:24] or f"Player {len(room.players) + 1}")
    return room, pid


def attach(room: Room, ws: WebSocket, pid: str) -> None:
    room.sockets[ws] = pid
    room.players[pid].sockets += 1
    room.empty_since = None


def detach(room: Room, ws: WebSocket) -> None:
    pid = room.sockets.pop(ws, None)
    if pid in room.players:
        room.players[pid].sockets = max(0, room.players[pid].sockets - 1)
    if not room.sockets:
        room.empty_since = time.time()


def sweep_empty_rooms() -> None:
    now = time.time()
    for code, room in list(ROOMS.items()):
        if room.empty_since and now - room.empty_since > EMPTY_ROOM_TTL:
            del ROOMS[code]
            log.info("room %s expired", code)
