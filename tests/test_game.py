import asyncio
import json
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from app import game
from app.main import app


# --- pure rules (same behaviour as contact.html) ------------------------------------

@pytest.mark.parametrize("a,b", [
    ("Lemon", "lemons"), ("box", "boxes"), ("LADDER", "ladder "), ("church", "churches"),
    ("ice-cream", "icecream"),
])
def test_words_match(a, b):
    assert game.words_match(a, b)


@pytest.mark.parametrize("a,b", [("lamp", "lamb"), ("", ""), (None, "x"),
                                 ("berry", "berries"), ("glass", "glasses")])  # original's quirks kept
def test_words_dont_match(a, b):
    assert not game.words_match(a, b)


def test_outcomes():
    assert game.outcome_of("lemon", "lemons", "STATIC") == "contact"
    assert game.outcome_of("lemon", "lemon", "Lemons") == "blocked"
    assert game.outcome_of("lemon", "lime", "lemon") == "miss"
    assert game.outcome_of("lemon", "lemon", None) == "contact"


def test_local_terms_filtered_by_prefix():
    dictionary = [{"term": "lekker", "meaning": "great"},
                  {"term": "braai", "meaning": "barbecue"},
                  {"term": "Lappie", "meaning": "cloth"}]
    assert [t["term"] for t in game.relevant_terms(dictionary, "LE")] == ["lekker"]
    assert [t["term"] for t in game.relevant_terms(dictionary, "L")] == ["lekker", "Lappie"]


def test_saved_reference_reaches_guess_prompt(monkeypatch):
    seen = {}

    async def fake_complete(system, user, **kw):
        seen["user"] = user
        return '{"guess": "GARBAGE", "reasoning": "Local knowledge."}'

    monkeypatch.setattr(game.llm, "complete", fake_complete)
    dictionary = [{"term": "garbage", "meaning": "adopted child"}, {"term": "lekker", "meaning": "great"}]
    reply = asyncio.run(game.wordmaster_guess("GA", "picked up as a baby", dictionary))
    assert '"garbage" = adopted child' in seen["user"] and "lekker" not in seen["user"]
    assert reply["guess"] == "GARBAGE"


def test_plain_text_guess_is_salvaged():
    # Free models sometimes skip the JSON; that shouldn't cost CONTROL its block.
    assert asyncio.run(game.wordmaster_guess("LA", "climb it fakeplain:ladder", []))["guess"] == "LADDER"


def test_dictionary_db_keeps_every_example(tmp_path, monkeypatch):
    from app import db
    monkeypatch.setattr(db, "SQLITE_PATH", tmp_path / "d.db")
    ana = [{"id": "dev-ana", "name": "Ana"}, {"id": "dev-ben", "name": "Ben"}]

    async def main():
        await db.add_reference("Fun Friday", "garbage", "adopted child", "GARDEN", "Ana", "dev-ana", ana)
        await db.add_reference("fun  friday", "Garbage", "picked up as a baby", None, "Ben", "dev-ben", ana)
        await db.add_reference("Other group", "garbage", "trash", None, "Cy", "dev-cy", [])
        d = await db.load_dictionary("FUN FRIDAY")
        assert d == [{"term": "Garbage", "meaning": "picked up as a baby / adopted child", "examples": 2}]
        await db.delete_term("Fun Friday", "GARBAGE")
        assert await db.load_dictionary("Fun Friday") == []
        assert len(await db.load_dictionary("Other group")) == 1

    asyncio.run(main())


def test_shared_quota_caps(monkeypatch):
    from app import llm
    monkeypatch.setenv("SHARED_DAILY_LIMIT", "3")
    monkeypatch.setenv("ROOM_DAILY_LIMIT", "2")
    monkeypatch.setattr(llm, "_usage", {"day": llm.date.today().isoformat(), "calls": 0,
                                        "failures": 0, "shared": 0, "rooms": {}})

    async def main():
        for _ in range(2):
            await llm.complete("sys", "user", room="AAAA")
        with pytest.raises(llm.LLMUnavailable, match="this room used its share"):
            await llm.complete("sys", "user", room="AAAA")
        # A room with its own key is never capped.
        await llm.complete("sys", "user", room="AAAA", key="sk-or-v1-theirownkey")
        await llm.complete("sys", "user", room="BBBB")
        with pytest.raises(llm.LLMUnavailable, match="shared daily quota"):
            await llm.complete("sys", "user", room="CCCC")

    asyncio.run(main())
    assert llm.usage()["shared"] == 3


def test_fallbacks_when_llm_unavailable(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    word, flavor, source = asyncio.run(game.pick_secret_word("hard", []))
    assert source == "wordlist" and 8 <= len(word) <= 12 and word.isupper()
    assert flavor == game.FALLBACK_FLAVOR
    assert asyncio.run(game.wordmaster_guess("L", "sour fruit", [])) is None


# --- full multiplayer game over websockets ----------------------------------------

class Client:
    def __init__(self, ws):
        self.ws, self.raw = ws, []

    def send(self, **msg):
        self.ws.send_json(msg)

    def recv(self):
        text = self.ws.receive_text()
        self.raw.append(text)
        return json.loads(text)

    def until(self, pred, limit=40):
        for _ in range(limit):
            msg = self.recv()
            assert msg["type"] != "error", msg
            if msg["type"] == "state" and pred(msg):
                return msg
        raise AssertionError("condition never met")

    def expect(self, kind, limit=10):
        for _ in range(limit):
            msg = self.recv()
            if msg["type"] == kind:
                return msg
        raise AssertionError(f"expected {kind}")


def clue_round(giver, caller, text, word, guess):
    """One clue: giver writes it, caller calls Contact. Returns the result state."""
    giver.send(type="open_clue")
    giver.until(lambda s: s["phase"] == "clue-entry")
    giver.send(type="submit_clue", text=text, word=word)
    caller.until(lambda s: s["phase"] == "clue-shown" and s["currentClue"]["text"] == text)
    caller.send(type="open_contact")
    caller.until(lambda s: s["phase"] == "contact-entry")
    caller.send(type="submit_contact", word=guess)
    return caller.until(lambda s: s["phase"] == "result" and s["lastResult"]["giverWord"] == word)


def test_multiplayer_game_and_no_leaks():
    # One shared event loop for both sockets, like a real uvicorn server.
    with TestClient(app) as tc, tc.websocket_connect("/ws") as ws_a, tc.websocket_connect("/ws") as ws_b:
        a, b = Client(ws_a), Client(ws_b)
        a.send(type="create", name="Ana", group="Test Crew",
               playerId="device-ana", apiKey="sk-or-v1-anas-own-free-key")
        joined = a.expect("joined")
        code, = (joined["code"],)
        assert joined["playerId"] == "device-ana"     # the browser's lasting id is kept
        b.send(type="join", code=code, name="Ben", playerId="device-ben")
        b.expect("joined")
        s = a.until(lambda s: len(s["players"]) == 2)
        assert s["ownKey"] is True                    # room runs on its own quota
        assert "own-free-key" not in json.dumps(s)    # and the key never reaches players

        b.send(type="begin_game")
        assert b.expect("error")["message"] == "Only the host can do that."
        a.send(type="begin_game")
        s = b.until(lambda s: s["phase"] == "idle" and not s["loading"])
        assert s["revealed"] == "L" and s["secretWord"] == "" and s["dictionary"] == []

        # 1) CONTACT - CONTROL guesses STATIC, Ben matches Ana's word.
        s = clue_round(a, b, "sour yellow fruit", "lemon", "lemons")
        assert s["lastResult"]["outcome"] == "contact" and s["revealed"] == "LA"
        assert s["lastResult"]["controlGuess"] == "STATIC" and s["scoreboard"] == {"Ben": 1}
        assert s["log"][0] == {"type": "contact", "clue": "sour yellow fruit",
                               "note": '"lemon" confirmed. Next letter released.'}
        assert s["lastResult"]["secretHit"] is None
        assert [g["word"] for g in s["guesses"]] == ["LEMON", "LEMONS", "STATIC"]

        # CONTROL missed, so anyone can save it as a local reference (once).
        assert s["lastResult"]["canSave"] and not s["lastResult"]["saved"]
        b.send(type="save_reference", term="lemon", meaning="our word for a dud car")
        s = a.until(lambda s: s["lastResult"] and s["lastResult"]["saved"])
        assert s["dictionary"] == [{"term": "lemon", "meaning": "our word for a dud car", "examples": 1}]
        a.send(type="save_reference")
        assert a.expect("error")["message"] == "Already saved."

        # 2) BLOCKED - the fake provider guesses whatever follows "fake:".
        b.send(type="continue")
        s = clue_round(b, a, "climb it fake:ladder", "ladder", "ladder")
        assert s["lastResult"]["outcome"] == "blocked" and s["revealed"] == "LA"
        assert s["lastResult"]["controlLine"] == "Signal's clear. I've heard that one before."
        assert not s["lastResult"]["canSave"]
        a.send(type="save_reference")
        assert a.expect("error")["message"] == "CONTROL got that one, nothing to teach it."

        # 3) NO CONTACT, plus the prefix check.
        a.send(type="continue")
        a.until(lambda s: s["phase"] == "idle")
        a.send(type="open_clue")
        a.until(lambda s: s["phase"] == "clue-entry")
        a.send(type="submit_clue", text="x", word="moon")
        assert a.expect("error")["message"] == "Your word has to start with LA."
        a.send(type="back_to_idle")
        s = clue_round(a, b, "it gives light", "lamp", "lamb")
        assert s["lastResult"]["outcome"] == "miss"

        # 4) Nobody got it.
        b.send(type="continue")
        b.until(lambda s: s["phase"] == "idle" and s["lastResult"] is None)
        b.send(type="open_clue")
        b.until(lambda s: s["phase"] == "clue-entry")
        b.send(type="submit_clue", text="hot rock", word="lava")
        a.until(lambda s: s["phase"] == "clue-shown" and s["currentClue"]["text"] == "hot rock")
        a.send(type="no_contact")
        s = a.until(lambda s: s["phase"] == "idle" and s["log"] and s["log"][0]["clue"] == "hot rock")
        assert s["log"][0]["note"] == "No one called it."

        # Nothing Ben received so far contains the secret; the log never carries hidden words.
        assert all("lantern" not in r.lower() for r in b.raw), "secret leaked"
        assert all("lava" not in r.lower() for r in a.raw), "unrevealed clue word leaked"

        # 5) Full-word guess: a miss is private to the guesser, a hit ends the round.
        b.send(type="full_guess", word="lamppost")
        assert b.expect("fullGuessMiss")["message"] == "Not it. The channel stays open."
        b.send(type="full_guess", word="Lantern")
        s = a.until(lambda s: s["phase"] == "won")
        assert s["secretWord"] == "LANTERN" and s["winner"] == "Ben" and s["scoreboard"]["Ben"] == 3
        assert s["flavor"].startswith("Fake flavor")

        # 6) Next round: new word, log and guesses cleared, scoreboard and dictionary kept.
        a.send(type="new_round")
        s = b.until(lambda s: s["phase"] == "idle" and not s["loading"] and s["round"] == 2)
        assert s["revealed"] == "P" and s["log"] == [] and s["guesses"] == []
        assert s["scoreboard"]["Ben"] == 3 and s["dictionary"][0]["term"] == "lemon"


@contextmanager
def started_game(group):
    """Two players in a fresh room with round 1 underway."""
    with TestClient(app) as tc, tc.websocket_connect("/ws") as ws_a, tc.websocket_connect("/ws") as ws_b:
        a, b = Client(ws_a), Client(ws_b)
        yield from _start(a, b, group)


def _start(a, b, group):
    a.send(type="create", name="Ana", group=group)
    code = a.expect("joined")["code"]
    b.send(type="join", code=code, name="Ben")
    b.expect("joined")
    a.until(lambda s: len(s["players"]) == 2)
    a.send(type="begin_game")
    b.until(lambda s: s["phase"] == "idle" and not s["loading"])
    yield a, b


def test_secret_word_in_blocked_contact_ends_round_players_win():
    with started_game("Blockers") as (a, b):
        # Ana's word IS the secret; CONTROL blocks it - the round still ends, players win.
        s = clue_round(a, b, "glows fake:lantern", "lantern", "lantern")
        r = s["lastResult"]
        assert r["outcome"] == "blocked" and r["secretHit"] == {"word": "LANTERN", "names": ["Ana", "Ben"]}
        b.send(type="continue")
        s = a.until(lambda s: s["phase"] == "won")
        assert s["secretWord"] == "LANTERN" and s["winner"] == "Ana & Ben"
        assert s["scoreboard"] == {"Ana": 2, "Ben": 2}


def test_callers_word_being_the_secret_ends_round_on_a_miss():
    with started_game("Missers") as (a, b):
        s = clue_round(a, b, "gives light", "lamp", "lantern")
        r = s["lastResult"]
        assert r["outcome"] == "miss" and r["secretHit"]["names"] == ["Ben"]
        b.send(type="continue")
        s = a.until(lambda s: s["phase"] == "won")
        assert s["winner"] == "Ben" and s["scoreboard"] == {"Ben": 2}


def test_reconnect_keeps_player():
    with TestClient(app) as tc:
        with tc.websocket_connect("/ws") as ws:
            ws.send_json({"type": "create", "name": "Ana", "group": "Reconnectors", "playerId": "device-ana"})
            joined = ws.receive_json()
        with tc.websocket_connect("/ws") as ws:
            ws.send_json({"type": "join", "code": joined["code"], "playerId": joined["playerId"]})
            again = ws.receive_json()
            assert again["playerId"] == joined["playerId"]
            state = ws.receive_json()
            assert state["host"] == joined["playerId"] and len(state["players"]) == 1
