# Contact — with CONTROL, an AI Wordmaster

The party game Contact, played on phones. CONTROL picks a secret word, listens to your
clues, and races to guess the described word before two players can confirm it.

Everyone joins a room with a 4-letter code, so you can play in one room passing a single
phone around, or from different cities.

## Running it on your own machine

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt     # macOS/Linux: .venv/bin/pip
copy .env.example .env                            # then fill in a key (see below)
.venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000>. Phones on the same Wi-Fi use `http://<your-ip>:8000`.
On Windows, set the Wi-Fi network profile to **Private** or phones can't reach your laptop.

Set `LLM_PROVIDER=fake` in `.env` to develop without spending any API quota.

## How CONTROL is powered

CONTROL needs a language model. Supported providers (`LLM_PROVIDER`):

| Value | Notes |
|---|---|
| `openrouter` | Default. Free models; a key from openrouter.ai/keys needs no card. 50 calls/day per key. |
| `gemini` | Google AI Studio free tier, roughly 1,500 calls/day. Keep billing off. |
| `anthropic` | Claude via the official SDK. Paid after the new-account trial credit. |
| `fake` | Canned replies for development. Spends nothing. |

**Each room can bring its own key.** Whoever creates a room can paste their own free key,
and that room then spends its own allowance instead of the server's. Rooms without one
share `SHARED_DAILY_LIMIT` calls per day across the whole site, with `ROOM_DAILY_LIMIT`
per room. When the allowance runs out CONTROL stops guessing and the game continues
without it — secret words then come from the built-in word list.

A room's key is held in memory for that room only. It is never stored, logged, or sent
to other players.

**Calls per round:** 1 to pick the word, plus 1 per clue.

## The local dictionary

When CONTROL fails to guess a clue-giver's word, the result screen offers **Save as local
reference**, pre-filled with that word and the clue. Saves are per group name and are
stored one row per save, keeping the word, the clue, CONTROL's wrong guess, who saved it,
and who was in the room. When the revealed letters match a saved word, its meaning goes
into CONTROL's prompt, so it learns your group's slang, nicknames and in-jokes.

`GET /api/dictionary/<group>` returns a group's dictionary.

## Deploying to Render (free)

1. **Database (Neon, free, no card):** create a project at neon.tech and copy the
   connection string. Without `DATABASE_URL` the app writes to a local SQLite file, which
   Render erases on every restart.
2. **Web service:** in Render, **New > Blueprint**, pick this repo. `render.yaml` sets it up.
3. **Environment variables:** paste `DATABASE_URL`. Optionally add `OPENROUTER_API_KEY` as
   the shared key. Leave `DEBUG_ENDPOINTS` unset so the quota-spending test endpoints stay off.
4. Open the `onrender.com` URL and share it. The first visit after an idle spell takes about
   a minute while the service wakes up.

Games live in memory, so a restart or sleep returns players to the lobby. The dictionary
and everything else in the database are unaffected.

## Tests

```bash
.venv\Scripts\python -m pytest
```

Covers the match rules, CONTROL's fallbacks, quota caps, and a full two-player game over
websockets, including a check that the secret word never reaches players' phones.
