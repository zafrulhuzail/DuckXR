# OpenClaw Quest Relay

Small Express relay so your Quest or Unity app does **not** need to store the real OpenClaw gateway token.

## Endpoints

- `GET /health`
- `POST /quest/ask`

## Request body

```json
{
  "input": "Tell me a short robot joke.",
  "user": "meta-quest-test"
}
```

## Response shape

```json
{
  "ok": true,
  "text": "...assistant reply...",
  "raw": { "...": "full upstream response" }
}
```

## Setup

1. Copy `.env.example` to `.env`
2. Set `OPENCLAW_TOKEN`
3. Run:
   - `npm install`
   - `npm start`

## Default local config

- Relay: `http://localhost:3001/quest/ask`
- OpenClaw upstream: `http://127.0.0.1:18789/v1/responses`

## Why this is safer

- token stays on the PC/server
- Quest only talks to the relay
- easier to add auth, rate limits, logging, or request validation later
