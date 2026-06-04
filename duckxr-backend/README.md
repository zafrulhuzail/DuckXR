# DuckXR Backend (Local MVP)

Tiny local backend for testing DuckXR -> backend -> reply.

## Start

```powershell
cd C:\Users\Zafrul Huzail\.openclaw\workspace\DuckXR\duckxr-backend
node server.js
```

Or:

```powershell
npm start
```

Server URL:

- Editor testing: `http://localhost:8080/duck/chat`
- Health check: `http://localhost:8080/health`

## Unity setup

In `DuckAIClient` set:

- `backendUrl = http://localhost:8080/duck/chat`

If testing from Quest on the same Wi-Fi, use your PC LAN IP instead:

- `http://YOUR_PC_IP:8080/duck/chat`

Example:

- `http://192.168.1.23:8080/duck/chat`

## Current behavior

This is a mock backend.
It does **not** call OpenClaw yet.
It just returns a structured JSON response so you can validate the end-to-end flow.

## Next step

Once Unity can talk to this backend reliably, replace the mock `buildReply()` logic in `server.js` with real OpenClaw forwarding.
