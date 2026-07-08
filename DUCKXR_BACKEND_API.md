# DuckXR Backend API (MVP)

This is the minimal, safer integration shape for DuckXR.

## Goal

Keep Quest/Unity as the client UI and voice layer.
Keep OpenClaw behind a backend service.
Do **not** put OpenClaw gateway tokens or privileged credentials in the app.

---

## Recommended flow

1. User speaks in DuckXR
2. DuckXR gets transcript from Whisper / Meta Voice / another voice layer
3. DuckXR sends a request to the DuckXR backend
4. Backend validates the request and user/session
5. Backend calls OpenClaw privately
6. Backend returns a safe response payload to DuckXR
7. DuckXR shows the response in-world and optionally speaks it

---

## Endpoint

`POST /duck/chat`

### Request headers

- `Content-Type: application/json`
- `Authorization: Bearer <duckxr-app-token-or-user-jwt>`

For local development, you can temporarily allow a fixed dev token.

---

## Request body

```json
{
  "sessionId": "optional-existing-session-id",
  "userId": "optional-user-id",
  "duckName": "Ducky",
  "userName": "Zafrul",
  "message": "I keep getting a null reference when I grab the object.",
  "sceneName": "DuckXR_NewFlow",
  "context": {
    "platform": "Quest",
    "appVersion": "0.1.0",
    "notes": [
      "Grab works in editor",
      "Fails only on device"
    ]
  }
}
```

### Required fields

- `message`

### Optional fields

- `sessionId`
- `userId`
- `duckName`
- `userName`
- `sceneName`
- `context`

---

## Response body

```json
{
  "success": true,
  "sessionId": "duckxr-session-123",
  "reply": "This sounds like a missing reference on the Quest build. Check whether the object is assigned in the inspector and whether the prefab differs from the editor version.",
  "shouldSpeak": true,
  "mood": "helpful",
  "hints": [
    "Check serialized references on the prefab",
    "Compare device scene setup with editor scene"
  ],
  "error": null
}
```

### Response fields

- `success` - boolean
- `sessionId` - backend session id for continuity
- `reply` - text for DuckXR to display or speak
- `shouldSpeak` - whether the app should run TTS for this response
- `mood` - optional presentation hint
- `hints` - optional short bullet points for UI chips
- `error` - nullable string

---

## Error response

```json
{
  "success": false,
  "sessionId": "duckxr-session-123",
  "reply": "",
  "shouldSpeak": false,
  "mood": "error",
  "hints": [],
  "error": "Backend could not reach OpenClaw"
}
```

Use normal HTTP status codes too:

- `400` bad request
- `401` unauthorized
- `429` rate limited
- `500` internal error
- `503` OpenClaw unavailable

---

## Backend responsibilities

The backend should:

- authenticate the client or user
- rate limit requests
- sanitize incoming text
- map app sessions to backend/OpenClaw sessions
- hide OpenClaw tokens and private endpoints from the app
- optionally inject project/system prompts
- log enough for debugging, but avoid storing sensitive raw data by default

---

## What the backend should send to OpenClaw

A simple first pass is enough:

- one internal OpenClaw session per DuckXR conversation
- one prompt template for DuckXR assistant behavior
- append app context only when relevant

Example internal message:

```text
You are the AI duck inside DuckXR.
Be concise, practical, and conversational.
The user is speaking from a Quest app.
Scene: DuckXR_NewFlow
Duck name: Ducky
User name: Zafrul

Problem:
I keep getting a null reference when I grab the object.

Context notes:
- Grab works in editor
- Fails only on device
```

---

## Security notes

Avoid these in the client app:

- OpenClaw gateway token
- direct OpenClaw public endpoint
- file-writing privileges
- shell-like actions
- broad tool access

For MVP, keep OpenClaw in **advice mode**, not **action mode**.

---

## MVP scope

Ship only this first:

- one chat endpoint
- one text reply
- one session id
- optional TTS flag

Add later if useful:

- streaming responses
- structured cards
- conversation history fetch
- saved summaries
- multimodal context
- guarded action requests
