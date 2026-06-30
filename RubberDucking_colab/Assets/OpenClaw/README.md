# OpenClawBoard

Isolated OpenClaw-specific board summarization prototype for DuckXR.

## Contents

- `Scripts/NoteItem.cs` — reads note text from a TMP component
- `Scripts/BoardSummarizer.cs` — collects notes, sends summary request, updates output panel/text
- `Scripts/OpenClawRelayConnect.cs` — simple relay client for OpenClaw ask/execute endpoints
- `Scripts/SummarizeButton3D.cs` — optional 3D click trigger for testing

## Intended integration

Use this feature inside `Assets/Scenes/DuckXR_NewFlow.unity` or another DuckXR scene without mixing the scripts into the project's general-purpose `Assets/Scripts/` folder.

## Notes

- This is intentionally separate from `DuckAIClient.cs`.
- If DuckXR later standardizes on one backend path, consider adapting `BoardSummarizer` to talk through a shared client abstraction.
