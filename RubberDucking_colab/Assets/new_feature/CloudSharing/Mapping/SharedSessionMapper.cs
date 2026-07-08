using System;
using System.Collections.Generic;
using System.Linq;

public static class SharedSessionMapper
{
    public static SharedSessionRecord FromLocalSession(SavedSessionData local, CloudAuthUser owner, SharedSessionVisibility visibility = SharedSessionVisibility.InviteOnly)
    {
        if (local == null)
            return null;

        var shared = new SharedSessionRecord
        {
            id = local.id,
            ownerUserId = owner != null ? owner.userId : string.Empty,
            ownerEmail = owner != null ? owner.email : string.Empty,
            ownerDisplayName = owner != null ? owner.displayName : string.Empty,
            title = local.title,
            userName = local.userName,
            duckName = local.duckName,
            createdAtUtc = local.createdAtUtc,
            updatedAtUtc = local.updatedAtUtc,
            visibility = visibility,
            notes = new List<SharedTranscriptNoteData>()
        };

        if (owner != null && !string.IsNullOrWhiteSpace(owner.userId))
            shared.allowedUserIds.Add(owner.userId);

        if (local.notes != null)
        {
            foreach (var note in local.notes)
            {
                shared.notes.Add(new SharedTranscriptNoteData
                {
                    id = note.id,
                    text = note.text,
                    createdAtUtc = note.createdAtUtc,
                    localPosition = note.localPosition,
                    localRotationEuler = note.localRotationEuler,
                    localScale = note.localScale,
                    siblingIndex = note.siblingIndex
                });
            }
        }

        return shared;
    }

    public static SavedSessionData ToLocalSession(SharedSessionRecord shared)
    {
        if (shared == null)
            return null;

        var local = new SavedSessionData
        {
            id = shared.id,
            title = shared.title,
            userName = shared.userName,
            duckName = shared.duckName,
            createdAtUtc = shared.createdAtUtc,
            updatedAtUtc = shared.updatedAtUtc,
            notes = new List<SavedTranscriptNote>()
        };

        if (shared.notes != null)
        {
            foreach (var note in shared.notes)
            {
                local.notes.Add(new SavedTranscriptNote
                {
                    id = note.id,
                    text = note.text,
                    createdAtUtc = note.createdAtUtc,
                    localPosition = note.localPosition,
                    localRotationEuler = note.localRotationEuler,
                    localScale = note.localScale,
                    siblingIndex = note.siblingIndex
                });
            }
        }

        return local;
    }

    public static SharedSessionSummaryRecord ToSummary(SharedSessionRecord shared)
    {
        if (shared == null)
            return null;

        var latestText = shared.notes != null && shared.notes.Count > 0
            ? shared.notes[shared.notes.Count - 1].text
            : string.Empty;

        return new SharedSessionSummaryRecord
        {
            id = shared.id,
            title = shared.title,
            ownerUserId = shared.ownerUserId,
            ownerDisplayName = shared.ownerDisplayName,
            ownerEmail = shared.ownerEmail,
            updatedAtUtc = shared.updatedAtUtc,
            noteCount = shared.notes != null ? shared.notes.Count : 0,
            latestNotePreview = BuildPreview(latestText),
            visibility = shared.visibility
        };
    }

    private static string BuildPreview(string text)
    {
        text = string.IsNullOrWhiteSpace(text) ? string.Empty : text.Trim();
        if (text.Length <= 80)
            return text;
        return text.Substring(0, 77) + "...";
    }
}
