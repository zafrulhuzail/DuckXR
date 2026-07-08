using System;
using System.Collections.Generic;
using UnityEngine;

public enum SharedSessionVisibility
{
    Private = 0,
    InviteOnly = 1,
    AnyoneWithCode = 2
}

[Serializable]
public class SharedUserRef
{
    public string userId;
    public string email;
    public string displayName;
}

[Serializable]
public class SharedTranscriptNoteData
{
    public string id;
    public string text;
    public string createdAtUtc;
    public SavedVector3Data localPosition;
    public SavedVector3Data localRotationEuler;
    public SavedVector3Data localScale;
    public int siblingIndex;
}

[Serializable]
public class SharedSessionRecord
{
    public string id;
    public string ownerUserId;
    public string ownerEmail;
    public string ownerDisplayName;
    public string title;
    public string userName;
    public string duckName;
    public string createdAtUtc;
    public string updatedAtUtc;
    public SharedSessionVisibility visibility = SharedSessionVisibility.Private;
    public List<string> allowedUserIds = new List<string>();
    public List<string> invitedEmails = new List<string>();
    public List<SharedTranscriptNoteData> notes = new List<SharedTranscriptNoteData>();
}

[Serializable]
public class SharedSessionSummaryRecord
{
    public string id;
    public string title;
    public string ownerUserId;
    public string ownerDisplayName;
    public string ownerEmail;
    public string updatedAtUtc;
    public int noteCount;
    public string latestNotePreview;
    public SharedSessionVisibility visibility;
}

[Serializable]
public class CloudAuthUser
{
    public string userId;
    public string email;
    public string displayName;
    public bool isAuthenticated;

    public static CloudAuthUser SignedOut => new CloudAuthUser
    {
        userId = string.Empty,
        email = string.Empty,
        displayName = string.Empty,
        isAuthenticated = false
    };
}
