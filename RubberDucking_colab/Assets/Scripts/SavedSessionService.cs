using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

[Serializable]
public class SavedTranscriptNote
{
    public string id;
    public string text;
    public string createdAtUtc;
}

[Serializable]
public class SavedSessionData
{
    public string id;
    public string title;
    public string userName;
    public string duckName;
    public string createdAtUtc;
    public string updatedAtUtc;
    public List<SavedTranscriptNote> notes = new List<SavedTranscriptNote>();
}

[Serializable]
public class SavedSessionSummary
{
    public string id;
    public string title;
    public string userName;
    public string duckName;
    public string createdAtUtc;
    public string updatedAtUtc;
    public int noteCount;
    public string latestNotePreview;
}

[Serializable]
public class SavedSessionIndex
{
    public List<SavedSessionSummary> sessions = new List<SavedSessionSummary>();
}

public static class SavedSessionService
{
    private static SavedSessionData _currentSession;
    private static SavedSessionIndex _cachedIndex;

    public static SavedSessionData CurrentSession => _currentSession;

    private static string BaseFolder => Path.Combine(Application.persistentDataPath, "saved-sessions");
    private static string SessionsFolder => Path.Combine(BaseFolder, "sessions");
    private static string IndexPath => Path.Combine(BaseFolder, "index.json");

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void InitializeOnLoad()
    {
        EnsureStorage();
        EnsureCurrentSession();
    }

    public static SavedSessionData EnsureCurrentSession()
    {
        EnsureStorage();

        if (_currentSession != null)
            return _currentSession;

        _currentSession = CreateNewSession();
        SaveCurrentSession();
        return _currentSession;
    }

    public static SavedSessionData StartNewSession(string title = null)
    {
        EnsureStorage();
        _currentSession = CreateNewSession(title);
        SaveCurrentSession();
        return _currentSession;
    }

    public static void SetUserName(string userName)
    {
        var session = EnsureCurrentSession();
        session.userName = Sanitize(userName);
        RefreshTitle(session);
        SaveCurrentSession();
    }

    public static void SetDuckName(string duckName)
    {
        var session = EnsureCurrentSession();
        session.duckName = Sanitize(duckName);
        RefreshTitle(session);
        SaveCurrentSession();
    }

    public static SavedTranscriptNote AddTranscriptNote(string text)
    {
        text = Sanitize(text);
        if (string.IsNullOrWhiteSpace(text))
            return null;

        var session = EnsureCurrentSession();
        var note = new SavedTranscriptNote
        {
            id = Guid.NewGuid().ToString("N"),
            text = text,
            createdAtUtc = DateTime.UtcNow.ToString("o")
        };

        session.notes.Add(note);
        SaveCurrentSession();
        return note;
    }

    public static SavedSessionIndex LoadIndex()
    {
        EnsureStorage();

        if (_cachedIndex != null)
            return _cachedIndex;

        if (!File.Exists(IndexPath))
        {
            _cachedIndex = new SavedSessionIndex();
            return _cachedIndex;
        }

        try
        {
            var json = File.ReadAllText(IndexPath);
            _cachedIndex = JsonUtility.FromJson<SavedSessionIndex>(json) ?? new SavedSessionIndex();
        }
        catch (Exception e)
        {
            Debug.LogWarning($"SavedSessionService: Failed to load index: {e.Message}");
            _cachedIndex = new SavedSessionIndex();
        }

        return _cachedIndex;
    }

    public static SavedSessionData LoadSession(string sessionId)
    {
        EnsureStorage();
        if (string.IsNullOrWhiteSpace(sessionId))
            return null;

        var path = GetSessionPath(sessionId);
        if (!File.Exists(path))
            return null;

        try
        {
            var json = File.ReadAllText(path);
            return JsonUtility.FromJson<SavedSessionData>(json);
        }
        catch (Exception e)
        {
            Debug.LogWarning($"SavedSessionService: Failed to load session '{sessionId}': {e.Message}");
            return null;
        }
    }

    public static void SaveCurrentSession()
    {
        if (_currentSession == null)
            return;

        EnsureStorage();

        _currentSession.updatedAtUtc = DateTime.UtcNow.ToString("o");
        RefreshTitle(_currentSession);

        try
        {
            File.WriteAllText(GetSessionPath(_currentSession.id), JsonUtility.ToJson(_currentSession, true));
            UpsertSummary(_currentSession);
            Debug.Log($"SavedSessionService: Saved session {_currentSession.id} ({_currentSession.notes.Count} notes)");
        }
        catch (Exception e)
        {
            Debug.LogError($"SavedSessionService: Failed to save current session: {e.Message}");
        }
    }

    private static SavedSessionData CreateNewSession(string title = null)
    {
        var now = DateTime.UtcNow.ToString("o");
        var session = new SavedSessionData
        {
            id = Guid.NewGuid().ToString("N"),
            createdAtUtc = now,
            updatedAtUtc = now,
            title = string.IsNullOrWhiteSpace(title) ? "Untitled session" : title.Trim()
        };
        return session;
    }

    private static void UpsertSummary(SavedSessionData session)
    {
        var index = LoadIndex();
        var summary = index.sessions.Find(s => s.id == session.id);
        if (summary == null)
        {
            summary = new SavedSessionSummary();
            index.sessions.Add(summary);
        }

        summary.id = session.id;
        summary.title = session.title;
        summary.userName = session.userName;
        summary.duckName = session.duckName;
        summary.createdAtUtc = session.createdAtUtc;
        summary.updatedAtUtc = session.updatedAtUtc;
        summary.noteCount = session.notes != null ? session.notes.Count : 0;
        summary.latestNotePreview = summary.noteCount > 0 ? BuildPreview(session.notes[summary.noteCount - 1].text) : string.Empty;

        index.sessions.Sort((a, b) => string.CompareOrdinal(b.updatedAtUtc, a.updatedAtUtc));
        _cachedIndex = index;

        File.WriteAllText(IndexPath, JsonUtility.ToJson(index, true));
    }

    private static void RefreshTitle(SavedSessionData session)
    {
        if (session == null)
            return;

        string duck = Sanitize(session.duckName);
        string user = Sanitize(session.userName);

        if (!string.IsNullOrWhiteSpace(duck) && !string.IsNullOrWhiteSpace(user))
            session.title = $"{duck} · {user}";
        else if (!string.IsNullOrWhiteSpace(duck))
            session.title = duck;
        else if (!string.IsNullOrWhiteSpace(user))
            session.title = user;
        else if (string.IsNullOrWhiteSpace(session.title))
            session.title = "Untitled session";
    }

    private static string BuildPreview(string text)
    {
        text = Sanitize(text);
        if (text.Length <= 80)
            return text;
        return text.Substring(0, 77) + "...";
    }

    private static string Sanitize(string value)
    {
        return string.IsNullOrWhiteSpace(value) ? string.Empty : value.Trim();
    }

    private static void EnsureStorage()
    {
        Directory.CreateDirectory(BaseFolder);
        Directory.CreateDirectory(SessionsFolder);
    }

    private static string GetSessionPath(string sessionId)
    {
        return Path.Combine(SessionsFolder, sessionId + ".json");
    }
}
