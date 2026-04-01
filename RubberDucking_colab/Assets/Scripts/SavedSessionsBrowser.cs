using System;
using System.Text;
using TMPro;
using UnityEngine;
using UnityEngine.Events;
using System.Collections.Generic;

public class SavedSessionsBrowser : MonoBehaviour
{
    [Header("List UI")]
    [SerializeField] private TMP_Text sessionsListText;
    [SerializeField] private TMP_Text selectedSessionText;
    [SerializeField] private TMP_Text selectedNotesText;
    [SerializeField] private TMP_Text headerText;
    [SerializeField] private TMP_Text helperText;

    [Header("Selection")]
    [SerializeField] private int selectedIndex = 0;
    [SerializeField] private bool wrapSelection = true;

    [Header("Session Creation")]
    [SerializeField] private string newSessionTitle = "New session";
    [SerializeField] private bool clearPreviewWhenStartingNewSession = true;

    [Header("Restore")]
    [SerializeField] private GameObject notePrefab;
    [SerializeField] private Transform notesParent;
    [SerializeField] private string noteTextChildName = "Transcription1";
    [SerializeField] private bool clearExistingNotesOnResume = true;

    [Header("Events")]
    [SerializeField] private UnityEvent onSessionResumed;
    [SerializeField] private UnityEvent onSessionStarted;
    [SerializeField] private UnityEvent onSelectionChanged;

    private SavedSessionIndex _index;
    private SavedSessionData _selectedSession;

    public int SessionCount => _index?.sessions?.Count ?? 0;
    public SavedSessionData SelectedSession => _selectedSession;

    private void OnEnable()
    {
        Refresh();
    }

    public void Refresh()
    {
        _index = SavedSessionService.LoadIndex();
        var hasSessions = _index != null && _index.sessions.Count > 0;

        if (!hasSessions)
        {
            selectedIndex = 0;
            _selectedSession = null;
            SetText(headerText, "Saved Sessions");
            SetText(helperText, "No saved sessions yet. Start a new one to begin.");
            SetText(sessionsListText, "No saved sessions yet.");
            SetText(selectedSessionText, "No session selected.");
            SetText(selectedNotesText, string.Empty);
            return;
        }

        selectedIndex = Mathf.Clamp(selectedIndex, 0, _index.sessions.Count - 1);
        LoadSelected();
        RenderList();
        RenderSelection();
        onSelectionChanged?.Invoke();
    }

    public void StartNewSession()
    {
        SavedSessionService.StartNewSession(newSessionTitle);
        _index = SavedSessionService.LoadIndex();
        selectedIndex = 0;
        LoadSelected();
        RenderList();
        RenderSelection();

        if (clearPreviewWhenStartingNewSession)
            ClearSpawnedNotes();

        Debug.Log("SavedSessionsBrowser: Started a new session.");
        onSessionStarted?.Invoke();
        onSelectionChanged?.Invoke();
    }

    public void SelectNext()
    {
        if (_index == null || _index.sessions.Count == 0)
            return;

        if (wrapSelection)
            selectedIndex = (selectedIndex + 1) % _index.sessions.Count;
        else
            selectedIndex = Mathf.Min(selectedIndex + 1, _index.sessions.Count - 1);

        LoadSelected();
        RenderList();
        RenderSelection();
        onSelectionChanged?.Invoke();
    }

    public void SelectPrevious()
    {
        if (_index == null || _index.sessions.Count == 0)
            return;

        if (wrapSelection)
            selectedIndex = (selectedIndex - 1 + _index.sessions.Count) % _index.sessions.Count;
        else
            selectedIndex = Mathf.Max(selectedIndex - 1, 0);

        LoadSelected();
        RenderList();
        RenderSelection();
        onSelectionChanged?.Invoke();
    }

    public void SelectByIndex(int index)
    {
        if (_index == null || _index.sessions.Count == 0)
            return;

        selectedIndex = Mathf.Clamp(index, 0, _index.sessions.Count - 1);
        LoadSelected();
        RenderList();
        RenderSelection();
        onSelectionChanged?.Invoke();
    }

    public void ResumeSelected()
    {
        if (_selectedSession == null)
            return;

        if (SavedSessionService.ResumeSession(_selectedSession.id))
        {
            RestoreNotes(_selectedSession);
            Debug.Log($"SavedSessionsBrowser: Resumed {_selectedSession.title}");
            onSessionResumed?.Invoke();
        }
    }

    private void LoadSelected()
    {
        if (_index == null || _index.sessions.Count == 0)
        {
            _selectedSession = null;
            return;
        }

        var summary = _index.sessions[selectedIndex];
        _selectedSession = SavedSessionService.LoadSession(summary.id);
    }

    private void RenderList()
    {
        if (_index == null)
            return;

        SetText(headerText, $"Saved Sessions ({_index.sessions.Count})");
        SetText(helperText, BuildHelperText());

        if (sessionsListText == null)
            return;

        var sb = new StringBuilder();
        for (int i = 0; i < _index.sessions.Count; i++)
        {
            var session = _index.sessions[i];
            var marker = i == selectedIndex ? ">" : "-";
            var label = string.IsNullOrWhiteSpace(session.title) ? "Untitled session" : session.title;
            var owner = BuildOwnerLabel(session.userName, session.duckName);

            sb.AppendLine($"{marker} {i + 1}. {label}");
            sb.AppendLine($"   {session.noteCount} notes · {FormatDate(session.updatedAtUtc)}");
            if (!string.IsNullOrWhiteSpace(owner))
                sb.AppendLine($"   {owner}");
            if (!string.IsNullOrWhiteSpace(session.latestNotePreview))
                sb.AppendLine($"   \"{session.latestNotePreview}\"");
            if (i < _index.sessions.Count - 1)
                sb.AppendLine();
        }

        sessionsListText.text = sb.ToString().TrimEnd();
    }

    private void RenderSelection()
    {
        if (_selectedSession == null)
        {
            SetText(selectedSessionText, "No session selected.");
            SetText(selectedNotesText, string.Empty);
            return;
        }

        var title = string.IsNullOrWhiteSpace(_selectedSession.title) ? "Untitled session" : _selectedSession.title;
        var summary = new StringBuilder();
        summary.AppendLine(title);
        summary.AppendLine($"User: {Fallback(_selectedSession.userName)}");
        summary.AppendLine($"Duck: {Fallback(_selectedSession.duckName)}");
        summary.AppendLine($"Created: {FormatDate(_selectedSession.createdAtUtc)}");
        summary.AppendLine($"Updated: {FormatDate(_selectedSession.updatedAtUtc)}");
        summary.AppendLine($"Notes: {_selectedSession.notes.Count}");
        SetText(selectedSessionText, summary.ToString().TrimEnd());

        var notes = new StringBuilder();
        if (_selectedSession.notes.Count == 0)
        {
            notes.Append("No notes saved in this session yet.");
        }
        else
        {
            for (int i = 0; i < _selectedSession.notes.Count; i++)
            {
                var note = _selectedSession.notes[i];
                notes.AppendLine($"[{i + 1}] {FormatDate(note.createdAtUtc)}");
                notes.AppendLine(note.text);
                if (i < _selectedSession.notes.Count - 1)
                    notes.AppendLine().AppendLine();
            }
        }

        SetText(selectedNotesText, notes.ToString().TrimEnd());
    }

    private string BuildHelperText()
    {
        if (_index == null || _index.sessions.Count == 0)
            return "No saved sessions yet. Start a new one to begin.";

        return _selectedSession == null
            ? "Choose a saved session to inspect it."
            : $"Showing {selectedIndex + 1} of {_index.sessions.Count}. Use previous/next to browse, then resume when ready.";
    }

    private static string BuildOwnerLabel(string userName, string duckName)
    {
        var user = string.IsNullOrWhiteSpace(userName) ? null : userName.Trim();
        var duck = string.IsNullOrWhiteSpace(duckName) ? null : duckName.Trim();

        if (!string.IsNullOrEmpty(user) && !string.IsNullOrEmpty(duck))
            return $"{user} + {duck}";
        return user ?? duck ?? string.Empty;
    }

    private void RestoreNotes(SavedSessionData session)
    {
        if (session == null || notePrefab == null || notesParent == null)
            return;

        ClearSpawnedNotes();

        foreach (var note in session.notes)
        {
            var instance = Instantiate(notePrefab, notesParent);
            instance.name = notePrefab.name;

            var noteInstance = instance.GetComponent<SavedNoteInstance>();
            if (noteInstance == null)
                noteInstance = instance.AddComponent<SavedNoteInstance>();

            var tmp = FindNoteText(instance.transform);
            if (tmp != null)
                noteInstance.SetNoteTextTarget(tmp);

            noteInstance.Apply(note);

            if (note.siblingIndex >= 0 && note.siblingIndex < notesParent.childCount)
                instance.transform.SetSiblingIndex(note.siblingIndex);
        }
    }

    private void ClearSpawnedNotes()
    {
        if (!clearExistingNotesOnResume || notesParent == null)
            return;

        var toDestroy = new List<GameObject>();
        for (int i = 0; i < notesParent.childCount; i++)
            toDestroy.Add(notesParent.GetChild(i).gameObject);

        foreach (var go in toDestroy)
            Destroy(go);
    }

    private TMP_Text FindNoteText(Transform root)
    {
        if (root == null)
            return null;

        if (!string.IsNullOrWhiteSpace(noteTextChildName))
        {
            var child = root.Find(noteTextChildName);
            if (child != null)
                return child.GetComponent<TMP_Text>();
        }

        return root.GetComponentInChildren<TMP_Text>(true);
    }

    private static string Fallback(string value)
    {
        return string.IsNullOrWhiteSpace(value) ? "—" : value;
    }

    private static string FormatDate(string isoUtc)
    {
        if (DateTime.TryParse(isoUtc, out var dt))
            return dt.ToLocalTime().ToString("dd MMM yyyy, HH:mm");
        return isoUtc;
    }

    private static void SetText(TMP_Text target, string value)
    {
        if (target != null)
            target.text = value;
    }
}
