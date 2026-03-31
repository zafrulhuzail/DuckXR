using System;
using System.Text;
using TMPro;
using UnityEngine;
using UnityEngine.Events;

public class SavedSessionsBrowser : MonoBehaviour
{
    [Header("List UI")]
    [SerializeField] private TMP_Text sessionsListText;
    [SerializeField] private TMP_Text selectedSessionText;
    [SerializeField] private TMP_Text selectedNotesText;

    [Header("Selection")]
    [SerializeField] private int selectedIndex = 0;

    [Header("Events")]
    [SerializeField] private UnityEvent onSessionResumed;

    private SavedSessionIndex _index;
    private SavedSessionData _selectedSession;

    private void OnEnable()
    {
        Refresh();
    }

    public void Refresh()
    {
        _index = SavedSessionService.LoadIndex();

        if (_index.sessions.Count == 0)
        {
            SetText(sessionsListText, "No saved sessions yet.");
            SetText(selectedSessionText, "No session selected.");
            SetText(selectedNotesText, string.Empty);
            _selectedSession = null;
            return;
        }

        selectedIndex = Mathf.Clamp(selectedIndex, 0, _index.sessions.Count - 1);
        LoadSelected();
        RenderList();
        RenderSelection();
    }

    public void SelectNext()
    {
        if (_index == null || _index.sessions.Count == 0)
            return;

        selectedIndex = (selectedIndex + 1) % _index.sessions.Count;
        LoadSelected();
        RenderList();
        RenderSelection();
    }

    public void SelectPrevious()
    {
        if (_index == null || _index.sessions.Count == 0)
            return;

        selectedIndex = (selectedIndex - 1 + _index.sessions.Count) % _index.sessions.Count;
        LoadSelected();
        RenderList();
        RenderSelection();
    }

    public void ResumeSelected()
    {
        if (_selectedSession == null)
            return;

        if (SavedSessionService.ResumeSession(_selectedSession.id))
        {
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
        if (sessionsListText == null || _index == null)
            return;

        var sb = new StringBuilder();
        for (int i = 0; i < _index.sessions.Count; i++)
        {
            var session = _index.sessions[i];
            var marker = i == selectedIndex ? "▶" : "•";
            var label = string.IsNullOrWhiteSpace(session.title) ? "Untitled session" : session.title;
            sb.AppendLine($"{marker} {label}");
            sb.AppendLine($"   {session.noteCount} notes · {FormatDate(session.updatedAtUtc)}");
            if (!string.IsNullOrWhiteSpace(session.latestNotePreview))
                sb.AppendLine($"   {session.latestNotePreview}");
            if (i < _index.sessions.Count - 1)
                sb.AppendLine();
        }

        sessionsListText.text = sb.ToString();
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
