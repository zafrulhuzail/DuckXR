using System;
using TMPro;
using UnityEngine;

public class SavedSessionsListController : MonoBehaviour
{
    [Header("Optional UI")]
    [SerializeField] private TMP_Text outputText;

    private void OnEnable()
    {
        Refresh();
    }

    public void Refresh()
    {
        var index = SavedSessionService.LoadIndex();
        if (outputText == null)
        {
            Debug.Log($"SavedSessionsListController: {index.sessions.Count} saved sessions loaded.");
            return;
        }

        if (index.sessions.Count == 0)
        {
            outputText.text = "No saved sessions yet.";
            return;
        }

        outputText.text = BuildList(index);
    }

    private string BuildList(SavedSessionIndex index)
    {
        var lines = new System.Text.StringBuilder();
        foreach (var session in index.sessions)
        {
            var label = string.IsNullOrWhiteSpace(session.title) ? "Untitled session" : session.title;
            var when = FormatDate(session.updatedAtUtc);
            lines.AppendLine($"• {label}");
            lines.AppendLine($"  {session.noteCount} notes · {when}");
            if (!string.IsNullOrWhiteSpace(session.latestNotePreview))
                lines.AppendLine($"  {session.latestNotePreview}");
            lines.AppendLine();
        }

        return lines.ToString().TrimEnd();
    }

    private string FormatDate(string isoUtc)
    {
        if (DateTime.TryParse(isoUtc, out var dt))
            return dt.ToLocalTime().ToString("dd MMM yyyy, HH:mm");
        return isoUtc;
    }
}
