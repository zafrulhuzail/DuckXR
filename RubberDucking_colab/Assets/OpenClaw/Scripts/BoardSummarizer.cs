using System.Text;
using TMPro;
using UnityEngine;

public enum Panel3DResizeAxis
{
    X,
    Y,
    Z
}

public class BoardSummarizer : MonoBehaviour
{
    [Header("Notes")]
    [SerializeField] private NoteItem[] notes;
    [SerializeField] private Transform notesRoot;
    [SerializeField] private bool autoFindNotesFromChildren = true;

    [Header("OpenClaw")]
    [SerializeField] private OpenClawRelayConnect relay;
    [SerializeField] private string summaryInstruction = "Summarize these board notes. Return plain text only. Do not use markdown, bold, asterisks, headers, or code formatting. Use simple readable lines. Include: 1) key themes, 2) action items, 3) blockers or risks, and 4) a one-sentence takeaway. Keep it concise and easy to scan.";

    [Header("Output")]
    [SerializeField] private TMP_Text outputText;
    [SerializeField] private RectTransform outputPanel;
    [SerializeField] private Transform outputPanel3D;
    [SerializeField] private Panel3DResizeAxis outputPanel3DResizeAxis = Panel3DResizeAxis.Y;
    [SerializeField] private float outputPanelMinHeight = 120f;
    [SerializeField] private float outputPanelPaddingY = 40f;
    [SerializeField] private float outputPanel3DMinHeight = 1.25f;
    [SerializeField] private float outputPanel3DPaddingY = 0.3f;
    [SerializeField] private float outputPanel3DHeightMultiplier = 0.02f;
    [SerializeField] private float outputTextMaxWidth = 700f;
    [SerializeField] private string loadingText = "Summarizing notes...";
    [SerializeField] private string emptyStateText = "No notes found to summarize.";

    private void OnEnable()
    {
        if (relay != null)
        {
            relay.OnRelayTextReceived += HandleRelayTextReceived;
            relay.OnRelayRequestFailed += HandleRelayRequestFailed;
        }
    }

    private void OnDisable()
    {
        if (relay != null)
        {
            relay.OnRelayTextReceived -= HandleRelayTextReceived;
            relay.OnRelayRequestFailed -= HandleRelayRequestFailed;
        }
    }

    [ContextMenu("Summarize Notes")]
    public void SummarizeNotes()
    {
        if (relay == null)
        {
            Debug.LogWarning("BoardSummarizer: relay reference is missing.");
            return;
        }

        var prompt = BuildSummaryPrompt();
        if (string.IsNullOrWhiteSpace(prompt))
        {
            SetOutput(emptyStateText);
            return;
        }

        SetOutput(loadingText);
        relay.SendText(prompt);
    }

    private string BuildSummaryPrompt()
    {
        var activeNotes = GetActiveNotes();
        if (activeNotes.Length == 0)
        {
            return string.Empty;
        }

        var builder = new StringBuilder();
        builder.AppendLine(summaryInstruction);
        builder.AppendLine();
        builder.AppendLine("Notes:");

        var noteCount = 0;
        for (var i = 0; i < activeNotes.Length; i++)
        {
            var text = activeNotes[i] != null ? activeNotes[i].GetText() : string.Empty;
            if (string.IsNullOrWhiteSpace(text))
            {
                continue;
            }

            noteCount++;
            builder.AppendLine($"- Note {noteCount}: {text.Trim()}");
        }

        return noteCount > 0 ? builder.ToString() : string.Empty;
    }

    private NoteItem[] GetActiveNotes()
    {
        if (autoFindNotesFromChildren && notesRoot != null)
        {
            return notesRoot.GetComponentsInChildren<NoteItem>(true);
        }

        return notes ?? System.Array.Empty<NoteItem>();
    }

    private void HandleRelayTextReceived(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            SetOutput("No summary returned.");
            return;
        }

        SetOutput(SanitizeMarkdown(text));
    }

    private void HandleRelayRequestFailed(string error)
    {
        SetOutput("Summary failed: " + error);
    }

    private string SanitizeMarkdown(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return string.Empty;
        }

        var cleaned = text;
        cleaned = cleaned.Replace("**", string.Empty);
        cleaned = cleaned.Replace("__", string.Empty);
        cleaned = cleaned.Replace("```", string.Empty);
        cleaned = cleaned.Replace("`", string.Empty);
        cleaned = cleaned.Replace("#", string.Empty);

        return cleaned.Trim();
    }

    private void SetOutput(string text)
    {
        if (outputText != null)
        {
            outputText.text = text;
            ResizeOutputPanel();
        }
    }

    private void ResizeOutputPanel()
    {
        if (outputText == null)
        {
            return;
        }

        var outputTextRect = outputText.GetComponent<RectTransform>();
        var isUsing3DPanel = outputPanel == null && outputPanel3D != null;
        var preferredWidth = isUsing3DPanel ? GetPreferredOutputWidth(outputTextRect) : outputTextMaxWidth;

        outputText.ForceMeshUpdate();
        var preferred = outputText.GetPreferredValues(outputText.text, preferredWidth, 0f);

        if (outputPanel != null)
        {
            var height = Mathf.Max(outputPanelMinHeight, preferred.y + outputPanelPaddingY);
            outputPanel.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, height);
            return;
        }

        if (outputPanel3D != null)
        {
            var newSize = Mathf.Max(outputPanel3DMinHeight, (preferred.y * outputPanel3DHeightMultiplier) + outputPanel3DPaddingY);
            var currentScale = outputPanel3D.localScale;
            var nextScale = currentScale;

            switch (outputPanel3DResizeAxis)
            {
                case Panel3DResizeAxis.X:
                    nextScale.x = newSize;
                    break;
                case Panel3DResizeAxis.Z:
                    nextScale.z = newSize;
                    break;
                default:
                    nextScale.y = newSize;
                    break;
            }

            outputPanel3D.localScale = nextScale;
            ResizeOutputTextBounds(outputTextRect, preferredWidth, preferred.y);
            PositionOutputTextInFrontOfPanel(outputTextRect);
        }
    }

    private float GetPreferredOutputWidth(RectTransform outputTextRect)
    {
        if (outputPanel3D != null && outputTextRect != null)
        {
            var textScaleX = Mathf.Abs(outputText.transform.localScale.x);
            var panelScaleX = Mathf.Abs(outputPanel3D.localScale.x);

            if (!Mathf.Approximately(textScaleX, 0f) && !Mathf.Approximately(panelScaleX, 0f))
            {
                var panelWidth = Mathf.Max(0.1f, panelScaleX - outputPanel3DPaddingY);
                return panelWidth / textScaleX;
            }
        }

        return outputTextMaxWidth;
    }

    private void ResizeOutputTextBounds(RectTransform outputTextRect, float preferredWidth, float preferredHeight)
    {
        if (outputTextRect == null)
        {
            return;
        }

        var width = Mathf.Max(0.1f, preferredWidth);
        var height = Mathf.Max(outputTextRect.sizeDelta.y, preferredHeight + outputPanel3DPaddingY);
        outputTextRect.sizeDelta = new Vector2(width, height);
    }

    private void PositionOutputTextInFrontOfPanel(RectTransform outputTextRect)
    {
        if (outputPanel3D == null || outputTextRect == null)
        {
            return;
        }

        outputTextRect.SetParent(outputPanel3D.parent, false);
        outputTextRect.localRotation = outputPanel3D.localRotation;
        outputTextRect.localScale = Vector3.one;
        outputTextRect.localPosition = outputPanel3D.localPosition + new Vector3(0f, 0f, -0.17f);
    }
}
