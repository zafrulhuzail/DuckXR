using System.Text;
using TMPro;
using UnityEngine;

public class SharedWithMeDebugView : MonoBehaviour
{
    [SerializeField] private CloudSharingFacade cloudSharingFacade;
    [SerializeField] private TMP_Text outputText;
    [SerializeField] private bool autoRefreshOnEnable = true;

    private async void OnEnable()
    {
        if (autoRefreshOnEnable)
            await RefreshAsync();
    }

    public async System.Threading.Tasks.Task RefreshAsync()
    {
        if (cloudSharingFacade == null)
        {
            SetOutput("SharedWithMeDebugView: Missing CloudSharingFacade reference.");
            return;
        }

        var items = await cloudSharingFacade.LoadSharedWithMeAsync();
        if (items == null || items.Count == 0)
        {
            SetOutput("Shared With Me\n\nNo shared sessions available for the current signed-in user.");
            return;
        }

        var sb = new StringBuilder();
        sb.AppendLine("Shared With Me");
        sb.AppendLine();

        for (int i = 0; i < items.Count; i++)
        {
            var item = items[i];
            sb.AppendLine($"[{i + 1}] {item.title}");
            sb.AppendLine($"Owner: {Fallback(item.ownerDisplayName, item.ownerEmail)}");
            sb.AppendLine($"Updated: {item.updatedAtUtc}");
            sb.AppendLine($"Notes: {item.noteCount}");
            if (!string.IsNullOrWhiteSpace(item.latestNotePreview))
                sb.AppendLine($"Preview: {item.latestNotePreview}");
            sb.AppendLine();
        }

        SetOutput(sb.ToString().TrimEnd());
    }

    private static string Fallback(string displayName, string email)
    {
        if (!string.IsNullOrWhiteSpace(displayName))
            return displayName;
        if (!string.IsNullOrWhiteSpace(email))
            return email;
        return "Unknown owner";
    }

    private void SetOutput(string value)
    {
        if (outputText != null)
            outputText.text = value;
        else
            Debug.Log(value);
    }
}
