using TMPro;
using UnityEngine;

public class SavedSessionsListController : MonoBehaviour
{
    [Header("Optional UI")]
    [SerializeField] private TMP_Text outputText;
    [SerializeField] private SavedSessionsBrowser browser;

    private void OnEnable()
    {
        Refresh();
    }

    public void Refresh()
    {
        if (browser != null)
        {
            browser.Refresh();
            return;
        }

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

        outputText.text = $"{index.sessions.Count} saved sessions loaded.";
    }
}
