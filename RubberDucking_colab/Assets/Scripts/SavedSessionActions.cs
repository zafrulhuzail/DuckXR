using UnityEngine;

public class SavedSessionActions : MonoBehaviour
{
    [Header("Optional Browser Reference")]
    [SerializeField] private SavedSessionsBrowser savedSessionsBrowser;

    public void SaveCurrentSession()
    {
        SavedSessionService.SaveCurrentSession();
        Debug.Log("SavedSessionActions: Saved current local session.");
    }

    public void StartNewSession()
    {
        SavedSessionService.StartNewSession();
        Debug.Log("SavedSessionActions: Started a new local session.");

        if (savedSessionsBrowser != null)
            savedSessionsBrowser.Refresh();
    }

    public void ResumeSelectedSessionFromBrowser()
    {
        if (savedSessionsBrowser == null)
        {
            Debug.LogWarning("SavedSessionActions: No SavedSessionsBrowser assigned.");
            return;
        }

        savedSessionsBrowser.ResumeSelected();
    }

    public void RefreshBrowser()
    {
        if (savedSessionsBrowser == null)
        {
            Debug.LogWarning("SavedSessionActions: No SavedSessionsBrowser assigned.");
            return;
        }

        savedSessionsBrowser.Refresh();
    }

    public void RefreshSharedWithMeBrowser()
    {
        if (savedSessionsBrowser == null)
        {
            Debug.LogWarning("SavedSessionActions: No SavedSessionsBrowser assigned.");
            return;
        }

        savedSessionsBrowser.RefreshSharedWithMe();
    }

    public void RefreshSharedOwnedBrowser()
    {
        if (savedSessionsBrowser == null)
        {
            Debug.LogWarning("SavedSessionActions: No SavedSessionsBrowser assigned.");
            return;
        }

        savedSessionsBrowser.RefreshSharedOwnedByMe();
    }

    public void RefreshLocalBrowser()
    {
        if (savedSessionsBrowser == null)
        {
            Debug.LogWarning("SavedSessionActions: No SavedSessionsBrowser assigned.");
            return;
        }

        savedSessionsBrowser.RefreshLocalSessions();
    }
}
