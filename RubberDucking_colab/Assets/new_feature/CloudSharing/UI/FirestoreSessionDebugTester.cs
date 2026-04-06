using System.Text;
using System.Threading.Tasks;
using TMPro;
using UnityEngine;

public class FirestoreSessionDebugTester : MonoBehaviour
{
    [SerializeField] private CloudSharingFacade facade;
    [SerializeField] private TMP_Text outputText;
    [SerializeField] private bool shareCurrentSessionOnStart;
    [SerializeField] private bool loadOwnedSessionsOnStart;
    [SerializeField] private bool loadSharedWithMeOnStart;

    [Header("Invite Debug")]
    [SerializeField] private string invitedEmail = "";
    [SerializeField] private TMP_InputField invitedEmailInput;
    [SerializeField] private string sharedSessionId = "";
    [SerializeField] private TMP_InputField sharedSessionIdInput;

    private async void Start()
    {
        if (shareCurrentSessionOnStart)
            await ShareCurrentSessionAsync();

        if (loadOwnedSessionsOnStart)
            await LoadOwnedSessionsAsync();

        if (loadSharedWithMeOnStart)
            await LoadSharedWithMeAsync();
    }

    public void ShareCurrentSession()
    {
        _ = ShareCurrentSessionAsync();
    }

    public async Task ShareCurrentSessionAsync()
    {
        if (facade == null)
        {
            LogToUi("No CloudSharingFacade assigned.");
            Debug.LogWarning("FirestoreSessionDebugTester: No CloudSharingFacade assigned.");
            return;
        }

        var result = await facade.ShareCurrentLocalSessionAsync();
        if (result == null)
        {
            LogToUi("Share current session returned null.");
            Debug.LogWarning("FirestoreSessionDebugTester: ShareCurrentLocalSessionAsync returned null.");
            return;
        }

        sharedSessionId = result.id;
        if (sharedSessionIdInput != null)
            sharedSessionIdInput.text = sharedSessionId;

        var message = $"Shared current session\nID: {result.id}\nTitle: {result.title}\nNotes: {result.notes.Count}";
        LogToUi(message);
        Debug.Log($"FirestoreSessionDebugTester: Shared current session. id={result.id}, title={result.title}, notes={result.notes.Count}");
    }

    public void LoadOwnedSessions()
    {
        _ = LoadOwnedSessionsAsync();
    }

    public async Task LoadOwnedSessionsAsync()
    {
        if (facade == null)
        {
            LogToUi("No CloudSharingFacade assigned.");
            Debug.LogWarning("FirestoreSessionDebugTester: No CloudSharingFacade assigned.");
            return;
        }

        var sessions = await facade.LoadOwnedByMeAsync();
        Debug.Log($"FirestoreSessionDebugTester: Loaded {sessions.Count} owned shared sessions.");

        var sb = new StringBuilder();
        sb.AppendLine($"Owned Sessions ({sessions.Count})");

        foreach (var session in sessions)
        {
            Debug.Log($"FirestoreSessionDebugTester: Owned session {session.id} | {session.title} | notes={session.noteCount}");
            sb.AppendLine();
            sb.AppendLine(session.title);
            sb.AppendLine($"ID: {session.id}");
            sb.AppendLine($"Notes: {session.noteCount}");
        }

        LogToUi(sb.ToString().Trim());
    }

    public void InviteByEmail()
    {
        _ = InviteByEmailAsync();
    }

    public async Task InviteByEmailAsync()
    {
        if (facade == null)
        {
            LogToUi("No CloudSharingFacade assigned.");
            Debug.LogWarning("FirestoreSessionDebugTester: No CloudSharingFacade assigned.");
            return;
        }

        string email = ResolveInvitedEmail();
        string sessionId = ResolveSharedSessionId();

        if (string.IsNullOrWhiteSpace(email))
        {
            LogToUi("Invite email is empty.");
            Debug.LogWarning("FirestoreSessionDebugTester: Invite email is empty.");
            return;
        }

        if (string.IsNullOrWhiteSpace(sessionId))
        {
            LogToUi("Shared session id is empty. Share a session first or paste a session id.");
            Debug.LogWarning("FirestoreSessionDebugTester: Shared session id is empty. Share a session first or paste a session id.");
            return;
        }

        bool success = await facade.GrantInviteByEmailAsync(sessionId, email);
        var message = success
            ? $"Invited '{email}'\nTo session: {sessionId}"
            : $"Failed to invite '{email}'\nTo session: {sessionId}";

        LogToUi(message);
        Debug.Log(success
            ? $"FirestoreSessionDebugTester: Invited '{email}' to session '{sessionId}'."
            : $"FirestoreSessionDebugTester: Failed to invite '{email}' to session '{sessionId}'.");
    }

    public void LoadSharedWithMe()
    {
        _ = LoadSharedWithMeAsync();
    }

    public async Task LoadSharedWithMeAsync()
    {
        if (facade == null)
        {
            LogToUi("No CloudSharingFacade assigned.");
            Debug.LogWarning("FirestoreSessionDebugTester: No CloudSharingFacade assigned.");
            return;
        }

        var sessions = await facade.LoadSharedWithMeAsync();
        Debug.Log($"FirestoreSessionDebugTester: Loaded {sessions.Count} shared-with-me sessions.");

        var sb = new StringBuilder();
        sb.AppendLine($"Shared With Me ({sessions.Count})");

        foreach (var session in sessions)
        {
            Debug.Log($"FirestoreSessionDebugTester: Shared with me {session.id} | {session.title} | owner={session.ownerEmail} | notes={session.noteCount}");
            sb.AppendLine();
            sb.AppendLine(session.title);
            sb.AppendLine($"Owner: {Fallback(session.ownerDisplayName, session.ownerEmail)}");
            sb.AppendLine($"ID: {session.id}");
            sb.AppendLine($"Notes: {session.noteCount}");
        }

        LogToUi(sb.ToString().Trim());
    }

    private void LogToUi(string message)
    {
        if (outputText != null)
            outputText.text = string.IsNullOrWhiteSpace(message) ? "(empty)" : message;
    }

    private string ResolveInvitedEmail()
    {
        if (invitedEmailInput != null && !string.IsNullOrWhiteSpace(invitedEmailInput.text))
            return invitedEmailInput.text.Trim();

        return string.IsNullOrWhiteSpace(invitedEmail) ? string.Empty : invitedEmail.Trim();
    }

    private string ResolveSharedSessionId()
    {
        if (sharedSessionIdInput != null && !string.IsNullOrWhiteSpace(sharedSessionIdInput.text))
            return sharedSessionIdInput.text.Trim();

        return string.IsNullOrWhiteSpace(sharedSessionId) ? string.Empty : sharedSessionId.Trim();
    }

    private static string Fallback(string displayName, string email)
    {
        if (!string.IsNullOrWhiteSpace(displayName))
            return displayName;

        return string.IsNullOrWhiteSpace(email) ? "Unknown" : email;
    }
}
