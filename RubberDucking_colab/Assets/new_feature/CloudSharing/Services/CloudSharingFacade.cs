using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;

public class CloudSharingFacade : MonoBehaviour
{
    [Header("Temporary wiring")]
    [SerializeField] private bool useEditorStubAuth = false;
    [SerializeField] private bool useFirestoreRepository = true;

    private ICloudAuthService _authService;
    private ISharedSessionRepository _repository;

    public CloudAuthUser CurrentUser => _authService != null ? _authService.CurrentUser : CloudAuthUser.SignedOut;

    private void Awake()
    {
        _authService = useEditorStubAuth
            ? new EditorStubGoogleAuthService()
            : new FirebaseEmailAuthService();

        _repository = useFirestoreRepository
            ? new FirestoreSharedSessionRepository()
            : new InMemorySharedSessionRepository();
    }

    public async Task<CloudAuthUser> EnsureSignedInAsync()
    {
        if (_authService == null)
        {
            Debug.LogWarning("CloudSharingFacade: No auth service configured yet.");
            return CloudAuthUser.SignedOut;
        }

        if (_authService.IsSignedIn)
            return _authService.CurrentUser;

        return await _authService.SignInWithGoogleAsync();
    }

    public async Task<SharedSessionRecord> ShareCurrentLocalSessionAsync(string invitedEmail = null)
    {
        var user = await EnsureSignedInAsync();
        if (user == null || !user.isAuthenticated)
            return null;

        var current = SavedSessionService.CurrentSession;
        if (current == null)
        {
            Debug.LogWarning("CloudSharingFacade: No current local session to share.");
            return null;
        }

        var shared = SharedSessionMapper.FromLocalSession(current, user, SharedSessionVisibility.InviteOnly);
        var uploaded = await _repository.UpsertOwnedSessionAsync(shared, user);

        if (!string.IsNullOrWhiteSpace(invitedEmail))
            await _repository.GrantInviteByEmailAsync(uploaded.id, invitedEmail, user);

        return uploaded;
    }

    public async Task<bool> GrantInviteByEmailAsync(string sessionId, string invitedEmail)
    {
        var user = await EnsureSignedInAsync();
        if (user == null || !user.isAuthenticated)
            return false;

        return await _repository.GrantInviteByEmailAsync(sessionId, invitedEmail, user);
    }

    public async Task<IReadOnlyList<SharedSessionSummaryRecord>> LoadSharedWithMeAsync()
    {
        var user = await EnsureSignedInAsync();
        if (user == null || !user.isAuthenticated)
            return Array.Empty<SharedSessionSummaryRecord>();

        return await _repository.GetSharedWithMeAsync(user);
    }

    public async Task<IReadOnlyList<SharedSessionSummaryRecord>> LoadOwnedByMeAsync()
    {
        var user = await EnsureSignedInAsync();
        if (user == null || !user.isAuthenticated)
            return Array.Empty<SharedSessionSummaryRecord>();

        return await _repository.GetOwnedByMeAsync(user);
    }

    public async Task<SharedSessionRecord> LoadSharedSessionAsync(string sessionId)
    {
        var user = await EnsureSignedInAsync();
        if (user == null || !user.isAuthenticated)
            return null;

        return await _repository.GetSharedSessionAsync(sessionId, user);
    }
}
