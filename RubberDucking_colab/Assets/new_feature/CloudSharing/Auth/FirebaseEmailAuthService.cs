using System;
using System.Threading.Tasks;
using Firebase.Auth;
using UnityEngine;

public class FirebaseEmailAuthService : ICloudAuthService
{
    public CloudAuthUser CurrentUser { get; private set; } = CloudAuthUser.SignedOut;
    public bool IsSignedIn => CurrentUser != null && CurrentUser.isAuthenticated;
    public event Action<CloudAuthUser> AuthStateChanged;

    private FirebaseAuth _auth;

    public FirebaseEmailAuthService()
    {
        TryInitialize();
    }

    public Task<CloudAuthUser> SignInWithGoogleAsync()
    {
        TryInitialize();
        RefreshCurrentUser();

        if (!IsSignedIn)
            Debug.LogWarning("FirebaseEmailAuthService: No Firebase email/password user is currently signed in. Sign in first using FirebaseEmailPasswordAuthTester.");

        return Task.FromResult(CurrentUser);
    }

    public Task SignOutAsync()
    {
        TryInitialize();

        if (_auth != null)
            _auth.SignOut();

        SetCurrentUser(CloudAuthUser.SignedOut);
        return Task.CompletedTask;
    }

    private void TryInitialize()
    {
        if (_auth != null)
            return;

        try
        {
            _auth = FirebaseAuth.DefaultInstance;
            _auth.StateChanged += HandleAuthStateChanged;
            RefreshCurrentUser();
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"FirebaseEmailAuthService: FirebaseAuth not ready yet. {ex.Message}");
        }
    }

    private void HandleAuthStateChanged(object sender, EventArgs e)
    {
        RefreshCurrentUser();
    }

    private void RefreshCurrentUser()
    {
        if (_auth == null)
        {
            SetCurrentUser(CloudAuthUser.SignedOut);
            return;
        }

        var firebaseUser = _auth.CurrentUser;
        if (firebaseUser == null)
        {
            SetCurrentUser(CloudAuthUser.SignedOut);
            return;
        }

        string displayName = !string.IsNullOrWhiteSpace(firebaseUser.DisplayName)
            ? firebaseUser.DisplayName
            : PlayerPrefs.GetString("LastUserName", string.Empty).Trim();

        SetCurrentUser(new CloudAuthUser
        {
            userId = firebaseUser.UserId,
            email = firebaseUser.Email ?? string.Empty,
            displayName = displayName,
            isAuthenticated = true
        });
    }

    private void SetCurrentUser(CloudAuthUser user)
    {
        CurrentUser = user ?? CloudAuthUser.SignedOut;
        AuthStateChanged?.Invoke(CurrentUser);
    }
}
