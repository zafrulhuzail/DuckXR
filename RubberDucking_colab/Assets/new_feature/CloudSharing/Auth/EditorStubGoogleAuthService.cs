using System;
using System.Threading.Tasks;
using UnityEngine;

public class EditorStubGoogleAuthService : ICloudAuthService
{
    private CloudAuthUser _currentUser = CloudAuthUser.SignedOut;

    public CloudAuthUser CurrentUser => _currentUser;
    public bool IsSignedIn => _currentUser != null && _currentUser.isAuthenticated;
    public event Action<CloudAuthUser> AuthStateChanged;

    public Task<CloudAuthUser> SignInWithGoogleAsync()
    {
        _currentUser = new CloudAuthUser
        {
            userId = "editor-user-demo",
            email = "demo.user@example.com",
            displayName = "Demo User",
            isAuthenticated = true
        };

        Debug.Log("EditorStubGoogleAuthService: Signed in with stub Google user. Replace with real Firebase Auth later.");
        AuthStateChanged?.Invoke(_currentUser);
        return Task.FromResult(_currentUser);
    }

    public Task SignOutAsync()
    {
        _currentUser = CloudAuthUser.SignedOut;
        AuthStateChanged?.Invoke(_currentUser);
        return Task.CompletedTask;
    }
}
