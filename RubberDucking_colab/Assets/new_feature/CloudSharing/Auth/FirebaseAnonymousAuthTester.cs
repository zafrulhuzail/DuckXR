using System;
using Firebase;
using Firebase.Auth;
using UnityEngine;

public class FirebaseAnonymousAuthTester : MonoBehaviour
{
    [Header("Firebase Auth Test")]
    [SerializeField] private bool signInOnStart = true;
    [SerializeField] private bool signOutFirst = false;

    private FirebaseAuth _auth;

    private async void Start()
    {
        if (!signInOnStart)
            return;

        await InitializeAndSignInAsync();
    }

    [ContextMenu("Initialize And Sign In Anonymously")]
    public async void InitializeAndSignInFromContextMenu()
    {
        await InitializeAndSignInAsync();
    }

    [ContextMenu("Sign Out Firebase User")]
    public void SignOutFromContextMenu()
    {
        if (_auth == null)
            _auth = FirebaseAuth.DefaultInstance;

        if (_auth.CurrentUser != null)
        {
            Debug.Log($"FirebaseAnonymousAuthTester: Signing out user {_auth.CurrentUser.UserId}");
            _auth.SignOut();
        }
        else
        {
            Debug.Log("FirebaseAnonymousAuthTester: No Firebase user is currently signed in.");
        }
    }

    private async System.Threading.Tasks.Task InitializeAndSignInAsync()
    {
        try
        {
            Debug.Log("FirebaseAnonymousAuthTester: Checking Firebase dependencies...");
            var dependencyStatus = await FirebaseApp.CheckAndFixDependenciesAsync();

            if (dependencyStatus != DependencyStatus.Available)
            {
                Debug.LogError($"FirebaseAnonymousAuthTester: Firebase dependencies not available: {dependencyStatus}");
                return;
            }

            var app = FirebaseApp.DefaultInstance;
            _auth = FirebaseAuth.DefaultInstance;

            Debug.Log($"FirebaseAnonymousAuthTester: Firebase ready for app '{app.Name}'.");

            if (signOutFirst && _auth.CurrentUser != null)
            {
                Debug.Log($"FirebaseAnonymousAuthTester: signOutFirst enabled. Signing out current user {_auth.CurrentUser.UserId}.");
                _auth.SignOut();
            }

            if (_auth.CurrentUser != null)
            {
                Debug.Log(
                    $"FirebaseAnonymousAuthTester: Already signed in. UID={_auth.CurrentUser.UserId}, Anonymous={_auth.CurrentUser.IsAnonymous}, Email={_auth.CurrentUser.Email}");
                return;
            }

            Debug.Log("FirebaseAnonymousAuthTester: Signing in anonymously...");
            var credential = await _auth.SignInAnonymouslyAsync();
            var user = credential.User;

            Debug.Log(
                $"FirebaseAnonymousAuthTester: Anonymous sign-in success. UID={user.UserId}, Anonymous={user.IsAnonymous}, Email={user.Email}");
        }
        catch (Exception ex)
        {
            Debug.LogError($"FirebaseAnonymousAuthTester: Anonymous auth failed. {ex}");
        }
    }
}
