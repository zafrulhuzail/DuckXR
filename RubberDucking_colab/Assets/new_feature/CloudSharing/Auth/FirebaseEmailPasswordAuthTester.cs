using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Firebase;
using Firebase.Auth;
using Firebase.Extensions;
using Firebase.Firestore;
using UnityEngine;

public class FirebaseEmailPasswordAuthTester : MonoBehaviour
{
    private const string LastUserNameKey = "LastUserName";
    private const string LastDuckNameKey = "LastDuckName";

    [Header("Firebase")]
    [SerializeField] private bool initializeFirebaseOnStart = true;

    [Header("Test Credentials")]
    [SerializeField] private string email = "";
    [SerializeField] private string password = "";
    [SerializeField] private bool signInOnStart = false;
    [SerializeField] private bool createUserIfMissing = false;
    [SerializeField] private bool signOutFirst = false;

    [Header("User Profile")]
    [SerializeField] private bool syncUserProfileToFirestore = true;

    private FirebaseAuth _auth;
    private FirebaseFirestore _firestore;

    private async void Start()
    {
        if (initializeFirebaseOnStart)
            await InitializeFirebaseAsync();

        if (signInOnStart)
            SignInWithEmailPassword();
    }

    [ContextMenu("Initialize Firebase")]
    public async void InitializeFirebaseFromContextMenu()
    {
        await InitializeFirebaseAsync();
    }

    public async void SignInWithEmailPassword()
    {
        if (!ValidateCredentials())
            return;

        var initialized = await InitializeFirebaseAsync();
        if (!initialized)
            return;

        try
        {
            if (signOutFirst && _auth.CurrentUser != null)
            {
                Debug.Log($"FirebaseEmailPasswordAuthTester: signOutFirst enabled. Signing out Firebase user {_auth.CurrentUser.UserId}.");
                _auth.SignOut();
            }

            Debug.Log($"FirebaseEmailPasswordAuthTester: Signing in with email '{email}'...");
            var authResult = await _auth.SignInWithEmailAndPasswordAsync(email, password);
            var user = authResult.User;
            LogSignedInUser("FirebaseEmailPasswordAuthTester: Email sign-in success", user);
            await SyncUserProfileAsync(user, isNewUser: false);
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"FirebaseEmailPasswordAuthTester: Sign-in failed. {ex.Message}");

            if (!createUserIfMissing)
            {
                Debug.LogError($"FirebaseEmailPasswordAuthTester: Email/password sign-in failed. {ex}");
                return;
            }

            try
            {
                Debug.Log($"FirebaseEmailPasswordAuthTester: createUserIfMissing enabled. Creating Firebase user '{email}'...");
                var createdAuthResult = await _auth.CreateUserWithEmailAndPasswordAsync(email, password);
                var createdUser = createdAuthResult.User;
                LogSignedInUser("FirebaseEmailPasswordAuthTester: User creation success", createdUser);
                await SyncUserProfileAsync(createdUser, isNewUser: true);
            }
            catch (Exception createEx)
            {
                Debug.LogError($"FirebaseEmailPasswordAuthTester: User creation failed. {createEx}");
            }
        }
    }

    public async void CreateUser()
    {
        if (!ValidateCredentials())
            return;

        var initialized = await InitializeFirebaseAsync();
        if (!initialized)
            return;

        try
        {
            Debug.Log($"FirebaseEmailPasswordAuthTester: Creating Firebase user '{email}'...");
            var authResult = await _auth.CreateUserWithEmailAndPasswordAsync(email, password);
            var user = authResult.User;
            LogSignedInUser("FirebaseEmailPasswordAuthTester: User creation success", user);
            await SyncUserProfileAsync(user, isNewUser: true);
        }
        catch (Exception ex)
        {
            Debug.LogError($"FirebaseEmailPasswordAuthTester: User creation failed. {ex}");
        }
    }

    public void SignOut()
    {
        try
        {
            if (_auth == null)
                _auth = FirebaseAuth.DefaultInstance;
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"FirebaseEmailPasswordAuthTester: FirebaseAuth.DefaultInstance unavailable during sign-out: {ex.Message}");
        }

        if (_auth != null && _auth.CurrentUser != null)
        {
            Debug.Log($"FirebaseEmailPasswordAuthTester: Signing out Firebase user {_auth.CurrentUser.UserId}");
            _auth.SignOut();
            return;
        }

        Debug.Log("FirebaseEmailPasswordAuthTester: No Firebase user is currently signed in.");
    }

    public async void LogCurrentFirebaseUser()
    {
        var initialized = await InitializeFirebaseAsync();
        if (!initialized)
            return;

        if (_auth.CurrentUser == null)
        {
            Debug.Log("FirebaseEmailPasswordAuthTester: No Firebase user is currently signed in.");
            return;
        }

        LogSignedInUser("FirebaseEmailPasswordAuthTester: Current user", _auth.CurrentUser);
    }

    private async Task<bool> InitializeFirebaseAsync()
    {
        try
        {
            Debug.Log("FirebaseEmailPasswordAuthTester: Checking Firebase dependencies...");
            var dependencyStatus = await FirebaseApp.CheckAndFixDependenciesAsync();
            if (dependencyStatus != DependencyStatus.Available)
            {
                Debug.LogError($"FirebaseEmailPasswordAuthTester: Firebase dependencies not available: {dependencyStatus}");
                return false;
            }

            var app = FirebaseApp.DefaultInstance;
            _auth = FirebaseAuth.DefaultInstance;
            _firestore = FirebaseFirestore.DefaultInstance;
            Debug.Log($"FirebaseEmailPasswordAuthTester: Firebase ready for app '{app.Name}'.");
            return true;
        }
        catch (Exception ex)
        {
            Debug.LogError($"FirebaseEmailPasswordAuthTester: Firebase initialization failed. {ex}");
            return false;
        }
    }

    private async Task SyncUserProfileAsync(FirebaseUser user, bool isNewUser)
    {
        if (!syncUserProfileToFirestore)
            return;

        if (user == null)
        {
            Debug.LogWarning("FirebaseEmailPasswordAuthTester: Cannot sync user profile because FirebaseUser is null.");
            return;
        }

        if (_firestore == null)
        {
            Debug.LogWarning("FirebaseEmailPasswordAuthTester: Cannot sync user profile because Firestore is not initialized.");
            return;
        }

        try
        {
            string displayName = GetSavedUserName();
            string duckName = GetSavedDuckName();
            string normalizedEmail = string.IsNullOrWhiteSpace(user.Email) ? string.Empty : user.Email.Trim().ToLowerInvariant();

            var profile = new Dictionary<string, object>
            {
                { "uid", user.UserId },
                { "email", user.Email ?? string.Empty },
                { "emailLower", normalizedEmail },
                { "displayName", displayName },
                { "duckName", duckName },
                { "isAnonymous", user.IsAnonymous },
                { "lastLoginAt", Timestamp.GetCurrentTimestamp() },
                { "updatedAt", Timestamp.GetCurrentTimestamp() }
            };

            if (isNewUser)
                profile["createdAt"] = Timestamp.GetCurrentTimestamp();

            var userDoc = _firestore.Collection("users").Document(user.UserId);
            await userDoc.SetAsync(profile, SetOptions.MergeAll);

            Debug.Log($"FirebaseEmailPasswordAuthTester: Synced Firestore user profile for UID={user.UserId}, displayName='{displayName}', duckName='{duckName}'.");
        }
        catch (Exception ex)
        {
            Debug.LogError($"FirebaseEmailPasswordAuthTester: Failed to sync Firestore user profile. {ex}");
        }
    }

    private bool ValidateCredentials()
    {
        if (string.IsNullOrWhiteSpace(email) || string.IsNullOrWhiteSpace(password))
        {
            Debug.LogError("FirebaseEmailPasswordAuthTester: Email and password must both be filled in Inspector before testing.");
            return false;
        }

        return true;
    }

    private static string GetSavedUserName()
    {
        return Sanitize(PlayerPrefs.GetString(LastUserNameKey, string.Empty));
    }

    private static string GetSavedDuckName()
    {
        return Sanitize(PlayerPrefs.GetString(LastDuckNameKey, string.Empty));
    }

    private static string Sanitize(string value)
    {
        return string.IsNullOrWhiteSpace(value) ? string.Empty : value.Trim();
    }

    private static void LogSignedInUser(string prefix, FirebaseUser user)
    {
        Debug.Log($"{prefix}. UID={user.UserId}, Email={user.Email}, DisplayName={user.DisplayName}, Anonymous={user.IsAnonymous}");
    }
}
