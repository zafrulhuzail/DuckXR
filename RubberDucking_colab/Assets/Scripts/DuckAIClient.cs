using System;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.Networking;

/// <summary>
/// Minimal DuckXR client for talking to a backend bridge.
/// This script intentionally talks to your own backend, not directly to OpenClaw.
/// </summary>
public class DuckAIClient : MonoBehaviour
{
    [Header("Backend")]
    [Tooltip("Backend endpoint, for example https://your-backend.example.com/duck/chat")]
    public string backendUrl = "http://localhost:8080/duck/chat";

    [Tooltip("Optional bearer token for local/dev use. Do not ship real secrets in the client.")]
    public string authToken = "";

    [Tooltip("Request timeout in seconds")]
    public float requestTimeoutSeconds = 20f;

    [Header("Session")]
    [Tooltip("Keeps the backend conversation thread alive across requests")]
    public string sessionId = "";

    [Tooltip("Optional stable user id if you have auth")]
    public string userId = "";

    [Header("Context")]
    public string duckName = "Ducky";
    public string userName = "";
    public bool includeCurrentSceneName = true;
    public bool saveMessagesToSavedSessionService = true;

    [Header("UI")]
    public TextMeshProUGUI statusText;
    public TextMeshProUGUI responseText;
    public GameObject loadingIndicator;

    [Header("Events")]
    public StringEvent OnReplyReceived;
    public StringEvent OnErrorReceived;

    private bool isSending;

    [Serializable]
    public class StringEvent : UnityEvent<string> { }

    [Serializable]
    private class DuckChatRequest
    {
        public string sessionId;
        public string userId;
        public string duckName;
        public string userName;
        public string message;
        public string sceneName;
        public DuckChatContext context;
    }

    [Serializable]
    private class DuckChatContext
    {
        public string platform;
        public string appVersion;
        public List<string> notes;
    }

    [Serializable]
    private class DuckChatResponse
    {
        public bool success;
        public string sessionId;
        public string reply;
        public bool shouldSpeak;
        public string mood;
        public string error;
        public List<string> hints;
    }

    public void SendMessageToDuck(string message)
    {
        message = Sanitize(message);
        if (string.IsNullOrWhiteSpace(message))
        {
            UpdateStatus("No message to send");
            return;
        }

        if (isSending)
        {
            UpdateStatus("Already waiting for reply");
            return;
        }

        StartCoroutine(SendMessageCoroutine(message));
    }

    [ContextMenu("Test Send Message")]
    public void TestSendMessage()
    {
        SendMessageToDuck("Help me reason through this bug in DuckXR.");
    }

    private IEnumerator SendMessageCoroutine(string message)
    {
        isSending = true;
        SetLoading(true);
        UpdateStatus("Sending to duck backend...");

        if (saveMessagesToSavedSessionService)
        {
            if (SavedSessionService.EnsureCurrentSession() == null)
                SavedSessionService.StartNewSession();

            SavedSessionService.AddTranscriptNote("User: " + message);
        }

        var requestBody = new DuckChatRequest
        {
            sessionId = sessionId,
            userId = Sanitize(userId),
            duckName = Sanitize(duckName),
            userName = ResolveUserName(),
            message = message,
            sceneName = includeCurrentSceneName ? UnityEngine.SceneManagement.SceneManager.GetActiveScene().name : string.Empty,
            context = new DuckChatContext
            {
                platform = Application.platform.ToString(),
                appVersion = Application.version,
                notes = new List<string>()
            }
        };

        string json = JsonUtility.ToJson(requestBody);
        using (var request = new UnityWebRequest(backendUrl, UnityWebRequest.kHttpVerbPOST))
        {
            byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(json);
            request.uploadHandler = new UploadHandlerRaw(bodyRaw);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");

            if (!string.IsNullOrWhiteSpace(authToken))
                request.SetRequestHeader("Authorization", "Bearer " + authToken.Trim());

            request.timeout = Mathf.CeilToInt(requestTimeoutSeconds);

            yield return request.SendWebRequest();

            if (request.result != UnityWebRequest.Result.Success)
            {
                HandleError("Request failed: " + request.error);
                yield break;
            }

            DuckChatResponse response = null;
            try
            {
                response = JsonUtility.FromJson<DuckChatResponse>(request.downloadHandler.text);
            }
            catch (Exception e)
            {
                HandleError("Failed to parse response: " + e.Message);
                yield break;
            }

            if (response == null)
            {
                HandleError("Backend returned an empty response");
                yield break;
            }

            if (!string.IsNullOrWhiteSpace(response.sessionId))
                sessionId = response.sessionId;

            if (!response.success)
            {
                HandleError(string.IsNullOrWhiteSpace(response.error) ? "Backend returned an error" : response.error);
                yield break;
            }

            string reply = Sanitize(response.reply);
            if (string.IsNullOrWhiteSpace(reply))
                reply = "No reply received.";

            if (responseText != null)
                responseText.text = reply;

            UpdateStatus("Duck replied");
            OnReplyReceived?.Invoke(reply);

            if (saveMessagesToSavedSessionService)
                SavedSessionService.AddTranscriptNote("Duck: " + reply);

            // Hook your TTS system here if needed.
            // if (response.shouldSpeak) { ... }
        }

        SetLoading(false);
        isSending = false;
    }

    private string ResolveUserName()
    {
        string explicitName = Sanitize(userName);
        if (!string.IsNullOrWhiteSpace(explicitName))
            return explicitName;

        return Sanitize(PlayerPrefs.GetString("LastUserName", string.Empty));
    }

    private void HandleError(string error)
    {
        error = Sanitize(error);
        UpdateStatus(error);

        if (responseText != null)
            responseText.text = error;

        OnErrorReceived?.Invoke(error);
        SetLoading(false);
        isSending = false;
    }

    private void SetLoading(bool value)
    {
        if (loadingIndicator != null)
            loadingIndicator.SetActive(value);
    }

    private void UpdateStatus(string value)
    {
        if (statusText != null)
            statusText.text = value;

        Debug.Log("[DuckAIClient] " + value);
    }

    private static string Sanitize(string value)
    {
        return string.IsNullOrWhiteSpace(value) ? string.Empty : value.Trim();
    }
}
