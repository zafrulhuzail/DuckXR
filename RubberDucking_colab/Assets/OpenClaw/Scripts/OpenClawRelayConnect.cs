using System;
using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

public class OpenClawRelayConnect : MonoBehaviour
{
    public Action<string> OnRelayTextReceived;
    public Action<string> OnRelayRequestFailed;

    [Header("Chat Relay")]
    [SerializeField] private string askEndpoint = "http://127.0.0.1:3001/quest/ask";
    [SerializeField] private string sessionUser = "duckxr-openclaw-board";

    [Header("Execute Relay")]
    [SerializeField] private string executeEndpoint = "http://127.0.0.1:3001/quest/execute";
    [SerializeField] private string projectName = "DuckXR";
    [SerializeField] private string projectPath = "C:/Users/Zafrul Huzail/.openclaw/workspace/DuckXR/RubberDucking_colab";
    [SerializeField] private string executeUser = "duckxr-openclaw-execute";

    [Header("Debug")]
    [SerializeField] private bool logFullJsonResponse = false;

    public void SendText(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            Debug.LogWarning("OpenClawRelayConnect: text was empty.");
            return;
        }

        StartCoroutine(SendAskRequest(text));
    }

    public void ExecuteTask(string prompt)
    {
        if (string.IsNullOrWhiteSpace(prompt))
        {
            Debug.LogWarning("OpenClawRelayConnect: execute prompt was empty.");
            return;
        }

        if (string.IsNullOrWhiteSpace(projectPath))
        {
            Debug.LogWarning("OpenClawRelayConnect: projectPath is missing.");
            return;
        }

        StartCoroutine(SendExecuteRequest(prompt));
    }

    private IEnumerator SendAskRequest(string text)
    {
        var body = new AskRequestBody
        {
            input = text,
            user = sessionUser
        };

        yield return SendJsonRequest(askEndpoint, body, "Relay response");
    }

    private IEnumerator SendExecuteRequest(string prompt)
    {
        var body = new ExecuteRequestBody
        {
            prompt = prompt,
            projectName = projectName,
            projectPath = projectPath,
            user = executeUser
        };

        yield return SendJsonRequest(executeEndpoint, body, "Execute response");
    }

    private IEnumerator SendJsonRequest(string endpoint, object body, string successPrefix)
    {
        var json = JsonUtility.ToJson(body);
        var bodyRaw = Encoding.UTF8.GetBytes(json);

        using var request = new UnityWebRequest(endpoint, UnityWebRequest.kHttpVerbPOST);
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result != UnityWebRequest.Result.Success)
        {
            var errorText = $"{request.error} {request.downloadHandler.text}".Trim();
            Debug.LogError($"Relay request failed: {request.error}\n{request.downloadHandler.text}");
            OnRelayRequestFailed?.Invoke(errorText);
            yield break;
        }

        var responseJson = request.downloadHandler.text;

        if (logFullJsonResponse)
        {
            Debug.Log(responseJson);
        }

        try
        {
            var parsed = JsonUtility.FromJson<RelayResponseBody>(responseJson);
            Debug.Log(successPrefix + ": " + parsed.text);
            OnRelayTextReceived?.Invoke(parsed.text);
        }
        catch (Exception ex)
        {
            Debug.LogWarning("Relay parse warning: " + ex.Message);
            Debug.Log(responseJson);
            OnRelayRequestFailed?.Invoke("Failed to parse relay response.");
        }
    }

    [Serializable]
    private class AskRequestBody
    {
        public string input;
        public string user;
    }

    [Serializable]
    private class ExecuteRequestBody
    {
        public string prompt;
        public string projectName;
        public string projectPath;
        public string user;
    }

    [Serializable]
    private class RelayResponseBody
    {
        public bool ok;
        public string text;
    }
}
