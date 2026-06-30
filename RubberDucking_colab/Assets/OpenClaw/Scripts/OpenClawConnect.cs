using System;
using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

public class OpenClawConnect : MonoBehaviour
{
    [Header("OpenClaw")]
    [SerializeField] private string endpoint = "https://your-endpoint.example/v1/responses";
    [SerializeField] private string bearerToken = "";
    [SerializeField] private string model = "openclaw:main";
    [SerializeField] private string sessionUser = "meta-quest-test";

    [Header("Debug")]
    [SerializeField] private bool logFullJsonResponse = false;

    public void SendText(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            Debug.LogWarning("OpenClawConnect: text was empty.");
            return;
        }

        StartCoroutine(SendRequest(text));
    }

    private IEnumerator SendRequest(string text)
    {
        var body = new RequestBody
        {
            model = model,
            user = sessionUser,
            input = text
        };

        var json = JsonUtility.ToJson(body);
        var bodyRaw = Encoding.UTF8.GetBytes(json);

        using var request = new UnityWebRequest(endpoint, UnityWebRequest.kHttpVerbPOST);
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        if (!string.IsNullOrWhiteSpace(bearerToken))
        {
            request.SetRequestHeader("Authorization", "Bearer " + bearerToken);
        }

        yield return request.SendWebRequest();

        if (request.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError($"OpenClaw request failed: {request.error}\n{request.downloadHandler.text}");
            yield break;
        }

        var responseJson = request.downloadHandler.text;

        if (logFullJsonResponse)
        {
            Debug.Log(responseJson);
        }

        var parsed = TryExtractOutputText(responseJson);
        Debug.Log("OpenClaw response: " + parsed);
    }

    private string TryExtractOutputText(string responseJson)
    {
        try
        {
            var response = JsonUtility.FromJson<ResponseBody>(responseJson);
            if (response?.output != null && response.output.Length > 0)
            {
                var firstOutput = response.output[0];
                if (firstOutput.content != null && firstOutput.content.Length > 0)
                {
                    return firstOutput.content[0].text;
                }
            }
        }
        catch (Exception ex)
        {
            Debug.LogWarning("OpenClaw response parse warning: " + ex.Message);
        }

        return "[No parsable output_text found]";
    }

    [Serializable]
    private class RequestBody
    {
        public string model;
        public string user;
        public string input;
    }

    [Serializable]
    private class ResponseBody
    {
        public OutputItem[] output;
    }

    [Serializable]
    private class OutputItem
    {
        public ContentItem[] content;
    }

    [Serializable]
    private class ContentItem
    {
        public string text;
    }
}
