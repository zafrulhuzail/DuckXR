using TMPro;
using UnityEngine;

public class QuestVoicePlaceholder : MonoBehaviour
{
    [SerializeField] private OpenClawRelayConnect openClawRelay;

    [Header("Optional UI Input")]
    [SerializeField] private TMP_InputField promptInputField;
    [SerializeField] private TMP_InputField executeInputField;

    [Header("Fallback Prompts")]
    [SerializeField] private string testPrompt = "Tell me a short robot joke.";
    [SerializeField] private string testExecutePrompt = "Create a new MonoBehaviour in Assets/Scripts called HelloQuest.cs that logs a short message in Start.";

    [ContextMenu("Send Test Prompt")]
    public void SendTestPrompt()
    {
        if (openClawRelay == null)
        {
            Debug.LogWarning("QuestVoicePlaceholder: OpenClawRelayConnect reference is missing.");
            return;
        }

        var prompt = GetPromptText(promptInputField, testPrompt);
        if (string.IsNullOrWhiteSpace(prompt))
        {
            Debug.LogWarning("QuestVoicePlaceholder: prompt is empty.");
            return;
        }

        openClawRelay.SendText(prompt);
    }

    [ContextMenu("Execute Test Prompt")]
    public void ExecuteTestPrompt()
    {
        if (openClawRelay == null)
        {
            Debug.LogWarning("QuestVoicePlaceholder: OpenClawRelayConnect reference is missing.");
            return;
        }

        var prompt = GetPromptText(executeInputField, testExecutePrompt);
        if (string.IsNullOrWhiteSpace(prompt))
        {
            Debug.LogWarning("QuestVoicePlaceholder: execute prompt is empty.");
            return;
        }

        openClawRelay.ExecuteTask(prompt);
    }

    private static string GetPromptText(TMP_InputField inputField, string fallback)
    {
        if (inputField != null && !string.IsNullOrWhiteSpace(inputField.text))
        {
            return inputField.text;
        }

        return fallback;
    }
}
