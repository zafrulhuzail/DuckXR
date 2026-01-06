using UnityEngine;
using UnityEngine.UI;
using TMPro;                 // If using TextMeshPro input field
using LLMUnity;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

public class SummarizeText : MonoBehaviour
{
    [Header("LLM References")]
    // Assign this in the Inspector to your LLMCharacter GameObject
    public LLMCharacter llmCharacter;

    [Header("Input References")]
    // If using TextMeshPro InputField
    public TMP_Text tmpInputField;

    // Summary result
    
    

    [Header("Output References")]
    public TMP_Text tmpOutputField;
    public TextCollisionSwitcher TextCollisionSwitcher;
    [TextArea] private string summaryResult;

    public void Start()
    {
        SummarizeCurrentText();
    }
    /// <summary>
    /// Call this (e.g., from a button) to summarize the current text input
    /// </summary>
    public async void SummarizeCurrentText()
    {
        // 1) Get the text from whichever input field is present
        string userText = tmpInputField.text;
        if (string.IsNullOrWhiteSpace(userText))
        {
            Debug.LogWarning("No input text found to summarize.");
            return;
        }

        // 2) Construct a prompt designed to produce a short key-word-like summary
        //    Adjust the prompt to focus the model on concise summary
        //string prompt = $"Summarize this text into a short set of key phrases. " + $"Keep it concise with no extra punctuation:\n\n\"{userText}\"";

        // 3) Send the prompt to the LLM and await the result
        string llmOutput = await llmCharacter.Chat(userText, null, null);

        // 4) Clean up the output: remove unwanted punctuation if necessary
        summaryResult = CleanUpSummary(llmOutput);
        tmpOutputField.text = summaryResult;
        TextCollisionSwitcher.summaryText = summaryResult;

        Debug.Log("Summary: " + summaryResult);
    }

    /// <summary>
    /// Cleans up model output so it’s concise with minimal punctuation
    /// </summary>
    private string CleanUpSummary(string raw)
    {
        // remove newlines, trailing punctuation, extra spaces
        string cleaned = raw.Trim();

        // remove punctuation except alphanumeric and spaces
        cleaned = Regex.Replace(cleaned, @"[^\w\s]", "");

        // collapse multiple spaces
        cleaned = Regex.Replace(cleaned, @"\s+", " ").Trim();

        return cleaned;
    }

    /// <summary>
    /// Allows external access to the last summary
    /// </summary>
    public string GetSummary()
    {
        return summaryResult;
    }
}
