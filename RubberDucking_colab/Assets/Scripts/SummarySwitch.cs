using UnityEngine;
using TMPro;

public class TextCollisionSwitcher : MonoBehaviour
{
    public SummarizeText SummarizeText;

    [Header("Text References")]
    public TMP_Text textDisplay;

    [Header("Text Content")]
    [TextArea] public string originalText;
    [TextArea] public string summaryText;

    [Header("Collision")]
    public string targetTag = "SummaryZone";

    private void Start()
    {
        // Ensure original text is shown at start
        originalText = textDisplay.text;
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag(targetTag))
        {
            SwitchToSummary();
        }
    }

    private void OnTriggerExit(Collider other)
    {
        if (other.CompareTag(targetTag))
        {
            SwitchToOriginal();
        }
    }

    private void SwitchToSummary()
    {
        if (!string.IsNullOrEmpty(summaryText))
        {
            textDisplay.text = summaryText;
        }
        else
        {
            SummarizeText.SummarizeCurrentText();
            SwitchToSummary();
        }
    }

    private void SwitchToOriginal()
    {
        textDisplay.text = originalText;
    }
}