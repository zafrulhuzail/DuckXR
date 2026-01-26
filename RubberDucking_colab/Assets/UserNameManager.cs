using UnityEngine;
using TMPro;

public class UserNameManager : MonoBehaviour
{
    public RunWhisper whisper;

    [Header("UI")]
    public TMP_Text nameText;   
    public GameObject animation;

    public string userName { get; private set; } = "";

    void OnEnable()
    {
        if (whisper != null)
            whisper.OnTranscriptionFinished += HandleWhisperFinished;
    }

    void OnDisable()
    {
        if (whisper != null)
            whisper.OnTranscriptionFinished -= HandleWhisperFinished;
    }

    void HandleWhisperFinished(string text)
    {
        userName = text.Trim().TrimEnd('.');

        Debug.Log("USERNAME SET TO: " + userName);

        if (nameText != null)
            nameText.text = "Nice to meet you, \n" + userName + "!";
            animation.SetActive(false);
    }
}
