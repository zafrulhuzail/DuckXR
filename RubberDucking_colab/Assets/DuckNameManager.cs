using UnityEngine;
using TMPro;

public class DuckNameManager : MonoBehaviour
{
    public RunWhisper whisper;

    [Header("UI")]
    public TMP_Text nameText;
    public GameObject nameTagObject;
    public GameObject animation;
    public UserNameManager userNameManager;

    public string duckName { get; private set; } = "";

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
        duckName = text.Trim();

        Debug.Log("DUCK NAME SET TO: " + duckName);

        if (nameText != null)
            nameText.text = "Hi, " + userNameManager.userName + "!\n" + duckName + " is a great name, thanks!";
            nameTagObject.SetActive(true);
            animation.SetActive(false);
    }
}
