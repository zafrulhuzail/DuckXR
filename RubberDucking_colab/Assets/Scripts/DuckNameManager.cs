using UnityEngine;
using TMPro;

public class DuckNameManager : MonoBehaviour
{
    public RunWhisper whisper;

    [Header("UI")]
    public TMP_Text nameText;
    public GameObject nameTagObject;
    public GameObject duckChatBubble;
    public TMP_Text duckChatBubbleText;
    public GameObject animation;
    public UserNameManager userNameManager;
    [TextArea]
    public string duckChatBubbleFormat = "Hi {userName}!\nI am {duckName}.";
    public bool hideChatBubbleUntilNamed = true;

    public string duckName { get; private set; } = "";

    void OnEnable()
    {
        if (whisper != null)
            whisper.OnTranscriptionFinished += HandleWhisperFinished;

        if (hideChatBubbleUntilNamed && duckChatBubble != null)
            duckChatBubble.SetActive(false);
    }

    void OnDisable()
    {
        if (whisper != null)
            whisper.OnTranscriptionFinished -= HandleWhisperFinished;
    }

    void HandleWhisperFinished(string text)
    {
        duckName = text.Trim().TrimEnd('.');
        SavedSessionService.SetDuckName(duckName);
        PlayerPrefs.SetString("LastDuckName", duckName);
        PlayerPrefs.Save();

        Debug.Log("DUCK NAME SET TO: " + duckName);

        string userName = userNameManager != null ? userNameManager.userName : string.Empty;

        if (nameText != null)
            nameText.text = "Hi, " + userName + "!\n" + duckName + " is a great name, thanks!";

        if (duckChatBubbleText != null)
        {
            duckChatBubbleText.text = duckChatBubbleFormat
                .Replace("{userName}", userName)
                .Replace("{duckName}", duckName);
        }

        if (duckChatBubble != null)
            duckChatBubble.SetActive(true);
            
        if (nameTagObject != null)
            nameTagObject.SetActive(true);
            
        if (animation != null)
            animation.SetActive(false);
    }
}
