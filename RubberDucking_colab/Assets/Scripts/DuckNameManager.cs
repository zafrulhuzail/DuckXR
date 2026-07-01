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

    [Header("Sounds")]
    [SerializeField] private AudioClip quackClip;
    [SerializeField] private AudioSource quackAudioSource;
    [SerializeField] [Range(0f, 1f)] private float quackVolume = 1f;

    public string duckName { get; private set; } = "";

    void Awake()
    {
        EnsureAudioSource();
    }

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

        PlayQuack();

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

    private void PlayQuack()
    {
        if (quackClip == null)
            return;

        EnsureAudioSource();
        if (quackClip.loadState == AudioDataLoadState.Unloaded)
            quackClip.LoadAudioData();

        quackAudioSource.PlayOneShot(quackClip, quackVolume);
    }

    private void EnsureAudioSource()
    {
        if (quackAudioSource != null)
            return;

        quackAudioSource = GetComponent<AudioSource>();
        if (quackAudioSource == null)
            quackAudioSource = gameObject.AddComponent<AudioSource>();

        quackAudioSource.playOnAwake = false;
        quackAudioSource.spatialBlend = 0f;
    }
}
