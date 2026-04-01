using TMPro;
using UnityEngine;

public class ProfileLoader : MonoBehaviour
{
    private const string LastUserNameKey = "LastUserName";
    private const string LastDuckNameKey = "LastDuckName";

    [Header("Optional UI")]
    [SerializeField] private TMP_Text userNameText;
    [SerializeField] private TMP_Text duckNameText;
    [SerializeField] private GameObject namedProfileRoot;
    [SerializeField] private GameObject needsNamingRoot;

    [Header("Behavior")]
    [SerializeField] private bool applyToSavedSessionService = true;
    [SerializeField] private bool logLoadedProfile = true;

    public string LastUserName => PlayerPrefs.GetString(LastUserNameKey, string.Empty);
    public string LastDuckName => PlayerPrefs.GetString(LastDuckNameKey, string.Empty);
    public bool HasSavedProfile => !string.IsNullOrWhiteSpace(LastUserName) || !string.IsNullOrWhiteSpace(LastDuckName);

    private void Start()
    {
        LoadProfile();
    }

    public void LoadProfile()
    {
        string lastUserName = LastUserName;
        string lastDuckName = LastDuckName;

        if (applyToSavedSessionService)
        {
            if (!string.IsNullOrWhiteSpace(lastUserName))
                SavedSessionService.SetUserName(lastUserName);

            if (!string.IsNullOrWhiteSpace(lastDuckName))
                SavedSessionService.SetDuckName(lastDuckName);
        }

        if (userNameText != null)
            userNameText.text = lastUserName;

        if (duckNameText != null)
            duckNameText.text = lastDuckName;

        bool hasCompleteProfile = !string.IsNullOrWhiteSpace(lastUserName) && !string.IsNullOrWhiteSpace(lastDuckName);

        if (namedProfileRoot != null)
            namedProfileRoot.SetActive(hasCompleteProfile);

        if (needsNamingRoot != null)
            needsNamingRoot.SetActive(!hasCompleteProfile);

        if (logLoadedProfile)
            Debug.Log($"ProfileLoader: Loaded profile User='{lastUserName}', Duck='{lastDuckName}'");
    }

    public void ClearProfile()
    {
        PlayerPrefs.DeleteKey(LastUserNameKey);
        PlayerPrefs.DeleteKey(LastDuckNameKey);
        PlayerPrefs.Save();

        if (userNameText != null)
            userNameText.text = string.Empty;

        if (duckNameText != null)
            duckNameText.text = string.Empty;

        if (namedProfileRoot != null)
            namedProfileRoot.SetActive(false);

        if (needsNamingRoot != null)
            needsNamingRoot.SetActive(true);

        Debug.Log("ProfileLoader: Cleared saved profile.");
    }
}
