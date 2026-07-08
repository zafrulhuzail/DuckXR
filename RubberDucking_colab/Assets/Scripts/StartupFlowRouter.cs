using UnityEngine;

public class StartupFlowRouter : MonoBehaviour
{
    private const string HasCompletedOnboardingKey = "HasCompletedOnboarding";
    private const string LastUserNameKey = "LastUserName";
    private const string LastDuckNameKey = "LastDuckName";

    [Header("State Roots")]
    [SerializeField] private GameObject onboardingRoot;
    [SerializeField] private GameObject sessionBrowserRoot;

    [Header("Optional Helpers")]
    [SerializeField] private ProfileLoader profileLoader;
    [SerializeField] private SavedSessionsListController savedSessionsListController;

    private void Start()
    {
        Route();
    }

    public void Route()
    {
        bool shouldShowSessionBrowser = ShouldShowSessionBrowser();

        if (onboardingRoot != null)
            onboardingRoot.SetActive(!shouldShowSessionBrowser);

        if (sessionBrowserRoot != null)
            sessionBrowserRoot.SetActive(shouldShowSessionBrowser);

        if (!shouldShowSessionBrowser)
            return;

        if (profileLoader != null)
            profileLoader.LoadProfile();

        if (savedSessionsListController != null)
            savedSessionsListController.Refresh();
    }

    public void MarkOnboardingComplete()
    {
        PlayerPrefs.SetInt(HasCompletedOnboardingKey, 1);
        PlayerPrefs.Save();
        Route();
    }

    public void CompleteOnboarding(string userName, string duckName)
    {
        if (!string.IsNullOrWhiteSpace(userName))
            PlayerPrefs.SetString(LastUserNameKey, userName.Trim());

        if (!string.IsNullOrWhiteSpace(duckName))
            PlayerPrefs.SetString(LastDuckNameKey, duckName.Trim());

        MarkOnboardingComplete();
    }

    public void ResetProfile()
    {
        SavedSessionService.ClearAllSavedSessions();

        PlayerPrefs.DeleteKey(HasCompletedOnboardingKey);
        PlayerPrefs.DeleteKey(LastUserNameKey);
        PlayerPrefs.DeleteKey(LastDuckNameKey);
        PlayerPrefs.Save();

        if (profileLoader != null)
            profileLoader.ClearProfile();

        if (savedSessionsListController != null)
            savedSessionsListController.Refresh();

        Route();
    }

    public bool HasCompletedOnboarding()
    {
        return PlayerPrefs.GetInt(HasCompletedOnboardingKey, 0) == 1;
    }

    public bool ShouldShowSessionBrowser()
    {
        bool hasCompletedOnboarding = HasCompletedOnboarding();
        bool hasUserName = !string.IsNullOrWhiteSpace(PlayerPrefs.GetString(LastUserNameKey, string.Empty));
        bool hasDuckName = !string.IsNullOrWhiteSpace(PlayerPrefs.GetString(LastDuckNameKey, string.Empty));

        Debug.Log($"Routing decision: HasCompletedOnboarding={hasCompletedOnboarding}, HasUserName={hasUserName}, HasDuckName={hasDuckName}");

        return hasCompletedOnboarding && hasUserName && hasDuckName;
    }
}
