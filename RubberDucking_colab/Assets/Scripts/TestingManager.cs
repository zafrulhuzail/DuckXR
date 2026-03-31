using System.Collections;
using UnityEngine;

public class TestingManager : MonoBehaviour
{
    public enum TestMode
    {
        Off,
        ScreenIndex,
        FinalNoteTaking,
        WhisperOnly,
        SavedSessions
    }

    [Header("Testing")]
    [SerializeField] private bool enableTesting = false;
    [SerializeField] private TestMode testMode = TestMode.Off;

    [Header("Flow references")]
    [SerializeField] private ScreenFlowManager screenFlowManager;
    [SerializeField] private RunWhisper runWhisper;
    [SerializeField] private SavedSessionsListController savedSessionsListController;

    [Header("Optional test targets")]
    [SerializeField] private GameObject onboardingRoot;
    [SerializeField] private GameObject finalNoteTakingRoot;
    [SerializeField] private GameObject savedSessionsRoot;
    [SerializeField] private GameObject whisperTestRoot;

    [Header("Screen jump")]
    [SerializeField] private int targetScreenIndex = 0;
    [SerializeField] private int finalNoteTakingScreenIndex = 6;

    [Header("Seeded test data")]
    [SerializeField] private string fakeUserName = "Tester";
    [SerializeField] private string fakeDuckName = "Ducky";
    [SerializeField] private bool autoTriggerWhisper = false;

    private IEnumerator Start()
    {
        if (!enableTesting || testMode == TestMode.Off)
            yield break;

        // Let other scene Start() methods finish first, especially ScreenFlowManager.Start()
        // which otherwise resets the screen back to index 0.
        yield return null;

        ApplySeedData();
        ApplyMode();
    }

    private void ApplySeedData()
    {
        if (!string.IsNullOrWhiteSpace(fakeUserName))
            SavedSessionService.SetUserName(fakeUserName);

        if (!string.IsNullOrWhiteSpace(fakeDuckName))
            SavedSessionService.SetDuckName(fakeDuckName);
    }

    private void ApplyMode()
    {
        switch (testMode)
        {
            case TestMode.ScreenIndex:
                ActivateOnly(onboardingRoot);
                JumpToScreen(targetScreenIndex);
                break;

            case TestMode.FinalNoteTaking:
                ActivateOnly(onboardingRoot, finalNoteTakingRoot);
                JumpToScreen(finalNoteTakingScreenIndex);
                if (finalNoteTakingRoot != null)
                    finalNoteTakingRoot.SetActive(true);
                TryAutoTriggerWhisper();
                break;

            case TestMode.WhisperOnly:
                ActivateOnly(whisperTestRoot, finalNoteTakingRoot);
                TryAutoTriggerWhisper();
                break;

            case TestMode.SavedSessions:
                ActivateOnly(savedSessionsRoot);
                if (savedSessionsRoot != null)
                    savedSessionsRoot.SetActive(true);
                if (savedSessionsListController != null)
                    savedSessionsListController.Refresh();
                break;
        }
    }

    private void JumpToScreen(int index)
    {
        if (screenFlowManager != null)
            screenFlowManager.ShowScreenForTesting(index);
    }

    private void TryAutoTriggerWhisper()
    {
        if (autoTriggerWhisper && runWhisper != null)
            runWhisper.TriggerRecordingFromInteraction();
    }

    private void ActivateOnly(params GameObject[] keepActive)
    {
        if (onboardingRoot != null)
            onboardingRoot.SetActive(false);
        if (finalNoteTakingRoot != null)
            finalNoteTakingRoot.SetActive(false);
        if (savedSessionsRoot != null)
            savedSessionsRoot.SetActive(false);
        if (whisperTestRoot != null)
            whisperTestRoot.SetActive(false);

        foreach (var go in keepActive)
        {
            if (go != null)
                go.SetActive(true);
        }
    }
}
