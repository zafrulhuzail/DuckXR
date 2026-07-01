using UnityEngine;
using TMPro;

public class ScreenFlowManager : MonoBehaviour
{
    [Header("Screens in order")]
    [SerializeField] private GameObject[] screens;

    [Header("Screen Sounds")]
    [SerializeField] private AudioClip quackClip;
    [SerializeField] private AudioSource quackAudioSource;
    [SerializeField] [Range(0f, 1f)] private float quackVolume = 1f;

    [Header("Final Screen")]
    [SerializeField] private StartupFlowRouter startupFlowRouter;
    [SerializeField] private GameObject finalScreenFallbackRoot;
    [SerializeField] private GameObject[] finalScreenRootsToHide;

    [Header("Instruction Navigation Cue")]
    [SerializeField] private bool createInstructionNavigationCues = true;
    [SerializeField] private string navigationButtonObjectName = "Skip_Button (1)";

    private int currentScreenIndex = 0;
    private Material _cueMaterial;

    private void Awake()
    {
        EnsureAudioSource();
    }

    private void Start()
    {
        ShowScreen(0);
    }

    public void ShowScreenForTesting(int index)
    {
        if (screens == null || screens.Length == 0)
        {
            Debug.LogWarning("ScreenFlowManager: no screens assigned.");
            return;
        }

        index = Mathf.Clamp(index, 0, screens.Length - 1);
        currentScreenIndex = index;
        ShowScreen(currentScreenIndex);
    }

    /// <summary>
    /// Go to the next screen in the flow
    /// </summary>
    public void NextScreen()
    {
        if (currentScreenIndex < screens.Length - 1)
        {
            currentScreenIndex++;
            ShowScreen(currentScreenIndex);
        }
        else
        {
            CompleteFinalScreen();
        }
    }

    /// <summary>
    /// Go to the previous screen (optional)
    /// </summary>
    public void PreviousScreen()
    {
        if (currentScreenIndex > 0)
        {
            currentScreenIndex--;
            ShowScreen(currentScreenIndex);
        }
    }

    /// <summary>
    /// Show a specific screen by index
    /// </summary>
    private void ShowScreen(int index)
    {
        for (int i = 0; i < screens.Length; i++)
        {
            screens[i].SetActive(i == index);
        }

        PlayQuackIfNeeded(screens[index]);
        EnsureInstructionNavigationCue(screens[index], index >= screens.Length - 1);
    }

    private void PlayQuackIfNeeded(GameObject screen)
    {
        if (quackClip == null || screen == null)
            return;

        string screenName = screen.name.ToLowerInvariant();
        bool shouldQuack =
            screenName.Contains("splash") ||
            screenName.Contains("placeduckinfront") ||
            screenName.Contains("duckname");

        if (!shouldQuack)
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

    private void CompleteFinalScreen()
    {
        if (startupFlowRouter == null)
            startupFlowRouter = GetComponent<StartupFlowRouter>();

        if (startupFlowRouter != null)
            startupFlowRouter.MarkOnboardingComplete();

        HideAllScreens();
        HideFinalCompletionRoots();

        if (finalScreenFallbackRoot != null)
            finalScreenFallbackRoot.SetActive(true);

        Debug.Log("Reached final screen.");
    }

    private void HideAllScreens()
    {
        if (screens == null)
            return;

        for (int i = 0; i < screens.Length; i++)
        {
            if (screens[i] != null)
                screens[i].SetActive(false);
        }
    }

    private void HideFinalCompletionRoots()
    {
        if (finalScreenRootsToHide == null)
            return;

        for (int i = 0; i < finalScreenRootsToHide.Length; i++)
        {
            if (finalScreenRootsToHide[i] != null)
                finalScreenRootsToHide[i].SetActive(false);
        }
    }

    private void EnsureInstructionNavigationCue(GameObject screen, bool isFinalScreen)
    {
        if (!createInstructionNavigationCues || screen == null)
            return;

        string screenName = screen.name;
        if (!screenName.Contains("Instruction_to_DoubleTap"))
            return;

        Transform buttonRoot = FindChildRecursive(screen.transform, navigationButtonObjectName);
        if (buttonRoot == null)
            return;

        Transform existingCue = buttonRoot.Find("__OnboardingNavigationCue");
        string label = isFinalScreen ? "Start Draft Board" : "Next";

        if (existingCue != null)
        {
            TMP_Text existingText = existingCue.GetComponentInChildren<TMP_Text>(true);
            if (existingText != null)
                existingText.text = label;
            return;
        }

        GameObject cueRoot = new GameObject("__OnboardingNavigationCue");
        cueRoot.transform.SetParent(buttonRoot, false);
        cueRoot.transform.localPosition = new Vector3(0f, 0f, -0.02f);
        cueRoot.transform.localRotation = Quaternion.identity;
        cueRoot.transform.localScale = Vector3.one;

        GameObject background = GameObject.CreatePrimitive(PrimitiveType.Cube);
        background.name = "Background";
        background.transform.SetParent(cueRoot.transform, false);
        background.transform.localPosition = Vector3.zero;
        background.transform.localRotation = Quaternion.identity;
        background.transform.localScale = new Vector3(isFinalScreen ? 0.5f : 0.28f, 0.12f, 0.02f);

        Collider backgroundCollider = background.GetComponent<Collider>();
        if (backgroundCollider != null)
            Destroy(backgroundCollider);

        MeshRenderer renderer = background.GetComponent<MeshRenderer>();
        if (renderer != null)
        {
            Material cueMaterial = GetCueMaterial();
            if (cueMaterial != null)
                renderer.sharedMaterial = cueMaterial;
        }

        GameObject labelObject = new GameObject("Label");
        labelObject.transform.SetParent(cueRoot.transform, false);
        labelObject.transform.localPosition = new Vector3(0f, -0.025f, -0.02f);
        labelObject.transform.localRotation = Quaternion.identity;
        labelObject.transform.localScale = Vector3.one;

        TextMeshPro labelText = labelObject.AddComponent<TextMeshPro>();
        labelText.text = label;
        labelText.fontSize = isFinalScreen ? 0.055f : 0.07f;
        labelText.alignment = TextAlignmentOptions.Center;
        labelText.color = new Color(0.15f, 0.15f, 0.15f, 1f);
        labelText.enableWordWrapping = false;
        labelText.rectTransform.sizeDelta = new Vector2(isFinalScreen ? 0.48f : 0.26f, 0.1f);
    }

    private Material GetCueMaterial()
    {
        if (_cueMaterial != null)
            return _cueMaterial;

        Shader shader = Shader.Find("Universal Render Pipeline/Unlit") ??
                        Shader.Find("Unlit/Color") ??
                        Shader.Find("Standard") ??
                        Shader.Find("Sprites/Default");

        if (shader == null)
            return null;

        _cueMaterial = new Material(shader);
        _cueMaterial.color = new Color(1f, 0.86f, 0.25f, 1f);
        return _cueMaterial;
    }

    private static Transform FindChildRecursive(Transform root, string childName)
    {
        if (root == null)
            return null;

        for (int i = 0; i < root.childCount; i++)
        {
            Transform child = root.GetChild(i);
            if (child.name == childName)
                return child;

            Transform result = FindChildRecursive(child, childName);
            if (result != null)
                return result;
        }

        return null;
    }
}
