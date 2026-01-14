using UnityEngine;

public class ScreenFlowManager : MonoBehaviour
{
    [Header("Screens in order")]
    [SerializeField] private GameObject[] screens;

    private int currentScreenIndex = 0;

    private void Start()
    {
        ShowScreen(0);
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
            Debug.Log("Reached final screen.");
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
    }
}
