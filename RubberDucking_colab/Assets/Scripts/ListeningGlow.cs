using UnityEngine;

public class ListeningGlow : MonoBehaviour
{
    [Header("Glow GameObjects")]
    public GameObject[] glowObjects = new GameObject[9];

    // Enables all assigned GameObjects
    public void EnableGlowObjects()
    {
        foreach (var obj in glowObjects)
        {
            if (obj != null)
                obj.SetActive(true);
        }
    }

    // Disables all assigned GameObjects
    public void DisableGlowObjects()
    {
        foreach (var obj in glowObjects)
        {
            if (obj != null)
                obj.SetActive(false);
        }
    }
}
