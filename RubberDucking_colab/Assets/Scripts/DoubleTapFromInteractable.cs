using UnityEngine;

public class DoubleTapFromInteractable : MonoBehaviour
{
    public float doubleTapWindow = 0.35f;
    private float lastTapTime = -1f;

    [Header("Whisper")]
    public RunWhisper whisper;   // assign in Inspector


    // Called from the event wrapper
    public void OnTap()
    {
        Debug.Log($"First tap on {gameObject.name}");
        float now = Time.time;

        if (lastTapTime > 0f && now - lastTapTime <= doubleTapWindow)
        {
            lastTapTime = -1f;
            OnDoubleTap();
        }
        else
        {
            lastTapTime = now;
        }
    }

    void OnDoubleTap()
    {
        Debug.Log($"Double tap on {gameObject.name}");
        
        if (whisper != null)
        {
            whisper.TriggerRecordingFromInteraction();
        }
        else
        {
            Debug.LogWarning("DoubleTapFromInteractable: Whisper reference not set.");
        }
        transform.localScale *= 1.2f;
    }
}
