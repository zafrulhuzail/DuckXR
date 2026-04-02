using UnityEngine;
using System;
using UnityEngine.Events;

public class VoiceControl : MonoBehaviour
{
    [Header("Key Press Events")]
    public bool enableKeyboardTesting = true;
    public UnityEvent OnKey1Pressed;
    public UnityEvent OnKey2Pressed;
    public UnityEvent OnKey3Pressed;
    public UnityEvent OnKey4Pressed;
    public UnityEvent OnKey5Pressed;
    public UnityEvent OnKey6Pressed;
    public UnityEvent OnKey7Pressed;
    public UnityEvent OnKey8Pressed;
    public UnityEvent OnKey9Pressed;

    void Update()
    {
        if (!enableKeyboardTesting)
            return;

        if (Input.GetKeyDown(KeyCode.Alpha1))
            OnKey1Pressed?.Invoke();

        if (Input.GetKeyDown(KeyCode.Alpha2))
            OnKey2Pressed?.Invoke();

        if (Input.GetKeyDown(KeyCode.Alpha3))
            OnKey3Pressed?.Invoke();

        if (Input.GetKeyDown(KeyCode.Alpha4))
            OnKey4Pressed?.Invoke();

        if (Input.GetKeyDown(KeyCode.Alpha5))
            OnKey5Pressed?.Invoke();

        if (Input.GetKeyDown(KeyCode.Alpha6))
            OnKey6Pressed?.Invoke();

        if (Input.GetKeyDown(KeyCode.Alpha7))
            OnKey7Pressed?.Invoke();

        if (Input.GetKeyDown(KeyCode.Alpha8))
            OnKey8Pressed?.Invoke();
        if (Input.GetKeyDown(KeyCode.Alpha9))
            OnKey9Pressed?.Invoke();

    }
}
