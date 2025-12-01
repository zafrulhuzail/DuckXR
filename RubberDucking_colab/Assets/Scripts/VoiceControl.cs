using UnityEngine;
using System;
using UnityEngine.Events;

public class VoiceControl : MonoBehaviour
{
    [Header("Key Press Events")]
    public UnityEvent OnKey1Pressed;
    public UnityEvent OnKey2Pressed;
    public UnityEvent OnKey3Pressed;
    public UnityEvent OnKey4Pressed;
    public UnityEvent OnKey5Pressed;

    void Update()
    {
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
    }
}
