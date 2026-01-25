using UnityEngine;
using System;

public class WhisperStopSignal : MonoBehaviour
{
    public static event Action StopRequested;

    public static void RequestStop()
    {
        StopRequested?.Invoke();
    }
    public static void StopRecordingFromButton()
    {
        RequestStop();
    }
}
