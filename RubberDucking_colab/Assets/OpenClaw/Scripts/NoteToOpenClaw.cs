using UnityEngine;

public class NoteToOpenClaw : MonoBehaviour
{
    [SerializeField] private OpenClawRelayConnect openClawRelay;
    [SerializeField] private NoteItem noteItem;
    [SerializeField] private string promptPrefix = "Help me with this note:\n\n";

    private void Awake()
    {
        if (noteItem == null)
            noteItem = GetComponent<NoteItem>();

        if (openClawRelay == null)
            openClawRelay = FindFirstObjectByType<OpenClawRelayConnect>();
    }

    public void SendNoteToOpenClaw()
    {
        if (openClawRelay == null)
        {
            Debug.LogWarning("NoteToOpenClaw: OpenClawRelayConnect not found.");
            return;
        }

        if (noteItem == null)
        {
            Debug.LogWarning("NoteToOpenClaw: NoteItem not found.");
            return;
        }

        var noteText = noteItem.GetText();

        if (string.IsNullOrWhiteSpace(noteText))
        {
            Debug.LogWarning("NoteToOpenClaw: note text is empty.");
            return;
        }

        openClawRelay.SendText(promptPrefix + noteText);
    }

    public void ExecuteFromNote()
    {
        if (openClawRelay == null)
        {
            Debug.LogWarning("NoteToOpenClaw: OpenClawRelayConnect not found.");
            return;
        }

        if (noteItem == null)
        {
            Debug.LogWarning("NoteToOpenClaw: NoteItem not found.");
            return;
        }

        var noteText = noteItem.GetText();

        if (string.IsNullOrWhiteSpace(noteText))
        {
            Debug.LogWarning("NoteToOpenClaw: note text is empty.");
            return;
        }

        openClawRelay.ExecuteTask(noteText);
    }
}
