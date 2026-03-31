using TMPro;
using UnityEngine;

public class SavedNoteInstance : MonoBehaviour
{
    [SerializeField] private TMP_Text noteText;

    public string noteId;

    public void SetNoteTextTarget(TMP_Text target)
    {
        noteText = target;
    }

    private void Awake()
    {
        if (GetComponent<SavedNoteAutosave>() == null)
            gameObject.AddComponent<SavedNoteAutosave>();
    }

    public void Apply(SavedTranscriptNote note)
    {
        if (note == null)
            return;

        noteId = note.id;

        if (noteText != null)
            noteText.text = note.text;

        if (note.localPosition != null)
            transform.localPosition = new Vector3(note.localPosition.x, note.localPosition.y, note.localPosition.z);

        if (note.localRotationEuler != null)
            transform.localEulerAngles = new Vector3(note.localRotationEuler.x, note.localRotationEuler.y, note.localRotationEuler.z);

        // Keep the prefab's authored scale so restored notes stay readable.
        // We intentionally do not overwrite localScale from saved data here.
    }
}
