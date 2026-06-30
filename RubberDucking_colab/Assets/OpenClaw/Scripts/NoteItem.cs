using TMPro;
using UnityEngine;

public class NoteItem : MonoBehaviour
{
    [SerializeField] private TMP_Text noteText;

    public string GetText()
    {
        return noteText != null ? noteText.text : string.Empty;
    }
}
