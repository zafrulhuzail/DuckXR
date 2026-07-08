using UnityEngine;

public class SummarizeButton3D : MonoBehaviour
{
    [SerializeField] private BoardSummarizer summarizer;

    private void OnMouseDown()
    {
        if (summarizer != null)
        {
            summarizer.SummarizeNotes();
        }
    }
}
