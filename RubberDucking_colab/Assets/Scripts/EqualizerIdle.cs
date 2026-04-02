using UnityEngine;

public class EqualizerIdle : MonoBehaviour
{
    public RectTransform[] bars;

    public float baseHeight = 40f;
    public float heightRange = 30f;
    public float speed = 3f;

    void Update()
    {
        for (int i = 0; i < bars.Length; i++)
        {
            float t = Time.time * speed + i * 0.6f;
            float h = baseHeight + (Mathf.Sin(t) * 0.5f + 0.5f) * heightRange;

            Vector2 size = bars[i].sizeDelta;
            bars[i].sizeDelta = new Vector2(size.x, h);
        }
    }

    public void ActivateAnimation()
    {
        for (int i = 0; i < bars.Length; i++)
        {
            float t = Time.time * speed + i * 0.6f;
            float h = baseHeight + (Mathf.Sin(t) * 0.5f + 0.5f) * heightRange;

            Vector2 size = bars[i].sizeDelta;
            bars[i].sizeDelta = new Vector2(size.x, h);
        }
    }
}
