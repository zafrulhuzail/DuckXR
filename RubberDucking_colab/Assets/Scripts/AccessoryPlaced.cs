using UnityEngine;

public class AccessoryPlaced : MonoBehaviour
{
    public bool isPlaced = false;

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Duck"))
        {
            isPlaced = true;
        }
    }

    private void OnTriggerExit(Collider other)
    {
        if (other.CompareTag("Duck"))
        {
            isPlaced = false;
        }
    }
}
