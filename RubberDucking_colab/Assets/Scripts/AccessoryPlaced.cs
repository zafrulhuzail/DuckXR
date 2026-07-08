using UnityEngine;

public class AccessoryPlaced : MonoBehaviour
{
    public bool isPlaced = false;
     [Header("Assign the Accessory Manager here")]
    public AccessoryFinalizer accessoryManager;
    [Header("Assign the Accessory Model From Duck Placeholder here")]
    public GameObject accessoryModel;

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Duck"))
        {
            // isPlaced = true;
            
            accessoryModel.gameObject.SetActive(true);
            this.gameObject.SetActive(false);
            Debug.Log("Accessory placed on duck!");
            if (accessoryManager != null)
            {
                if (accessoryManager != null)
                {   
                    Debug.Log("Notifying Accessory Manager to hide unplaced accessories.");
                    accessoryManager.HideUnplacedAccessories();
                }
            }
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
