using UnityEngine;

public class AccessoryFinalizer : MonoBehaviour
{
    [Header("Assign all accessories here")]
    public AccessoryPlaced[] accessories;

    // Map this to your button
    public void HideUnplacedAccessories()
    {
        foreach (AccessoryPlaced accessory in accessories)
        {
            if (accessory == null)
                continue;

            if (!accessory.isPlaced)
            {
                accessory.gameObject.SetActive(false);
                // Or Destroy(accessory.gameObject);
            }
        }
    }
}
