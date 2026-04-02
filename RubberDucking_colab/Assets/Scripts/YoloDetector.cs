using System.Collections.Generic;

using UnityEngine;

using UnityEngine.XR.ARFoundation;

using UnityEngine.XR.ARSubsystems;


[RequireComponent(typeof(ARTrackedImageManager))]

public class SmartImageSpawner : MonoBehaviour

{

    [Header("Einstellungen")]

    [Tooltip("Das Würfel-Prefab, das gespawnt werden soll.")]

    public GameObject prefabToSpawn;


    [Tooltip("Simulierter Threshold: Nur anzeigen, wenn Tracking perfekt ist.")]

    public bool nurBeiPerfektemTracking = true;


    // Referenz zum AR Manager

    private ARTrackedImageManager _trackedImageManager;

    

    // Wir speichern für jedes erkannte Bild den dazugehörigen Würfel

    private Dictionary<string, GameObject> _spawnedObjects = new Dictionary<string, GameObject>();


    void Awake()

    {

        _trackedImageManager = GetComponent<ARTrackedImageManager>();

    }


    void OnEnable()

    {

        // Abonnieren der Events, wenn sich am Tracking etwas ändert

        _trackedImageManager.trackedImagesChanged += OnTrackedImagesChanged;

    }


    void OnDisable()

    {

        // Abbestellen der Events

        _trackedImageManager.trackedImagesChanged -= OnTrackedImagesChanged;

    }


    private void OnTrackedImagesChanged(ARTrackedImagesChangedEventArgs eventArgs)

    {

        // 1. Neue Bilder erkannt (ADDED)

        foreach (var trackedImage in eventArgs.added)

        {

            UpdateSpawnedObject(trackedImage);

        }


        // 2. Bestehende Bilder aktualisiert (UPDATED - Position hat sich geändert)

        foreach (var trackedImage in eventArgs.updated)

        {

            UpdateSpawnedObject(trackedImage);

        }


        // 3. Bilder verloren (REMOVED)

        foreach (var trackedImage in eventArgs.removed)

        {

            DestroySpawnedObject(trackedImage.referenceImage.name);

        }

    }


    private void UpdateSpawnedObject(ARTrackedImage trackedImage)

    {

        string imageName = trackedImage.referenceImage.name;


        // Logik für den Threshold / Genauigkeit

        // ARFoundation nutzt 'TrackingState'. 

        // 'Tracking' = Hohe Genauigkeit (entspricht > 0.6)

        // 'Limited' = Schlechte Genauigkeit / Unsicher (entspricht < 0.6)

        bool isTrackingReliable = trackedImage.trackingState == TrackingState.Tracking;


        // Wenn wir noch keinen Würfel für dieses Bild haben, erstellen wir einen

        if (!_spawnedObjects.ContainsKey(imageName))

        {

            // Erstelle den Würfel

            GameObject newObject = Instantiate(prefabToSpawn, trackedImage.transform.position, trackedImage.transform.rotation);

            

            // Mache den Würfel zu einem Kind des TrackedImage, damit er sich automatisch mitbewegt

            // (Alternativ kann man das Parenting weglassen und Position manuell setzen, Parenting ist aber performanter)

            newObject.transform.parent = trackedImage.transform;

            

            _spawnedObjects.Add(imageName, newObject);

        }


        GameObject currentObject = _spawnedObjects[imageName];


        // Sichtbarkeit steuern basierend auf dem "Threshold" (TrackingState)

        if (nurBeiPerfektemTracking)

        {

            // Zeige Würfel nur, wenn Tracking absolut sicher ist

            currentObject.SetActive(isTrackingReliable);

        }

        else

        {

            // Zeige Würfel auch bei "Limited" Tracking (etwas wackeliger)

            currentObject.SetActive(trackedImage.trackingState != TrackingState.None);

        }

    }


    private void DestroySpawnedObject(string imageName)

    {

        if (_spawnedObjects.ContainsKey(imageName))

        {

            Destroy(_spawnedObjects[imageName]);

            _spawnedObjects.Remove(imageName);

        }

    }

} 