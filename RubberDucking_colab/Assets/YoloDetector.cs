using UnityEngine;
using Unity.Sentis;
using System.Linq;

public class YoloDetector : MonoBehaviour
{
    [Header("Model Settings")]
    public ModelAsset modelAsset;
    public Texture2D testImage;
    
    [Header("Detection Settings")]
    [Range(0.0f, 1.0f)] 
    public float confidenceThreshold = 0.6f; 

    [Header("Interaktion")]
    public GameObject objectToSpawn; 
    private GameObject currentSpawnedObject; 

    private Model runtimeModel;
    private Worker worker;
    private const int ImageSize = 640; 
    private int numClasses = 80; 
    private int numBoxes = 8400; 
    
    void Start()
    {
        if (modelAsset == null) return;
        runtimeModel = ModelLoader.Load(modelAsset);
        worker = new Worker(runtimeModel, BackendType.GPUCompute);
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            if (testImage != null) Detect(testImage);
        }
    }

    public void Detect(Texture sourceTexture)
    {
        TensorShape shape = new TensorShape(1, 3, ImageSize, ImageSize);
        using Tensor<float> inputTensor = new Tensor<float>(shape);
        TextureConverter.ToTensor(sourceTexture, inputTensor, new TextureTransform().SetDimensions(width: ImageSize, height: ImageSize));

        worker.Schedule(inputTensor);

        using Tensor<float> outputTensor = worker.PeekOutput("output0") as Tensor<float>;
        float[] data = outputTensor.DownloadToArray();
        
        ParseYoloOutput(data);
    }

    void ParseYoloOutput(float[] data)
    {
        float maxScore = 0f;
        int bestBoxIndex = -1;

        for (int i = 0; i < numBoxes; i++)
        {
            for (int c = 0; c < numClasses; c++)
            {
                float score = data[(4 + c) * numBoxes + i];
                
                if (score < confidenceThreshold) continue; 

                if (score > maxScore)
                {
                    maxScore = score;
                    bestBoxIndex = i;
                }
            }
        }

        if (bestBoxIndex != -1)
        {
            Debug.Log($"Objekt erkannt! Sicherheit: {maxScore * 100:0}%");
            
            float x = data[0 * numBoxes + bestBoxIndex];
            float y = data[1 * numBoxes + bestBoxIndex];
            
            // HIER WAR DER FEHLER: Jetzt sauber getrennt
            float normalizedX = x / ImageSize;
            float normalizedY = 1.0f - (y / ImageSize); 

            PlaceObject(normalizedX, normalizedY);
        }
    }

    void PlaceObject(float xPercent, float yPercent)
    {
        Vector3 screenPos = new Vector3(xPercent * Screen.width, yPercent * Screen.height, 0);
        Ray ray = Camera.main.ScreenPointToRay(screenPos);
        
        Vector3 finalPos;
        if (Physics.Raycast(ray, out RaycastHit hit)) finalPos = hit.point; 
        else finalPos = ray.GetPoint(1.0f); 

        if (currentSpawnedObject == null && objectToSpawn != null)
        {
            currentSpawnedObject = Instantiate(objectToSpawn, finalPos, Quaternion.identity);
        }
        else if (currentSpawnedObject != null)
        {
            currentSpawnedObject.transform.position = finalPos;
        }
    }

    private void OnDestroy()
    {
        worker?.Dispose();
    }
}