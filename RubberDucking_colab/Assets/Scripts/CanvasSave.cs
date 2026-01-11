using System.IO;
using UnityEngine;

public class CameraCapture : MonoBehaviour
{
    public int fileCounter;
    public KeyCode screenshotKey;
    public Camera captureCamera;

    private void LateUpdate()
    {
        if (Input.GetKeyDown(screenshotKey))
        {
            Capture();
        }
    }

    public void Capture()
    {
        int width = Screen.width;
        int height = Screen.height;

        RenderTexture rt = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32);
        rt.antiAliasing = 1; // URP-safe
        rt.Create();

        RenderTexture prevRT = RenderTexture.active;
        captureCamera.targetTexture = rt;
        RenderTexture.active = rt;

        captureCamera.Render();

        Texture2D image = new Texture2D(width, height, TextureFormat.RGB24, false);
        image.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        image.Apply();

        captureCamera.targetTexture = null;
        RenderTexture.active = prevRT;

        byte[] bytes = image.EncodeToPNG();
        Destroy(image);
        rt.Release();
        Destroy(rt);

        string path = Application.dataPath + "/" + fileCounter + ".png";
        File.WriteAllBytes(path, bytes);
        Debug.Log("Saved Camera Capture to: " + path);

        fileCounter++;
    }

    public void Start()
    {
        Capture();
    }
}