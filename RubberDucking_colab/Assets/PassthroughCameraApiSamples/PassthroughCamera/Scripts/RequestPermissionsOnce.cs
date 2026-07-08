// Copyright (c) Meta Platforms, Inc. and affiliates.

using UnityEngine;
using UnityEngine.SceneManagement;

namespace PassthroughCameraSamples
{
    internal static class RequestPermissionsOnce
    {
        private static bool s_permissionsRequestedOnce = false;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void AfterSceneLoad()
        {
            // Check the currently loaded scene immediately (in case this is the first scene)
            var currentScene = SceneManager.GetActiveScene();
            if (currentScene.name != "StartScene" && !s_permissionsRequestedOnce)
            {
                s_permissionsRequestedOnce = true;
                OVRPermissionsRequester.Request(new[]
                {
                    OVRPermissionsRequester.Permission.Scene,
                    OVRPermissionsRequester.Permission.PassthroughCameraAccess
                });
            }

            // Also subscribe to future scene loads
            SceneManager.sceneLoaded += (scene, _) =>
            {
                if (scene.name != "StartScene" && !s_permissionsRequestedOnce)
                {
                    s_permissionsRequestedOnce = true;
                    OVRPermissionsRequester.Request(new[]
                    {
                        OVRPermissionsRequester.Permission.Scene,
                        OVRPermissionsRequester.Permission.PassthroughCameraAccess
                    });
                }
            };
        }
    }
}
