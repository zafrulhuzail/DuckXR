using UnityEngine;

public class SavedNoteAutosave : MonoBehaviour
{
    [SerializeField] private float saveDelaySeconds = 0.4f;
    [SerializeField] private float positionThreshold = 0.0005f;
    [SerializeField] private float rotationThreshold = 0.05f;

    private SavedNoteInstance _savedNoteInstance;
    private Vector3 _lastPosition;
    private Vector3 _lastRotation;
    private float _lastChangeTime;
    private bool _pendingSave;
    private bool _autoSave = false;

    private void Awake()
    {
        _savedNoteInstance = GetComponent<SavedNoteInstance>();
        _lastPosition = transform.localPosition;
        _lastRotation = transform.localEulerAngles;
    }

    private void Update()
    {
        if (_savedNoteInstance == null || string.IsNullOrWhiteSpace(_savedNoteInstance.noteId))
            return;

        if (HasTransformChanged())
        {
            _lastPosition = transform.localPosition;
            _lastRotation = transform.localEulerAngles;
            _lastChangeTime = Time.unscaledTime;
            _pendingSave = true;
        }

        if (_pendingSave && Time.unscaledTime - _lastChangeTime >= saveDelaySeconds)
        {
            _pendingSave = false;
            SavedSessionService.UpdateNoteTransform(_savedNoteInstance.noteId, transform);
        }
    }

    private bool HasTransformChanged()
    {
        return Vector3.Distance(_lastPosition, transform.localPosition) > positionThreshold ||
               Vector3.Distance(_lastRotation, transform.localEulerAngles) > rotationThreshold;
    }
}