using System.Collections;
using System.Collections.Generic;
using Oculus.Interaction;
using UnityEngine;

public class DraftBoardSnapper : MonoBehaviour
{
    [SerializeField] private RectTransform boardRect;
    [SerializeField] private RectTransform[] boardRects;
    [SerializeField] private string fallbackBoardName = "Canvas Draft Board";
    [SerializeField] private string[] fallbackBoardNames = { "Canvas Draft Board", "Canvas Final Board" };
    [SerializeField] private bool snapEnabled = true;

    [Header("Snap Tuning")]
    [SerializeField] private float snapPlaneDistance = 650f;
    [SerializeField] private float snapCatchPadding = 180f;
    [SerializeField] private float boardEdgePadding = 40f;
    [SerializeField] private float cellWidth = 0f;
    [SerializeField] private float cellHeight = 0f;
    [SerializeField] private float cellGap = 40f;
    [SerializeField] private float snapLocalZ = 0f;
    [SerializeField] private bool alignRotationToBoard = true;
    [SerializeField] private bool persistOnSnap = true;

    [Header("Fallback Release Detection")]
    [SerializeField] private float settleDelaySeconds = 0.15f;
    [SerializeField] private float movementThreshold = 0.0005f;
    [SerializeField] private float rotationThresholdDegrees = 0.05f;

    private readonly List<IInteractableView> _interactableViews = new List<IInteractableView>();
    private RectTransform _rectTransform;
    private Coroutine _snapRoutine;
    private Vector3 _lastPosition;
    private Quaternion _lastRotation;
    private float _lastMoveTime;
    private bool _hasInteractableEvents;
    private bool _snapAttemptedSinceLastMove;

    private struct SnapCandidate
    {
        public RectTransform Board;
        public Vector3 SnappedLocalPoint;
        public float Score;
    }

    public void Configure(params RectTransform[] snapTargets)
    {
        boardRects = snapTargets;
        boardRect = FirstValidBoard(snapTargets);
    }

    private void Awake()
    {
        _rectTransform = transform as RectTransform;
        _lastPosition = transform.position;
        _lastRotation = transform.rotation;
    }

    private void OnEnable()
    {
        ResolveBoardsIfNeeded();
        SubscribeInteractables();
    }

    private void OnDisable()
    {
        UnsubscribeInteractables();
    }

    private void Update()
    {
        if (!snapEnabled || _hasInteractableEvents)
            return;

        if (HasMoved())
        {
            _lastPosition = transform.position;
            _lastRotation = transform.rotation;
            _lastMoveTime = Time.unscaledTime;
            _snapAttemptedSinceLastMove = false;
            return;
        }

        if (!_snapAttemptedSinceLastMove && Time.unscaledTime - _lastMoveTime >= settleDelaySeconds)
        {
            _snapAttemptedSinceLastMove = true;
            TrySnapToBoard();
        }
    }

    private void SubscribeInteractables()
    {
        UnsubscribeInteractables();

        var components = GetComponentsInChildren<MonoBehaviour>(true);
        for (int i = 0; i < components.Length; i++)
        {
            if (components[i] is IInteractableView view)
            {
                _interactableViews.Add(view);
                view.WhenStateChanged += HandleInteractableStateChanged;
            }
        }

        _hasInteractableEvents = _interactableViews.Count > 0;
    }

    private void UnsubscribeInteractables()
    {
        for (int i = 0; i < _interactableViews.Count; i++)
            _interactableViews[i].WhenStateChanged -= HandleInteractableStateChanged;

        _interactableViews.Clear();
        _hasInteractableEvents = false;
    }

    private void HandleInteractableStateChanged(InteractableStateChangeArgs args)
    {
        if (!snapEnabled || args.PreviousState != InteractableState.Select || args.NewState == InteractableState.Select)
            return;

        if (AnyInteractableStillSelected())
            return;

        if (_snapRoutine != null)
            StopCoroutine(_snapRoutine);

        _snapRoutine = StartCoroutine(SnapAfterInteractionFrame());
    }

    private IEnumerator SnapAfterInteractionFrame()
    {
        yield return null;
        TrySnapToBoard();
        _snapRoutine = null;
    }

    private bool AnyInteractableStillSelected()
    {
        for (int i = 0; i < _interactableViews.Count; i++)
        {
            if (_interactableViews[i].State == InteractableState.Select)
                return true;
        }

        return false;
    }

    private bool TrySnapToBoard()
    {
        if (!snapEnabled || !ResolveBoardsIfNeeded())
            return false;

        bool hasCandidate = false;
        SnapCandidate bestCandidate = new SnapCandidate();
        RectTransform[] snapTargets = GetSnapTargets();

        for (int i = 0; i < snapTargets.Length; i++)
        {
            if (snapTargets[i] == null || !TryGetSnapCandidate(snapTargets[i], out SnapCandidate candidate))
                continue;

            if (!hasCandidate || candidate.Score < bestCandidate.Score)
            {
                bestCandidate = candidate;
                hasCandidate = true;
            }
        }

        if (!hasCandidate)
            return false;

        transform.position = bestCandidate.Board.TransformPoint(bestCandidate.SnappedLocalPoint);
        if (alignRotationToBoard)
            transform.rotation = bestCandidate.Board.rotation;

        _lastPosition = transform.position;
        _lastRotation = transform.rotation;
        _lastMoveTime = Time.unscaledTime;
        _snapAttemptedSinceLastMove = true;

        PersistTransform();
        return true;
    }

    private bool TryGetSnapCandidate(RectTransform targetBoard, out SnapCandidate candidate)
    {
        candidate = new SnapCandidate();

        Vector3 localPoint = targetBoard.InverseTransformPoint(transform.position);
        Rect snapRect = ExpandedRect(targetBoard.rect, snapCatchPadding);
        Vector2 boardPoint = new Vector2(localPoint.x, localPoint.y);

        if (Mathf.Abs(localPoint.z - snapLocalZ) > snapPlaneDistance || !snapRect.Contains(boardPoint))
            return false;

        Vector2 footprint = GetFootprintInBoardSpace(targetBoard);
        float halfWidth = Mathf.Max(0f, footprint.x * 0.5f);
        float halfHeight = Mathf.Max(0f, footprint.y * 0.5f);
        Rect board = targetBoard.rect;

        float minX = board.xMin + boardEdgePadding + halfWidth;
        float maxX = board.xMax - boardEdgePadding - halfWidth;
        float minY = board.yMin + boardEdgePadding + halfHeight;
        float maxY = board.yMax - boardEdgePadding - halfHeight;

        if (minX > maxX)
        {
            float centerX = board.center.x;
            minX = centerX;
            maxX = centerX;
        }

        if (minY > maxY)
        {
            float centerY = board.center.y;
            minY = centerY;
            maxY = centerY;
        }

        float resolvedCellWidth = cellWidth > 0f ? cellWidth : footprint.x + cellGap;
        float resolvedCellHeight = cellHeight > 0f ? cellHeight : footprint.y + cellGap;
        float startX = minX;
        float startY = maxY;

        float snappedX = SnapFromStart(localPoint.x, startX, resolvedCellWidth);
        float snappedY = SnapFromStart(localPoint.y, startY, -resolvedCellHeight);
        snappedX = Mathf.Clamp(snappedX, minX, maxX);
        snappedY = Mathf.Clamp(snappedY, minY, maxY);

        candidate = new SnapCandidate
        {
            Board = targetBoard,
            SnappedLocalPoint = new Vector3(snappedX, snappedY, snapLocalZ),
            Score = Mathf.Abs(localPoint.z - snapLocalZ) + Vector2.Distance(boardPoint, new Vector2(snappedX, snappedY))
        };
        return true;
    }

    private bool ResolveBoardsIfNeeded()
    {
        if (HasValidBoard(boardRects) || boardRect != null)
            return true;

        List<RectTransform> resolvedBoards = new List<RectTransform>();
        AddFallbackBoard(resolvedBoards, fallbackBoardName);

        if (fallbackBoardNames != null)
        {
            for (int i = 0; i < fallbackBoardNames.Length; i++)
                AddFallbackBoard(resolvedBoards, fallbackBoardNames[i]);
        }

        if (resolvedBoards.Count == 0)
            return false;

        boardRects = resolvedBoards.ToArray();
        boardRect = boardRects[0];
        return true;
    }

    private RectTransform[] GetSnapTargets()
    {
        List<RectTransform> snapTargets = new List<RectTransform>();

        if (boardRects != null)
        {
            for (int i = 0; i < boardRects.Length; i++)
                AddUniqueBoard(snapTargets, boardRects[i]);
        }

        AddUniqueBoard(snapTargets, boardRect);
        return snapTargets.ToArray();
    }

    private Vector2 GetFootprintInBoardSpace(RectTransform targetBoard)
    {
        if (_rectTransform == null || targetBoard == null)
            return Vector2.zero;

        Rect rect = _rectTransform.rect;
        Vector3 tileScale = transform.lossyScale;
        Vector3 boardScale = targetBoard.lossyScale;
        float boardScaleX = Mathf.Approximately(boardScale.x, 0f) ? 1f : Mathf.Abs(boardScale.x);
        float boardScaleY = Mathf.Approximately(boardScale.y, 0f) ? 1f : Mathf.Abs(boardScale.y);

        return new Vector2(
            rect.width * Mathf.Abs(tileScale.x) / boardScaleX,
            rect.height * Mathf.Abs(tileScale.y) / boardScaleY);
    }

    private static bool HasValidBoard(RectTransform[] snapTargets)
    {
        if (snapTargets == null)
            return false;

        for (int i = 0; i < snapTargets.Length; i++)
        {
            if (snapTargets[i] != null)
                return true;
        }

        return false;
    }

    private static RectTransform FirstValidBoard(RectTransform[] snapTargets)
    {
        if (snapTargets == null)
            return null;

        for (int i = 0; i < snapTargets.Length; i++)
        {
            if (snapTargets[i] != null)
                return snapTargets[i];
        }

        return null;
    }

    private static void AddUniqueBoard(List<RectTransform> snapTargets, RectTransform snapTarget)
    {
        if (snapTarget == null || snapTargets.Contains(snapTarget))
            return;

        snapTargets.Add(snapTarget);
    }

    private static void AddFallbackBoard(List<RectTransform> snapTargets, string boardName)
    {
        if (string.IsNullOrWhiteSpace(boardName))
            return;

        GameObject boardObject = GameObject.Find(boardName);
        if (boardObject == null)
            return;

        AddUniqueBoard(snapTargets, boardObject.GetComponent<RectTransform>());
    }

    private static Rect ExpandedRect(Rect rect, float padding)
    {
        rect.xMin -= padding;
        rect.xMax += padding;
        rect.yMin -= padding;
        rect.yMax += padding;
        return rect;
    }

    private static float SnapFromStart(float value, float start, float step)
    {
        if (Mathf.Approximately(step, 0f))
            return start;

        return start + Mathf.Round((value - start) / step) * step;
    }

    private bool HasMoved()
    {
        return Vector3.Distance(_lastPosition, transform.position) > movementThreshold ||
               Quaternion.Angle(_lastRotation, transform.rotation) > rotationThresholdDegrees;
    }

    private void PersistTransform()
    {
        if (!persistOnSnap)
            return;

        SavedNoteInstance noteInstance = GetComponent<SavedNoteInstance>();
        if (noteInstance == null || string.IsNullOrWhiteSpace(noteInstance.noteId))
            return;

        SavedSessionService.UpdateNoteTransform(noteInstance.noteId, transform);
    }
}
