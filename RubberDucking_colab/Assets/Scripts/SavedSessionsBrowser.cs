using System;
using System.Text;
using TMPro;
using UnityEngine;
using UnityEngine.Events;
using System.Collections.Generic;
using UnityEngine.UI;
using System.Collections;

public class SavedSessionsBrowser : MonoBehaviour
{
    private enum BrowserDataMode
    {
        LocalSavedSessions = 0,
        SharedWithMe = 1,
        SharedOwnedByMe = 2
    }

    private sealed class BrowserSessionItem
    {
        public string id;
        public string title;
        public string userName;
        public string duckName;
        public string ownerDisplayName;
        public string ownerEmail;
        public string createdAtUtc;
        public string updatedAtUtc;
        public int noteCount;
        public string latestNotePreview;
        public bool isShared;
    }

    [Header("Data Source")]
    [SerializeField] private BrowserDataMode dataMode = BrowserDataMode.LocalSavedSessions;
    [SerializeField] private CloudSharingFacade cloudSharingFacade;

    [Header("List UI")]
    [SerializeField] private TMP_Text sessionsListText;
    [SerializeField] private TMP_Text selectedSessionText;
    [SerializeField] private TMP_Text selectedNotesText;
    [SerializeField] private TMP_Text headerText;
    [SerializeField] private TMP_Text helperText;

    [Header("Scrollable Slots")]
    [SerializeField] private SavedSessionListItemView[] visibleItems;
    [SerializeField] private Button scrollUpButton;
    [SerializeField] private Button scrollDownButton;
    [SerializeField] private bool showLegacyTextListWhenNoSlots = false;

    [Header("Scroll Animation")]
    [SerializeField] private bool animateScrolling = true;
    [SerializeField] private float scrollAnimationDuration = 0.2f;
    [SerializeField] private float rowSpacing = 120f;

    [Header("Selection")]
    [SerializeField] private int selectedIndex = 0;
    [SerializeField] private bool wrapSelection = true;
    [SerializeField] private int scrollOffset = 0;

    [Header("Session Creation")]
    [SerializeField] private string newSessionTitle = "New session";
    [SerializeField] private bool clearPreviewWhenStartingNewSession = true;

    [Header("Restore")]
    [SerializeField] private GameObject notePrefab;
    [SerializeField] private Transform notesParent;
    [SerializeField] private string noteTextChildName = "Transcription1";
    [SerializeField] private bool clearExistingNotesOnResume = true;

    [Header("Events")]
    [SerializeField] private UnityEvent onSessionResumed;
    [SerializeField] private UnityEvent onSessionStarted;
    [SerializeField] private UnityEvent onSelectionChanged;

    private SavedSessionIndex _index;
    private SavedSessionData _selectedSession;
    private SharedSessionRecord _selectedSharedSession;
    private List<BrowserSessionItem> _browserItems = new List<BrowserSessionItem>();
    private Vector2[] _slotBasePositions;
    private bool _isAnimatingScroll;

    public int SessionCount => _browserItems?.Count ?? 0;
    public SavedSessionData SelectedSession => _selectedSession;
    public int VisibleItemCount => visibleItems == null ? 0 : visibleItems.Length;

    private void OnEnable()
    {
        CacheSlotBasePositions();
        Refresh();
    }

    public void Refresh()
    {
        _ = RefreshAsync();
    }

    public async void RefreshSharedWithMe()
    {
        dataMode = BrowserDataMode.SharedWithMe;
        await RefreshAsync();
    }

    public async void RefreshSharedOwnedByMe()
    {
        dataMode = BrowserDataMode.SharedOwnedByMe;
        await RefreshAsync();
    }

    public async void RefreshLocalSessions()
    {
        dataMode = BrowserDataMode.LocalSavedSessions;
        await RefreshAsync();
    }

    private async System.Threading.Tasks.Task RefreshAsync()
    {
        await LoadBrowserItemsAsync();
        var hasSessions = _browserItems != null && _browserItems.Count > 0;

        if (!hasSessions)
        {
            selectedIndex = 0;
            scrollOffset = 0;
            _selectedSession = null;
            _selectedSharedSession = null;
            SetText(headerText, BuildHeaderTitle(0));
            SetText(helperText, BuildEmptyHelperText());
            SetText(sessionsListText, showLegacyTextListWhenNoSlots ? "No sessions yet." : string.Empty);
            SetText(selectedSessionText, "No session selected.");
            SetText(selectedNotesText, string.Empty);
            ClearVisibleItems();
            ResetVisibleItemPositions();
            UpdateScrollButtons();
            return;
        }

        selectedIndex = Mathf.Clamp(selectedIndex, 0, _browserItems.Count - 1);
        ClampScrollOffset();
        EnsureSelectionVisible();
        await LoadSelectedAsync();
        RenderList();
        RenderSelection();
        UpdateScrollButtons();
        onSelectionChanged?.Invoke();
    }

    public void StartNewSession()
    {
        if (dataMode != BrowserDataMode.LocalSavedSessions)
        {
            Debug.LogWarning("SavedSessionsBrowser: StartNewSession is only supported in local mode.");
            return;
        }

        SavedSessionService.StartNewSession(newSessionTitle);
        _index = SavedSessionService.LoadIndex();
        selectedIndex = 0;
        scrollOffset = 0;
        Refresh();

        if (clearPreviewWhenStartingNewSession)
            ClearSpawnedNotes();

        Debug.Log("SavedSessionsBrowser: Started a new session.");
        onSessionStarted?.Invoke();
        onSelectionChanged?.Invoke();
    }

    public void SelectNext()
    {
        if (_browserItems == null || _browserItems.Count == 0)
            return;

        if (wrapSelection)
            selectedIndex = (selectedIndex + 1) % _browserItems.Count;
        else
            selectedIndex = Mathf.Min(selectedIndex + 1, _browserItems.Count - 1);

        EnsureSelectionVisible();
        _ = ReloadSelectionAndRenderAsync();
    }

    public void SelectPrevious()
    {
        if (_browserItems == null || _browserItems.Count == 0)
            return;

        if (wrapSelection)
            selectedIndex = (selectedIndex - 1 + _browserItems.Count) % _browserItems.Count;
        else
            selectedIndex = Mathf.Max(selectedIndex - 1, 0);

        EnsureSelectionVisible();
        _ = ReloadSelectionAndRenderAsync();
    }

    public void SelectByIndex(int index)
    {
        if (_browserItems == null || _browserItems.Count == 0)
            return;

        selectedIndex = Mathf.Clamp(index, 0, _browserItems.Count - 1);
        EnsureSelectionVisible();
        _ = ReloadSelectionAndRenderAsync();
    }

    public async System.Threading.Tasks.Task<bool> SelectAndResumeByIndexAsync(int index)
    {
        if (_browserItems == null || _browserItems.Count == 0)
            return false;

        selectedIndex = Mathf.Clamp(index, 0, _browserItems.Count - 1);
        EnsureSelectionVisible();
        await ReloadSelectionAndRenderAsync();
        ResumeSelected();
        return _selectedSession != null || _selectedSharedSession != null;
    }

    public async void SelectAndResumeByIndex(int index)
    {
        await SelectAndResumeByIndexAsync(index);
    }

    public void ScrollUp()
    {
        if (_browserItems == null || _browserItems.Count == 0 || _isAnimatingScroll)
            return;

        var targetOffset = Mathf.Max(0, scrollOffset - 1);
        if (targetOffset == scrollOffset)
            return;

        if (ShouldAnimateSlots())
            StartCoroutine(AnimateScrollToOffset(targetOffset));
        else
        {
            scrollOffset = targetOffset;
            RenderList();
            UpdateScrollButtons();
        }
    }

    public void ScrollDown()
    {
        if (_browserItems == null || _browserItems.Count == 0 || _isAnimatingScroll)
            return;

        var targetOffset = Mathf.Min(GetMaxScrollOffset(), scrollOffset + 1);
        if (targetOffset == scrollOffset)
            return;

        if (ShouldAnimateSlots())
            StartCoroutine(AnimateScrollToOffset(targetOffset));
        else
        {
            scrollOffset = targetOffset;
            RenderList();
            UpdateScrollButtons();
        }
    }

    public void ResumeSelected()
    {
        if (_selectedSession != null)
        {
            if (SavedSessionService.ResumeSession(_selectedSession.id))
            {
                RestoreNotes(_selectedSession);
                Debug.Log($"SavedSessionsBrowser: Resumed {_selectedSession.title}");
                onSessionResumed?.Invoke();
            }
            return;
        }

        if (_selectedSharedSession != null)
        {
            var local = SharedSessionMapper.ToLocalSession(_selectedSharedSession);
            RestoreNotes(local);
            Debug.Log($"SavedSessionsBrowser: Restored shared session {_selectedSharedSession.title}");
            onSessionResumed?.Invoke();
        }
    }

    private async System.Threading.Tasks.Task ReloadSelectionAndRenderAsync()
    {
        await LoadSelectedAsync();
        RenderList();
        RenderSelection();
        UpdateScrollButtons();
        onSelectionChanged?.Invoke();
    }

    private async System.Threading.Tasks.Task LoadBrowserItemsAsync()
    {
        _selectedSession = null;
        _selectedSharedSession = null;
        _browserItems = new List<BrowserSessionItem>();

        if (dataMode == BrowserDataMode.LocalSavedSessions)
        {
            _index = SavedSessionService.LoadIndex();
            if (_index?.sessions == null)
                return;

            foreach (var session in _index.sessions)
            {
                _browserItems.Add(new BrowserSessionItem
                {
                    id = session.id,
                    title = session.title,
                    userName = session.userName,
                    duckName = session.duckName,
                    createdAtUtc = session.createdAtUtc,
                    updatedAtUtc = session.updatedAtUtc,
                    noteCount = session.noteCount,
                    latestNotePreview = session.latestNotePreview,
                    isShared = false
                });
            }

            return;
        }

        if (cloudSharingFacade == null)
        {
            Debug.LogWarning("SavedSessionsBrowser: CloudSharingFacade is required for shared session modes.");
            return;
        }

        IReadOnlyList<SharedSessionSummaryRecord> summaries = dataMode == BrowserDataMode.SharedOwnedByMe
            ? await cloudSharingFacade.LoadOwnedByMeAsync()
            : await cloudSharingFacade.LoadSharedWithMeAsync();

        foreach (var session in summaries)
        {
            _browserItems.Add(new BrowserSessionItem
            {
                id = session.id,
                title = session.title,
                ownerDisplayName = session.ownerDisplayName,
                ownerEmail = session.ownerEmail,
                updatedAtUtc = session.updatedAtUtc,
                noteCount = session.noteCount,
                latestNotePreview = session.latestNotePreview,
                isShared = true
            });
        }
    }

    private async System.Threading.Tasks.Task LoadSelectedAsync()
    {
        _selectedSession = null;
        _selectedSharedSession = null;

        if (_browserItems == null || _browserItems.Count == 0)
            return;

        var summary = _browserItems[selectedIndex];
        if (!summary.isShared)
        {
            _selectedSession = SavedSessionService.LoadSession(summary.id);
            return;
        }

        if (cloudSharingFacade == null)
            return;

        _selectedSharedSession = await cloudSharingFacade.LoadSharedSessionAsync(summary.id);
    }

    private void RenderList()
    {
        SetText(headerText, BuildHeaderTitle(_browserItems.Count));
        SetText(helperText, BuildHelperText());

        RenderVisibleItems();
        RenderLegacyTextListIfNeeded();
    }

    private void RenderVisibleItems()
    {
        if (visibleItems == null || visibleItems.Length == 0)
            return;

        if (_slotBasePositions == null || _slotBasePositions.Length != visibleItems.Length)
            CacheSlotBasePositions();
        ResetVisibleItemPositions();

        for (int i = 0; i < visibleItems.Length; i++)
        {
            var slot = visibleItems[i];
            if (slot == null)
                continue;

            var sessionIndex = scrollOffset + i;
            if (sessionIndex >= 0 && sessionIndex < _browserItems.Count)
            {
                var summary = _browserItems[sessionIndex];
                slot.Bind(this, summary.title, BuildOwnerLabel(summary), summary.noteCount, summary.updatedAtUtc, summary.latestNotePreview, sessionIndex, sessionIndex == selectedIndex);
            }
            else
            {
                slot.Clear();
            }
        }
    }

    private void RenderLegacyTextListIfNeeded()
    {
        if (sessionsListText == null)
            return;

        if (!showLegacyTextListWhenNoSlots || visibleItems == null || visibleItems.Length > 0)
        {
            sessionsListText.text = string.Empty;
            return;
        }

        var sb = new StringBuilder();
        for (int i = 0; i < _browserItems.Count; i++)
        {
            var session = _browserItems[i];
            var marker = i == selectedIndex ? ">" : "-";
            var label = string.IsNullOrWhiteSpace(session.title) ? "Untitled session" : session.title;
            var owner = BuildOwnerLabel(session);

            sb.AppendLine($"{marker} {i + 1}. {label}");
            sb.AppendLine($"   {session.noteCount} notes · {FormatDate(session.updatedAtUtc)}");
            if (!string.IsNullOrWhiteSpace(owner)) sb.AppendLine($"   {owner}");
            if (!string.IsNullOrWhiteSpace(session.latestNotePreview)) sb.AppendLine($"   \"{session.latestNotePreview}\"");
            if (i < _browserItems.Count - 1) sb.AppendLine();
        }

        sessionsListText.text = sb.ToString().TrimEnd();
    }

    private void RenderSelection()
    {
        if (_selectedSession == null && _selectedSharedSession == null)
        {
            SetText(selectedSessionText, "No session selected.");
            SetText(selectedNotesText, string.Empty);
            return;
        }

        if (_selectedSession != null)
        {
            RenderSavedSessionSelection(_selectedSession);
            return;
        }

        RenderSharedSessionSelection(_selectedSharedSession);
    }

    private void RenderSavedSessionSelection(SavedSessionData session)
    {
        var title = string.IsNullOrWhiteSpace(session.title) ? "Untitled session" : session.title;
        var summary = new StringBuilder();
        summary.AppendLine(title);
        summary.AppendLine($"User: {Fallback(session.userName)}");
        summary.AppendLine($"Duck: {Fallback(session.duckName)}");
        summary.AppendLine($"Created: {FormatDate(session.createdAtUtc)}");
        summary.AppendLine($"Updated: {FormatDate(session.updatedAtUtc)}");
        summary.AppendLine($"Notes: {session.notes.Count}");
        SetText(selectedSessionText, summary.ToString().TrimEnd());
        SetText(selectedNotesText, BuildNotesText(session.notes));
    }

    private void RenderSharedSessionSelection(SharedSessionRecord session)
    {
        var title = string.IsNullOrWhiteSpace(session.title) ? "Untitled session" : session.title;
        var summary = new StringBuilder();
        summary.AppendLine(title);
        summary.AppendLine($"Owner: {Fallback(session.ownerDisplayName)}");
        summary.AppendLine($"Owner Email: {Fallback(session.ownerEmail)}");
        summary.AppendLine($"User: {Fallback(session.userName)}");
        summary.AppendLine($"Duck: {Fallback(session.duckName)}");
        summary.AppendLine($"Created: {FormatDate(session.createdAtUtc)}");
        summary.AppendLine($"Updated: {FormatDate(session.updatedAtUtc)}");
        summary.AppendLine($"Notes: {session.notes.Count}");
        SetText(selectedSessionText, summary.ToString().TrimEnd());
        SetText(selectedNotesText, BuildSharedNotesText(session.notes));
    }

    private static string BuildNotesText(List<SavedTranscriptNote> notesData)
    {
        var notes = new StringBuilder();
        if (notesData == null || notesData.Count == 0)
        {
            notes.Append("No notes saved in this session yet.");
        }
        else
        {
            for (int i = 0; i < notesData.Count; i++)
            {
                var note = notesData[i];
                notes.AppendLine($"[{i + 1}] {FormatDate(note.createdAtUtc)}");
                notes.AppendLine(note.text);
                if (i < notesData.Count - 1)
                    notes.AppendLine().AppendLine();
            }
        }
        return notes.ToString().TrimEnd();
    }

    private static string BuildSharedNotesText(List<SharedTranscriptNoteData> notesData)
    {
        var notes = new StringBuilder();
        if (notesData == null || notesData.Count == 0)
        {
            notes.Append("No notes saved in this session yet.");
        }
        else
        {
            for (int i = 0; i < notesData.Count; i++)
            {
                var note = notesData[i];
                notes.AppendLine($"[{i + 1}] {FormatDate(note.createdAtUtc)}");
                notes.AppendLine(note.text);
                if (i < notesData.Count - 1)
                    notes.AppendLine().AppendLine();
            }
        }
        return notes.ToString().TrimEnd();
    }

    private string BuildHelperText()
    {
        if (_browserItems == null || _browserItems.Count == 0)
            return BuildEmptyHelperText();

        var top = Mathf.Min(scrollOffset + 1, _browserItems.Count);
        var bottom = Mathf.Min(scrollOffset + Mathf.Max(VisibleItemCount, 1), _browserItems.Count);

        return (_selectedSession == null && _selectedSharedSession == null)
            ? "Choose a session to inspect it."
            : $"Showing {top}-{bottom} of {_browserItems.Count}. Selected: {selectedIndex + 1}.";
    }

    private string BuildEmptyHelperText()
    {
        return dataMode switch
        {
            BrowserDataMode.SharedWithMe => "No shared sessions yet.",
            BrowserDataMode.SharedOwnedByMe => "No cloud sessions uploaded yet.",
            _ => "No saved sessions yet. Start a new one to begin."
        };
    }

    private string BuildHeaderTitle(int count)
    {
        return dataMode switch
        {
            BrowserDataMode.SharedWithMe => $"Shared With Me ({count})",
            BrowserDataMode.SharedOwnedByMe => $"My Shared Sessions ({count})",
            _ => $"Saved Sessions ({count})"
        };
    }

    private static string BuildOwnerLabel(BrowserSessionItem item)
    {
        if (item == null)
            return string.Empty;

        if (item.isShared)
        {
            if (!string.IsNullOrWhiteSpace(item.ownerDisplayName) && !string.IsNullOrWhiteSpace(item.ownerEmail))
                return $"Owner: {item.ownerDisplayName} ({item.ownerEmail})";
            if (!string.IsNullOrWhiteSpace(item.ownerDisplayName))
                return $"Owner: {item.ownerDisplayName}";
            if (!string.IsNullOrWhiteSpace(item.ownerEmail))
                return $"Owner: {item.ownerEmail}";
            return string.Empty;
        }

        var user = string.IsNullOrWhiteSpace(item.userName) ? null : item.userName.Trim();
        var duck = string.IsNullOrWhiteSpace(item.duckName) ? null : item.duckName.Trim();

        if (!string.IsNullOrEmpty(user) && !string.IsNullOrEmpty(duck))
            return $"Owner: {user}, Duck Name: {duck}";
        return user ?? duck ?? string.Empty;
    }

    private void RestoreNotes(SavedSessionData session)
    {
        if (session == null || notePrefab == null || notesParent == null)
            return;

        ClearSpawnedNotes();

        foreach (var note in session.notes)
        {
            var instance = Instantiate(notePrefab, notesParent);
            instance.name = notePrefab.name;

            var noteInstance = instance.GetComponent<SavedNoteInstance>();
            if (noteInstance == null)
                noteInstance = instance.AddComponent<SavedNoteInstance>();

            var tmp = FindNoteText(instance.transform);
            if (tmp != null)
                noteInstance.SetNoteTextTarget(tmp);

            noteInstance.Apply(note);

            if (note.siblingIndex >= 0 && note.siblingIndex < notesParent.childCount)
                instance.transform.SetSiblingIndex(note.siblingIndex);
        }
    }

    private void EnsureSelectionVisible()
    {
        if (_browserItems == null || _browserItems.Count == 0)
        {
            scrollOffset = 0;
            return;
        }

        var visibleCount = Mathf.Max(VisibleItemCount, 1);
        if (selectedIndex < scrollOffset)
            scrollOffset = selectedIndex;
        else if (selectedIndex >= scrollOffset + visibleCount)
            scrollOffset = selectedIndex - visibleCount + 1;

        ClampScrollOffset();
    }

    private bool ShouldAnimateSlots()
    {
        return animateScrolling && visibleItems != null && visibleItems.Length > 0 && rowSpacing > 0.001f && gameObject.activeInHierarchy;
    }

    private IEnumerator AnimateScrollToOffset(int targetOffset)
    {
        _isAnimatingScroll = true;
        UpdateScrollButtons();
        CacheSlotBasePositions();

        bool scrollingDown = targetOffset > scrollOffset;
        int count = visibleItems.Length;
        if (count == 0)
        {
            _isAnimatingScroll = false;
            yield break;
        }

        SavedSessionListItemView outgoingItem = scrollingDown ? visibleItems[0] : visibleItems[count - 1];
        if (outgoingItem != null)
            outgoingItem.gameObject.SetActive(false);

        var animatedItems = new List<SavedSessionListItemView>();
        var startPositions = new List<Vector2>();
        var targetPositions = new List<Vector2>();

        if (scrollingDown)
        {
            for (int i = 1; i < count; i++)
            {
                var item = visibleItems[i];
                if (item == null)
                    continue;

                animatedItems.Add(item);
                startPositions.Add(_slotBasePositions[i]);
                targetPositions.Add(_slotBasePositions[i - 1]);
            }
        }
        else
        {
            for (int i = 0; i < count - 1; i++)
            {
                var item = visibleItems[i];
                if (item == null)
                    continue;

                animatedItems.Add(item);
                startPositions.Add(_slotBasePositions[i]);
                targetPositions.Add(_slotBasePositions[i + 1]);
            }
        }

        float duration = Mathf.Max(0.01f, scrollAnimationDuration);
        float elapsed = 0f;

        while (elapsed < duration)
        {
            elapsed += Time.unscaledDeltaTime;
            float t = Mathf.Clamp01(elapsed / duration);
            float eased = Mathf.SmoothStep(0f, 1f, t);

            for (int i = 0; i < animatedItems.Count; i++)
            {
                var item = animatedItems[i];
                if (item == null || !item.gameObject.activeSelf)
                    continue;

                item.SetAnchoredPosition(Vector2.Lerp(startPositions[i], targetPositions[i], eased));
            }

            yield return null;
        }

        scrollOffset = targetOffset;
        RotateVisibleItems(scrollingDown);
        ResetVisibleItemPositions();
        RefreshVisibleItemBindings();

        SavedSessionListItemView enteringItem = scrollingDown ? visibleItems[count - 1] : visibleItems[0];
        if (enteringItem != null)
            enteringItem.gameObject.SetActive(true);

        RenderLegacyTextListIfNeeded();
        _ = LoadSelectedAsync();
        RenderSelection();
        _isAnimatingScroll = false;
        UpdateScrollButtons();
    }

    private void CacheSlotBasePositions()
    {
        if (visibleItems == null)
        {
            _slotBasePositions = Array.Empty<Vector2>();
            return;
        }

        if (_slotBasePositions == null || _slotBasePositions.Length != visibleItems.Length)
            _slotBasePositions = new Vector2[visibleItems.Length];

        for (int i = 0; i < visibleItems.Length; i++)
        {
            var item = visibleItems[i];
            if (item != null)
                _slotBasePositions[i] = item.GetAnchoredPosition();
        }
    }

    private void RotateVisibleItems(bool scrollingDown)
    {
        if (visibleItems == null || visibleItems.Length <= 1)
            return;

        if (scrollingDown)
        {
            var first = visibleItems[0];
            for (int i = 0; i < visibleItems.Length - 1; i++)
                visibleItems[i] = visibleItems[i + 1];
            visibleItems[visibleItems.Length - 1] = first;
        }
        else
        {
            var last = visibleItems[visibleItems.Length - 1];
            for (int i = visibleItems.Length - 1; i > 0; i--)
                visibleItems[i] = visibleItems[i - 1];
            visibleItems[0] = last;
        }
    }

    private void RefreshVisibleItemBindings()
    {
        if (visibleItems == null || _browserItems == null)
            return;

        for (int i = 0; i < visibleItems.Length; i++)
        {
            var item = visibleItems[i];
            if (item == null)
                continue;

            int sessionIndex = scrollOffset + i;
            if (sessionIndex >= 0 && sessionIndex < _browserItems.Count)
            {
                var summary = _browserItems[sessionIndex];
                item.Bind(this, summary.title, BuildOwnerLabel(summary), summary.noteCount, summary.updatedAtUtc, summary.latestNotePreview, sessionIndex, sessionIndex == selectedIndex);
            }
            else
            {
                item.Clear();
            }
        }
    }

    private void ResetVisibleItemPositions()
    {
        if (visibleItems == null || _slotBasePositions == null)
            return;

        for (int i = 0; i < visibleItems.Length; i++)
        {
            var item = visibleItems[i];
            if (item != null && i < _slotBasePositions.Length)
                item.SetAnchoredPosition(_slotBasePositions[i]);
        }
    }

    private void ClampScrollOffset()
    {
        scrollOffset = Mathf.Clamp(scrollOffset, 0, GetMaxScrollOffset());
    }

    private int GetMaxScrollOffset()
    {
        if (_browserItems == null || _browserItems.Count == 0)
            return 0;

        var visibleCount = Mathf.Max(VisibleItemCount, 1);
        return Mathf.Max(0, _browserItems.Count - visibleCount);
    }

    private void ClearVisibleItems()
    {
        if (visibleItems == null)
            return;

        foreach (var item in visibleItems)
        {
            if (item != null)
                item.Clear();
        }
    }

    private void UpdateScrollButtons()
    {
        var canScroll = _browserItems != null && _browserItems.Count > Mathf.Max(VisibleItemCount, 1);

        if (scrollUpButton != null)
            scrollUpButton.interactable = canScroll && scrollOffset > 0;

        if (scrollDownButton != null)
            scrollDownButton.interactable = canScroll && scrollOffset < GetMaxScrollOffset();
    }

    private void ClearSpawnedNotes()
    {
        if (!clearExistingNotesOnResume || notesParent == null)
            return;

        var toDestroy = new List<GameObject>();
        for (int i = 0; i < notesParent.childCount; i++)
            toDestroy.Add(notesParent.GetChild(i).gameObject);

        foreach (var go in toDestroy)
            Destroy(go);
    }

    private TMP_Text FindNoteText(Transform root)
    {
        if (root == null)
            return null;

        if (!string.IsNullOrWhiteSpace(noteTextChildName))
        {
            var child = root.Find(noteTextChildName);
            if (child != null)
                return child.GetComponent<TMP_Text>();
        }

        return root.GetComponentInChildren<TMP_Text>(true);
    }

    private static string Fallback(string value)
    {
        return string.IsNullOrWhiteSpace(value) ? "—" : value;
    }

    private static string FormatDate(string isoUtc)
    {
        if (DateTime.TryParse(isoUtc, out var dt))
            return dt.ToLocalTime().ToString("dd MMM yyyy, HH:mm");
        return isoUtc;
    }

    private static void SetText(TMP_Text target, string value)
    {
        if (target != null)
            target.text = value;
    }
}
