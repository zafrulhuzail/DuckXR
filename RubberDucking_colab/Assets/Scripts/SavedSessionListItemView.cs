using TMPro;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;

public class SavedSessionListItemView : MonoBehaviour
{
    private RectTransform _rectTransform;
    [Header("UI")]
    [SerializeField] private TMP_Text titleText;
    [SerializeField] private TMP_Text metaText;
    [SerializeField] private TMP_Text previewText;
    [SerializeField] private GameObject selectedState;
    [SerializeField] private GameObject unselectedState;
    [SerializeField] private Image backgroundImage;

    [Header("Colors")]
    [SerializeField] private Color selectedBackgroundColor = new Color(0.22f, 0.18f, 0.16f, 0.95f);
    [SerializeField] private Color unselectedBackgroundColor = new Color(1f, 1f, 1f, 0.6f);
    [SerializeField] private Color selectedTextColor = Color.white;
    [SerializeField] private Color unselectedTextColor = new Color(0.18f, 0.16f, 0.2f, 1f);
    [SerializeField] private Color selectedMetaColor = new Color(1f, 0.94f, 0.84f, 1f);
    [SerializeField] private Color unselectedMetaColor = new Color(0.39f, 0.35f, 0.42f, 1f);

    [Header("Interaction")]
    [SerializeField] private Button button;
    [SerializeField] private UnityEvent onClicked;

    private SavedSessionsBrowser _browser;
    private int _sessionIndex = -1;

    private void Awake()
    {
        _rectTransform = transform as RectTransform;

        if (button != null)
        {
            button.onClick.RemoveListener(HandleClick);
            button.onClick.AddListener(HandleClick);
        }
    }

    public void Bind(SavedSessionsBrowser browser, SavedSessionSummary summary, int sessionIndex, bool selected)
    {
        _browser = browser;
        _sessionIndex = sessionIndex;

        gameObject.SetActive(true);

        var title = string.IsNullOrWhiteSpace(summary.title) ? "Untitled session" : summary.title;
        var owner = BuildOwner(summary.userName, summary.duckName);
        var meta = $"{summary.noteCount} notes · {FormatDate(summary.updatedAtUtc)}";

        SetText(titleText, title);
        SetText(metaText, string.IsNullOrWhiteSpace(owner) ? meta : owner + "\n" + meta);
        SetText(previewText, string.IsNullOrWhiteSpace(summary.latestNotePreview) ? "No saved notes yet." : summary.latestNotePreview);

        SetSelected(selected);
    }

    public void Clear()
    {
        _browser = null;
        _sessionIndex = -1;
        gameObject.SetActive(false);
    }

    public void SetSelected(bool selected)
    {
        if (selectedState != null)
            selectedState.SetActive(selected);
        if (unselectedState != null)
            unselectedState.SetActive(!selected);
        if (backgroundImage != null)
            backgroundImage.color = selected ? selectedBackgroundColor : unselectedBackgroundColor;

        SetColor(titleText, selected ? selectedTextColor : unselectedTextColor);
        SetColor(metaText, selected ? selectedMetaColor : unselectedMetaColor);
        SetColor(previewText, selected ? selectedMetaColor : unselectedMetaColor);
    }

    public RectTransform RectTransform
    {
        get
        {
            if (_rectTransform == null)
                _rectTransform = transform as RectTransform;
            return _rectTransform;
        }
    }

    public void SetAnchoredPosition(Vector2 position)
    {
        if (RectTransform != null)
            RectTransform.anchoredPosition = position;
    }

    public Vector2 GetAnchoredPosition()
    {
        return RectTransform != null ? RectTransform.anchoredPosition : Vector2.zero;
    }

    public void TriggerSelect()
    {
        HandleClick();
    }

    private void HandleClick()
    {
        if (_browser != null && _sessionIndex >= 0)
            _browser.SelectByIndex(_sessionIndex);

        onClicked?.Invoke();
    }

    private static string BuildOwner(string userName, string duckName)
    {
        var user = string.IsNullOrWhiteSpace(userName) ? null : userName.Trim();
        var duck = string.IsNullOrWhiteSpace(duckName) ? null : duckName.Trim();

        if (!string.IsNullOrEmpty(user) && !string.IsNullOrEmpty(duck))
            return $"{user} + {duck}";

        return user ?? duck ?? string.Empty;
    }

    private static string FormatDate(string isoUtc)
    {
        if (System.DateTime.TryParse(isoUtc, out var dt))
            return dt.ToLocalTime().ToString("dd MMM yyyy, HH:mm");
        return isoUtc;
    }

    private static void SetText(TMP_Text target, string value)
    {
        if (target != null)
            target.text = value;
    }

    private static void SetColor(TMP_Text target, Color color)
    {
        if (target != null)
            target.color = color;
    }
}
