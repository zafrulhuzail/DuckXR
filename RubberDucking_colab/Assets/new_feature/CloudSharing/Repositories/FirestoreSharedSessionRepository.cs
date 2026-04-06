using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Firebase.Firestore;
using UnityEngine;

public class FirestoreSharedSessionRepository : ISharedSessionRepository
{
    private const string CollectionName = "sharedSessions";

    private readonly FirebaseFirestore _firestore;

    public FirestoreSharedSessionRepository()
    {
        _firestore = FirebaseFirestore.DefaultInstance;
    }

    public async Task<SharedSessionRecord> UpsertOwnedSessionAsync(SharedSessionRecord session, CloudAuthUser owner)
    {
        if (session == null)
            throw new ArgumentNullException(nameof(session));

        if (owner == null || !owner.isAuthenticated)
            throw new InvalidOperationException("Owner must be authenticated before uploading shared sessions.");

        session.ownerUserId = owner.userId;
        session.ownerEmail = NormalizeEmail(owner.email);
        session.ownerDisplayName = owner.displayName ?? string.Empty;
        session.updatedAtUtc = DateTime.UtcNow.ToString("o");

        if (string.IsNullOrWhiteSpace(session.id))
            session.id = Guid.NewGuid().ToString("N");

        if (string.IsNullOrWhiteSpace(session.createdAtUtc))
            session.createdAtUtc = session.updatedAtUtc;

        session.allowedUserIds ??= new List<string>();
        session.invitedEmails ??= new List<string>();
        session.notes ??= new List<SharedTranscriptNoteData>();

        if (!session.allowedUserIds.Contains(owner.userId))
            session.allowedUserIds.Add(owner.userId);

        await GetCollection().Document(session.id).SetAsync(ToFirestoreMap(session), SetOptions.MergeAll);
        Debug.Log($"FirestoreSharedSessionRepository: Upserted shared session {session.id}");
        return session;
    }

    public async Task<SharedSessionRecord> GetSharedSessionAsync(string sessionId, CloudAuthUser requester)
    {
        if (string.IsNullOrWhiteSpace(sessionId) || requester == null || !requester.isAuthenticated)
            return null;

        var snapshot = await GetCollection().Document(sessionId).GetSnapshotAsync();
        if (!snapshot.Exists)
            return null;

        var session = FromSnapshot(snapshot);
        return CanRead(session, requester) ? session : null;
    }

    public async Task<IReadOnlyList<SharedSessionSummaryRecord>> GetSharedWithMeAsync(CloudAuthUser requester)
    {
        if (requester == null || !requester.isAuthenticated)
            return Array.Empty<SharedSessionSummaryRecord>();

        var snapshot = await GetCollection().GetSnapshotAsync();
        var summaries = snapshot.Documents
            .Select(FromSnapshot)
            .Where(s => s != null && s.ownerUserId != requester.userId && CanRead(s, requester))
            .Select(SharedSessionMapper.ToSummary)
            .OrderByDescending(s => s.updatedAtUtc)
            .ToList();

        return summaries;
    }

    public async Task<IReadOnlyList<SharedSessionSummaryRecord>> GetOwnedByMeAsync(CloudAuthUser requester)
    {
        if (requester == null || !requester.isAuthenticated)
            return Array.Empty<SharedSessionSummaryRecord>();

        Query query = GetCollection().WhereEqualTo("ownerUserId", requester.userId);
        var snapshot = await query.GetSnapshotAsync();
        var summaries = snapshot.Documents
            .Select(FromSnapshot)
            .Where(s => s != null)
            .Select(SharedSessionMapper.ToSummary)
            .OrderByDescending(s => s.updatedAtUtc)
            .ToList();

        return summaries;
    }

    public async Task<bool> GrantInviteByEmailAsync(string sessionId, string invitedEmail, CloudAuthUser owner)
    {
        invitedEmail = NormalizeEmail(invitedEmail);
        if (string.IsNullOrWhiteSpace(sessionId) || string.IsNullOrWhiteSpace(invitedEmail) || owner == null || !owner.isAuthenticated)
            return false;

        var session = await GetSharedSessionAsync(sessionId, owner);
        if (session == null || session.ownerUserId != owner.userId)
            return false;

        if (!session.invitedEmails.Contains(invitedEmail))
            session.invitedEmails.Add(invitedEmail);

        session.updatedAtUtc = DateTime.UtcNow.ToString("o");
        await GetCollection().Document(session.id).SetAsync(ToFirestoreMap(session), SetOptions.MergeAll);
        return true;
    }

    private CollectionReference GetCollection() => _firestore.Collection(CollectionName);

    private static Dictionary<string, object> ToFirestoreMap(SharedSessionRecord session)
    {
        return new Dictionary<string, object>
        {
            { "id", session.id ?? string.Empty },
            { "ownerUserId", session.ownerUserId ?? string.Empty },
            { "ownerEmail", NormalizeEmail(session.ownerEmail) },
            { "ownerDisplayName", session.ownerDisplayName ?? string.Empty },
            { "title", session.title ?? string.Empty },
            { "userName", session.userName ?? string.Empty },
            { "duckName", session.duckName ?? string.Empty },
            { "createdAtUtc", session.createdAtUtc ?? string.Empty },
            { "updatedAtUtc", session.updatedAtUtc ?? string.Empty },
            { "visibility", (int)session.visibility },
            { "allowedUserIds", session.allowedUserIds ?? new List<string>() },
            { "invitedEmails", (session.invitedEmails ?? new List<string>()).Select(NormalizeEmail).ToList() },
            { "notes", ToFirestoreNotes(session.notes) }
        };
    }

    private static List<Dictionary<string, object>> ToFirestoreNotes(List<SharedTranscriptNoteData> notes)
    {
        var output = new List<Dictionary<string, object>>();
        if (notes == null)
            return output;

        foreach (var note in notes)
        {
            output.Add(new Dictionary<string, object>
            {
                { "id", note.id ?? string.Empty },
                { "text", note.text ?? string.Empty },
                { "createdAtUtc", note.createdAtUtc ?? string.Empty },
                { "localPosition", ToVectorMap(note.localPosition) },
                { "localRotationEuler", ToVectorMap(note.localRotationEuler) },
                { "localScale", ToVectorMap(note.localScale) },
                { "siblingIndex", note.siblingIndex }
            });
        }

        return output;
    }

    private static Dictionary<string, object> ToVectorMap(SavedVector3Data vector)
    {
        return new Dictionary<string, object>
        {
            { "x", vector != null ? vector.x : 0f },
            { "y", vector != null ? vector.y : 0f },
            { "z", vector != null ? vector.z : 0f }
        };
    }

    private static SharedSessionRecord FromSnapshot(DocumentSnapshot snapshot)
    {
        if (snapshot == null || !snapshot.Exists)
            return null;

        var data = snapshot.ToDictionary();
        var session = new SharedSessionRecord
        {
            id = ReadString(data, "id", snapshot.Id),
            ownerUserId = ReadString(data, "ownerUserId"),
            ownerEmail = ReadString(data, "ownerEmail"),
            ownerDisplayName = ReadString(data, "ownerDisplayName"),
            title = ReadString(data, "title"),
            userName = ReadString(data, "userName"),
            duckName = ReadString(data, "duckName"),
            createdAtUtc = ReadString(data, "createdAtUtc"),
            updatedAtUtc = ReadString(data, "updatedAtUtc"),
            visibility = (SharedSessionVisibility)ReadInt(data, "visibility", 0),
            allowedUserIds = ReadStringList(data, "allowedUserIds"),
            invitedEmails = ReadStringList(data, "invitedEmails").Select(NormalizeEmail).ToList(),
            notes = ReadNotes(data, "notes")
        };

        return session;
    }

    private static List<SharedTranscriptNoteData> ReadNotes(IDictionary<string, object> data, string key)
    {
        var output = new List<SharedTranscriptNoteData>();
        if (!data.TryGetValue(key, out var value) || value is not IEnumerable<object> items)
            return output;

        foreach (var item in items)
        {
            if (item is not IDictionary<string, object> map)
                continue;

            output.Add(new SharedTranscriptNoteData
            {
                id = ReadString(map, "id"),
                text = ReadString(map, "text"),
                createdAtUtc = ReadString(map, "createdAtUtc"),
                localPosition = ReadVector(map, "localPosition"),
                localRotationEuler = ReadVector(map, "localRotationEuler"),
                localScale = ReadVector(map, "localScale"),
                siblingIndex = ReadInt(map, "siblingIndex", 0)
            });
        }

        return output;
    }

    private static SavedVector3Data ReadVector(IDictionary<string, object> data, string key)
    {
        if (!data.TryGetValue(key, out var value) || value is not IDictionary<string, object> map)
            return null;

        return new SavedVector3Data
        {
            x = ReadFloat(map, "x"),
            y = ReadFloat(map, "y"),
            z = ReadFloat(map, "z")
        };
    }

    private static List<string> ReadStringList(IDictionary<string, object> data, string key)
    {
        if (!data.TryGetValue(key, out var value) || value is not IEnumerable<object> items)
            return new List<string>();

        return items.Select(item => item?.ToString() ?? string.Empty)
            .Where(item => !string.IsNullOrWhiteSpace(item))
            .ToList();
    }

    private static string ReadString(IDictionary<string, object> data, string key, string fallback = "")
    {
        return data.TryGetValue(key, out var value) && value != null ? value.ToString() : fallback;
    }

    private static int ReadInt(IDictionary<string, object> data, string key, int fallback)
    {
        if (!data.TryGetValue(key, out var value) || value == null)
            return fallback;

        return value switch
        {
            int i => i,
            long l => (int)l,
            double d => (int)d,
            _ => int.TryParse(value.ToString(), out var parsed) ? parsed : fallback
        };
    }

    private static float ReadFloat(IDictionary<string, object> data, string key)
    {
        if (!data.TryGetValue(key, out var value) || value == null)
            return 0f;

        return value switch
        {
            float f => f,
            double d => (float)d,
            long l => l,
            int i => i,
            _ => float.TryParse(value.ToString(), out var parsed) ? parsed : 0f
        };
    }

    private static bool CanRead(SharedSessionRecord session, CloudAuthUser requester)
    {
        if (session == null || requester == null || !requester.isAuthenticated)
            return false;

        if (session.ownerUserId == requester.userId)
            return true;

        if (session.allowedUserIds != null && session.allowedUserIds.Contains(requester.userId))
            return true;

        var email = NormalizeEmail(requester.email);
        if (!string.IsNullOrWhiteSpace(email) && session.invitedEmails != null && session.invitedEmails.Contains(email))
            return true;

        return session.visibility == SharedSessionVisibility.AnyoneWithCode;
    }

    private static string NormalizeEmail(string email)
    {
        return string.IsNullOrWhiteSpace(email) ? string.Empty : email.Trim().ToLowerInvariant();
    }
}
