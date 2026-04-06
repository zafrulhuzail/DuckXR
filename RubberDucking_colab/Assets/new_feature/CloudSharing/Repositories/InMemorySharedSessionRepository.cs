using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;

public class InMemorySharedSessionRepository : ISharedSessionRepository
{
    private static readonly Dictionary<string, SharedSessionRecord> Sessions = new Dictionary<string, SharedSessionRecord>();

    public Task<SharedSessionRecord> UpsertOwnedSessionAsync(SharedSessionRecord session, CloudAuthUser owner)
    {
        if (session == null)
            throw new ArgumentNullException(nameof(session));

        if (owner == null || !owner.isAuthenticated)
            throw new InvalidOperationException("Owner must be authenticated before uploading shared sessions.");

        session.ownerUserId = owner.userId;
        session.ownerEmail = owner.email;
        session.ownerDisplayName = owner.displayName;
        session.updatedAtUtc = DateTime.UtcNow.ToString("o");

        if (string.IsNullOrWhiteSpace(session.createdAtUtc))
            session.createdAtUtc = session.updatedAtUtc;

        if (!session.allowedUserIds.Contains(owner.userId))
            session.allowedUserIds.Add(owner.userId);

        Sessions[session.id] = session;
        Debug.Log($"InMemorySharedSessionRepository: Upserted shared session {session.id}");
        return Task.FromResult(session);
    }

    public Task<SharedSessionRecord> GetSharedSessionAsync(string sessionId, CloudAuthUser requester)
    {
        if (string.IsNullOrWhiteSpace(sessionId) || requester == null || !requester.isAuthenticated)
            return Task.FromResult<SharedSessionRecord>(null);

        if (!Sessions.TryGetValue(sessionId, out var session))
            return Task.FromResult<SharedSessionRecord>(null);

        return Task.FromResult(CanRead(session, requester) ? session : null);
    }

    public Task<IReadOnlyList<SharedSessionSummaryRecord>> GetSharedWithMeAsync(CloudAuthUser requester)
    {
        if (requester == null || !requester.isAuthenticated)
            return Task.FromResult((IReadOnlyList<SharedSessionSummaryRecord>)Array.Empty<SharedSessionSummaryRecord>());

        var summaries = Sessions.Values
            .Where(s => s.ownerUserId != requester.userId && CanRead(s, requester))
            .Select(SharedSessionMapper.ToSummary)
            .OrderByDescending(s => s.updatedAtUtc)
            .ToList();

        return Task.FromResult((IReadOnlyList<SharedSessionSummaryRecord>)summaries);
    }

    public Task<IReadOnlyList<SharedSessionSummaryRecord>> GetOwnedByMeAsync(CloudAuthUser requester)
    {
        if (requester == null || !requester.isAuthenticated)
            return Task.FromResult((IReadOnlyList<SharedSessionSummaryRecord>)Array.Empty<SharedSessionSummaryRecord>());

        var summaries = Sessions.Values
            .Where(s => s.ownerUserId == requester.userId)
            .Select(SharedSessionMapper.ToSummary)
            .OrderByDescending(s => s.updatedAtUtc)
            .ToList();

        return Task.FromResult((IReadOnlyList<SharedSessionSummaryRecord>)summaries);
    }

    public Task<bool> GrantInviteByEmailAsync(string sessionId, string invitedEmail, CloudAuthUser owner)
    {
        invitedEmail = NormalizeEmail(invitedEmail);
        if (string.IsNullOrWhiteSpace(sessionId) || string.IsNullOrWhiteSpace(invitedEmail) || owner == null || !owner.isAuthenticated)
            return Task.FromResult(false);

        if (!Sessions.TryGetValue(sessionId, out var session))
            return Task.FromResult(false);

        if (session.ownerUserId != owner.userId)
            return Task.FromResult(false);

        if (!session.invitedEmails.Contains(invitedEmail))
            session.invitedEmails.Add(invitedEmail);

        session.updatedAtUtc = DateTime.UtcNow.ToString("o");
        return Task.FromResult(true);
    }

    private static bool CanRead(SharedSessionRecord session, CloudAuthUser requester)
    {
        if (session == null || requester == null || !requester.isAuthenticated)
            return false;

        if (session.ownerUserId == requester.userId)
            return true;

        if (session.allowedUserIds.Contains(requester.userId))
            return true;

        var email = NormalizeEmail(requester.email);
        if (!string.IsNullOrWhiteSpace(email) && session.invitedEmails.Contains(email))
            return true;

        return session.visibility == SharedSessionVisibility.AnyoneWithCode;
    }

    private static string NormalizeEmail(string email)
    {
        return string.IsNullOrWhiteSpace(email) ? string.Empty : email.Trim().ToLowerInvariant();
    }
}
