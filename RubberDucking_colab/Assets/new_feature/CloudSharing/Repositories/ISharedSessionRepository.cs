using System.Collections.Generic;
using System.Threading.Tasks;

public interface ISharedSessionRepository
{
    Task<SharedSessionRecord> UpsertOwnedSessionAsync(SharedSessionRecord session, CloudAuthUser owner);
    Task<SharedSessionRecord> GetSharedSessionAsync(string sessionId, CloudAuthUser requester);
    Task<IReadOnlyList<SharedSessionSummaryRecord>> GetSharedWithMeAsync(CloudAuthUser requester);
    Task<IReadOnlyList<SharedSessionSummaryRecord>> GetOwnedByMeAsync(CloudAuthUser requester);
    Task<bool> GrantInviteByEmailAsync(string sessionId, string invitedEmail, CloudAuthUser owner);
}
