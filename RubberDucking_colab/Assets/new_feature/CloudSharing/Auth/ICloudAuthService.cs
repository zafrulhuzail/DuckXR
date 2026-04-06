using System;
using System.Threading.Tasks;

public interface ICloudAuthService
{
    CloudAuthUser CurrentUser { get; }
    bool IsSignedIn { get; }
    event Action<CloudAuthUser> AuthStateChanged;

    Task<CloudAuthUser> SignInWithGoogleAsync();
    Task SignOutAsync();
}
