# CloudSharing new_feature scaffold

This folder is an isolated first-pass scaffold for DuckXR cloud sharing.

## What is implemented here
- compile-safe data models for shared sessions/users
- auth abstraction (`ICloudAuthService`)
- repository abstraction (`ISharedSessionRepository`)
- mapper between local saved sessions and shared-session records
- temporary in-memory repository for testing flow without Firebase
- temporary stub Google auth service for editor testing
- `CloudSharingFacade` as a simple integration point
- `SharedWithMeDebugView` for rough UI/debug output

## What is NOT implemented yet
- real Firebase Auth
- real Firestore repository
- security rules
- invite lookup by real users
- scene wiring
- deep links / share UI

## Suggested next steps
1. Add Firebase SDK / package strategy for Unity.
2. Replace `EditorStubGoogleAuthService` with a real Google/Firebase auth implementation.
3. Replace `InMemorySharedSessionRepository` with Firestore-backed reads/writes.
4. Add a dedicated Shared With Me screen using the same visual language as the local saved-session browser.
5. Keep shared sessions read-only for recipients in v1.

## Current limitation
The in-memory repository only exists for structural testing during play mode; data will not persist across app restarts.
