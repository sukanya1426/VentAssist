import { create } from "zustand";
import * as api from "../api/client";
import { apiErrorMessage } from "../api/client";
import { clearSession, loadSession, onSessionExpired, saveSession } from "../api/session";
import { useRoster } from "./useRoster";
import type { AuthUser, LoginPayload, SignupPayload } from "../types/auth";

/**
 * Who is signed in. The gate in front of the whole application.
 *
 * `status` is three-valued on purpose. A stored token is not trusted on sight —
 * it may have expired, or the account may have been deleted — so the app starts
 * in `checking` while `restore()` asks the backend, and only then settles on
 * `authenticated` or `anonymous`. Without the middle state a returning clinician
 * would see the sign-in page flash before their session was confirmed.
 */
type Status = "checking" | "authenticated" | "anonymous";

interface AuthState {
  user: AuthUser | null;
  status: Status;
  error: string | null;
  restore: () => Promise<void>;
  login: (payload: LoginPayload) => Promise<void>;
  signup: (payload: SignupPayload) => Promise<void>;
  logout: () => void;
  clearError: () => void;
}

const stored = loadSession();

export const useAuth = create<AuthState>()((set) => ({
  // A stored session is shown as `checking`, not `authenticated` — see above.
  user: stored?.user ?? null,
  status: stored ? "checking" : "anonymous",
  error: null,

  restore: async () => {
    const session = loadSession();
    if (!session) {
      set({ user: null, status: "anonymous" });
      return;
    }
    try {
      set({ user: await api.getMe(), status: "authenticated", error: null });
    } catch (e: any) {
      // 401 → the token is dead, so sign out. Anything else (backend restarting,
      // network blip) is not a reason to throw away a session that may still work.
      if (e?.response?.status === 401) {
        clearSession();
        set({ user: null, status: "anonymous" });
      } else {
        set({ user: session.user, status: "authenticated" });
      }
    }
  },

  login: async (payload) => {
    set({ error: null });
    try {
      const auth = await api.login(payload);
      saveSession(auth);
      set({ user: auth.user, status: "authenticated", error: null });
    } catch (e) {
      const message = apiErrorMessage(e, "Could not sign in");
      set({ error: message });
      throw new Error(message);
    }
  },

  signup: async (payload) => {
    set({ error: null });
    try {
      const auth = await api.signup(payload);
      saveSession(auth);
      set({ user: auth.user, status: "authenticated", error: null });
    } catch (e) {
      const message = apiErrorMessage(e, "Could not create the account");
      set({ error: message });
      throw new Error(message);
    }
  },

  logout: () => {
    clearSession();
    set({ user: null, status: "anonymous", error: null });
    // The roster is patient data held in memory. Signing out has to drop it, or
    // the next person to sign in on this machine sees the previous clinician's
    // ward for as long as the refetch takes.
    useRoster.setState({ patients: [], loaded: false, error: null });
  },

  clearError: () => set({ error: null }),
}));

// A token the server rejects mid-session (expired, or the account was removed)
// signs the user out wherever they are, instead of leaving them on a page whose
// every request quietly fails.
onSessionExpired(() => {
  if (useAuth.getState().status !== "anonymous") useAuth.getState().logout();
});
