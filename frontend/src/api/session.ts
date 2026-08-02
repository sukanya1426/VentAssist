import type { AuthResponse, AuthUser } from "../types/auth";

/**
 * Where the signed-in session lives between page loads.
 *
 * This module is deliberately tiny and dependency-free: both the axios client
 * (which attaches the token) and the auth store (which owns the UI state) need
 * the session, and routing it through one small module keeps them from importing
 * each other in a cycle.
 *
 * `localStorage` rather than a cookie, because the token is a bearer token the
 * client attaches explicitly — nothing is sent ambiently, so there is no CSRF
 * surface. The trade-off is XSS exposure, which for a prototype served from its
 * own origin with no third-party scripts is the accepted one.
 */

const KEY = "ventassist.session";

export interface StoredSession {
  token: string;
  user: AuthUser;
  expires_at?: string | null;
}

let cached: StoredSession | null | undefined;   // undefined = not read yet

export function loadSession(): StoredSession | null {
  if (cached !== undefined) return cached;
  try {
    const raw = localStorage.getItem(KEY);
    const parsed = raw ? (JSON.parse(raw) as StoredSession) : null;
    // A half-written or hand-edited entry must not wedge the app on every load.
    cached = parsed?.token && parsed?.user?.username ? parsed : null;
  } catch {
    cached = null;
  }
  return cached;
}

export function saveSession(auth: AuthResponse): StoredSession {
  const session: StoredSession = {
    token: auth.token,
    user: auth.user,
    expires_at: auth.expires_at ?? null,
  };
  cached = session;
  try {
    localStorage.setItem(KEY, JSON.stringify(session));
  } catch {
    // Private browsing with storage denied: the session still works for this tab,
    // it just will not survive a reload. Not worth failing the sign-in over.
  }
  return session;
}

export function clearSession(): void {
  cached = null;
  try {
    localStorage.removeItem(KEY);
  } catch {
    /* see saveSession */
  }
}

export function getToken(): string | null {
  return loadSession()?.token ?? null;
}

/**
 * Called when the server rejects the held token (401 on a guarded route). The
 * auth store registers here so an expired session flips the UI back to the
 * sign-in page from wherever the user was, rather than leaving them staring at a
 * page whose every request is failing.
 */
type ExpiryListener = () => void;
const listeners = new Set<ExpiryListener>();

export function onSessionExpired(fn: ExpiryListener): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function notifySessionExpired(): void {
  clearSession();
  listeners.forEach((fn) => fn());
}
