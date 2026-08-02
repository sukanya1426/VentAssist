import axios from "axios";
import type { RecommendationRequest, RecommendationResponse } from "../types/recommendation";
import type { ValidationResponse } from "../types/validation";
import type { Patient, PatientCreate, RecommendationRecord } from "../types/patient";
import type { AuthResponse, AuthUser, LoginPayload, SignupPayload } from "../types/auth";
import { getToken, notifySessionExpired } from "./session";

const api = axios.create({ baseURL: "/api" });

// Every clinical route on the backend is guarded, so the session token rides on
// every request. Read per-request rather than captured once: signing in or out
// takes effect immediately, without rebuilding the client.
api.interceptors.request.use((cfg) => {
  const token = getToken();
  if (token) cfg.headers.Authorization = `Bearer ${token}`;
  return cfg;
});

// A 401 from a guarded route means the session is gone (expired, or the account
// was deleted) — drop it and let the app fall back to the sign-in page. The auth
// endpoints are exempt: a 401 from /auth/login is a wrong password, and clearing
// the session there would sign out a user who is merely mistyping.
api.interceptors.response.use(
  (res) => res,
  (err) => {
    const url: string = err?.config?.url ?? "";
    if (err?.response?.status === 401 && !url.startsWith("/auth/")) {
      notifySessionExpired();
    }
    return Promise.reject(err);
  }
);

/**
 * FastAPI puts the readable reason in `detail`, which is a string for our own
 * HTTPExceptions and a list of `{loc, msg}` for 422s. Axios' own message ("Request
 * failed with status code 503") never says *why*, so unwrap it here.
 */
export function apiErrorMessage(e: any, fallback = "Request failed"): string {
  const detail = e?.response?.data?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d: any) => {
        const field = Array.isArray(d?.loc) ? d.loc[d.loc.length - 1] : "request";
        return `${field}: ${d?.msg ?? "invalid"}`;
      })
      .join("; ");
  }
  return e?.message ?? fallback;
}

// --- Auth ---

/** Register a clinician. Succeeds signed in, so no second login round-trip. */
export async function signup(payload: SignupPayload): Promise<AuthResponse> {
  const { data } = await api.post<AuthResponse>("/auth/signup", payload);
  return data;
}

export async function login(payload: LoginPayload): Promise<AuthResponse> {
  const { data } = await api.post<AuthResponse>("/auth/login", payload);
  return data;
}

/** Re-validate a stored token on page load; 401 means it is no longer good. */
export async function getMe(): Promise<AuthUser> {
  const { data } = await api.get<AuthUser>("/auth/me");
  return data;
}

export async function getRecommendation(
  req: RecommendationRequest
): Promise<RecommendationResponse> {
  const { data } = await api.post<RecommendationResponse>("/recommend", req);
  return data;
}

export interface TrackMeta {
  id: string; label: string; available: boolean; waveform_required: boolean;
}
export async function getTracks(): Promise<TrackMeta[]> {
  const { data } = await api.get<{ tracks: TrackMeta[] }>("/tracks");
  return data.tracks;
}

/** Model-level, not patient-level — safe to fetch once and reuse across patients. */
export async function getValidation(track = "a"): Promise<ValidationResponse> {
  const { data } = await api.get<ValidationResponse>("/validation", { params: { track } });
  return data;
}

// --- Roster (MongoDB-backed) ---

export async function listPatients(): Promise<Patient[]> {
  const { data } = await api.get<{ patients: Patient[] }>("/patients");
  return data.patients;
}

export async function getPatient(id: string): Promise<Patient> {
  const { data } = await api.get<Patient>(`/patients/${encodeURIComponent(id)}`);
  return data;
}

/** Save an uploaded patient. Re-posting the same id updates it in place. */
export async function createPatient(p: PatientCreate): Promise<Patient> {
  const { data } = await api.post<Patient>("/patients", p);
  return data;
}

/** Removes the patient and every recommendation saved for them. */
export async function deletePatient(id: string): Promise<{ recommendations_deleted: number }> {
  const { data } = await api.delete<{ id: string; deleted: boolean; recommendations_deleted: number }>(
    `/patients/${encodeURIComponent(id)}`
  );
  return data;
}

/** Newest first — every set of settings this patient has been asked about. */
export async function getPatientRecommendations(
  id: string, limit = 25
): Promise<RecommendationRecord[]> {
  const { data } = await api.get<{ records: RecommendationRecord[] }>(
    `/patients/${encodeURIComponent(id)}/recommendations`, { params: { limit } }
  );
  return data.records;
}
