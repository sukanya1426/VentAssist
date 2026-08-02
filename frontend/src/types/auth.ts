/** Mirrors `backend/api/models.py` — AuthUser / AuthResponse. */

export interface AuthUser {
  id: string;
  username: string;
  full_name?: string | null;
  role?: string | null;
  created_at?: string | null;
  last_login_at?: string | null;
}

export interface AuthResponse {
  token: string;
  token_type: "bearer";
  expires_at?: string | null;
  user: AuthUser;
}

export interface SignupPayload {
  username: string;
  password: string;
  full_name?: string;
  role?: string;
}

export interface LoginPayload {
  username: string;
  password: string;
}
