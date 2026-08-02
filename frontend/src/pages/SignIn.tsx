import { useEffect, useState, type FormEvent } from "react";
import { Navigate, useLocation } from "react-router-dom";
import {
  AlertCircle, Eye, EyeOff, KeyRound, Loader2, LogIn, ShieldCheck, UserPlus,
} from "lucide-react";
import { Wordmark } from "../components/shared/Wordmark";
import { useAuth } from "../store/useAuth";

/**
 * The door to VentAssist: sign in, or create the account and be signed in.
 *
 * One page with two modes rather than two routes — a clinician who mistypes their
 * username on a system they have never used should be one click from registering,
 * not one navigation and a re-type. Switching modes keeps what has been typed.
 */
type Mode = "signin" | "signup";

const MIN_PASSWORD = 8;

export function SignIn() {
  const status = useAuth((s) => s.status);
  const error = useAuth((s) => s.error);
  const clearError = useAuth((s) => s.clearError);
  const doLogin = useAuth((s) => s.login);
  const doSignup = useAuth((s) => s.signup);
  const location = useLocation();

  const [mode, setMode] = useState<Mode>("signin");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [role, setRole] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [busy, setBusy] = useState(false);

  // A stale error from a previous attempt must not sit under a form the user has
  // since changed — it reads as though the new input was rejected.
  useEffect(() => { clearError(); }, [mode, clearError]);

  // Already signed in (or signed in just now): go where they were headed.
  if (status === "authenticated") {
    const from = (location.state as { from?: string } | null)?.from;
    return <Navigate to={from && from !== "/signin" ? from : "/"} replace />;
  }

  const trimmed = username.trim();
  const usernameValid = /^[A-Za-z0-9][A-Za-z0-9._-]{2,39}$/.test(trimmed);
  const passwordValid = password.length >= MIN_PASSWORD;
  const canSubmit = mode === "signin"
    ? trimmed.length > 0 && password.length > 0
    : usernameValid && passwordValid;

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (!canSubmit || busy) return;
    setBusy(true);
    try {
      if (mode === "signin") {
        await doLogin({ username: trimmed, password });
      } else {
        await doSignup({
          username: trimmed,
          password,
          full_name: fullName.trim() || undefined,
          role: role.trim() || undefined,
        });
      }
      // On success the store flips to `authenticated` and the <Navigate/> above
      // takes over; nothing further to do here.
    } catch {
      // Already surfaced through the store's `error`.
      setBusy(false);
      return;
    }
    setBusy(false);
  }

  return (
    <div className="mx-auto grid min-h-screen max-w-5xl place-items-center px-6 py-10">
      <div className="w-full max-w-md">
        <div className="mb-8 text-center">
          <Wordmark className="text-5xl sm:text-6xl" />
          <p className="mt-3 text-sm leading-relaxed text-slate-500">
            Ventilator settings guidance at the bedside, with a safety filter on
            every recommendation. Sign in to open the ward.
          </p>
        </div>

        <div className="panel overflow-hidden">
          {/* Mode switch */}
          <div className="grid grid-cols-2 border-b border-slate-200/80 bg-white/50">
            <ModeTab active={mode === "signin"} onClick={() => setMode("signin")}
              icon={<LogIn size={13} />} label="Sign in" />
            <ModeTab active={mode === "signup"} onClick={() => setMode("signup")}
              icon={<UserPlus size={13} />} label="Create account" />
          </div>

          <form onSubmit={submit} className="space-y-4 p-6">
            <Field label="Username" hint={mode === "signup" ? "3–40 characters · letters, digits, . _ -" : undefined}>
              <input
                className="field font-sans"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                autoComplete="username"
                autoFocus
                placeholder="dr.chen"
                aria-invalid={mode === "signup" && trimmed.length > 0 && !usernameValid}
              />
              {mode === "signup" && trimmed.length > 0 && !usernameValid && (
                <Hint tone="warn">
                  Use 3–40 characters: letters, digits, dots, underscores or hyphens.
                </Hint>
              )}
            </Field>

            <Field label="Password" hint={mode === "signup" ? `At least ${MIN_PASSWORD} characters` : undefined}>
              <div className="relative">
                <input
                  className="field pr-10 font-sans"
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete={mode === "signin" ? "current-password" : "new-password"}
                  placeholder="••••••••"
                  aria-invalid={mode === "signup" && password.length > 0 && !passwordValid}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((v) => !v)}
                  aria-label={showPassword ? "Hide password" : "Show password"}
                  className="absolute inset-y-0 right-0 grid w-9 place-items-center text-slate-400 transition hover:text-slate-600"
                >
                  {showPassword ? <EyeOff size={14} /> : <Eye size={14} />}
                </button>
              </div>
              {mode === "signup" && password.length > 0 && !passwordValid && (
                <Hint tone="warn">
                  {MIN_PASSWORD - password.length} more character
                  {MIN_PASSWORD - password.length === 1 ? "" : "s"} needed.
                </Hint>
              )}
            </Field>

            {mode === "signup" && (
              <div className="grid grid-cols-2 gap-3">
                <Field label="Full name" hint="optional">
                  <input className="field font-sans" value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    autoComplete="name" placeholder="Dr. Mei Chen" />
                </Field>
                <Field label="Role" hint="optional">
                  <input className="field font-sans" value={role}
                    onChange={(e) => setRole(e.target.value)}
                    placeholder="ICU consultant" />
                </Field>
              </div>
            )}

            {error && (
              <div className="flex items-start gap-2 rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-[12px] leading-relaxed text-rose-700">
                <AlertCircle size={14} className="mt-px shrink-0" />
                <span>{error}</span>
              </div>
            )}

            <button type="submit" disabled={!canSubmit || busy} className="btn-primary">
              {busy ? (
                <span className="inline-flex items-center gap-2">
                  <Loader2 size={15} className="animate-spin" />
                  {mode === "signin" ? "Signing in…" : "Creating account…"}
                </span>
              ) : (
                <span className="inline-flex items-center gap-2">
                  {mode === "signin" ? <LogIn size={15} /> : <UserPlus size={15} />}
                  {mode === "signin" ? "Sign in" : "Create account"}
                </span>
              )}
            </button>

            <p className="text-center text-[11px] text-slate-500">
              {mode === "signin" ? (
                <>
                  No account yet?{" "}
                  <button type="button" onClick={() => setMode("signup")}
                    className="font-semibold text-cyan-700 hover:underline">
                    Create one
                  </button>
                </>
              ) : (
                <>
                  Already registered?{" "}
                  <button type="button" onClick={() => setMode("signin")}
                    className="font-semibold text-cyan-700 hover:underline">
                    Sign in
                  </button>
                </>
              )}
            </p>
          </form>
        </div>

        <div className="mt-5 space-y-1.5 px-1">
          <p className="flex items-start gap-2 text-[11px] leading-relaxed text-slate-500">
            <KeyRound size={12} className="mt-px shrink-0 text-slate-400" />
            Passwords are stored only as a salted PBKDF2 hash. There is no reset
            path in this prototype — keep the one you choose.
          </p>
          <p className="flex items-start gap-2 text-[11px] leading-relaxed text-slate-500">
            <ShieldCheck size={12} className="mt-px shrink-0 text-slate-400" />
            Research and educational system. Every recommendation is advisory and
            is not approved for clinical use.
          </p>
        </div>
      </div>
    </div>
  );
}

function ModeTab({ active, onClick, icon, label }: {
  active: boolean; onClick: () => void; icon: React.ReactNode; label: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={`flex items-center justify-center gap-1.5 py-3 font-display text-[13px] font-semibold tracking-tight transition ${
        active
          ? "bg-white text-slate-900 shadow-[inset_0_-2px_0_0_rgb(8,145,178)]"
          : "text-slate-500 hover:bg-white/60 hover:text-slate-700"
      }`}
    >
      {icon}
      {label}
    </button>
  );
}

function Field({ label, hint, children }: {
  label: string; hint?: string; children: React.ReactNode;
}) {
  return (
    <label className="block">
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <span className="caption">{label}</span>
        {hint && <span className="text-[10px] text-slate-400">{hint}</span>}
      </div>
      {children}
    </label>
  );
}

function Hint({ tone, children }: { tone: "warn"; children: React.ReactNode }) {
  return (
    <p className={`mt-1 text-[11px] ${tone === "warn" ? "text-amber-600" : "text-slate-500"}`}>
      {children}
    </p>
  );
}
