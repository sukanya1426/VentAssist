import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { ClipboardCheck, LogOut, UserRound } from "lucide-react";
import { useAuth } from "../../store/useAuth";

/**
 * Who is signed in, and the way out. Sits in the header of every signed-in page.
 *
 * The name is shown rather than only an avatar: a recommendation gets filed under
 * whoever is signed in, so on a shared ward machine it should be obvious at a
 * glance whose account that will be.
 */
export function UserMenu() {
  const user = useAuth((s) => s.user);
  const logout = useAuth((s) => s.logout);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Click-away and Escape both close it — a menu that can only be dismissed by
  // clicking the same button is a trap on a touchscreen.
  useEffect(() => {
    if (!open) return;
    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", onClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  if (!user) return null;

  const display = user.full_name?.trim() || user.username;

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-haspopup="menu"
        title={`Signed in as ${user.username}`}
        className="panel flex items-center gap-2 px-2.5 py-1.5 transition hover:border-cyan-400/50 hover:bg-white"
      >
        <span className="grid h-6 w-6 place-items-center rounded-lg bg-gradient-to-br from-cyan-100 to-indigo-100 font-display text-[10px] font-bold text-slate-700 ring-1 ring-inset ring-slate-900/5">
          {initials(display)}
        </span>
        <span className="max-w-[10rem] truncate text-[11px] font-medium text-slate-700">
          {display}
        </span>
      </button>

      {open && (
        <div role="menu"
          className="panel absolute right-0 z-20 mt-2 w-56 overflow-hidden p-1 shadow-lg">
          <div className="border-b border-slate-200/70 px-3 py-2.5">
            <div className="flex items-center gap-1.5">
              <UserRound size={11} className="text-slate-400" />
              <span className="num truncate text-[11px] text-slate-600">{user.username}</span>
            </div>
            {user.role && (
              <p className="mt-0.5 truncate text-[10px] text-slate-400">{user.role}</p>
            )}
            <p className="mt-1 text-[10px] leading-relaxed text-slate-400">
              Recommendations you request are filed under this account.
            </p>
          </div>
          {/* The model-evidence page is model-level, not patient-level, so it used to
              be reachable only from inside a patient record — which meant a clinician
              on the roster had no way to find it at all. It belongs in the header menu
              that is present on every signed-in page. */}
          <Link
            role="menuitem"
            to="/validation?track=a"
            onClick={() => setOpen(false)}
            className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-[12px] text-slate-700 transition hover:bg-cyan-50 hover:text-cyan-800"
          >
            <ClipboardCheck size={13} /> How this model was evaluated
          </Link>
          <button
            role="menuitem"
            onClick={() => { setOpen(false); logout(); }}
            className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-[12px] text-slate-700 transition hover:bg-rose-50 hover:text-rose-700"
          >
            <LogOut size={13} /> Sign out
          </button>
        </div>
      )}
    </div>
  );
}

function initials(name: string): string {
  const words = name.trim().split(/\s+/).filter(Boolean);
  if (words.length === 0) return "?";
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase();
  return (words[0][0] + words[words.length - 1][0]).toUpperCase();
}
