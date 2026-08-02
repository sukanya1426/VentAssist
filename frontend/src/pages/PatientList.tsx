import { useEffect, useState } from "react";
import { ArrowUpRight, FileText, History, Loader2, Trash2, X } from "lucide-react";
import { Link } from "react-router-dom";
import { apiErrorMessage } from "../api/client";
import { Wordmark } from "../components/shared/Wordmark";
import { Waveform } from "../components/shared/Waveform";
import { UploadPatientCard } from "../components/patients/UploadPatientCard";
import { UserMenu } from "../components/auth/UserMenu";
import { useAllPatients, useRoster } from "../store/useRoster";
import type { Patient } from "../types/patient";
import { LEVEL_DOT, vitalLevel, worstLevel, type VitalLevel } from "../data/vitalRanges";
import type { TabularState } from "../types/recommendation";

const ACUITY: Record<VitalLevel, { rail: string; label: string; chip: string; tone: "cyan" | "amber" | "rose" }> = {
  ok:   { rail: "from-emerald-500 to-teal-300", label: "Stable",   chip: "text-emerald-700 border-emerald-200 bg-emerald-50", tone: "cyan" },
  warn: { rail: "from-amber-500 to-orange-300", label: "Watch",    chip: "text-amber-700 border-amber-200 bg-amber-50",       tone: "amber" },
  crit: { rail: "from-rose-500 to-rose-300",    label: "Critical", chip: "text-rose-700 border-rose-200 bg-rose-50",          tone: "rose" },
};

export function PatientList() {
  const patients = useAllPatients();
  const loading = useRoster((s) => s.loading);
  const loaded = useRoster((s) => s.loaded);
  const rosterError = useRoster((s) => s.error);
  const reload = useRoster((s) => s.load);

  // Coming back from a patient page, the saved-recommendation counts on the cards
  // are stale. `load` no-ops while a first fetch is already in flight, so this
  // costs nothing on the initial render.
  useEffect(() => { reload(true); }, [reload]);

  // Acuity travels with the patient so filtering can't desynchronise the two.
  const rows = patients.map((p) => ({ patient: p, level: worstLevel(p.state) }));
  const counts = {
    crit: rows.filter((r) => r.level === "crit").length,
    warn: rows.filter((r) => r.level === "warn").length,
    ok: rows.filter((r) => r.level === "ok").length,
  };

  // null = show every bed. Clicking the active chip clears it.
  const [filter, setFilter] = useState<VitalLevel | null>(null);
  const visible = filter ? rows.filter((r) => r.level === filter) : rows;
  const toggle = (l: VitalLevel) => setFilter((cur) => (cur === l ? null : l));

  return (
    <div className="mx-auto max-w-6xl px-6 py-10">
      <header className="mb-10 flex flex-wrap items-end justify-between gap-6">
        <div>
          <Wordmark className="text-6xl sm:text-7xl" />
          <p className="mt-2 max-w-md text-sm leading-relaxed text-slate-500">
            Dual-track offline reinforcement learning for mechanical ventilation —
            settings guidance at the bedside, with a safety filter on every recommendation.
          </p>
        </div>
        <div className="flex flex-col items-end gap-3">
          <UserMenu />
          <div className="flex items-center gap-2 text-[11px]">
            <Stat level="crit" n={counts.crit} label="Critical" dot="bg-rose-500"
              active={filter === "crit"} onClick={() => toggle("crit")} />
            <Stat level="warn" n={counts.warn} label="Watch" dot="bg-amber-500"
              active={filter === "warn"} onClick={() => toggle("warn")} />
            <Stat level="ok" n={counts.ok} label="Stable" dot="bg-emerald-500"
              active={filter === "ok"} onClick={() => toggle("ok")} />
          </div>
        </div>
      </header>

      <div className="mb-4 flex items-center gap-4">
        <h2 className="font-display text-sm font-semibold tracking-[0.2em] text-slate-500">
          {filter ? `${ACUITY[filter].label.toUpperCase()} PATIENTS` : "VENTILATED PATIENTS"}
        </h2>
        <div className="h-px flex-1 bg-gradient-to-r from-slate-300 to-transparent" />
        {filter ? (
          <button onClick={() => setFilter(null)} className="btn-ghost">
            <X size={12} /> Clear filter
          </button>
        ) : (
          <span className="num text-xs text-slate-400">
            {loading && !loaded ? "loading…" : `${patients.length} beds`}
          </span>
        )}
      </div>

      {/* The roster is served from MongoDB — if it can't be read, say so rather
          than showing an empty ward that looks like every bed was discharged. */}
      {rosterError && (
        <div className="panel mb-4 flex flex-wrap items-center justify-between gap-3 border-rose-200 bg-rose-50/80 p-4 text-sm text-rose-700">
          <span>{rosterError}</span>
          <button onClick={() => reload(true)} className="btn-ghost">Retry</button>
        </div>
      )}

      {loading && !loaded ? (
        <div className="panel grid min-h-[220px] place-items-center p-10 text-slate-400">
          <Loader2 className="animate-spin" size={22} />
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
          {visible.map((r, i) => (
            <PatientCard key={r.patient.id} patient={r.patient} level={r.level} index={i} />
          ))}
          {/* Adding a patient isn't a filter result — only offer it on the full roster. */}
          {!filter && <UploadPatientCard />}
        </div>
      )}

      {filter && visible.length === 0 && (
        <div className="panel p-10 text-center text-sm text-slate-500">
          No {ACUITY[filter].label.toLowerCase()} patients right now.{" "}
          <button onClick={() => setFilter(null)} className="text-cyan-700 hover:underline">
            Show all beds
          </button>
        </div>
      )}
    </div>
  );
}

function PatientCard({ patient, level, index }: { patient: Patient; level: VitalLevel; index: number }) {
  const removePatient = useRoster((s) => s.removePatient);
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const acuity = ACUITY[level];
  const uploaded = patient.source === "upload";
  const saved = patient.recommendation_count ?? 0;

  // The card is a link, so anything interactive inside it has to stop the click
  // from navigating away mid-decision.
  const stop = (e: React.MouseEvent) => { e.preventDefault(); e.stopPropagation(); };

  async function confirmDelete(e: React.MouseEvent) {
    stop(e);
    setBusy(true);
    setError(null);
    try {
      await removePatient(patient.id);   // card unmounts on success
    } catch (err) {
      setError(apiErrorMessage(err, "Could not delete this patient"));
      setBusy(false);
      setConfirming(false);
    }
  }

  return (
    <Link
      to={`/patients/${patient.id}`}
      style={{ animationDelay: `${index * 45}ms` }}
      className="panel panel-hover group relative animate-rise overflow-hidden p-5"
    >
      {/* acuity rail */}
      <span className={`absolute inset-y-0 left-0 w-[3px] bg-gradient-to-b ${acuity.rail}`} />

      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="grid h-11 w-11 place-items-center rounded-xl bg-gradient-to-br from-cyan-100 to-indigo-100 font-display text-lg font-bold text-slate-800 ring-1 ring-inset ring-slate-900/5">
            {initials(patient.name)}
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <span className="font-display text-base font-semibold text-slate-900">{patient.name}</span>
              {uploaded && (
                <span className="inline-flex items-center gap-1 rounded border border-slate-200 bg-slate-50 px-1 py-px text-[9px] font-semibold uppercase tracking-wide text-slate-500">
                  <FileText size={9} /> file
                </span>
              )}
            </div>
            <div className="num mt-0.5 text-[11px] text-slate-500">
              {patient.bed}
              {patient.age ? ` · ${patient.age}y ${patient.sex}` : ""} · {patient.weight} kg
            </div>
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          <span className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider ${acuity.chip}`}>
            {acuity.label}
          </span>
          {/* Every bed can be discharged, uploaded or not. Kept always visible
              rather than hover-revealed: a hover-only control is unreachable on
              touch and invisible to anyone scanning the card by keyboard. */}
          <button
            aria-label={`Remove ${patient.name} from the roster`}
            title="Remove this patient"
            onClick={(e) => { stop(e); setConfirming(true); setError(null); }}
            className="rounded-md p-1 text-slate-300 transition hover:bg-rose-50 hover:text-rose-600 focus-visible:text-rose-600"
          >
            <Trash2 size={14} />
          </button>
        </div>
      </div>

      <p className="mt-3 min-h-[2.5rem] text-[13px] leading-relaxed text-slate-600">
        {patient.summary}
      </p>

      <Waveform state={patient.state} tone={acuity.tone} className="mt-1 h-10 w-full opacity-70" />

      <div className="mt-3 grid grid-cols-4 gap-2 border-t border-slate-200/70 pt-3">
        <Vital label="PEEP" k="PEEP" state={patient.state} />
        <Vital label="TV" k="TV" state={patient.state} />
        <Vital label="FiO₂" k="FiO2" state={patient.state} format={(v) => v.toFixed(2)} />
        <Vital label="SpO₂" k="SpO2" state={patient.state} />
      </div>

      {confirming ? (
        // Deleting takes the saved recommendations with it and can't be undone, so
        // name the cost before asking for the click.
        <div className="mt-4 rounded-xl border border-rose-200 bg-rose-50 p-3" onClick={stop}>
          <p className="text-[11px] leading-relaxed text-rose-800">
            Remove <span className="font-semibold">{patient.name}</span> from the roster?
            {saved > 0
              ? ` Their ${saved} saved recommendation${saved > 1 ? "s" : ""} will be deleted too.`
              : " This also deletes their record from the database."}
          </p>
          <div className="mt-2 flex items-center gap-2">
            <button onClick={confirmDelete} disabled={busy}
              className="inline-flex items-center gap-1 rounded-lg bg-rose-600 px-2.5 py-1 text-[11px] font-semibold text-white transition hover:bg-rose-700 disabled:opacity-60">
              {busy ? <Loader2 size={12} className="animate-spin" /> : <Trash2 size={12} />}
              {busy ? "Deleting…" : "Delete"}
            </button>
            <button onClick={(e) => { stop(e); setConfirming(false); }} disabled={busy}
              className="btn-ghost">
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <div className="mt-4 flex items-center justify-between">
          <span className="flex items-center gap-1 text-[11px] font-medium text-slate-400 transition group-hover:text-cyan-700">
            Open patient
            <ArrowUpRight size={13} className="transition group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
          </span>
          {saved > 0 && (
            <span className="num flex items-center gap-1 text-[10px] text-slate-400"
              title={`${saved} recommendation${saved > 1 ? "s" : ""} saved for this patient`}>
              <History size={11} /> {saved}
            </span>
          )}
        </div>
      )}

      {error && (
        <p className="mt-2 rounded-lg border border-rose-200 bg-rose-50 px-2 py-1 text-[11px] text-rose-700">
          {error}
        </p>
      )}
    </Link>
  );
}

/** "Patient A" → "A"; an uploaded "John Doe" → "JD". */
function initials(name: string): string {
  const words = name.trim().split(/\s+/);
  if (words.length > 1 && words[0].toLowerCase() === "patient") return words[words.length - 1].slice(0, 2);
  return words.map((w) => w[0]).join("").slice(0, 2).toUpperCase();
}

function Vital({ label, k, state, format }: {
  label: string; k: keyof TabularState; state: TabularState; format?: (v: number) => string;
}) {
  const v = state[k];
  const level = vitalLevel(k, v);
  return (
    <div>
      <div className="flex items-center gap-1">
        <span className={`h-1 w-1 rounded-full ${LEVEL_DOT[level]}`} />
        <span className="caption">{label}</span>
      </div>
      <div className={`num mt-0.5 text-sm font-semibold ${
        level === "crit" ? "text-rose-600" : level === "warn" ? "text-amber-600" : "text-slate-900"}`}>
        {format ? format(v) : v}
      </div>
    </div>
  );
}

/** Acuity count that doubles as a filter toggle for the roster below. */
function Stat({ n, label, dot, active, onClick }: {
  level: VitalLevel; n: number; label: string; dot: string;
  active: boolean; onClick: () => void;
}) {
  const empty = n === 0;
  return (
    <button
      onClick={onClick}
      disabled={empty}
      aria-pressed={active}
      title={empty ? `No ${label.toLowerCase()} patients`
        : active ? "Show all beds" : `Show only ${label.toLowerCase()} patients`}
      className={`panel flex items-center gap-2 px-3 py-2 transition ${
        empty ? "cursor-not-allowed opacity-40"
          : active
            // Ring + the ✕ mark the active state, so it never reads by colour alone.
            ? "border-cyan-400/70 bg-white ring-1 ring-cyan-400/40"
            : "hover:border-cyan-400/50 hover:bg-white"}`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
      <span className="num text-base font-semibold text-slate-900">{n}</span>
      <span className={`caption ${active ? "text-slate-700" : ""}`}>{label}</span>
      {active && <X size={11} className="text-slate-400" />}
    </button>
  );
}
