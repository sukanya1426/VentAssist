import { ArrowUpRight, FileText, Trash2 } from "lucide-react";
import { Link } from "react-router-dom";
import { Wordmark } from "../components/shared/Wordmark";
import { Waveform } from "../components/shared/Waveform";
import { UploadPatientCard } from "../components/patients/UploadPatientCard";
import { useAllPatients, useRoster } from "../store/useRoster";
import type { Patient } from "../data/patients";
import { LEVEL_DOT, vitalLevel, worstLevel, type VitalLevel } from "../data/vitalRanges";
import type { TabularState } from "../types/recommendation";

const ACUITY: Record<VitalLevel, { rail: string; label: string; chip: string; tone: "cyan" | "amber" | "rose" }> = {
  ok:   { rail: "from-emerald-500 to-teal-300", label: "Stable",   chip: "text-emerald-700 border-emerald-200 bg-emerald-50", tone: "cyan" },
  warn: { rail: "from-amber-500 to-orange-300", label: "Watch",    chip: "text-amber-700 border-amber-200 bg-amber-50",       tone: "amber" },
  crit: { rail: "from-rose-500 to-rose-300",    label: "Critical", chip: "text-rose-700 border-rose-200 bg-rose-50",          tone: "rose" },
};

export function PatientList() {
  const patients = useAllPatients();
  const levels = patients.map((p) => worstLevel(p.state));
  const counts = {
    crit: levels.filter((l) => l === "crit").length,
    warn: levels.filter((l) => l === "warn").length,
    ok: levels.filter((l) => l === "ok").length,
  };

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
        <div className="flex items-center gap-2 text-[11px]">
          <Stat n={counts.crit} label="Critical" dot="bg-rose-500" />
          <Stat n={counts.warn} label="Watch" dot="bg-amber-500" />
          <Stat n={counts.ok} label="Stable" dot="bg-emerald-500" />
        </div>
      </header>

      <div className="mb-4 flex items-center gap-4">
        <h2 className="font-display text-sm font-semibold tracking-[0.2em] text-slate-500">
          VENTILATED PATIENTS
        </h2>
        <div className="h-px flex-1 bg-gradient-to-r from-slate-300 to-transparent" />
        <span className="num text-xs text-slate-400">{patients.length} beds</span>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
        {patients.map((p, i) => (
          <PatientCard key={p.id} patient={p} level={levels[i]} index={i} />
        ))}
        <UploadPatientCard />
      </div>
    </div>
  );
}

function PatientCard({ patient, level, index }: { patient: Patient; level: VitalLevel; index: number }) {
  const removePatient = useRoster((s) => s.removePatient);
  const acuity = ACUITY[level];
  const uploaded = patient.source === "upload";
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
        <span className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider ${acuity.chip}`}>
          {acuity.label}
        </span>
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

      <div className="mt-4 flex items-center justify-between">
        <span className="flex items-center gap-1 text-[11px] font-medium text-slate-400 transition group-hover:text-cyan-700">
          Open patient
          <ArrowUpRight size={13} className="transition group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
        </span>
        {uploaded && (
          <button
            title="Remove this uploaded patient"
            onClick={(e) => { e.preventDefault(); removePatient(patient.id); }}
            className="rounded p-1 text-slate-300 transition hover:bg-rose-50 hover:text-rose-600"
          >
            <Trash2 size={13} />
          </button>
        )}
      </div>
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

function Stat({ n, label, dot }: { n: number; label: string; dot: string }) {
  return (
    <div className="panel flex items-center gap-2 px-3 py-2">
      <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
      <span className="num text-base font-semibold text-slate-900">{n}</span>
      <span className="caption">{label}</span>
    </div>
  );
}
