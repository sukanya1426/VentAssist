import { useEffect, type ReactNode } from "react";
import { Link, useParams } from "react-router-dom";
import {
  AlertTriangle, ArrowLeft, CheckCircle2, Gauge, Loader2, RefreshCw, RotateCcw,
  ShieldCheck, Sparkles, Timer,
} from "lucide-react";
import { useStore } from "../store/useStore";
import { TrackBadge } from "../components/shared/TrackBadge";
import { Waveform } from "../components/shared/Waveform";
import { Wordmark } from "../components/shared/Wordmark";
import { ConfidenceBar } from "../components/recommendation/ConfidenceBar";
import { TrackSelector } from "../components/recommendation/TrackSelector";
import { ValidationLink } from "../components/recommendation/ValidationLink";
import { RecommendationHistory } from "../components/recommendation/RecommendationHistory";
import { UserMenu } from "../components/auth/UserMenu";
import { usePatientById, useRoster } from "../store/useRoster";
import { CLINICAL_FIELDS, fieldIsValid, validateTabular } from "../data/patientFile";
import { LEVEL_DOT, LEVEL_TEXT, vitalLevel, worstLevel } from "../data/vitalRanges";
import type { TabularState } from "../types/recommendation";

// key, label, unit, input step
const FIELDS: [keyof TabularState, string, string, number][] = [
  ["PEEP", "PEEP", "cmH₂O", 1], ["TV", "Tidal Vol", "mL", 10], ["FiO2", "FiO₂", "frac", 0.05],
  ["SpO2", "SpO₂", "%", 1], ["PaO2", "PaO₂", "mmHg", 1], ["PaCO2", "PaCO₂", "mmHg", 1],
  ["pH", "pH", "", 0.01], ["HR", "HR", "bpm", 1], ["SBP", "SBP", "mmHg", 1],
  ["RR", "RR", "/min", 1], ["RASS", "RASS", "", 1], ["Temp", "Temp", "°C", 0.1],
];

const WAVE_FIELDS = [
  ["ecgHRV", "HRV SDNN"], ["ecgArr", "Arrhythmia"], ["pleth", "Perfusion Idx"],
  ["rrv", "RRV"], ["breathReg", "Breath Reg"], ["asynchrony", "Asynchrony"],
] as const;

export function PatientDetail() {
  const { patientId } = useParams<{ patientId: string }>();
  const s = useStore();
  const loadPatient = useStore((st) => st.loadPatient);
  const patient = usePatientById(patientId);
  // The roster comes from MongoDB, so on a direct page load it is briefly empty.
  const rosterLoaded = useRoster((st) => st.loaded);
  const rosterError = useRoster((st) => st.error);

  useEffect(() => {
    if (patient) loadPatient(patient);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [patient?.id, loadPatient]);

  if (!patient) {
    return (
      <div className="mx-auto max-w-6xl px-6 py-16">
        <div className="panel grid min-h-[160px] place-items-center p-10 text-center text-slate-600">
          {!rosterLoaded ? (
            <Loader2 className="animate-spin text-slate-300" size={22} />
          ) : (
            <div>
              {/* A database that's down is not the same as a patient who isn't
                  there — a wrong message here sends the user hunting the wrong bug. */}
              {rosterError ? `Could not load the roster — ${rosterError}` : "No such patient."}{" "}
              <Link to="/" className="text-cyan-700 hover:underline">Back to the roster</Link>
            </div>
          )}
        </div>
      </div>
    );
  }

  const rec = s.result?.recommendation;
  const invalidFields = validateTabular(s.tabular, s.patientWeight);
  // Fields edited since the displayed recommendation was computed. The policy's
  // action is coarse (5 levels per dimension) and "hold" covers a wide band, so an
  // edit often leaves the action identical — this makes clear the panel is current,
  // not stuck, and offers a one-click re-run.
  const staleFields = s.resultState
    ? (Object.keys(s.tabular) as (keyof TabularState)[])
        .filter((k) => s.resultState![k] !== s.tabular[k])
        .map((k) => ({ key: k, was: s.resultState![k], now: s.tabular[k] }))
    : [];
  const level = worstLevel(s.tabular);
  const tone = level === "crit" ? "rose" : level === "warn" ? "amber" : "cyan";

  return (
    <div className="mx-auto max-w-6xl px-6 py-8">
      <div className="mb-6 flex items-center justify-between gap-4">
        <Link to="/" className="btn-ghost">
          <ArrowLeft size={14} /> Roster
        </Link>
        <div className="flex items-center gap-3">
          <Wordmark className="text-xl" />
          <UserMenu />
        </div>
      </div>

      {/* ---- Patient banner ---- */}
      <header className="panel relative mb-6 overflow-hidden p-6">
        <Waveform
          state={s.tabular}
          tone={tone}
          className="pointer-events-none absolute inset-x-0 bottom-0 h-20 w-full opacity-40"
        />
        <div className="relative flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div className="grid h-14 w-14 place-items-center rounded-2xl bg-gradient-to-br from-cyan-100 to-indigo-100 font-display text-2xl font-bold text-slate-800 ring-1 ring-inset ring-slate-900/5">
              {patient.name.slice(-1)}
            </div>
            <div>
              <h1 className="font-display text-3xl font-bold tracking-tight text-slate-900">
                {patient.name}
              </h1>
              <p className="num mt-1 text-xs text-slate-500">
                {patient.bed}
                {patient.age ? ` · ${patient.age}y ${patient.sex}` : ""} · {patient.weight} kg
                {patient.source === "upload" && " · from file"}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="panel px-3 py-1.5 text-[11px] text-slate-600">
              <span className={`mr-1.5 inline-block h-1.5 w-1.5 rounded-full align-middle ${LEVEL_DOT[level]}`} />
              {s.edited ? "Edited — unsaved" : "Recorded state"}
            </span>
            {s.edited && (
              <button onClick={s.resetPatient} className="btn-ghost">
                <RotateCcw size={13} /> Reset
              </button>
            )}
          </div>
        </div>
        <p className="relative mt-3 max-w-2xl text-sm text-slate-600">{patient.summary}</p>
      </header>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        {/* ---- Inputs ---- */}
        <div className="space-y-5">
          <section className="panel p-5">
            <SectionTitle icon={<Gauge size={14} />} title="Patient State">
              {!s.edited && patient.hint && (
                <span className="text-[11px] text-slate-500">{patient.hint}</span>
              )}
            </SectionTitle>
            <div className="mt-4 grid grid-cols-3 gap-3">
              {FIELDS.map(([k, label, unit, step]) => {
                const v = s.tabular[k];
                const lvl = vitalLevel(k, v);
                // Out-of-range or cleared: the API would reject it, so say so here.
                const invalid = !fieldIsValid(k, v);
                const b = CLINICAL_FIELDS.find((f) => f.key === k)!;
                return (
                  <label key={k} className="block">
                    <span className="flex items-center gap-1">
                      <span className={`h-1 w-1 rounded-full ${LEVEL_DOT[lvl]}`} />
                      <span className="caption">{label}</span>
                      {unit && <span className="text-[9px] text-slate-400">{unit}</span>}
                    </span>
                    <input
                      type="number" step={step} min={b.min} max={b.max}
                      // NaN would render as an empty-but-uncontrolled input and be
                      // serialised as null in the request body.
                      value={Number.isNaN(v) ? "" : v}
                      onChange={(e) => s.setField(k, e.target.value === "" ? NaN : parseFloat(e.target.value))}
                      title={invalid ? `Must be between ${b.min} and ${b.max}` : undefined}
                      className={`field mt-1 ${invalid
                        ? "border-rose-400 bg-rose-50 text-rose-700 focus:border-rose-400 focus:ring-rose-500/20"
                        : LEVEL_TEXT[lvl]}`}
                    />
                    {invalid && (
                      <span className="mt-0.5 block text-[10px] text-rose-600">
                        {Number.isNaN(v) ? "required" : `allowed ${b.min}–${b.max}`}
                      </span>
                    )}
                  </label>
                );
              })}
            </div>
          </section>

          <section className="panel p-5">
            <SectionTitle icon={<Sparkles size={14} />} title="Policy Track">
              <label className="flex items-center gap-1.5 text-[11px] text-slate-500">
                <input
                  type="checkbox" checked={s.waveformAvailable}
                  className="accent-cyan-600"
                  onChange={(e) => s.setMeta({
                    waveformAvailable: e.target.checked,
                    selectedTrack: e.target.checked ? s.selectedTrack : "track_a",
                  })}
                />
                waveform available
              </label>
            </SectionTitle>
            <div className="mt-4">
              <TrackSelector current={s.selectedTrack}
                onSelect={(t) => s.setMeta({ selectedTrack: t })}
                waveformAvailable={s.waveformAvailable} />
            </div>

            {s.selectedTrack === "track_b" && (
              <div className="mt-4 grid grid-cols-3 gap-3">
                {WAVE_FIELDS.map(([k, label]) => (
                  <label key={k} className="block">
                    <span className="caption">{label}</span>
                    <input type="number" step={0.01} value={(s as any)[k]}
                      onChange={(e) => s.setMeta({ [k]: parseFloat(e.target.value) } as any)}
                      className="field mt-1" />
                  </label>
                ))}
              </div>
            )}

            <div className="mt-6">
              <div className="flex items-baseline justify-between">
                <span className="caption">Responsiveness</span>
                <span className="num text-xs font-semibold text-slate-700">
                  {s.responsiveness === 0 ? "Conservative (deployed)"
                    : `${Math.round(s.responsiveness * 100)}% eager to act`}
                </span>
              </div>
              <input type="range" min={0} max={1} step={0.05} value={s.responsiveness}
                onChange={(e) => s.setMeta({ responsiveness: parseFloat(e.target.value) })}
                className="mt-2 w-full accent-cyan-600" />
              <p className="mt-1.5 text-[11px] leading-relaxed text-slate-500">
                Lowers the bar to leave “hold” — never changes which change is chosen when acting,
                and the safety filter still applies.
              </p>
            </div>

            <div className="mt-5">
              <div className="flex items-center justify-between gap-3">
                <span className="caption">Ventilation mode</span>
                <select
                  value={s.ventilationMode}
                  onChange={(e) => s.setMeta({ ventilationMode: e.target.value })}
                  className="field w-auto py-1"
                >
                  <option value="unknown">Unknown / not set</option>
                  <option value="volume_control">Volume control</option>
                  <option value="pressure_control">Pressure control</option>
                </select>
              </div>
              <p className="mt-1.5 text-[11px] leading-relaxed text-slate-500">
                In pressure control, tidal volume isn’t directly set — ΔTV recommendations are masked out.
              </p>
            </div>

            <button onClick={() => s.fetch()} disabled={s.loading || invalidFields.length > 0}
              className="btn-primary mt-6">
              {s.loading ? "Computing…"
                : invalidFields.length ? `Fix ${invalidFields.length} field${invalidFields.length > 1 ? "s" : ""} to continue`
                : "Get Recommendation"}
            </button>
            {s.error && (
              <p className="mt-2 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-700">
                {s.error}
              </p>
            )}
          </section>
        </div>

        {/* ---- Output ---- */}
        <div className="space-y-5">
          {!rec && (
            <div className="panel grid min-h-[280px] place-items-center p-10 text-center">
              <div>
                <Sparkles className="mx-auto text-slate-300" size={26} />
                <p className="mt-3 text-sm text-slate-500">
                  Review {patient.name}’s state, then request a recommendation.
                </p>
              </div>
            </div>
          )}

          {rec && s.result && (
            <>
              {staleFields.length > 0 && (
                <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-amber-200 bg-amber-50 px-4 py-2.5 text-[11px] text-amber-800">
                  <span>
                    Showing the result for{" "}
                    <span className="num font-semibold">
                      {staleFields.map((f) => `${f.key} ${f.was}`).join(", ")}
                    </span>{" "}
                    — the state has since changed to{" "}
                    <span className="num font-semibold">
                      {staleFields.map((f) => `${f.now}`).join(", ")}
                    </span>
                  </span>
                  <button onClick={() => s.fetch()} disabled={s.loading} className="btn-ghost">
                    <RefreshCw size={12} /> Re-run
                  </button>
                </div>
              )}

              <section className="panel animate-rise relative overflow-hidden p-6">
                <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-cyan-400/60 to-transparent" />
                <div className="mb-5 flex items-center justify-between gap-3">
                  <TrackBadge track={rec.track_info.track} label={rec.track_info.track_label}
                    waveformUsed={rec.track_info.waveform_used} />
                  <span className="num flex items-center gap-1 text-[11px] text-slate-500">
                    <Timer size={12} /> {s.result.metadata.latency_ms} ms
                    {" · "}
                    {new Date(s.result.metadata.timestamp).toLocaleTimeString()}
                  </span>
                </div>

                <div className="grid grid-cols-3 gap-4">
                  <Delta label="Δ PEEP" value={rec.delta_PEEP} unit="cmH₂O" />
                  <Delta label="Δ Tidal Vol" value={rec.delta_TV} unit="mL" />
                  <Delta label="Δ FiO₂" value={rec.delta_FiO2} unit="frac" decimals={2} />
                </div>

                <p className="mt-5 border-l-2 border-cyan-400 pl-3 text-sm leading-relaxed text-slate-700">
                  {rec.action_text}
                </p>

                {rec.track_info.tv_masked && (
                  <Note>Pressure-control mode: tidal-volume changes masked out (ΔTV not directly settable).</Note>
                )}
                {rec.track_info.in_support === false && (
                  <Note>
                    Out of training support (coverage {Math.round((rec.track_info.support_ratio ?? 0) * 100)}%) —
                    this state is unlike the training data; treat with extra caution.
                  </Note>
                )}

                {rec.alternatives?.length > 0 && (
                  <div className="mt-4 border-t border-slate-200/70 pt-3">
                    <div className="caption">
                      Next-best alternative{rec.alternatives.length > 1 ? "s" : ""}
                    </div>
                    {rec.alternatives.map((a, i) => (
                      <p key={i} className="mt-1 flex items-baseline justify-between gap-3 text-xs text-slate-600">
                        <span>{a.action_text}</span>
                        <span className="num shrink-0 text-slate-400">
                          margin {a.margin_from_best.toFixed(2)}
                        </span>
                      </p>
                    ))}
                  </div>
                )}

                <div className="mt-5"><ConfidenceBar confidence={rec.track_info.confidence}
                  track={rec.track_info.track} imputationUsed={rec.track_info.imputation_used}
                  decisionMargin={rec.track_info.decision_margin} /></div>
              </section>

              <section className={`panel animate-rise p-5 ${
                s.result.safety.all_clear
                  ? "border-emerald-200 bg-emerald-50/80"
                  : "border-rose-200 bg-rose-50/80"}`}>
                <div className="flex items-center gap-2 font-display text-sm font-semibold">
                  {s.result.safety.all_clear
                    ? <><ShieldCheck className="text-emerald-600" size={16} />
                        <span className="text-emerald-800">Safety filter — all clear</span></>
                    : <><AlertTriangle className="text-rose-600" size={16} />
                        <span className="text-rose-800">Safety alerts</span></>}
                </div>
                {s.result.safety.flags.map((f, i) => (
                  <p key={i} className={`mt-2 flex gap-2 text-xs leading-relaxed ${
                    f.level === "CRITICAL" ? "text-rose-700" : "text-amber-700"}`}>
                    <span className="mt-px shrink-0 rounded border border-current px-1 text-[9px] uppercase tracking-wide opacity-70">
                      {f.level}
                    </span>
                    {f.message}
                  </p>
                ))}
                {s.result.safety.all_clear && s.result.safety.flags.length === 0 && (
                  <p className="mt-1.5 flex items-center gap-1.5 text-xs text-emerald-700/80">
                    <CheckCircle2 size={12} /> No constraint violated by this action.
                  </p>
                )}
              </section>

              <section className="panel animate-rise p-5">
                <SectionTitle icon={<Sparkles size={14} />} title="Why this recommendation" />
                <ul className="mt-4 space-y-2">
                  {s.result.explanation.top_features.map((f, i) => {
                    const mag = Math.min(1, Math.abs(f.shap_value) /
                      Math.max(...s.result!.explanation.top_features.map((x) => Math.abs(x.shap_value) || 1)));
                    const up = f.direction === "up";
                    return (
                      <li key={i}>
                        <div className="flex items-baseline justify-between gap-3 text-xs">
                          <span className="text-slate-700">{f.display}</span>
                          <span className={`num shrink-0 font-semibold ${up ? "text-emerald-600" : "text-rose-600"}`}>
                            {f.shap_value > 0 ? "+" : ""}{f.shap_value.toFixed(3)}
                          </span>
                        </div>
                        <div className="mt-1 h-1 w-full overflow-hidden rounded-full bg-slate-200">
                          <div className={`h-full rounded-full ${up ? "bg-emerald-500" : "bg-rose-500"}`}
                            style={{ width: `${Math.max(4, mag * 100)}%` }} />
                        </div>
                      </li>
                    );
                  })}
                </ul>
                <p className="mt-4 rounded-lg border border-slate-200 bg-slate-50 p-3 font-mono text-[11px] leading-relaxed text-slate-600">
                  {s.result.explanation.decision_rule}
                </p>
              </section>
            </>
          )}

          {/* Everything previously asked for this patient, read back from MongoDB. */}
          <RecommendationHistory patientId={patient.id} refreshKey={s.savedCount} />

          {/* Model-level evidence — the same for every patient, so it lives on its
              own page; this is only the doorway to it. */}
          <ValidationLink track={s.selectedTrack === "track_b" ? "b" : "a"} />
        </div>
      </div>
    </div>
  );
}

function SectionTitle({ icon, title, children }: {
  icon: ReactNode; title: string; children?: ReactNode;
}) {
  return (
    <div className="flex items-center justify-between gap-3">
      <h2 className="flex items-center gap-2 font-display text-sm font-semibold tracking-wide text-slate-900">
        <span className="text-cyan-600">{icon}</span> {title}
      </h2>
      {children}
    </div>
  );
}

function Note({ children }: { children: ReactNode }) {
  return (
    <p className="mt-3 flex gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-[11px] leading-relaxed text-amber-800">
      <AlertTriangle size={13} className="mt-px shrink-0" />
      <span>{children}</span>
    </p>
  );
}

function Delta({ label, value, unit, decimals = 0 }:
  { label: string; value: number; unit: string; decimals?: number }) {
  const hold = value === 0;
  const color = hold ? "text-slate-400"
    : value > 0 ? "text-emerald-600" : "text-rose-600";
  const text = `${value > 0 ? "+" : ""}${value.toFixed(decimals)}`;
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50/80 px-3 py-3">
      <div className="caption">{label}</div>
      <div className={`num mt-1 text-[2rem] font-bold leading-none tracking-tight ${color}`}>
        {text}
      </div>
      <div className="mt-1 text-[10px] text-slate-400">{unit}</div>
    </div>
  );
}
