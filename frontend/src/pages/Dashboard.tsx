import { AlertTriangle, CheckCircle2, Wind } from "lucide-react";
import { useStore } from "../store/useStore";
import { TrackBadge } from "../components/shared/TrackBadge";
import { ConfidenceBar } from "../components/recommendation/ConfidenceBar";
import { TrackSelector } from "../components/recommendation/TrackSelector";
import { PATIENT_PRESETS, CUSTOM_LABEL } from "../data/patientPresets";
import type { TabularState } from "../types/recommendation";

const FIELDS: [keyof TabularState, string, number][] = [
  ["PEEP", "PEEP (cmH₂O)", 1], ["TV", "Tidal Vol (mL)", 10], ["FiO2", "FiO₂", 0.05],
  ["SpO2", "SpO₂ (%)", 1], ["PaO2", "PaO₂", 1], ["PaCO2", "PaCO₂", 1], ["pH", "pH", 0.01],
  ["HR", "HR", 1], ["SBP", "SBP", 1], ["RR", "RR", 1], ["RASS", "RASS", 1], ["Temp", "Temp (°C)", 0.1],
];

export function Dashboard() {
  const s = useStore();
  const rec = s.result?.recommendation;
  return (
    <div className="mx-auto max-w-6xl p-6">
      <header className="mb-6 flex items-center gap-3">
        <Wind className="text-blue-600" size={28} />
        <div>
          <h1 className="text-2xl font-bold">VentAssist</h1>
          <p className="text-sm text-slate-500">Dual-Track offline-RL ventilator decision support</p>
        </div>
      </header>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* ---- Inputs ---- */}
        <section className="rounded-xl bg-white p-5 shadow-sm">
          <div className="mb-3 flex items-center justify-between gap-3">
            <h2 className="font-semibold">Patient State</h2>
            <label className="flex items-center gap-2 text-xs text-slate-500">
              Preset
              <select
                value={s.presetKey}
                onChange={(e) => s.loadPreset(e.target.value)}
                className="rounded border border-slate-300 px-2 py-1 text-sm text-slate-800"
              >
                {s.presetKey === CUSTOM_LABEL && (
                  <option value={CUSTOM_LABEL} disabled>{CUSTOM_LABEL}</option>
                )}
                {PATIENT_PRESETS.map((p) => (
                  <option key={p.key} value={p.key}>{p.label}</option>
                ))}
              </select>
            </label>
          </div>
          {s.presetKey !== CUSTOM_LABEL && (
            <p className="mb-3 text-xs text-slate-400">
              {PATIENT_PRESETS.find((p) => p.key === s.presetKey)?.hint}
            </p>
          )}
          <div className="grid grid-cols-3 gap-3">
            {FIELDS.map(([k, label, step]) => (
              <label key={k} className="text-xs text-slate-500">
                {label}
                <input
                  type="number" step={step} value={s.tabular[k]}
                  onChange={(e) => s.setField(k, parseFloat(e.target.value))}
                  className="mt-1 w-full rounded border border-slate-300 px-2 py-1 text-sm text-slate-800"
                />
              </label>
            ))}
          </div>

          <div className="mt-5 mb-2 flex items-center justify-between">
            <h2 className="font-semibold">Policy Track</h2>
            <label className="flex items-center gap-2 text-xs text-slate-500">
              <input type="checkbox" checked={s.waveformAvailable}
                onChange={(e) => s.setMeta({ waveformAvailable: e.target.checked,
                  selectedTrack: e.target.checked ? s.selectedTrack : "track_a" })} />
              waveform available
            </label>
          </div>
          <TrackSelector current={s.selectedTrack}
            onSelect={(t) => s.setMeta({ selectedTrack: t })}
            waveformAvailable={s.waveformAvailable} />

          <div className="mt-4">
            <div className="flex items-center justify-between text-xs text-slate-500">
              <span>Responsiveness</span>
              <span className="font-semibold text-slate-700">
                {s.responsiveness === 0 ? "Conservative (deployed)"
                  : `${Math.round(s.responsiveness * 100)}% eager to act`}
              </span>
            </div>
            <input type="range" min={0} max={1} step={0.05} value={s.responsiveness}
              onChange={(e) => s.setMeta({ responsiveness: parseFloat(e.target.value) })}
              className="mt-1 w-full accent-blue-600" />
            <p className="mt-1 text-xs text-slate-400">
              Lowers the bar to leave “hold” — never changes which change is chosen when acting,
              and the safety filter still applies.
            </p>
          </div>

          <div className="mt-4">
            <label className="flex items-center justify-between text-xs text-slate-500">
              <span>Ventilation mode</span>
              <select
                value={s.ventilationMode}
                onChange={(e) => s.setMeta({ ventilationMode: e.target.value })}
                className="rounded border border-slate-300 px-2 py-1 text-sm text-slate-800"
              >
                <option value="unknown">Unknown / not set</option>
                <option value="volume_control">Volume control</option>
                <option value="pressure_control">Pressure control</option>
              </select>
            </label>
            <p className="mt-1 text-xs text-slate-400">
              In pressure control, tidal volume isn’t directly set — ΔTV recommendations are masked out.
            </p>
          </div>

          {s.selectedTrack === "track_b" && (
            <div className="mt-3 grid grid-cols-3 gap-3">
              {([["ecgHRV", "HRV SDNN"], ["ecgArr", "Arrhythmia"], ["pleth", "Perfusion Idx"],
                 ["rrv", "RRV"], ["breathReg", "Breath Reg"], ["asynchrony", "Asynchrony"]] as const)
                .map(([k, label]) => (
                <label key={k} className="text-xs text-slate-500">
                  {label}
                  <input type="number" step={0.01} value={(s as any)[k]}
                    onChange={(e) => s.setMeta({ [k]: parseFloat(e.target.value) } as any)}
                    className="mt-1 w-full rounded border border-slate-300 px-2 py-1 text-sm" />
                </label>
              ))}
            </div>
          )}

          <button onClick={() => s.fetch()} disabled={s.loading}
            className="mt-5 w-full rounded-lg bg-blue-600 py-2.5 font-semibold text-white hover:bg-blue-700 disabled:opacity-50">
            {s.loading ? "Computing…" : "Get Recommendation"}
          </button>
          {s.error && <p className="mt-2 text-sm text-red-600">Error: {s.error}</p>}
        </section>

        {/* ---- Output ---- */}
        <section className="space-y-4">
          {!rec && <div className="rounded-xl bg-white p-8 text-center text-slate-400 shadow-sm">
            Enter a patient state and request a recommendation.</div>}

          {rec && s.result && (
            <>
              <div className="rounded-xl bg-white p-5 shadow-sm">
                <div className="mb-3 flex items-center justify-between">
                  <TrackBadge track={rec.track_info.track} label={rec.track_info.track_label}
                    waveformUsed={rec.track_info.waveform_used} />
                  <span className="text-xs text-slate-400">{s.result.metadata.latency_ms} ms</span>
                </div>
                <div className="flex gap-6">
                  <Delta label="Δ PEEP" value={rec.delta_PEEP} unit="cmH₂O" />
                  <Delta label="Δ Tidal Volume" value={rec.delta_TV} unit="mL" />
                  <Delta label="Δ FiO₂" value={rec.delta_FiO2} unit="" decimals={2} />
                </div>
                <p className="mt-2 text-sm text-slate-600">{rec.action_text}</p>
                {rec.track_info.tv_masked && (
                  <p className="mt-1 text-xs text-amber-600">
                    Pressure-control mode: tidal-volume changes masked out (ΔTV not directly settable).
                  </p>
                )}
                {rec.track_info.in_support === false && (
                  <p className="mt-1 text-xs text-amber-600">
                    ⚠ Out of training support (coverage {Math.round((rec.track_info.support_ratio ?? 0) * 100)}%) —
                    this patient state is unlike the training data; treat the recommendation with extra caution.
                  </p>
                )}
                {rec.alternatives?.length > 0 && (
                  <div className="mt-3 border-t border-slate-100 pt-2">
                    <div className="text-xs font-semibold text-slate-400">Next-best alternative{rec.alternatives.length > 1 ? "s" : ""}</div>
                    {rec.alternatives.map((a, i) => (
                      <p key={i} className="text-xs text-slate-500">
                        {a.action_text} <span className="text-slate-400">(margin {a.margin_from_best.toFixed(2)})</span>
                      </p>
                    ))}
                  </div>
                )}
                <div className="mt-4"><ConfidenceBar confidence={rec.track_info.confidence}
                  track={rec.track_info.track} imputationUsed={rec.track_info.imputation_used}
                  decisionMargin={rec.track_info.decision_margin} /></div>
              </div>

              <div className={`rounded-xl p-4 shadow-sm ${
                s.result.safety.all_clear ? "bg-green-50" : "bg-red-50"}`}>
                <div className="flex items-center gap-2 font-semibold">
                  {s.result.safety.all_clear
                    ? <><CheckCircle2 className="text-green-600" size={18} /> Safety: all clear</>
                    : <><AlertTriangle className="text-red-600" size={18} /> Safety alerts</>}
                </div>
                {s.result.safety.flags.map((f, i) => (
                  <p key={i} className={`mt-1 text-sm ${
                    f.level === "CRITICAL" ? "text-red-700" : "text-amber-700"}`}>
                    [{f.level}] {f.message}</p>
                ))}
              </div>

              <div className="rounded-xl bg-white p-5 shadow-sm">
                <h3 className="mb-2 font-semibold">Why this recommendation</h3>
                <ul className="space-y-1 text-sm">
                  {s.result.explanation.top_features.map((f, i) => (
                    <li key={i} className="flex justify-between">
                      <span>{f.display}</span>
                      <span className={f.direction === "up" ? "text-green-600" : "text-red-600"}>
                        {f.shap_value > 0 ? "+" : ""}{f.shap_value.toFixed(3)}</span>
                    </li>
                  ))}
                </ul>
                <p className="mt-3 rounded bg-slate-50 p-2 font-mono text-xs text-slate-600">
                  {s.result.explanation.decision_rule}</p>
              </div>
            </>
          )}
        </section>
      </div>
    </div>
  );
}

function Delta({ label, value, unit, decimals = 0 }:
  { label: string; value: number; unit: string; decimals?: number }) {
  const color = value > 0 ? "text-green-600" : value < 0 ? "text-red-600" : "text-slate-500";
  const text = `${value > 0 ? "+" : ""}${value.toFixed(decimals)}`;
  return (
    <div>
      <div className="text-xs text-slate-500">{label}</div>
      <div className={`text-3xl font-bold ${color}`}>{text}</div>
      <div className="text-xs text-slate-400">{unit}</div>
    </div>
  );
}
