import { AlertTriangle, CheckCircle2, Wind } from "lucide-react";
import { useStore } from "../store/useStore";
import { TrackBadge } from "../components/shared/TrackBadge";
import { ConfidenceBar } from "../components/recommendation/ConfidenceBar";
import { TrackSelector } from "../components/recommendation/TrackSelector";
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
          <h2 className="mb-3 font-semibold">Patient State</h2>
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
                <div className="mt-4"><ConfidenceBar confidence={rec.track_info.confidence}
                  track={rec.track_info.track} imputationUsed={rec.track_info.imputation_used} /></div>
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
