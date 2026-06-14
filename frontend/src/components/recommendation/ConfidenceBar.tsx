import type { Track } from "../../types/recommendation";

export function ConfidenceBar({ confidence, track, imputationUsed }: {
  confidence: number; track: Track; imputationUsed?: boolean | null;
}) {
  const pct = Math.round(confidence * 100);
  const color =
    track === "track_a" ? "bg-slate-400"
    : imputationUsed ? "bg-amber-400" : "bg-green-500";
  const note =
    track === "track_a" ? "Based on clinical data only"
    : imputationUsed ? "Based on clinical + partial (imputed) waveform"
    : "Based on clinical + waveform data";
  return (
    <div>
      <div className="flex items-center justify-between text-xs text-slate-500">
        <span>Confidence</span>
        <span className="font-semibold text-slate-700">{pct}%</span>
      </div>
      <div className="mt-1 h-2 w-full rounded bg-slate-200">
        <div className={`h-2 rounded ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <p className="mt-1 text-xs text-slate-400">{note}</p>
    </div>
  );
}
