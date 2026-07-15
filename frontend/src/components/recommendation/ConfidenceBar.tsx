import type { Track } from "../../types/recommendation";

export function ConfidenceBar({ confidence, track, imputationUsed, decisionMargin }: {
  confidence: number; track: Track; imputationUsed?: boolean | null;
  decisionMargin?: number | null;
}) {
  const pct = Math.round(confidence * 100);
  // Colour by strength of the decision, not by track: red (toss-up) → amber → green.
  const color = pct >= 65 ? "bg-green-500" : pct >= 50 ? "bg-amber-400" : "bg-red-400";
  const source =
    track === "track_a" ? "clinical data"
    : imputationUsed ? "clinical + partial (imputed) waveform"
    : "clinical + waveform data";
  const margin = decisionMargin ?? null;
  const decisiveness =
    margin == null ? ""
    : margin >= 0.5 ? " · clear margin over next-best"
    : margin >= 0.2 ? " · modest margin over next-best"
    : " · near-tie with next-best";
  return (
    <div>
      <div className="flex items-center justify-between text-xs text-slate-500">
        <span>Confidence</span>
        <span className="font-semibold text-slate-700">{pct}%</span>
      </div>
      <div className="mt-1 h-2 w-full rounded bg-slate-200">
        <div className={`h-2 rounded ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <p className="mt-1 text-xs text-slate-400">
        Q-margin decision confidence on {source}{decisiveness}
      </p>
    </div>
  );
}
