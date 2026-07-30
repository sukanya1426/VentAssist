import type { Track } from "../../types/recommendation";

export function ConfidenceBar({ confidence, track, imputationUsed, decisionMargin }: {
  confidence: number; track: Track; imputationUsed?: boolean | null;
  decisionMargin?: number | null;
}) {
  const pct = Math.round(confidence * 100);
  // Colour by strength of the decision, not by track: red (toss-up) → amber → green.
  const bar =
    pct >= 65 ? "from-emerald-500 to-teal-400"
    : pct >= 50 ? "from-amber-500 to-yellow-400"
    : "from-rose-500 to-orange-400";
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
      <div className="flex items-baseline justify-between">
        <span className="caption">Confidence</span>
        <span className="num text-sm font-semibold text-slate-900">{pct}%</span>
      </div>
      <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-slate-200">
        <div
          className={`h-full rounded-full bg-gradient-to-r ${bar} transition-[width] duration-500`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <p className="mt-1.5 text-[11px] leading-relaxed text-slate-500">
        Q-margin decision confidence on {source}{decisiveness}
      </p>
    </div>
  );
}
