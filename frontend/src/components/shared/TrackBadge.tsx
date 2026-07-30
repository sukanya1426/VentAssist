import { Activity, Stethoscope } from "lucide-react";
import type { Track } from "../../types/recommendation";

export function TrackBadge({ track, label, waveformUsed }: {
  track: Track; label: string; waveformUsed: boolean;
}) {
  const isB = track === "track_b";
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium tracking-wide ${
        isB
          ? "border-indigo-200 bg-indigo-50 text-indigo-700"
          : "border-cyan-200 bg-cyan-50 text-cyan-700"
      }`}
    >
      {isB ? <Activity size={14} /> : <Stethoscope size={14} />}
      {label}
      {isB && !waveformUsed && <span className="opacity-60">· no waveform</span>}
    </span>
  );
}
