import { Activity, Stethoscope } from "lucide-react";
import type { Track } from "../../types/recommendation";

export function TrackBadge({ track, label, waveformUsed }: {
  track: Track; label: string; waveformUsed: boolean;
}) {
  const isB = track === "track_b";
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-sm font-medium ${
        isB ? "bg-blue-100 text-blue-800" : "bg-slate-200 text-slate-700"
      }`}
    >
      {isB ? <Activity size={15} /> : <Stethoscope size={15} />}
      {label}
      {isB && !waveformUsed && <span className="text-xs opacity-70">(no waveform)</span>}
    </span>
  );
}
