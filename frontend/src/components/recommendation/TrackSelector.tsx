import { Activity, Stethoscope } from "lucide-react";
import type { Track } from "../../types/recommendation";

export function TrackSelector({ current, onSelect, waveformAvailable }: {
  current: Track; onSelect: (t: Track) => void; waveformAvailable: boolean;
}) {
  const base = "flex-1 rounded-xl border px-3.5 py-3 text-left transition duration-200";
  const idle = "border-slate-200 bg-white/60 hover:border-slate-300 hover:bg-white";
  return (
    <div className="flex gap-3">
      <button
        onClick={() => onSelect("track_a")}
        className={`${base} ${current === "track_a"
          ? "border-cyan-400 bg-cyan-50 shadow-[0_10px_26px_-16px_rgba(8,145,178,0.9)]"
          : idle}`}
      >
        <div className="flex items-center gap-2 font-display text-sm font-semibold text-slate-900">
          <Stethoscope size={15} className="text-cyan-600" /> Clinical Policy
        </div>
        <div className="mt-0.5 text-[11px] text-slate-500">12 features · always available</div>
      </button>
      <button
        onClick={() => waveformAvailable && onSelect("track_b")}
        disabled={!waveformAvailable}
        title={!waveformAvailable ? "No waveform data for this patient" : ""}
        className={`${base} ${current === "track_b"
          ? "border-indigo-400 bg-indigo-50 shadow-[0_10px_26px_-16px_rgba(79,70,229,0.9)]"
          : idle} ${!waveformAvailable ? "cursor-not-allowed opacity-50" : ""}`}
      >
        <div className="flex items-center gap-2 font-display text-sm font-semibold text-slate-900">
          <Activity size={15} className="text-indigo-600" /> Waveform Policy
          <span className="rounded bg-slate-100 px-1 py-px text-[9px] font-medium tracking-wide text-slate-500">
            PoC
          </span>
        </div>
        <div className="mt-0.5 text-[11px] text-slate-500">
          18 features · waveform {waveformAvailable ? "available" : "unavailable"}
        </div>
      </button>
    </div>
  );
}
