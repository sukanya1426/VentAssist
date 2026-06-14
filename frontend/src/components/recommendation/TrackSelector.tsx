import type { Track } from "../../types/recommendation";

export function TrackSelector({ current, onSelect, waveformAvailable }: {
  current: Track; onSelect: (t: Track) => void; waveformAvailable: boolean;
}) {
  const base = "flex-1 rounded-lg border px-4 py-3 text-left transition";
  return (
    <div className="flex gap-3">
      <button
        onClick={() => onSelect("track_a")}
        className={`${base} ${current === "track_a"
          ? "border-slate-700 bg-slate-700 text-white"
          : "border-slate-300 bg-white hover:border-slate-400"}`}
      >
        <div className="font-semibold">📋 Clinical Policy</div>
        <div className="text-xs opacity-80">12 clinical features · always available</div>
      </button>
      <button
        onClick={() => waveformAvailable && onSelect("track_b")}
        disabled={!waveformAvailable}
        title={!waveformAvailable ? "No waveform data for this patient" : ""}
        className={`${base} ${current === "track_b"
          ? "border-blue-600 bg-blue-600 text-white"
          : "border-slate-300 bg-white hover:border-blue-400"} ${
          !waveformAvailable ? "cursor-not-allowed opacity-50" : ""}`}
      >
        <div className="font-semibold">〰️ Waveform Policy <span className="text-xs">(PoC)</span></div>
        <div className="text-xs opacity-80">
          18 features · {waveformAvailable ? "waveform available" : "unavailable"}
        </div>
      </button>
    </div>
  );
}
