import { useEffect, useState } from "react";
import { AlertTriangle, History, Loader2, ShieldCheck } from "lucide-react";
import { apiErrorMessage, getPatientRecommendations } from "../../api/client";
import type { RecommendationRecord } from "../../types/patient";

/**
 * Everything this patient has been asked about, newest first, read back from
 * the database. Each row is one `Get Recommendation` — change a setting, ask again, and
 * a second row appears under the same patient rather than replacing the first.
 *
 * `refreshKey` is bumped by the store on every saved result so a new ask shows up
 * without a page reload.
 */
export function RecommendationHistory({ patientId, refreshKey }: {
  patientId: string; refreshKey: number;
}) {
  const [records, setRecords] = useState<RecommendationRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    if (!patientId) return;
    let live = true;   // a fast patient switch must not let a stale response land
    setLoading(true);
    getPatientRecommendations(patientId)
      .then((r) => { if (live) { setRecords(r); setError(null); } })
      .catch((e) => { if (live) setError(apiErrorMessage(e, "Could not load saved recommendations")); })
      .finally(() => { if (live) setLoading(false); });
    return () => { live = false; };
  }, [patientId, refreshKey]);

  const shown = expanded ? records : records.slice(0, 5);

  return (
    <section className="panel p-5">
      <div className="flex items-center justify-between gap-3">
        <h2 className="flex items-center gap-2 font-display text-sm font-semibold tracking-wide text-slate-900">
          <span className="text-cyan-600"><History size={14} /></span> Saved history
        </h2>
        <span className="num text-[11px] text-slate-400">
          {loading ? "loading…" : `${records.length} saved`}
        </span>
      </div>

      {error && (
        <p className="mt-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-[11px] text-amber-800">
          {error}
        </p>
      )}

      {loading && records.length === 0 && !error && (
        <div className="mt-4 grid place-items-center py-6 text-slate-300">
          <Loader2 className="animate-spin" size={18} />
        </div>
      )}

      {!loading && !error && records.length === 0 && (
        <p className="mt-3 text-[12px] leading-relaxed text-slate-500">
          Nothing saved yet. Every recommendation you request for this patient is
          stored with the settings it was made from.
        </p>
      )}

      {shown.length > 0 && (
        <ul className="mt-4 space-y-2">
          {shown.map((r) => (
            <li key={r.id} className="rounded-xl border border-slate-200 bg-slate-50/70 p-3">
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <span className="num text-[11px] font-semibold text-slate-700">
                  {new Date(r.created_at).toLocaleString()}
                  {/* Who asked, on a ward machine several clinicians share. Older
                      records predate sign-in and carry no name — omit rather than
                      inventing one. */}
                  {r.clinician && (
                    <span className="ml-1.5 font-sans font-normal text-slate-400">
                      · {r.clinician}
                    </span>
                  )}
                </span>
                <span className="flex items-center gap-2 text-[10px]">
                  <span className="rounded border border-slate-200 bg-white px-1.5 py-px uppercase tracking-wide text-slate-500">
                    {r.track === "track_b" ? "Track B" : "Track A"}
                  </span>
                  {r.safety_all_clear ? (
                    <span className="flex items-center gap-1 text-emerald-600">
                      <ShieldCheck size={11} /> clear
                    </span>
                  ) : (
                    <span className="flex items-center gap-1 text-rose-600">
                      <AlertTriangle size={11} /> {r.safety_flags.length} flag
                      {r.safety_flags.length === 1 ? "" : "s"}
                    </span>
                  )}
                </span>
              </div>

              <p className="mt-1.5 text-[12px] leading-relaxed text-slate-700">{r.action_text}</p>

              {/* The settings this answer came from — what makes two rows for the
                  same patient tell apart. */}
              <div className="num mt-2 flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-slate-500">
                <span>PEEP {r.state.PEEP}</span>
                <span>TV {r.state.TV}</span>
                <span>FiO₂ {r.state.FiO2.toFixed(2)}</span>
                <span>SpO₂ {r.state.SpO2}</span>
                <span>PaCO₂ {r.state.PaCO2}</span>
                <span>pH {r.state.pH}</span>
                {r.responsiveness > 0 && <span>resp {Math.round(r.responsiveness * 100)}%</span>}
                {r.ventilation_mode && <span>{r.ventilation_mode.replace("_", " ")}</span>}
                {r.confidence != null && <span>conf {Math.round(r.confidence * 100)}%</span>}
              </div>
            </li>
          ))}
        </ul>
      )}

      {records.length > 5 && (
        <button onClick={() => setExpanded((v) => !v)} className="btn-ghost mt-3">
          {expanded ? "Show fewer" : `Show all ${records.length}`}
        </button>
      )}
    </section>
  );
}
