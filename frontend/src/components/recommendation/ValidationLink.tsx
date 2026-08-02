import { useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { ChevronRight, ClipboardCheck } from "lucide-react";
import { getValidation } from "../../api/client";
import type { ValidationResponse } from "../../types/validation";

/**
 * Entry point to the model-evidence page. The evaluation record is model-level —
 * identical for every patient — so it does not belong inline next to a
 * patient-specific recommendation; this is only the doorway to it.
 *
 * It still fetches, so the summary line carries real counts and the card can hide
 * itself when no evaluation record exists.
 */
export function ValidationLink({ track = "a" }: { track?: string }) {
  const [data, setData] = useState<ValidationResponse | null>(null);
  const [failed, setFailed] = useState(false);
  const location = useLocation();

  useEffect(() => {
    let live = true;
    getValidation(track)
      .then((d) => live && setData(d))
      .catch(() => live && setFailed(true));
    return () => { live = false; };
  }, [track]);

  // A missing evaluation record must not break the clinical view.
  if (failed || !data || (!data.estimators.length && !data.safety.length)) return null;

  return (
    <Link
      to={`/validation?track=${track}`}
      state={{ from: location.pathname }}
      className="panel panel-hover flex items-center justify-between gap-3 p-5 text-left"
    >
      <div className="flex items-center gap-2">
        <span className="text-cyan-600"><ClipboardCheck size={14} /></span>
        <div>
          <h2 className="font-display text-sm font-semibold tracking-wide text-slate-900">
            How this model was evaluated
          </h2>
          <p className="mt-0.5 text-[11px] text-slate-500">
            {data.cohort_stays
              ? `${data.cohort_stays.toLocaleString()} held-out ICU stays`
              : "Held-out offline evaluation"}
            {" · "}{data.estimators.length} estimator
            {data.estimators.length === 1 ? "" : "s"} vs the clinician baseline
          </p>
        </div>
      </div>
      <div className="flex shrink-0 items-center gap-2">
        {data.any_stale && (
          <span className="rounded border border-amber-300 bg-amber-50 px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-amber-700">
            stale
          </span>
        )}
        <ChevronRight size={16} className="text-slate-400" />
      </div>
    </Link>
  );
}
