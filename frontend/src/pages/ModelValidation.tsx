import { useEffect, useState } from "react";
import { Link, useLocation, useSearchParams } from "react-router-dom";
import { AlertTriangle, ArrowLeft, ClipboardCheck, Info } from "lucide-react";
import { getValidation } from "../api/client";
import { Wordmark } from "../components/shared/Wordmark";
import { UserMenu } from "../components/auth/UserMenu";
import type { OPEEstimator, ValidationResponse } from "../types/validation";

// Diverging pair for "above / below the clinician baseline", validated for CVD
// against this app's surface. Position relative to the baseline line and a signed
// direct label carry the same information, so colour is never the only encoding.
const ABOVE = "#0891b2";   // cyan-600 — the app accent
const BELOW = "#e11d48";   // rose-600

/** Full offline-evaluation record for the deployed policy (GET /api/validation). */
export function ModelValidation() {
  const [params] = useSearchParams();
  const location = useLocation() as { state?: { from?: string } };
  const track = params.get("track") === "b" ? "b" : "a";
  const back = location.state?.from ?? "/";

  const [data, setData] = useState<ValidationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    setData(null);
    setError(null);
    getValidation(track)
      .then((d) => live && setData(d))
      .catch(() => live && setError("Could not load the evaluation record."));
    return () => { live = false; };
  }, [track]);

  const base = data?.baseline?.v_hat ?? 0;

  return (
    <div className="mx-auto max-w-4xl px-6 py-8">
      <div className="mb-6 flex items-center justify-between gap-4">
        <Link to={back} className="btn-ghost">
          <ArrowLeft size={14} /> Back
        </Link>
        <div className="flex items-center gap-3">
          <Wordmark className="text-xl" />
          <UserMenu />
        </div>
      </div>

      <header className="panel mb-6 p-6">
        <div className="flex items-center gap-2 text-cyan-600">
          <ClipboardCheck size={16} />
          <span className="caption text-cyan-700">Model evidence</span>
        </div>
        <h1 className="mt-2 font-display text-3xl font-bold tracking-tight text-slate-900">
          How this model was evaluated
        </h1>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-600">
          The deployed policy is evaluated offline, on held-out data it never trained on,
          against what the clinicians in the record actually did. Nothing here is specific
          to the patient you came from — it describes the model itself.
        </p>
        {data && (
          <p className="num mt-3 text-xs text-slate-500">
            Track {data.track.toUpperCase()}
            {data.cohort_stays ? ` · ${data.cohort_stays.toLocaleString()} ICU stays` : ""}
            {` · ${data.estimators.length} estimator${data.estimators.length === 1 ? "" : "s"}`}
          </p>
        )}
      </header>

      {error && (
        <div className="panel p-10 text-center text-sm text-slate-500">{error}</div>
      )}
      {!error && !data && (
        <div className="panel p-10 text-center text-sm text-slate-400">Loading…</div>
      )}

      {data && (
        <div className="space-y-5">
          {data.any_stale && (
            <p className="flex gap-2 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-xs leading-relaxed text-amber-800">
              <AlertTriangle size={14} className="mt-px shrink-0" />
              <span>
                At least one evaluation predates the deployed checkpoint — it describes a
                model that is no longer being served. Re-run OPE before relying on it.
              </span>
            </p>
          )}

          {/* --- Safety metrics first: same episodes, directly comparable, and far
                  more legible to a clinician than any value estimate. Units differ
                  across rows, so this is a table rather than one shared-axis chart. --- */}
          {data.safety.length > 0 && (
            <section className="panel animate-rise p-6">
              <h2 className="font-display text-sm font-semibold tracking-wide text-slate-900">
                Policy vs clinician — matched rollout episodes
              </h2>
              <p className="mt-1 text-[11px] leading-relaxed text-slate-500">
                Retrospective behaviour over the same simulated episodes from the same
                starting states. This is a report card, not the live safety filter that
                vets each recommendation.
              </p>
              <table className="mt-4 w-full text-xs">
                <thead>
                  <tr className="text-[10px] uppercase tracking-wide text-slate-400">
                    <th className="pb-1.5 text-left font-semibold">Metric</th>
                    <th className="pb-1.5 text-right font-semibold">Clinician</th>
                    <th className="pb-1.5 text-right font-semibold">Policy</th>
                  </tr>
                </thead>
                <tbody>
                  {data.safety.map((m) => {
                    const fmt = (v: number) =>
                      m.unit === "percent" ? `${(v * 100).toFixed(1)}%` : v.toFixed(2);
                    const better = m.higher_is_better
                      ? m.policy > m.clinician
                      : m.policy < m.clinician;
                    return (
                      <tr key={m.key} className="border-t border-slate-100">
                        <td className="py-2.5 pr-3 text-slate-700">
                          {m.label}
                          {m.n != null && (
                            <span className="num ml-1.5 text-[10px] text-slate-400">n={m.n}</span>
                          )}
                        </td>
                        <td className="num py-2.5 text-right text-slate-500">{fmt(m.clinician)}</td>
                        <td className="num py-2.5 text-right font-semibold text-slate-900">
                          {fmt(m.policy)}
                          {/* Word, not colour alone — readable under any CVD. */}
                          <span className="ml-1.5 text-[9px] font-semibold uppercase tracking-wide text-slate-400">
                            {better ? "better" : "worse"}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <p className="mt-3 text-[10px] leading-relaxed text-slate-400">
                Hypoxaemic-start rows rest on a small subgroup — read n before the number.
              </p>
            </section>
          )}

          {/* --- Estimated value, drawn as a deviation from the clinician baseline --- */}
          {data.estimators.length > 0 && data.baseline && (
            <section className="panel animate-rise p-6">
              <h2 className="font-display text-sm font-semibold tracking-wide text-slate-900">
                Estimated value V̂ — deviation from the clinician baseline
              </h2>
              <p className="mt-1 text-[11px] leading-relaxed text-slate-500">
                Three independent estimators of what the policy would be worth if it were
                followed. Bars grow right of the baseline when the policy scores above the
                clinician, left when below.
              </p>
              <div className="mt-5 space-y-4">
                {data.estimators.map((e) => (
                  <EstimatorRow key={e.key} e={e} base={base} all={data.estimators}
                    baseline={data.baseline!} />
                ))}
              </div>
              <p className="num mt-4 border-t border-slate-100 pt-3 text-[10px] text-slate-400">
                Clinician baseline V̂ {base.toFixed(2)}
                {data.baseline.ci_low != null &&
                  ` (CI₉₅ ${data.baseline.ci_low.toFixed(2)} – ${data.baseline.ci_high?.toFixed(2)})`}
                {data.baseline.n != null && ` · ${data.baseline.n.toLocaleString()} episodes`}
              </p>
            </section>
          )}

          <section className="panel animate-rise p-6">
            <h2 className="font-display text-sm font-semibold tracking-wide text-slate-900">
              How to read these numbers
            </h2>
            <p className="mt-2 flex gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2.5 text-[11px] leading-relaxed text-slate-600">
              <Info size={13} className="mt-px shrink-0 text-slate-400" />
              <span>{data.caveat}</span>
            </p>
            {data.model && (
              <p className="num mt-4 text-[10px] leading-relaxed text-slate-400">
                Deployed checkpoint
                {data.model.trained_at &&
                  ` · trained ${new Date(data.model.trained_at).toLocaleDateString()}`}
                {data.model.n_transitions != null &&
                  ` · ${data.model.n_transitions.toLocaleString()} transitions`}
                {data.model.lam_causal != null && ` · λ_causal ${data.model.lam_causal}`}
                {data.model.cql_alpha != null && ` · CQL α ${data.model.cql_alpha}`}
                {data.model.w_outcome != null && ` · w_outcome ${data.model.w_outcome}`}
                {data.model.gamma != null && ` · γ ${data.model.gamma}`}
              </p>
            )}
          </section>
        </div>
      )}
    </div>
  );
}

/** One estimator as a bar growing from the baseline, plus its exact numbers. */
function EstimatorRow({ e, base, all, baseline }: {
  e: OPEEstimator; base: number; all: OPEEstimator[];
  baseline: NonNullable<ValidationResponse["baseline"]>;
}) {
  // Shared domain across every drawn value so the rows are comparable.
  const pts = [base, baseline.ci_low, baseline.ci_high,
    ...all.flatMap((x) => [x.v_hat, x.ci_low, x.ci_high, x.lcb])]
    .filter((v): v is number => v != null);
  const lo = Math.min(...pts), hi = Math.max(...pts);
  const pad = (hi - lo) * 0.08 || 1;
  const pos = (v: number) => ((v - (lo - pad)) / ((hi + pad) - (lo - pad))) * 100;

  const v = e.v_hat ?? base;
  const above = v >= base;
  const delta = v - base;
  const x0 = pos(base), x1 = pos(v);

  return (
    <div className={e.stale ? "opacity-50" : undefined}>
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-xs font-semibold text-slate-700">
          {e.label}
          <span className="ml-1.5 text-[10px] font-normal text-slate-400">{e.blurb}</span>
        </span>
        <span className="num shrink-0 text-xs font-semibold text-slate-900">
          {v.toFixed(2)}
          <span className="ml-1.5 text-[10px] font-normal text-slate-400">
            {delta >= 0 ? "+" : ""}{delta.toFixed(2)} vs clinician
          </span>
        </span>
      </div>

      <div
        className="relative mt-1.5 h-3 w-full rounded bg-slate-100"
        title={`${e.label} V̂ ${v.toFixed(4)}`
          + (e.ci_low != null ? ` · CI₉₅ [${e.ci_low.toFixed(2)}, ${e.ci_high?.toFixed(2)}]` : "")
          + (e.lcb != null ? ` · 5% LCB ${e.lcb.toFixed(2)}` : "")
          + (e.n != null ? ` · ${e.n.toLocaleString()} ${e.n_unit}` : "")
          + (e.timestamp ? ` · evaluated ${new Date(e.timestamp).toLocaleDateString()}` : "")}
      >
        {/* CI extent, drawn under the bar */}
        {e.ci_low != null && e.ci_high != null && (
          <div className="absolute inset-y-0 my-auto h-px bg-slate-300"
            style={{ left: `${pos(e.ci_low)}%`, width: `${pos(e.ci_high) - pos(e.ci_low)}%` }} />
        )}
        {/* The bar: grows from the baseline toward the estimate */}
        <div
          className="absolute inset-y-0 my-auto h-2"
          style={{
            left: `${Math.min(x0, x1)}%`,
            width: `${Math.max(Math.abs(x1 - x0), 0.6)}%`,
            background: above ? ABOVE : BELOW,
            // 4px rounded data-end, square against the baseline it grows from
            borderRadius: above ? "0 4px 4px 0" : "4px 0 0 4px",
          }}
        />
        {/* The baseline itself — the neutral midpoint of the diverging scale */}
        <div className="absolute inset-y-0 w-px bg-slate-400" style={{ left: `${x0}%` }} />
        {/* DFQE's pessimistic bound, the number worth trusting most */}
        {e.lcb != null && (
          <div className="absolute inset-y-0 my-auto h-3 w-0.5 rounded bg-slate-700 ring-2 ring-white"
            style={{ left: `${pos(e.lcb)}%` }} title={`5% lower bound ${e.lcb.toFixed(2)}`} />
        )}
      </div>

      <div className="mt-1 flex items-center justify-between gap-2 text-[10px] text-slate-400">
        <span className="num">
          {e.n != null && `${e.n.toLocaleString()} ${e.n_unit}`}
          {e.lcb != null && ` · 5% LCB ${e.lcb.toFixed(2)}`}
        </span>
        {e.stale
          ? <span className="font-semibold uppercase tracking-wide text-amber-600">stale</span>
          : e.timestamp && (
            <span className="num">{new Date(e.timestamp).toLocaleDateString()}</span>
          )}
      </div>
    </div>
  );
}
