"""Is the confidence number calibrated, and calibrated against *what*?

The UI shows a per-recommendation confidence. It is currently a **decision
confidence** — ``clip(info_weight × sigmoid(margin / temp), 0.30, 0.97)``, where
``margin`` is how far the chosen action's Q-value beats the best alternative. That
is a statement about the *policy's internal decisiveness*. It is not a
probability of anything in the world, and the SRS should not imply otherwise.

This module measures whether it behaves like one.

**CHOOSING THE TARGET EVENT.** "Calibrate against held-out outcomes" is
under-specified, because confidence could be calibrated against several
different events and only one of them is defensible:

* **Agreement with the clinician** (PRIMARY, used here). "When the system says
  0.8, how often does it pick the action the clinician actually chose?" This is
  directly measurable on held-out data, it is the event a clinician implicitly
  reads the number as, and it is the only one where miscalibration is genuinely
  the confidence's fault. Reported as a reliability diagram, ECE, MCE and Brier.

* **Patient mortality** (SECONDARY, descriptive only). Reported because the task
  asked for outcomes, but flagged as **confounded, not a calibration target**: a
  confident recommendation on a dying patient is not miscalibrated, and the
  confidence never claimed to predict survival. Treating a mortality association
  here as calibration error would be a category mistake. It is included so the
  association can be *stated and dismissed* rather than left as an open question.

A ceiling worth knowing before reading the numbers: Track A's ``info_weight`` is
capped at ``CONF_WEIGHT_CLINICAL = 0.85`` and the output is clipped to
``[0.30, 0.97]``, so the achievable range is roughly **[0.30, 0.85]**. Perfect
calibration against a 0–1 event is therefore *impossible by construction* — the
number cannot express certainty above 0.85 or doubt below 0.30. That is a design
choice (clinical-only data should never claim near-certainty), but it means the
right question is whether confidence is **monotonically informative**, not
whether ECE is near zero.

Run:  PYTHONPATH=. .venv/bin/python -m benchmark.confidence_calibration --track a
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from backend.mdp import dataset as D
from backend.mdp import normaliser as N
from backend.pipeline import config
from backend.pipeline.logging_utils import get_logger
from backend.router import policy_router as PR

log = get_logger("bench_calib")

RESULTS = Path(__file__).resolve().parent / "results"

N_BINS = 10


def _q_matrix(policy, z: np.ndarray, chunk: int = 32768) -> np.ndarray:
    """All 125 Q-values per state, chunked. ``q_values`` is single-row only."""
    out = np.empty((len(z), policy.action_dim), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(z), chunk):
            s = torch.as_tensor(z[i:i + chunk], dtype=torch.float32,
                                device=policy.device)
            out[i:i + chunk] = policy.Q(policy._phi(s)).cpu().numpy()
    return out


def _confidences(Q: np.ndarray, ood_ratio: np.ndarray | None) -> tuple:
    """Per-row (confidence, margin, chosen action) exactly as the router computes it.

    Delegates to ``PR._decision_confidence`` and ``PR._ood_weight`` row by row
    rather than reimplementing the formula: a calibration study of a *different*
    formula than the one being served would be worse than no study.
    """
    n = len(Q)
    conf = np.empty(n, dtype=float)
    margin = np.empty(n, dtype=float)
    chosen = Q.argmax(axis=1).astype(np.int64)
    for i in range(n):
        q = Q[i]
        iw = PR.CONF_WEIGHT_CLINICAL
        if ood_ratio is not None:
            iw = PR._ood_weight(iw, {"support_ratio": float(ood_ratio[i]),
                                     "in_support": True})
        c = PR._decision_confidence(q, iw, chosen=int(chosen[i]))
        conf[i], margin[i] = c["confidence"], c["decision_margin"]
    return conf, margin, chosen


def _reliability(conf: np.ndarray, hit: np.ndarray, n_bins: int = N_BINS) -> dict:
    """Equal-width reliability table + ECE / MCE / Brier.

    Equal-width bins over the *observed* confidence range rather than over [0, 1]:
    the confidence is structurally confined to ~[0.30, 0.85], so fixed [0, 1]
    deciles would leave most bins empty and make ECE look better than it is by
    averaging over bins that contain nothing.
    """
    lo, hi = float(conf.min()), float(conf.max())
    if hi - lo < 1e-9:
        return {"note": f"confidence is constant at {lo:.4f} — no spread to calibrate",
                "bins": [], "ece": None, "mce": None}
    edges = np.linspace(lo, hi + 1e-9, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    rows, ece, mce, n = [], 0.0, 0.0, len(conf)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        c_mean, h_mean, cnt = float(conf[m].mean()), float(hit[m].mean()), int(m.sum())
        gap = abs(c_mean - h_mean)
        ece += cnt / n * gap
        mce = max(mce, gap)
        rows.append({"bin": b,
                     "range": [round(float(edges[b]), 4), round(float(edges[b + 1]), 4)],
                     "n": cnt, "share": round(cnt / n, 4),
                     "mean_confidence": round(c_mean, 4),
                     "observed_rate": round(h_mean, 4),
                     "gap_confidence_minus_observed": round(c_mean - h_mean, 4)})
    return {"bins": rows,
            "ece": round(float(ece), 5),
            "mce": round(float(mce), 5),
            "brier": round(float(np.mean((conf - hit) ** 2)), 5),
            "base_rate": round(float(hit.mean()), 5),
            "confidence_range_observed": [round(lo, 4), round(hi, 4)]}


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation without scipy — the monotonic-informativeness check."""
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def evaluate(track: str = "a", write: bool = True, cap: int | None = None) -> dict:
    d = D.load_mdp(track)
    feats = list(d["feature_order"])
    nf = "normaliser_stats.json" if track == "a" else "normaliser_stats_track_b.json"
    stats = N.load(config.MODEL_PATH / nf)

    test = np.where(d["split"] == "test")[0]
    if len(test) == 0:
        raise RuntimeError("MDP has no test split.")
    if cap:
        test = test[:cap]

    raw = d["states"][test]
    # Serving encoding — the confidence being audited is the one the UI shows.
    z = N.transform_inference(raw, stats, feats).astype(np.float32)

    from backend.ope.fqe import _load_policy
    policy, _ = _load_policy(track)

    # The learned OOD detector tempers info_weight at serving time, so a study
    # that ignored it would audit a confidence nobody is ever shown.
    ood_ratio = None
    try:
        from backend.rl.ood_autoencoder import OODDetector
        det = OODDetector.load(track)
        ood_ratio = np.array([det.evaluate(zi)["support_ratio"] for zi in z])
    except Exception as e:                                   # detector optional
        log.info("OOD detector unavailable (%s) — auditing untempered confidence", e)

    Q = _q_matrix(policy, z)
    conf, margin, chosen = _confidences(Q, ood_ratio)
    clin = d["actions"][test]
    agree = (chosen == clin).astype(float)

    result = {
        "track": track,
        "n_test": int(len(test)),
        "n_transitions": int(len(d["actions"])),
        "structural_ceiling": {
            "info_weight_cap": PR.CONF_WEIGHT_CLINICAL,
            "clip": [PR.CONF_FLOOR, PR.CONF_CEIL],
            "achievable_range": [PR.CONF_FLOOR,
                                 round(PR.CONF_WEIGHT_CLINICAL * 0.999, 4)],
            "implication": "a 0-1 event cannot be perfectly calibrated against a "
                           "number confined to this range; read monotonicity "
                           "(spearman) as the primary result, not ECE",
        },
        "primary_agreement_with_clinician": {
            **_reliability(conf, agree),
            "spearman_confidence_vs_agreement": round(_spearman(conf, agree), 5),
            "target": "P(policy action == clinician action), 125-way",
        },
        "confidence_distribution": {
            "mean": round(float(conf.mean()), 4),
            "sd": round(float(conf.std()), 4),
            "p05": round(float(np.percentile(conf, 5)), 4),
            "median": round(float(np.median(conf)), 4),
            "p95": round(float(np.percentile(conf, 95)), 4),
            "share_at_floor": round(float(np.mean(conf <= PR.CONF_FLOOR + 1e-9)), 4),
            "share_at_ceiling": round(
                float(np.mean(conf >= PR.CONF_WEIGHT_CLINICAL - 1e-3)), 4),
        },
        "margin_distribution": {
            "mean": round(float(margin.mean()), 4),
            "median": round(float(np.median(margin)), 4),
            "share_negative": round(float(np.mean(margin < 0)), 4),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # --- secondary, explicitly confounded: association with mortality ----------
    out_path = config.PROCESSED_PATH / "outcomes.csv"
    if out_path.exists():
        oc = pd.read_csv(out_path)
        if "outcome_label_known" in oc.columns:
            oc = oc[oc["outcome_label_known"] == 1]
        died_by_stay = dict(zip(oc["stay_id"].astype(int),
                                pd.to_numeric(oc["died_horizon"], errors="coerce")))
        died = np.array([died_by_stay.get(int(s), np.nan) for s in d["stay_id"][test]])
        ok = np.isfinite(died)
        if ok.sum() > 100:
            surv = 1.0 - died[ok]
            result["secondary_outcome_association"] = {
                "n_with_outcome": int(ok.sum()),
                "survival_base_rate": round(float(surv.mean()), 5),
                "spearman_confidence_vs_survival": round(
                    _spearman(conf[ok], surv), 5),
                "mean_confidence_survivors": round(float(conf[ok][surv == 1].mean()), 4),
                "mean_confidence_died": round(float(conf[ok][surv == 0].mean()), 4),
                "interpretation": "DESCRIPTIVE ONLY, NOT a calibration target. "
                                  "Confidence measures decision sharpness, never "
                                  "P(survival); a confident recommendation on a "
                                  "dying patient is not miscalibrated. Any gap here "
                                  "is confounded by severity, which drives both the "
                                  "outcome and how decisive the Q-function is.",
            }

    result["verdict"] = (
        "confidence is monotonically informative about clinician agreement"
        if result["primary_agreement_with_clinician"].get(
            "spearman_confidence_vs_agreement", 0) > 0.1 else
        "confidence carries little information about clinician agreement — it "
        "should be presented as a decision margin, not as a probability")

    if write:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"confidence_calibration_track_{track}.json").write_text(
            json.dumps(result, indent=2))

    p = result["primary_agreement_with_clinician"]
    cd = result["confidence_distribution"]
    log.info("Confidence calibration on %d held-out states:", len(test))
    log.info("  achievable range %s (structural)", result["structural_ceiling"]["achievable_range"])
    log.info("  observed: mean %.4f  sd %.4f  p05 %.4f  p95 %.4f  at-floor %.3f",
             cd["mean"], cd["sd"], cd["p05"], cd["p95"], cd["share_at_floor"])
    log.info("  vs clinician agreement (base rate %.4f): ECE %.4f  MCE %.4f  "
             "Brier %.4f  spearman %+.4f",
             p["base_rate"], p["ece"], p["mce"], p["brier"],
             p["spearman_confidence_vs_agreement"])
    for b in p["bins"]:
        log.info("    conf %-15s n=%-7d conf %.4f  observed %.4f  gap %+.4f",
                 f"[{b['range'][0]:.3f},{b['range'][1]:.3f})", b["n"],
                 b["mean_confidence"], b["observed_rate"],
                 b["gap_confidence_minus_observed"])
    if "secondary_outcome_association" in result:
        s = result["secondary_outcome_association"]
        log.info("  [descriptive] confidence vs survival: spearman %+.4f  "
                 "survivors %.4f vs died %.4f  (CONFOUNDED — not calibration)",
                 s["spearman_confidence_vs_survival"],
                 s["mean_confidence_survivors"], s["mean_confidence_died"])
    log.info("  verdict: %s", result["verdict"])
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="a", choices=["a", "b"])
    ap.add_argument("--cap", type=int, default=None,
                    help="limit held-out rows (the per-row confidence loop is O(n))")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    evaluate(args.track, write=not args.no_write, cap=args.cap)


if __name__ == "__main__":
    main()
