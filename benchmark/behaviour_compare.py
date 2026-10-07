"""AI-vs-clinician behavioural analyses — agreement, per-setting deviation, churn.

IntelliLung Rule 6 / BENCHMARK_PLAN Phase 1 item 6. These are the *clinically
legible* secondary metrics: a clinician reading an OPE value of +3.19 learns
nothing, but "it agrees with me 57% of the time, and when it disagrees it is
almost always asking for a smaller tidal volume" is immediately interpretable.

Three families, each answering a question a reviewer will actually ask:

**1. Agreement** — how often does the policy pick the clinician's exact action
out of 125? Exact 125-way match is a harsh denominator, so it is reported
alongside per-knob agreement (did we agree on ΔPEEP at all?) and *directional*
agreement (did we agree on the sign, ignoring magnitude?). The gap between exact
and directional agreement is the interesting part: it separates "the policy
disagrees about what to do" from "the policy agrees but wants a different step
size".

**2. Per-setting deviation** — the signed distribution of
``policy_delta − clinician_delta`` per knob. The *sign of the mean* is the
clinical bias of the policy: a negative ΔTV bias means it systematically wants
lower tidal volumes than the clinician chose, which is the direction
lung-protective ventilation says is right and is a defensible finding. A bias
near zero with high spread means the policy is noisy rather than opinionated.

**3. Churn** — how often the recommendation *changes between consecutive hours*
within the same stay, compared with how often the clinician changed theirs. This
is the metric that kills jittery policies in practice: a policy that alternates
PEEP up / PEEP down hourly is unusable at the bedside no matter what its OPE
value says. Measured only on consecutive hour pairs inside one stay, so an
episode boundary never counts as a change.

Read the output as policy vs clinician on identical states. A policy with
clinician-level churn and a defensible deviation bias is clinically plausible;
one with far higher churn is not, whatever its value estimate.

NOTE ON CAUSALITY: every number here is computed at the clinician's *observed*
states. The policy is evaluated on the trajectory the clinician produced, so
churn measures "how much would the advice have jittered along this course",
not "how much would the course have jittered under this policy". The latter
needs the learned dynamics model and belongs in the OPE rollouts, not here.

Run:  PYTHONPATH=. .venv/bin/python -m benchmark.behaviour_compare --track a
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from backend.mdp import action_space
from backend.mdp import dataset as D
from backend.mdp import normaliser as N
from backend.pipeline import config
from backend.pipeline.logging_utils import get_logger

log = get_logger("bench_behaviour")

RESULTS = Path(__file__).resolve().parent / "results"

KNOBS = ("delta_PEEP", "delta_TV", "delta_FiO2")
HOLD_ACTION = action_space.encode_action(0, 0, 0.0)


def _decode(actions: np.ndarray) -> dict[str, np.ndarray]:
    """Action indices → the three deltas, as parallel arrays."""
    am = action_space.ACTION_MAP
    return {
        "delta_PEEP": np.array([am[int(a)][0] for a in actions], dtype=float),
        "delta_TV": np.array([am[int(a)][1] for a in actions], dtype=float),
        "delta_FiO2": np.array([am[int(a)][2] for a in actions], dtype=float),
    }


def _agreement(pi: np.ndarray, clin: np.ndarray) -> dict:
    """Exact, per-knob and directional agreement between two action sequences."""
    p, c = _decode(pi), _decode(clin)
    out = {
        "exact_125_way": round(float(np.mean(pi == clin)), 4),
        "per_knob": {},
        "directional": {},
    }
    for k in KNOBS:
        out["per_knob"][k] = round(float(np.mean(p[k] == c[k])), 4)
        out["directional"][k] = round(float(np.mean(np.sign(p[k]) == np.sign(c[k]))), 4)
    # Agreement on all three signs at once — "same clinical intent, maybe a
    # different step size".
    same_sign_all = np.ones(len(pi), dtype=bool)
    for k in KNOBS:
        same_sign_all &= (np.sign(p[k]) == np.sign(c[k]))
    out["directional"]["all_three"] = round(float(np.mean(same_sign_all)), 4)
    # Hold-vs-act is the single most consequential binary in this system.
    pi_hold, c_hold = (pi == HOLD_ACTION), (clin == HOLD_ACTION)
    out["hold_share_policy"] = round(float(np.mean(pi_hold)), 4)
    out["hold_share_clinician"] = round(float(np.mean(c_hold)), 4)
    out["hold_agreement"] = round(float(np.mean(pi_hold == c_hold)), 4)
    return out


def _deviation(pi: np.ndarray, clin: np.ndarray) -> dict:
    """Signed per-knob deviation ``policy − clinician``, with spread and quantiles."""
    p, c = _decode(pi), _decode(clin)
    out = {}
    for k in KNOBS:
        d = p[k] - c[k]
        out[k] = {
            "mean_signed": round(float(np.mean(d)), 5),
            "mean_absolute": round(float(np.mean(np.abs(d))), 5),
            "std": round(float(np.std(d)), 5),
            "p05": round(float(np.percentile(d, 5)), 5),
            "median": round(float(np.median(d)), 5),
            "p95": round(float(np.percentile(d, 95)), 5),
            "share_policy_lower": round(float(np.mean(d < 0)), 4),
            "share_equal": round(float(np.mean(d == 0)), 4),
            "share_policy_higher": round(float(np.mean(d > 0)), 4),
        }
    return out


def _churn(actions: np.ndarray, stay_id: np.ndarray, hour: np.ndarray) -> dict:
    """Rate of change between CONSECUTIVE hours within a stay.

    Only pairs that are genuinely adjacent in both stay and hour count, so a gap
    in charting or an episode boundary is never scored as a change. Without that
    guard the first hour of every stay would be compared against the last hour of
    the previous one and churn would be dominated by boundaries.
    """
    order = np.lexsort((hour, stay_id))
    a, s, h = actions[order], stay_id[order], hour[order]
    adjacent = (s[1:] == s[:-1]) & (h[1:] == h[:-1] + 1)
    if adjacent.sum() == 0:
        return {"n_consecutive_pairs": 0}
    prev, nxt = a[:-1][adjacent], a[1:][adjacent]
    p, n = _decode(prev), _decode(nxt)
    out = {
        "n_consecutive_pairs": int(adjacent.sum()),
        "action_changed": round(float(np.mean(prev != nxt)), 4),
        "per_knob_changed": {k: round(float(np.mean(p[k] != n[k])), 4) for k in KNOBS},
    }
    # A reversal is the pathological case: a non-zero move followed by a move in
    # the opposite direction on the same knob. This is what "jittery" means
    # clinically — it is strictly worse than simply changing often.
    for k in KNOBS:
        rev = (np.sign(p[k]) != 0) & (np.sign(n[k]) == -np.sign(p[k]))
        out.setdefault("per_knob_reversed", {})[k] = round(float(np.mean(rev)), 4)
    return out


def evaluate(track: str = "a", write: bool = True) -> dict:
    """Compare the policy with the clinician on the held-out split.

    ``write=False`` skips the artifact, matching the guard on the OPE entry
    points — a test must be able to run this without overwriting the canonical
    result that the report cites.
    """
    d = D.load_mdp(track)
    stats = N.load(config.MODEL_PATH / (
        "normaliser_stats.json" if track == "a" else "normaliser_stats_track_b.json"))
    feats = list(d["feature_order"])
    test = np.where(d["split"] == "test")[0]
    if len(test) == 0:
        raise RuntimeError("MDP has no test split.")

    raw = d["states"][test]
    # Serving encoding (transform_inference), not the training winsor clip, so
    # these are the actions the deployed system would really produce.
    from backend.ope.fqe import _load_policy
    policy, _ = _load_policy(track)
    z = N.transform_inference(raw, stats, feats).astype(np.float32)
    pi_actions = policy.act_batch(z)
    clin_actions = d["actions"][test]

    result = {
        "track": track,
        "n_test": int(len(test)),
        "n_transitions": int(len(d["actions"])),
        "agreement": _agreement(pi_actions, clin_actions),
        "deviation_policy_minus_clinician": _deviation(pi_actions, clin_actions),
        "churn": {
            "policy": _churn(pi_actions, d["stay_id"][test], d["hour"][test]),
            "clinician": _churn(clin_actions, d["stay_id"][test], d["hour"][test]),
        },
        "notes": {
            "encoding": "policy scored with transform_inference (serving encoding)",
            "churn_basis": "consecutive hours within one stay only; episode "
                           "boundaries and charting gaps excluded",
            "causality": "evaluated on the clinician's observed trajectory, so churn "
                         "is 'how much the advice would have jittered along this "
                         "course', not the churn of a counterfactual course",
            "responsiveness": "0.0 (deployed default) — a non-zero slider changes "
                              "the hold/act boundary and therefore every number here",
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if write:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"behaviour_compare_track_{track}.json").write_text(
            json.dumps(result, indent=2))

    a = result["agreement"]
    log.info("Behaviour comparison on %d held-out transitions:", len(test))
    log.info("  agreement  exact(125-way) %.4f   all-three-signs %.4f",
             a["exact_125_way"], a["directional"]["all_three"])
    log.info("  hold share  policy %.4f  clinician %.4f  (agree %.4f)",
             a["hold_share_policy"], a["hold_share_clinician"], a["hold_agreement"])
    for k in KNOBS:
        log.info("  %-11s exact %.4f  directional %.4f", k,
                 a["per_knob"][k], a["directional"][k])
    log.info("  deviation (policy - clinician):")
    for k, v in result["deviation_policy_minus_clinician"].items():
        log.info("    %-11s mean %+.4f  |mean| %.4f  lower %.3f / equal %.3f / higher %.3f",
                 k, v["mean_signed"], v["mean_absolute"], v["share_policy_lower"],
                 v["share_equal"], v["share_policy_higher"])
    cp, cc = result["churn"]["policy"], result["churn"]["clinician"]
    if cp.get("n_consecutive_pairs"):
        log.info("  churn over %d consecutive pairs: policy %.4f vs clinician %.4f",
                 cp["n_consecutive_pairs"], cp["action_changed"], cc["action_changed"])
        for k in KNOBS:
            log.info("    %-11s changed p/c %.4f/%.4f   reversed p/c %.4f/%.4f", k,
                     cp["per_knob_changed"][k], cc["per_knob_changed"][k],
                     cp["per_knob_reversed"][k], cc["per_knob_reversed"][k])
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="a", choices=["a", "b"])
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    evaluate(args.track, write=not args.no_write)


if __name__ == "__main__":
    main()
