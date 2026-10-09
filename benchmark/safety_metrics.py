"""Clinical safety-violation rates — policy vs clinician on the SAME held-out states.

IntelliLung Rule 9 (BENCHMARK_HANDOFF): *"the repo does NOT have these ... this is
the first thing a clinical reviewer asks and the strongest differentiator a
successor system can claim."* Their reward encodes physiological targets, but no
code checks whether the recommended **settings** violate lung-protective practice.

VentAssist already enforces these at serving time (`backend/router/safety_filter.py`);
this module turns that into an **evaluation metric**: for every held-out transition
we apply (a) the policy's recommended action and (b) the clinician's actual action
to the same state, and count how often the *resulting settings* violate each rule.

Rules checked (the ones our 12-dim tabular state can support):
  * **VT > 8 mL/kg PBW** — volutrauma (ARDSnet targets 6–8)
  * **PEEP > 15 cmH₂O** — over-distension risk
  * **FiO₂ > 0.8** — oxygen toxicity
  * **Needless hyperoxia**: FiO₂ ≥ 0.95 while SpO₂ ≥ 96 %
  * **PEEP below the ARDSnet minimum for the given FiO₂** — i.e. buying oxygenation
    with FiO₂ instead of PEEP (the low-PEEP/high-FiO₂ table)

NOT checked (needs the airway-pressure waveform we don't have — stated, not faked):
  * **Driving pressure > 15 cmH₂O** — requires plateau/inspiratory pressure.
    Reported as ``null`` rather than imputed.

Read the output as: `policy_rate` vs `clinician_rate` on identical states. A
successor that matches on FQE value while cutting the violation rate is a stronger
clinical result than a higher FQE value alone.

Run:  PYTHONPATH=. .venv/bin/python -m benchmark.safety_metrics --track a
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from backend.mdp import dataset as D
from backend.mdp import normaliser as N
from backend.pipeline import config
from backend.pipeline.logging_utils import get_logger
from backend.router import safety_rules as SR

log = get_logger("bench_safety")

RESULTS = Path(__file__).resolve().parent / "results"

# The rules themselves live in backend/router/safety_rules.py, so backend/ can
# check a violation rate without importing benchmark/ — the dependency points one
# way only (benchmark/ reads backend/, never the reverse, because backend/ is the
# serving path). They are re-exported here under their historical names because
# runner.py, the selection criterion and the tests all import them from this
# module; the logic is unchanged and the artifact is byte-identical.
MAX_TV_ML_PER_KG = SR.MAX_TV_ML_PER_KG
MAX_PEEP = SR.MAX_PEEP
MAX_FIO2 = SR.MAX_FIO2
HYPEROXIA_SPO2 = SR.HYPEROXIA_SPO2
HYPEROXIA_FIO2 = SR.HYPEROXIA_FIO2
_ARDSNET = SR._ARDSNET
ardsnet_min_peep = SR.ardsnet_min_peep
_resulting_settings = SR.resulting_settings
_violations = SR.violations


def evaluate(track: str = "a", write: bool = True) -> dict:
    """Policy-vs-clinician violation rates on the held-out split.

    ``write=False`` skips the artifact, so a test can exercise this without
    overwriting the canonical result the report cites.
    """
    d = D.load_mdp(track)
    stats = N.load(config.MODEL_PATH / (
        "normaliser_stats.json" if track == "a" else "normaliser_stats_track_b.json"))
    feats = list(d["feature_order"])
    test = np.where(d["split"] == "test")[0]
    if len(test) == 0:
        raise RuntimeError("MDP has no test split.")

    raw = d["states"][test]                       # RAW physiological values
    i_peep, i_tv, i_fio2, i_spo2 = (feats.index("PEEP"), feats.index("TV"),
                                    feats.index("FiO2"), feats.index("SpO2"))
    peep, tv, fio2, spo2 = raw[:, i_peep], raw[:, i_tv], raw[:, i_fio2], raw[:, i_spo2]
    weight = d["weight_kg"][test].astype(float)

    # The policy, encoded exactly as it is at SERVING time (transform_inference),
    # so these rates describe what the deployed system would actually recommend.
    from backend.ope.fqe import _load_policy
    policy, _ = _load_policy(track)
    z = N.transform_inference(raw, stats, feats).astype(np.float32)
    pi_actions = policy.act_batch(z)
    clin_actions = d["actions"][test]

    p_peep, p_tv, p_fio2 = _resulting_settings(peep, tv, fio2, pi_actions)
    c_peep, c_tv, c_fio2 = _resulting_settings(peep, tv, fio2, clin_actions)

    policy_v = _violations(p_peep, p_tv, p_fio2, spo2, weight)
    clin_v = _violations(c_peep, c_tv, c_fio2, spo2, weight)

    delta = {k: (None if policy_v[k] is None or clin_v[k] is None
                 else round(policy_v[k] - clin_v[k], 4))
             for k in policy_v}

    result = {
        "track": track,
        "n_test": int(len(test)),
        "n_transitions": int(len(d["actions"])),
        "policy": policy_v,
        "clinician": clin_v,
        # negative Δ = the policy violates LESS often than the clinician (good)
        "delta_policy_minus_clinician": delta,
        "notes": {
            "driving_pressure_gt_15": "requires airway-pressure waveform (not available) — not imputed",
            "encoding": "policy scored with transform_inference (serving encoding)",
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if write:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"safety_metrics_track_{track}.json").write_text(
            json.dumps(result, indent=2))

    log.info("Safety violation rates on %d held-out transitions (policy vs clinician):", len(test))
    for k in policy_v:
        if policy_v[k] is None:
            continue
        log.info("  %-48s policy=%.4f  clinician=%.4f  Δ=%+.4f",
                 k, policy_v[k], clin_v[k], delta[k])
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="a", choices=["a", "b"])
    args = ap.parse_args()
    evaluate(args.track)


if __name__ == "__main__":
    main()
