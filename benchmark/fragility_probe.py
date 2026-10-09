"""Does the recommendation flip when a CLINICALLY UNRELATED feature is swept?

This quantifies the residual effect of **confounding by indication** (SUMMARY
§15.4 item 1) on the deployed policy, so the report can state the limitation with
numbers instead of in the abstract.

THE TEST. Fix a patient whose correct management is unambiguous — hyperoxic at
SpO₂ 100 on FiO₂ 0.70, where the indicated action is to wean FiO₂ — then sweep one
feature at a time across its physiological range. A feature that is clinically
irrelevant to oxygen weaning must not change the decision. A flip means the policy
has attached weight to a correlate rather than to the indication.

WHY FLIPS AT RARE VALUES ARE THE SIGNATURE. Confounding by indication does not
produce uniform noise; it produces errors concentrated where the data is thin and
where unusual feature values co-occur with differently managed patients. So the
diagnostic is not "does it ever flip" but "does it flip only at the edges".

WHAT IS *NOT* CONFOUNDING. A change at extreme PaCO₂ is correct behaviour:
hypocapnia warrants lowering tidal volume and hypercapnia raising it, and because
the action space emits one combined action per step the policy must choose between
correcting CO₂ and weaning oxygen. `EXPECTED_TO_RESPOND` lists the features whose
flips are clinically defensible, and they are reported separately rather than
counted as fragility — otherwise the metric would punish the policy for being
right.

This probe is READ-ONLY with respect to the deployed model.

Run:  PYTHONPATH=. .venv/bin/python -m benchmark.fragility_probe
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from backend.mdp import action_space
from backend.mdp import dataset as D
from backend.mdp import normaliser as N
from backend.pipeline import config
from backend.pipeline.logging_utils import get_logger
from backend.rl.hybrid_iql import HybridIQL

log = get_logger("bench_fragility_probe")
RESULTS = Path(__file__).resolve().parent / "results"

# Unambiguously hyperoxic: SpO2 100 on FiO2 0.70. The indicated action is to wean
# FiO2, and every sweep below leaves that indication intact.
BASE = dict(PEEP=8, TV=480, FiO2=0.7, SpO2=100, PaO2=90, PaCO2=44, pH=7.37,
            HR=92, SBP=118, RR=22, RASS=-2, Temp=37.2)

SWEEPS = {
    "PaO2":  [70, 80, 90, 100, 120, 140, 160, 200, 300],
    "PaCO2": [25, 35, 44, 55, 70, 90],
    "pH":    [7.1, 7.25, 7.37, 7.45, 7.55, 7.7],
    "HR":    [50, 70, 92, 110, 130, 150, 180],
    "SBP":   [70, 90, 118, 140, 170, 200],
    "RR":    [8, 12, 22, 30, 40, 50],
    "RASS":  [-5, -4, -2, 0, 2, 4],
    "Temp":  [34, 35, 37.2, 38, 39, 41],
}

# Features whose flips are CLINICALLY CORRECT on this base, and therefore are not
# evidence of confounding. Counting them as fragility would penalise the policy for
# doing the right thing.
EXPECTED_TO_RESPOND = {
    "PaCO2": ("hypocapnia warrants lowering TV and hypercapnia raising it; the "
              "single-action space forces a choice between correcting CO2 and "
              "weaning oxygen"),
}


def _load_policy(track: str) -> tuple[HybridIQL, dict, list[str]]:
    ck = torch.load(config.MODEL_PATH / f"policy_track_{track}.pt",
                    map_location="cpu", weights_only=False)
    m = HybridIQL(ck["state_dim"], ck["action_dim"], ck["hidden_dim"])
    m.load_state_dict(ck["state_dict"])
    stats = N.load(config.MODEL_PATH / "normaliser_stats.json")
    feats = list(D.load_mdp(track)["feature_order"])
    return m, stats, feats


def evaluate(track: str = "a", write: bool = True) -> dict:
    model, stats, feats = _load_policy(track)

    def act(state: dict) -> tuple[int, int, float]:
        vec = np.array([float(state[f]) for f in feats])
        z = N.transform_inference(vec, stats, feats)
        return action_space.decode_action(int(np.argmax(model.q_values(z))))

    out: dict[str, dict] = {}
    for feat, values in SWEEPS.items():
        rows = []
        for v in values:
            dp, dt, df = act({**BASE, feat: v})
            rows.append({"value": v, "action": [dp, dt, df], "weans_fio2": bool(df < 0)})
        weans = [r["weans_fio2"] for r in rows]
        flips = [i for i in range(1, len(weans)) if weans[i] != weans[i - 1]]
        # Record the transition as a (from, to) pair. Reporting only the value AFTER
        # the flip is misleading: a hold at HR=50 that recovers by HR=70 reads as
        # "flips at 70" when the anomalous value is 50.
        out[feat] = {
            "n_values": len(values),
            "n_weaning": sum(weans),
            "n_flips": len(flips),
            "flip_boundaries": [{"from": values[i - 1], "to": values[i],
                                 "weans_before": weans[i - 1],
                                 "weans_after": weans[i]} for i in flips],
            "anomalous_values": [values[i - 1] if weans[i] else values[i]
                                 for i in flips],
            "clinically_expected_to_respond": feat in EXPECTED_TO_RESPOND,
            "expected_reason": EXPECTED_TO_RESPOND.get(feat),
            "sweep": rows,
        }

    spurious = sorted(f for f, r in out.items()
                      if r["n_flips"] and not r["clinically_expected_to_respond"])
    defensible = sorted(f for f, r in out.items()
                        if r["n_flips"] and r["clinically_expected_to_respond"])
    stable = sorted(f for f, r in out.items() if not r["n_flips"])

    result = {
        "track": track,
        "question": ("Does sweeping a clinically unrelated feature flip the "
                     "FiO2-weaning decision on an unambiguously hyperoxic patient?"),
        "base_state": BASE,
        "indicated_action": "wean FiO2 (SpO2 100 on FiO2 0.70)",
        "encoding": "serving (transform_inference)",
        "per_feature": out,
        "spurious_flips": spurious,
        "clinically_defensible_flips": defensible,
        "stable_features": stable,
        "n_spurious": len(spurious),
        "n_swept": len(SWEEPS),
        "interpretation": (
            "Flips concentrated at RARE / extreme feature values are the signature of "
            "residual confounding by indication: those regions are thin in the training "
            "data and co-occur with differently managed patients. This is a property of "
            "observational data, not a code defect — the causal effect of an action is "
            "not identifiable from retrospective records when unobserved severity drove "
            "both the action and the outcome. IPW was attempted and made the policy worse "
            "(SUMMARY §15.11 C8), and would only ever have addressed MEASURED confounding. "
            "The mitigation is therefore containment, not elimination: CQL bounds "
            "extrapolation, the OOD gate tempers confidence, and the safety filter is "
            "independent of the policy."),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if write:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"fragility_probe_track_{track}.json").write_text(
            json.dumps(result, indent=2))

    log.info("fragility probe — hyperoxic base, FiO2 weaning indicated:")
    for feat, r in out.items():
        tag = ("DEFENSIBLE" if r["clinically_expected_to_respond"]
               else "SPURIOUS") if r["n_flips"] else "stable"
        log.info("  %-6s weans %d/%d  flips=%d %-11s %s", feat, r["n_weaning"],
                 r["n_values"], r["n_flips"], tag,
                 f"non-weaning at {r['anomalous_values']}" if r["n_flips"] else "")
    log.info("  spurious: %s", spurious or "none")
    log.info("  clinically defensible: %s", defensible or "none")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="a", choices=["a", "b"])
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--json", action="store_true", help="print the full artifact")
    args = ap.parse_args()
    r = evaluate(args.track, write=not args.no_write)
    if args.json:
        print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
