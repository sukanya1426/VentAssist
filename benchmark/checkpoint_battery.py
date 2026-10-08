"""Does the model-selection criterion agree with clinical behaviour? (It does not.)

This module exists because of a result that reversed a session's premise. The plan
was: `checkpoint_every` was 10,000 and hid a better model, so lower it to 500,
retrain, and the retrain will beat the deployed artifact. The first half is true —
validation Q-loss improves from 1.5340 [1.5135, 1.5545] to 1.4802 [1.4740, 1.4864],
non-overlapping. The second half is false, and this module is the evidence.

Three criteria pick three different checkpoints:

  * **Validation Q-loss** (what the trainer early-stops on) picks step ~5,750.
  * **FQE value** peaks near step 10,000 and is NON-MONOTONIC in step.
  * **The 8-case clinical battery** (what the deploy gate uses) keeps improving
    out to the deployed artifact's step ~27,000.

The `high_peep` case is the sharp end of it: "PEEP is 18, lower it" is a safety
reflex, and it is acquired LATE. At step ~5,750 no seed has it; at step 10,000 two
of five do; the deployed step-~27,000 model has it. So selecting on validation
Q-loss — which is what the corrected config does — selects a model that has not yet
learned a clinical safety reflex the deployed one has.

Two consequences worth being explicit about:

1. **Do not deploy the retrain.** It would regress clinical behaviour on the very
   criterion the deploy gate enforces.
2. **The reward critique's absolute battery numbers are a FLOOR, not a ceiling.**
   `reward_critique.layer2_multiseed` trains 30,000 steps under the current config
   and so selects at ~5,750 for BOTH arms. Our reward scores 6.6/8 there; the
   deployed model, same reward, trained longer, scores 8/8. The paired CONTRAST
   (3.2/8 vs 6.6/8) is still valid — same budget, same selection rule, same seeds,
   only the reward differs — but 6.6/8 understates what our reward reaches at
   convergence, and should not be quoted as its capability.

Run:  PYTHONPATH=. .venv/bin/python -m benchmark.checkpoint_battery
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
from benchmark.reward_critique import BATTERY, _battery_verdict

log = get_logger("bench_checkpoint_battery")
RESULTS = Path(__file__).resolve().parent / "results"
SEED_DIR = RESULTS / "seeds"


def _load_policy(path: Path) -> HybridIQL:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = HybridIQL(ck["state_dim"], ck["action_dim"], ck["hidden_dim"])
    m.load_state_dict(ck["state_dict"])
    return m


def _battery(model, stats, feats) -> dict:
    """The 8-case battery at the SERVING encoding, as the deploy gate scores it."""
    out = {}
    for name, (state, expect) in BATTERY.items():
        vec = np.array([float(state[f]) for f in feats])
        z = N.transform_inference(vec, stats, feats)
        dp, dt, df = action_space.decode_action(int(np.argmax(model.q_values(z))))
        out[name] = {"expect": expect, "action": [dp, dt, df],
                     "pass": bool(_battery_verdict(name, dp, dt, df))}
    return out


def evaluate(track: str = "a", write: bool = True) -> dict:
    stats = N.load(config.MODEL_PATH / "normaliser_stats.json")
    feats = list(D.load_mdp(track)["feature_order"])

    families = {
        "deployed": {
            "approx_selected_step": 27000,
            "paths": [config.MODEL_PATH / f"policy_track_{track}.pt"],
            "note": "the live artifact; step-27,000 weights (the state_dict aliasing bug)",
        },
        "coarse_checkpoint_every_10000": {
            "approx_selected_step": 10000,
            "paths": [SEED_DIR / f"policy_track_{track}_seed{s}.pt" for s in range(5)],
            "note": "5 fresh seeds, all selected step 10,000 (the first checkpoint seen)",
        },
        "fine_checkpoint_every_500": {
            "approx_selected_step": 5750,
            "paths": [SEED_DIR / f"policy_track_{track}_ckpt500_seed{s}.pt"
                      for s in range(5)],
            "note": ("5 fresh seeds under the CORRECTED config, selected step 5,500-6,000. "
                     "THE WEIGHTS FOR THIS FAMILY WERE DELETED DELIBERATELY on 2026-10-08, "
                     "once this module established they score 6.60/8 on the battery against "
                     "the deployed model's 8.00/8 and fail `high_peep` in 0 of 5 seeds. They "
                     "were never deployment candidates and keeping them invited a mistake. "
                     "The measurement survives in checkpoint_battery_track_a.json and "
                     "runner_track_a_ckpt500.json, which is what the defence cites. This "
                     "family is therefore SKIPPED on a re-run, and that is expected, not a "
                     "bug: re-create it with "
                     "`python -m benchmark.runner --track a --seeds 5 --arm ckpt500`."),
        },
    }

    out = {}
    for fam, spec in families.items():
        present = [p for p in spec["paths"] if Path(p).exists()]
        if not present:
            log.warning("no checkpoints present for family %s — skipping", fam)
            continue
        per = []
        for p in present:
            b = _battery(_load_policy(Path(p)), stats, feats)
            per.append({"checkpoint": Path(p).name,
                        "passed": sum(v["pass"] for v in b.values()),
                        "battery": b})
        totals = [r["passed"] for r in per]
        out[fam] = {
            "approx_selected_step": spec["approx_selected_step"],
            "note": spec["note"],
            "n_checkpoints": len(per),
            "battery_passed_mean": round(float(np.mean(totals)), 2),
            "battery_passed_min": int(min(totals)),
            "battery_passed_max": int(max(totals)),
            "per_case_pass_rate": {
                case: round(float(np.mean([r["battery"][case]["pass"] for r in per])), 4)
                for case in BATTERY
            },
            "per_checkpoint": per,
        }

    # The headline: battery quality vs selected step, and the late-learned case.
    ordered = sorted(out.items(), key=lambda kv: kv[1]["approx_selected_step"])
    trend = [{"family": k, "approx_selected_step": v["approx_selected_step"],
              "battery_passed_mean": v["battery_passed_mean"],
              "high_peep_pass_rate": v["per_case_pass_rate"]["high_peep"]}
             for k, v in ordered]
    late = sorted(
        (case for case in BATTERY
         if len(ordered) > 1
         and ordered[0][1]["per_case_pass_rate"][case]
         < ordered[-1][1]["per_case_pass_rate"][case]),
        key=lambda c: ordered[-1][1]["per_case_pass_rate"][c]
        - ordered[0][1]["per_case_pass_rate"][c], reverse=True)

    result = {
        "track": track,
        "question": ("Does the checkpoint the trainer selects on validation Q-loss "
                     "behave best clinically?"),
        "answer": "No — the battery keeps improving well past the val-loss optimum.",
        "encoding": "serving (transform_inference), as the deploy gate scores it",
        "families": out,
        "battery_vs_selected_step": trend,
        "cases_learned_later_than_the_val_loss_optimum": late,
        "consequences": [
            "Do NOT deploy the retrain: it regresses clinical behaviour on the "
            "criterion the deploy gate enforces.",
            "reward_critique.layer2_multiseed selects at ~5,750 for BOTH arms, so its "
            "absolute battery numbers are a FLOOR. The paired contrast stays valid; "
            "6.6/8 is not our reward's capability at convergence (the deployed model, "
            "same reward, scores 8/8).",
            "Validation Q-loss, FQE value and the clinical battery disagree about which "
            "checkpoint is best. Model selection needs a criterion that is not purely "
            "TD-error.",
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if write:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"checkpoint_battery_track_{track}.json").write_text(
            json.dumps(result, indent=2))

    log.info("battery vs selected step (serving encoding):")
    for t in trend:
        log.info("  step ~%-6d %-32s battery %.2f/8   high_peep pass rate %.0f%%",
                 t["approx_selected_step"], t["family"], t["battery_passed_mean"],
                 t["high_peep_pass_rate"] * 100)
    if late:
        log.info("  cases learned later than the val-loss optimum: %s", ", ".join(late))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="a", choices=["a", "b"])
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    evaluate(args.track, write=not args.no_write)


if __name__ == "__main__":
    main()
