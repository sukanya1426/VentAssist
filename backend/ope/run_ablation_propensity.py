"""Static vs time-varying propensity — OPE ablation (Methodology §10.4, Task C).

Compares the NW model-based OPE under the two propensity definitions used as the
kernel's confounder-conditioning variable:

  * ``static``  — the per-patient 90-day-mortality ``propensity_z`` (unchanged
    default).
  * ``dynamic`` — the per-transition z_t = P(SpO₂ < 88 in next 6 h | state)
    (backend/pipeline/propensity_dynamic.py, §10.4).

Reports, for each: the held-out 1-step next-state MSE (from the bandwidth CV) and
the model-based rollout V̂. Writes ``logs/ablation_propensity_track_{track}.json``.
Evaluation-only and opt-in — never touches the deployed policy or the static
default OPE artifacts' meaning (the dynamic model/log are written to *_dynamic
sidecars).

Run:
    python -m backend.ope.run_ablation_propensity --track a
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from backend.ope import nwe
from backend.pipeline import config, propensity_dynamic
from backend.pipeline.logging_utils import get_logger

log = get_logger("ope_ablation_propensity")


def run(track: str = "a", n_starts: int = 200, T: int = 24) -> dict:
    dyn = propensity_dynamic.fit_dynamic(track)
    zt = dyn["z_t"]

    # 1-step held-out MSE under each propensity (from the bandwidth CV).
    bw_static = nwe.select_bandwidths(track)                       # static propensity_z
    bw_dynamic = nwe.select_bandwidths(track, z_override=zt)       # dynamic z_t

    # Model-based rollout V̂ under each propensity.
    roll_static = nwe.rollout_value(track, T=T, n_starts=n_starts, propensity="static")
    roll_dynamic = nwe.rollout_value(track, T=T, n_starts=n_starts, propensity="dynamic")

    result = {
        "track": track,
        "adverse_event": "SpO2<88 within next 6h",
        "dynamic_zt_auc": round(dyn["auc"], 4),
        "dynamic_zt_event_rate": round(dyn["event_rate"], 4),
        "static": {
            "val_1step_mse": bw_static["val_mse_selected"],
            "rollout_V_hat": roll_static["V_hat"],
            "bandwidths": {"hs": bw_static["hs"], "ha": bw_static["ha"],
                           "hz": bw_static["hz"]},
        },
        "dynamic": {
            "val_1step_mse": bw_dynamic["val_mse_selected"],
            "rollout_V_hat": roll_dynamic["V_hat"],
            "bandwidths": {"hs": bw_dynamic["hs"], "ha": bw_dynamic["ha"],
                           "hz": bw_dynamic["hz"]},
        },
        "n_transitions": roll_static["n_transitions"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out = config.LOGS_PATH / f"ablation_propensity_track_{track}.json"
    out.write_text(json.dumps(result, indent=2))
    log.info("Propensity ablation Track %s: 1-step MSE static=%.4f dynamic=%.4f | "
             "rollout V̂ static=%.3f dynamic=%.3f (z_t AUC=%.3f)", track.upper(),
             result["static"]["val_1step_mse"], result["dynamic"]["val_1step_mse"],
             result["static"]["rollout_V_hat"], result["dynamic"]["rollout_V_hat"],
             result["dynamic_zt_auc"])
    return result


def main():
    ap = argparse.ArgumentParser(description="Static vs dynamic propensity OPE ablation.")
    ap.add_argument("--track", default="a", choices=["a", "b"])
    args = ap.parse_args()
    run(args.track)


if __name__ == "__main__":
    main()
