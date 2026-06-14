"""Track A end-to-end pipeline (clinical, ~3,200-episode cohort).

Runs the clinical data pipeline, builds the MDP, trains the policy + baselines,
and runs OPE. Heavy stages (full chartevents/labevents scans) dominate runtime.

    python -m backend.scripts.run_track_a_pipeline [--skip-data] [--steps N]
"""

from __future__ import annotations

import argparse

from backend.pipeline import cohort, gp_imputation, propensity_score, state_builder
from backend.pipeline.logging_utils import get_logger

log = get_logger("run_track_a")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-data", action="store_true",
                    help="Skip the heavy data pipeline (reuse existing outputs).")
    ap.add_argument("--steps", type=int, default=30000)
    args = ap.parse_args()

    if not args.skip_data:
        log.info("=== cohort ===");          cohort.build_cohort()
        log.info("=== gp imputation ===");    gp_imputation.impute()
        log.info("=== state builder ===");    state_builder.build_states()
        log.info("=== propensity ===");       propensity_score.build_propensity()

    from backend.mdp import dataset
    from backend.rl import trainer
    from backend.ope import fqe, dfqe, nwe

    log.info("=== MDP (track A) ===");        dataset.build_track_a()
    log.info("=== train (track A) ===");      trainer.train("a", steps=args.steps)
    log.info("=== OPE: FQE ===");             fqe.fitted_q_evaluation("a", K=30)
    log.info("=== OPE: DFQE ===");            dfqe.distributional_fqe("a")
    log.info("=== OPE: NWE ===");             nwe.rollout_value("a", n_starts=100)
    log.info("Track A pipeline complete.")


if __name__ == "__main__":
    main()
