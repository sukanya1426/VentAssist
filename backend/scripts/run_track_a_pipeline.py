"""Track A end-to-end pipeline (clinical, ~3,200-episode cohort).

Runs the clinical data pipeline, builds the MDP, trains the policy + baselines,
and runs OPE. Heavy stages (full chartevents/labevents scans) dominate runtime.

    python -m backend.scripts.run_track_a_pipeline [--skip-data] [--steps N]
"""

from __future__ import annotations

import argparse
import shutil
import sys

from backend.pipeline import config, cohort, gp_imputation, propensity_score, state_builder
from backend.pipeline.logging_utils import get_logger

log = get_logger("run_track_a")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-data", action="store_true",
                    help="Skip the heavy data pipeline (reuse existing outputs).")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--kfold", type=int, default=5,
                    help="Patient-level k-fold CV (set 0/1 to disable).")
    args = ap.parse_args()

    if not args.skip_data:
        log.info("=== cohort ===");          cohort.build_cohort()
        log.info("=== gp imputation ===");    gp_imputation.impute()
        log.info("=== state builder ===");    state_builder.build_states()
        log.info("=== propensity ===");       propensity_score.build_propensity()

    from backend.mdp import dataset
    from backend.rl import trainer, ood_autoencoder
    from backend.ope import fqe, dfqe, nwe
    from backend.scripts.verify_before_deploy import verify

    log.info("=== MDP (track A) ===");        dataset.build_track_a()

    # Preserve the current good checkpoint so a degenerate retrain can't clobber it.
    model_path = config.MODEL_PATH / "policy_track_a.pt"
    prev_path = config.MODEL_PATH / "policy_track_a.pt.prev"
    if model_path.exists():
        shutil.copy2(model_path, prev_path)

    if args.kfold and args.kfold > 1:
        log.info("=== %d-fold CV + train (track A) ===", args.kfold)
        trainer.train_kfold("a", k=args.kfold, steps=args.steps)
    else:
        log.info("=== train (track A) ===");  trainer.train("a", steps=args.steps)

    # HARD DEPLOY GATE (§15.3 degeneracy guard). Refuse to keep — and to run OPE on —
    # a checkpoint that fails the holistic battery (e.g. the "always raise FiO₂"
    # collapse). On failure: quarantine the bad model, restore the previous good one,
    # and abort the pipeline. This is what turns the manual probe into an enforced gate.
    log.info("=== deploy gate (verify_before_deploy) ===")
    if not verify(model_path):
        rejected = config.MODEL_PATH / "policy_track_a.pt.rejected"
        shutil.move(str(model_path), str(rejected))
        if prev_path.exists():
            shutil.move(str(prev_path), str(model_path))
            log.error("DEPLOY BLOCKED: bad model → %s; restored previous good model.",
                      rejected.name)
        else:
            log.error("DEPLOY BLOCKED: bad model → %s; no previous model to restore.",
                      rejected.name)
        sys.exit(1)
    if prev_path.exists():
        prev_path.unlink()

    log.info("=== OOD/support autoencoder (track A) ==="); ood_autoencoder.train("a")

    log.info("=== OPE: FQE ===");             fqe.fitted_q_evaluation("a", K=30)
    log.info("=== OPE: DFQE ===");            dfqe.distributional_fqe("a")
    log.info("=== OPE: NWE ===");             nwe.rollout_value("a", n_starts=100)
    log.info("Track A pipeline complete.")


if __name__ == "__main__":
    main()
