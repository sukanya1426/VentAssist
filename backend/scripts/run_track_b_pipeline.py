"""Track B end-to-end pipeline (waveform-anchored, proof-of-concept).

Re-anchors the cohort on the 198 waveform records, extracts waveform features,
builds the 18-dim MDP, trains the (PoC) policy + imputer, and runs OPE.

    python -m backend.scripts.run_track_b_pipeline [--skip-data] [--steps N]
"""

from __future__ import annotations

import argparse

from backend.pipeline import cohort_track_b, gp_imputation, state_builder
from backend.pipeline.logging_utils import get_logger
from backend.waveform import aggregator

log = get_logger("run_track_b")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-data", action="store_true")
    # None → the step budget comes from track_b_config.yaml, so the fine-tuning
    # schedule lives with the warm-start settings it has to match.
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--kfold", type=int, default=5,
                    help="patient-level k-fold CV before the final fit; 0/1 to skip. "
                         "With only ~35 waveform stays the canonical split leaves a "
                         "2-stay validation set, so CV is the honest generalisation "
                         "estimate rather than a nicety.")
    args = ap.parse_args()

    if not args.skip_data:
        log.info("=== re-anchor cohort ==="); cohort_track_b.build()
        log.info("=== gp imputation ===")
        gp_imputation.impute(cohort_file="cohort_track_b.csv",
                             out_file="gp_imputed_labs_track_b.parquet")
        log.info("=== tabular states ===")
        state_builder.build_states(
            cohort_file="cohort_track_b.csv",
            labs_file="gp_imputed_labs_track_b.parquet",
            states_file="tabular_states_track_b.parquet",
            normaliser_file="normaliser_stats_track_b.json",
            split_file="train_val_test_split_track_b.json")
        log.info("=== waveform features ==="); aggregator.aggregate()

    from backend.mdp import dataset
    from backend.rl import trainer
    from backend.router.feature_imputer import FeatureImputer

    log.info("=== MDP (track B) ===")
    if dataset.build_track_b() is None:
        log.warning("Track B MDP empty — stopping (proof-of-concept data limit).")
        return
    log.info("=== feature imputer ==="); FeatureImputer.fit_from_files()
    if args.kfold and args.kfold > 1:
        log.info("=== %d-fold CV + train (track B) ===", args.kfold)
        trainer.train_kfold("b", k=args.kfold, steps=args.steps)
    else:
        log.info("=== train (track B) ==="); trainer.train("b", steps=args.steps)
    log.info("Track B pipeline complete (proof-of-concept scale).")


if __name__ == "__main__":
    main()
