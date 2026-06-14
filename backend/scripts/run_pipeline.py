"""Run the full data pipeline (cohort → GP imputation → states → propensity).

Examples:
    # Fast sample run (limited chartevents/labevents scan)
    python -m backend.scripts.run_pipeline --sample --max-stays 300

    # Full run over all data (long)
    python -m backend.scripts.run_pipeline
"""

from __future__ import annotations

import argparse
import time

from backend.pipeline import cohort, gp_imputation, propensity_score, state_builder
from backend.pipeline.logging_utils import get_logger

log = get_logger("run_pipeline")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the VentAssist data pipeline.")
    ap.add_argument("--sample", action="store_true",
                    help="Fast sample run with limited event-table scans.")
    ap.add_argument("--sample-chunks", type=int, default=60,
                    help="Chartevents chunks to scan in sample mode.")
    ap.add_argument("--max-stays", type=int, default=None,
                    help="Cap cohort size (sample mode).")
    args = ap.parse_args()

    sample = args.sample
    chunks = args.sample_chunks if sample else None

    t0 = time.time()
    log.info("=== Stage 1/4: cohort ===")
    cohort.build_cohort(sample_chunks=chunks, max_stays=args.max_stays)

    log.info("=== Stage 2/4: GP imputation ===")
    gp_imputation.impute(sample_chunks=1 if sample else None)

    log.info("=== Stage 3/4: state builder ===")
    state_builder.build_states(sample_chunks=chunks)

    log.info("=== Stage 4/4: propensity score ===")
    propensity_score.build_propensity()

    log.info("Pipeline complete in %.1f s.", time.time() - t0)


if __name__ == "__main__":
    main()
