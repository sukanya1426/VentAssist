"""Which criterion should pick the checkpoint? (§17: "needs a non-TD-error criterion".)

THE OPEN PROBLEM. ``benchmark/results/checkpoint_battery_track_a.json`` showed that
validation Q-loss, FQE value and the 8-case clinical battery pick three different
checkpoints of the same run, and that the one the trainer early-stops on (val
Q-loss, step ~5,750) is the worst of the three clinically: 6.60/8 against the
deployed step-~27,000 artifact's 8.00/8. That artifact established the
disagreement. It did not propose a fix, and selecting on the battery is not one —
the battery IS the deploy gate, so selecting on it would leave the gate certifying
a choice it had itself made.

WHAT THIS MODULE TESTS. ``backend/rl/selection.py`` proposes a criterion that is
neither TD-error nor the gate: the **clinical reflex score** (CRS), the
macro-averaged rate at which the policy moves in the clinically-indicated
direction across eight physiological strata of the **validation split's real
patient states**. This module asks whether it works, with the battery as the
held-out ground truth it must predict:

  for every checkpoint of a real training run, record
      (step, validation Q-loss, CRS, battery score)
  then compare the checkpoint each criterion would select against the checkpoint
  the battery actually prefers.

The battery is the TARGET here and never an input: ``selection.score`` cannot see
it (it takes states and actions, not the gate), and ``tests/test_selection.py``
pins that separation. If CRS tracked the battery only because it had been handed
the answer, the result would be worthless.

HOW THE TRACE IS TAKEN. Through ``trainer._run_training``'s ``on_checkpoint``
hook — the actual training loop, not a copy of it. A selection experiment run
against a reimplemented loop measures the reimplementation.

EARLY STOPPING IS DISABLED FOR THE TRACE, DELIBERATELY. We need the whole curve
out to the deployed artifact's region (~27,000 steps), and the point at issue is
precisely that early stopping on val Q-loss cuts the run off at ~5,750. A trace
that stopped where the incumbent criterion says to stop could not possibly see
past it. ``early_stop_patience`` is therefore raised above ``total_steps`` and the
override is recorded in the artifact.

WHAT A RESULT HERE DOES AND DOES NOT LICENSE. A CRS that ranks checkpoints the way
the battery does is evidence for a better *selection rule* — nothing more. CRS is
not a value estimate, it is not comparable across rewards, and it says nothing
about patient benefit. The claim boundary in SUMMARY §15.15 is unaffected. And
nothing here promotes, writes or touches ``backend/models/policy_track_a.pt``:
traces go to ``benchmark/results/``, and the deployed artifact is read-only input.

Run:  PYTHONPATH=. .venv/bin/python -m benchmark.selection_criterion --seeds 3
      PYTHONPATH=. .venv/bin/python -m benchmark.selection_criterion --seeds 1 --steps 6000 --no-write
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from backend.mdp import action_space
from backend.mdp import dataset as D
from backend.mdp import normaliser as N
from backend.pipeline import config
from backend.pipeline.logging_utils import get_logger
from backend.rl import selection as SEL
from backend.rl import trainer as T
from benchmark.reward_critique import BATTERY, _battery_verdict
from benchmark.runner import _ci95

log = get_logger("bench_selection")

RESULTS = Path(__file__).resolve().parent / "results"

HOLD_ACTION = action_space.encode_action(0, 0, 0.0)


def _battery_score(model, stats, feats) -> tuple[int, dict]:
    """The 8-case battery at the serving encoding, exactly as the deploy gate scores it."""
    per = {}
    for name, (state, _expect) in BATTERY.items():
        vec = np.array([float(state[f]) for f in feats])
        z = N.transform_inference(vec, stats, feats)
        dp, dt, df = action_space.decode_action(int(np.argmax(model.q_values(z))))
        per[name] = bool(_battery_verdict(name, dp, dt, df))
    return int(sum(per.values())), per


def _spearman(x: list[float], y: list[float]) -> float | None:
    """Rank correlation, ties averaged. Local so this module does not pull in scipy.

    Rank correlation rather than Pearson because the question is purely one of
    ORDERING — "does this criterion rank checkpoints the way the battery does" —
    and the battery is an integer count out of 8 with heavy ties, on which a
    linear correlation would be reading structure that is not there.
    """
    a, b = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None

    def rank(v: np.ndarray) -> np.ndarray:
        order = v.argsort()
        r = np.empty(len(v), float)
        r[order] = np.arange(len(v), dtype=float)
        # average ranks within each tie group
        for val in np.unique(v):
            m = v == val
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r

    ra, rb = rank(a), rank(b)
    return round(float(np.corrcoef(ra, rb)[0, 1]), 4)


def _trace_one_seed(d: dict, raw_val: np.ndarray, feats: list[str],
                    weight_val: np.ndarray, clin_val: np.ndarray,
                    z_serve_val: np.ndarray, stats: dict, cfg: dict,
                    train_idx: np.ndarray, val_idx: np.ndarray,
                    steps: int, seed: int, device: str) -> list[dict]:
    """Train one seed, scoring every criterion at every checkpoint."""
    trace: list[dict] = []
    z_train_val = d["states"][val_idx]          # already the TRAINING encoding

    def on_checkpoint(ev: dict) -> None:
        model = ev["model"]
        t0 = time.time()
        # CRS under both encodings. They are different quantities (the serving
        # encoding winsorises) and are never averaged — the trainer would select on
        # the training one, because that is what the policy was fitted on, while
        # the serving one is what the gate's battery sees. Reporting both is how we
        # find out whether the choice of encoding changes the ranking.
        cand_train = SEL.score(model, raw_val, z_train_val, feats, weight_val,
                               clinician_actions=clin_val,
                               val_q_loss=ev["val_q_loss"])
        crs_serve = SEL.clinical_reflex_score(
            model.act_batch(z_serve_val), raw_val, feats, weight_val)
        battery_n, battery_per = _battery_score(model, stats, feats)
        trace.append({
            "step": ev["step"],
            "val_q_loss": None if not np.isfinite(ev["val_q_loss"]) else round(float(ev["val_q_loss"]), 6),
            "selected_by_val_q_loss_at_this_point": ev["improved"],
            "crs_training_encoding": cand_train["crs"],
            "crs_serving_encoding": crs_serve["crs"],
            "selectable": cand_train["selectable"],
            "guards_failed": [n for n, g in cand_train["guards"].items() if not g["pass"]],
            "hold_share": cand_train["hold_share"],
            "n_distinct_actions": cand_train["n_distinct_actions"],
            "policy_any_violation": cand_train["safety"]["policy"]["any_violation"],
            "battery_passed_n": battery_n,
            "battery_per_case": battery_per,
            "per_stratum_compliance": {k: v["compliance"] for k, v
                                       in cand_train["reflex"]["per_stratum"].items()},
            "scoring_seconds": round(time.time() - t0, 2),
        })
        log.info("[seed %d] step %-6d val_q=%.4f  crs=%.4f  battery=%d/8  "
                 "hold=%.3f distinct=%d", seed, ev["step"], ev["val_q_loss"],
                 cand_train["crs"] or float("nan"), battery_n,
                 cand_train["hold_share"], cand_train["n_distinct_actions"])

    T._run_training(d, train_idx, val_idx, cfg, steps, device,
                    tag=f"[seed {seed}] ", seed=seed, on_checkpoint=on_checkpoint)
    return trace


def _null_baseline(trace: list[dict]) -> dict:
    """What a criterion that picked a checkpoint AT RANDOM would score.

    This is the baseline the headline needs, and it is not flattering to either
    criterion. Most checkpoints of a converged run already pass the whole battery,
    so "my criterion selected 8/8" is a weak claim on its own — the honest question
    is how each criterion compares to drawing a checkpoint out of a hat.

    It is also what makes the real finding visible: validation Q-loss selects BELOW
    this baseline. A criterion that is worse than chance at the thing you care about
    is not merely imperfect, it is pointed the wrong way.
    """
    bat = np.asarray([t["battery_passed_n"] for t in trace], dtype=float)
    return {
        "mean_battery_over_all_checkpoints": round(float(bat.mean()), 4),
        "share_of_checkpoints_at_full_battery": round(float((bat == 8).mean()), 4),
        "n_checkpoints": int(len(bat)),
        "interpretation": ("a criterion that selected uniformly at random from this "
                           "run's checkpoints would score this on the battery"),
    }


def _verdict_for_trace(trace: list[dict], crs_key: str) -> dict:
    """What each criterion would have selected, and what the battery thinks of it."""
    steps = [t["step"] for t in trace]
    vq = [t["val_q_loss"] if t["val_q_loss"] is not None else np.inf for t in trace]
    crs = [t[crs_key] if t[crs_key] is not None else -np.inf for t in trace]
    bat = [t["battery_passed_n"] for t in trace]

    # The incumbent: argmin validation Q-loss. The proposal: argmax CRS among
    # candidates that pass the guards, which is what `selection.best` does.
    i_vq = int(np.argmin(vq))
    eligible = [i for i, t in enumerate(trace) if t["selectable"]]
    i_crs = (max(eligible, key=lambda i: (crs[i], -vq[i])) if eligible
             else int(np.argmax(crs)))
    i_bat = int(np.argmax(bat))                      # earliest step achieving the max

    return {
        "val_q_loss_criterion": {"step": steps[i_vq], "val_q_loss": trace[i_vq]["val_q_loss"],
                                 "battery_at_that_step": bat[i_vq]},
        "crs_criterion": {"step": steps[i_crs], "crs": trace[i_crs][crs_key],
                          "battery_at_that_step": bat[i_crs],
                          "n_eligible_checkpoints": len(eligible)},
        "battery_oracle": {"step": steps[i_bat], "battery": bat[i_bat],
                           "note": "the best the run ever reaches; not a usable "
                                   "criterion (it is the deploy gate)"},
        "crs_beats_val_q_loss_on_the_battery": bool(bat[i_crs] > bat[i_vq]),
        "battery_gain": int(bat[i_crs] - bat[i_vq]),
        "null_baseline": _null_baseline(trace),
        "val_q_loss_is_worse_than_random": bool(
            bat[i_vq] < _null_baseline(trace)["mean_battery_over_all_checkpoints"]),
        "rank_correlation_with_battery": {
            "crs": _spearman(crs, bat),
            "negative_val_q_loss": _spearman([-v for v in vq], bat),
            "note": "Spearman over this run's checkpoints; higher means the "
                    "criterion orders checkpoints more like the battery does",
        },
    }


def evaluate(track: str = "a", seeds: int = 3, steps: int = 30000,
             device: str = "cpu", val_subsample: int | None = None,
             crs_encoding: str = "training", write: bool = True) -> dict:
    """Trace ``seeds`` training runs and compare the two selection criteria.

    ``write=False`` skips the artifact so a test can exercise this without
    overwriting the canonical result.
    """
    if crs_encoding not in ("training", "serving"):
        raise ValueError("crs_encoding must be 'training' or 'serving'")
    cfg_name = "track_a_config.yaml" if track == "a" else "track_b_config.yaml"
    cfg = yaml.safe_load((config.REPO_ROOT / "backend" / "configs" / cfg_name).read_text())
    # Trace the WHOLE curve: see the module docstring on why early stopping is off.
    patience_override = steps + 1
    cfg = {**cfg, "early_stop_patience": patience_override}

    raw_all = D.load_mdp(track)
    feats = list(raw_all["feature_order"])
    stats = N.load(config.MODEL_PATH / (
        "normaliser_stats.json" if track == "a" else "normaliser_stats_track_b.json"))

    d = T._normalise(D.load_mdp(track), track)
    train_idx = np.where(d["split"] == "train")[0]
    val_idx = np.where(d["split"] == "val")[0]
    if len(val_idx) == 0:
        raise RuntimeError("MDP has no val split — the criterion needs held-out states.")
    if val_subsample and val_subsample < len(val_idx):
        # Fixed seed so every seed's trace is scored on the SAME rows; otherwise a
        # criterion's run-to-run spread would include the sampling noise.
        val_idx = np.sort(np.random.default_rng(0).choice(
            val_idx, size=val_subsample, replace=False))

    raw_val = raw_all["states"][val_idx]
    weight_val = raw_all["weight_kg"][val_idx].astype(float)
    clin_val = raw_all["actions"][val_idx]
    z_serve_val = N.transform_inference(raw_val, stats, feats).astype(np.float32)

    masks = SEL.strata_masks(raw_val, feats, weight_val)
    log.info("Track %s: %d train / %d val transitions; strata n = %s", track.upper(),
             len(train_idx), len(val_idx),
             {k: int(v.sum()) for k, v in masks.items()})

    clinician = SEL.clinical_reflex_score(clin_val, raw_val, feats, weight_val)
    log.info("clinician CRS on the same strata = %s (the behaviour baseline)",
             clinician["crs"])

    crs_key = f"crs_{crs_encoding}_encoding"
    per_seed = {}
    t0 = time.time()
    for s in range(seeds):
        trace = _trace_one_seed(d, raw_val, feats, weight_val, clin_val, z_serve_val,
                                stats, cfg, train_idx, val_idx, steps, s, device)
        per_seed[f"seed{s}"] = {"verdict": _verdict_for_trace(trace, crs_key),
                                "trace": trace}

    verdicts = [v["verdict"] for v in per_seed.values()]
    agg = {
        "battery_at_val_q_loss_choice": _ci95(
            [v["val_q_loss_criterion"]["battery_at_that_step"] for v in verdicts]),
        "battery_at_crs_choice": _ci95(
            [v["crs_criterion"]["battery_at_that_step"] for v in verdicts]),
        "battery_oracle": _ci95([v["battery_oracle"]["battery"] for v in verdicts]),
        "step_selected_by_val_q_loss": _ci95(
            [v["val_q_loss_criterion"]["step"] for v in verdicts]),
        "step_selected_by_crs": _ci95([v["crs_criterion"]["step"] for v in verdicts]),
        "rank_correlation_crs_vs_battery": _ci95(
            [v["rank_correlation_with_battery"]["crs"] for v in verdicts]),
        "rank_correlation_neg_val_q_vs_battery": _ci95(
            [v["rank_correlation_with_battery"]["negative_val_q_loss"] for v in verdicts]),
        "n_seeds_where_crs_beats_val_q_loss": int(sum(
            v["crs_beats_val_q_loss_on_the_battery"] for v in verdicts)),
        "n_seeds_where_crs_ties_or_loses": int(sum(
            not v["crs_beats_val_q_loss_on_the_battery"] for v in verdicts)),
        "battery_if_a_checkpoint_were_picked_at_random": _ci95(
            [v["null_baseline"]["mean_battery_over_all_checkpoints"] for v in verdicts]),
        "share_of_checkpoints_at_full_battery": _ci95(
            [v["null_baseline"]["share_of_checkpoints_at_full_battery"] for v in verdicts]),
        "n_seeds_where_val_q_loss_is_worse_than_random": int(sum(
            v["val_q_loss_is_worse_than_random"] for v in verdicts)),
        "prob_all_seeds_at_full_battery_under_random_picking": round(float(np.prod(
            [v["null_baseline"]["share_of_checkpoints_at_full_battery"]
             for v in verdicts])), 5),
        "how_to_read_the_null": (
            "Most checkpoints of a converged run already pass the battery, so "
            "'CRS selected 8/8 in every seed' is suggestive rather than decisive on "
            "its own — see prob_all_seeds_at_full_battery_under_random_picking. The "
            "load-bearing result is the SIGN: validation Q-loss selects BELOW random "
            "and is negatively rank-correlated with the battery, while CRS is "
            "positively correlated and matches the oracle."),
    }

    result = {
        "track": track,
        "question": ("Does the clinical reflex score select checkpoints the deploy "
                     "gate's battery prefers, where validation Q-loss does not?"),
        "n_seeds": seeds,
        "steps_per_seed": steps,
        "crs_encoding_used_for_selection": crs_encoding,
        "budget": {
            "total_steps": steps,
            "checkpoint_every": cfg["checkpoint_every"],
            "early_stop_patience": patience_override,
            "early_stopping": ("DISABLED for the trace (patience > total_steps) so the "
                               "whole curve past the val-loss optimum is visible"),
            "batch_size": cfg["batch_size"], "lr": cfg["lr"],
            "cql_alpha": cfg.get("cql", {}).get("alpha"),
            "lam_causal": cfg.get("reward", {}).get("lam_causal"),
        },
        "n_val_transitions": int(len(val_idx)),
        "val_subsample": val_subsample,
        "strata_n": {k: int(v.sum()) for k, v in masks.items()},
        "clinician_baseline": clinician,
        "aggregate": agg,
        "per_seed": per_seed,
        "ground_truth": ("the 8-case battery at the serving encoding, scored exactly "
                         "as backend/scripts/verify_before_deploy.py scores it. It is "
                         "the TARGET of this experiment and is never an input to the "
                         "criterion — selection.score never sees it."),
        "boundary": ("CRS is a selection criterion, not a value estimate. It is not "
                     "comparable across rewards and implies nothing about patient "
                     "benefit. Nothing here promotes or writes a deployed model."),
        "elapsed_seconds": round(time.time() - t0, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if write:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"selection_criterion_track_{track}.json").write_text(
            json.dumps(result, indent=2))

    log.info("=== verdict over %d seed(s) ===", seeds)
    log.info("  val Q-loss picks step %s → battery %s/8",
             agg["step_selected_by_val_q_loss"]["mean"],
             agg["battery_at_val_q_loss_choice"]["mean"])
    log.info("  CRS        picks step %s → battery %s/8",
             agg["step_selected_by_crs"]["mean"], agg["battery_at_crs_choice"]["mean"])
    log.info("  oracle (the gate itself) reaches %s/8", agg["battery_oracle"]["mean"])
    log.info("  rank corr with battery: CRS %s vs -val_q %s",
             agg["rank_correlation_crs_vs_battery"]["mean"],
             agg["rank_correlation_neg_val_q_vs_battery"]["mean"])
    log.info("  CRS beats val Q-loss in %d/%d seeds",
             agg["n_seeds_where_crs_beats_val_q_loss"], seeds)
    return result


def recompute(track: str = "a", crs_encoding: str | None = None,
              write: bool = True) -> dict:
    """Rebuild the verdict and aggregate from an existing artifact's stored traces.

    The traces are the expensive part (five 30,000-step runs, ~12 minutes); the
    verdict is arithmetic over them. Separating the two means a change to how the
    comparison is reported — a new baseline, a different tie-break — can be applied
    to exactly the runs already published, rather than to a fresh set of runs that
    would differ for unrelated reasons. ``_verdict_for_trace`` is deterministic
    given a trace, so this reproduces the artifact bit-for-bit where the logic is
    unchanged.
    """
    f = RESULTS / f"selection_criterion_track_{track}.json"
    if not f.exists():
        raise FileNotFoundError(f"{f} not present — run the experiment first")
    result = json.loads(f.read_text())
    enc = crs_encoding or result.get("crs_encoding_used_for_selection", "training")
    crs_key = f"crs_{enc}_encoding"

    for seed_key, payload in result["per_seed"].items():
        payload["verdict"] = _verdict_for_trace(payload["trace"], crs_key)
    verdicts = [v["verdict"] for v in result["per_seed"].values()]

    result["crs_encoding_used_for_selection"] = enc
    result["aggregate"] = {
        "battery_at_val_q_loss_choice": _ci95(
            [v["val_q_loss_criterion"]["battery_at_that_step"] for v in verdicts]),
        "battery_at_crs_choice": _ci95(
            [v["crs_criterion"]["battery_at_that_step"] for v in verdicts]),
        "battery_oracle": _ci95([v["battery_oracle"]["battery"] for v in verdicts]),
        "step_selected_by_val_q_loss": _ci95(
            [v["val_q_loss_criterion"]["step"] for v in verdicts]),
        "step_selected_by_crs": _ci95([v["crs_criterion"]["step"] for v in verdicts]),
        "rank_correlation_crs_vs_battery": _ci95(
            [v["rank_correlation_with_battery"]["crs"] for v in verdicts]),
        "rank_correlation_neg_val_q_vs_battery": _ci95(
            [v["rank_correlation_with_battery"]["negative_val_q_loss"] for v in verdicts]),
        "n_seeds_where_crs_beats_val_q_loss": int(sum(
            v["crs_beats_val_q_loss_on_the_battery"] for v in verdicts)),
        "n_seeds_where_crs_ties_or_loses": int(sum(
            not v["crs_beats_val_q_loss_on_the_battery"] for v in verdicts)),
        "battery_if_a_checkpoint_were_picked_at_random": _ci95(
            [v["null_baseline"]["mean_battery_over_all_checkpoints"] for v in verdicts]),
        "share_of_checkpoints_at_full_battery": _ci95(
            [v["null_baseline"]["share_of_checkpoints_at_full_battery"] for v in verdicts]),
        "n_seeds_where_val_q_loss_is_worse_than_random": int(sum(
            v["val_q_loss_is_worse_than_random"] for v in verdicts)),
        "prob_all_seeds_at_full_battery_under_random_picking": round(float(np.prod(
            [v["null_baseline"]["share_of_checkpoints_at_full_battery"]
             for v in verdicts])), 5),
        "how_to_read_the_null": result.get("aggregate", {}).get("how_to_read_the_null"),
    }
    result["recomputed_at"] = datetime.now(timezone.utc).isoformat()
    if write:
        f.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="a", choices=["a", "b"])
    ap.add_argument("--recompute", action="store_true",
                    help="rebuild the verdict from the existing artifact's traces "
                         "instead of retraining")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--val-subsample", type=int, default=None,
                    help="score the criteria on this many val rows (default: all)")
    ap.add_argument("--crs-encoding", default="training", choices=["training", "serving"])
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    if args.recompute:
        recompute(args.track, crs_encoding=args.crs_encoding, write=not args.no_write)
        return
    evaluate(args.track, seeds=args.seeds, steps=args.steps,
             val_subsample=args.val_subsample, crs_encoding=args.crs_encoding,
             write=not args.no_write)


if __name__ == "__main__":
    main()
