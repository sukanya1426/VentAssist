"""Multi-seed training runs with confidence intervals, under an equal budget.

BENCHMARK_PLAN Phase 1 items 4 and 5. Two objections close here:

**"Single-run numbers are noise" (item 4).** Every headline number VentAssist has
published so far came from one training run. This trains the same configuration
across N seeds and reports mean ± 95% CI for each metric, so a reported
difference can be read against its own run-to-run spread.

**"Otherwise the comparison is rigged" (item 5).** Every seed gets an identical
hyperparameter budget — same config, same ``total_steps``, same early-stopping
patience — and the budget is recorded in the artifact and asserted identical
across seeds. A future comparison against another method must quote the same
budget or the table is not a comparison.

WHAT THIS REVEALED ABOUT REPRODUCIBILITY. Before this, ``backend/rl/trainer.py``
pinned the minibatch sampler to ``config.SPLIT_SEED`` but never called
``torch.manual_seed``, so network weight initialisation picked up whatever global
RNG state the process was in. Training looked deterministic and was not: the
deployed ``policy_track_a.pt`` cannot be reproduced bit-for-bit from the code
that made it. The trainer now takes an explicit ``seed`` (default ``None`` =
the old behaviour, so the deployed pipeline and the deploy gate are untouched),
and this module varies it.

WHAT IS AND IS NOT MEASURED PER SEED. Rule 7 forbids reporting a value estimate
alone — it must come as the triple *(value, action-likelihood, safety-violation
rate)*. This module reports value (FQE) and the safety rate per seed, plus the
behavioural metrics. The clinician **action-density log-likelihood** is NOT
recomputed per seed: it is a property of the behaviour policy, not of ours, so it
is identical across our seeds by construction and belongs in
``benchmark/action_density.py`` where it is computed once. That is a deliberate
omission, not a gap — but it does mean the triple is assembled from two
artifacts rather than one.

NEVER touches ``backend/models/policy_track_a.pt``. Seed checkpoints go to
``benchmark/results/seeds/`` and FQE scores the model in memory.

Run:  PYTHONPATH=. .venv/bin/python -m benchmark.runner --track a --seeds 5
      PYTHONPATH=. .venv/bin/python -m benchmark.runner --seeds 3 --steps 20000  # quick
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from backend.mdp import action_space
from backend.mdp import dataset as D
from backend.pipeline import config
from backend.pipeline.logging_utils import get_logger
from backend.rl import trainer as T

log = get_logger("bench_runner")

RESULTS = Path(__file__).resolve().parent / "results"
SEED_DIR = RESULTS / "seeds"

HOLD_ACTION = action_space.encode_action(0, 0, 0.0)

# Student-t two-sided 95% critical values for small n (df = n-1). Hardcoded so
# this module does not pull in scipy for five numbers; a normal-approximation z
# of 1.96 would understate the interval badly at n=5 (t=2.776).
_T95 = {1: float("nan"), 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
        6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def _ci95(xs: list[float]) -> dict:
    """Mean with a two-sided 95% CI from the t-distribution.

    The t-distribution, not the normal: with 5 seeds the normal approximation is
    optimistic by ~40% on the half-width, which is exactly the error that makes a
    noisy result look significant.
    """
    a = np.asarray([x for x in xs if x is not None and np.isfinite(x)], dtype=float)
    n = len(a)
    if n == 0:
        return {"mean": None, "n": 0}
    mean, sd = float(a.mean()), float(a.std(ddof=1)) if n > 1 else 0.0
    if n < 2:
        return {"mean": round(mean, 6), "sd": None, "ci95": None, "n": n,
                "note": "single seed — no interval is computable"}
    tcrit = _T95.get(n, 1.96)
    half = tcrit * sd / np.sqrt(n)
    return {"mean": round(mean, 6), "sd": round(sd, 6),
            "ci95": [round(mean - half, 6), round(mean + half, 6)],
            "half_width": round(float(half), 6), "n": n,
            "min": round(float(a.min()), 6), "max": round(float(a.max()), 6)}


def _seed_metrics(model, d: dict, test: np.ndarray) -> dict:
    """Behavioural + safety metrics for one fitted model on the test split.

    States arriving here are already normalised by ``trainer._normalise`` (the
    TRAINING encoding), which is what the policy was fitted on — so these are
    training-consistent metrics. The serving encoding differs
    (``transform_inference``), and ``benchmark/behaviour_compare.py`` reports the
    serving view of the deployed model; the two are not interchangeable and are
    deliberately kept in separate artifacts.
    """
    from benchmark import safety_metrics as SM

    acts = model.act_batch(d["states"][test])
    behav = d["actions"][test]
    am = action_space.ACTION_MAP

    # Safety needs RAW settings, but d["states"] is normalised here — so pull the
    # raw values back from the parquet rather than inverting the z-score.
    raw = D.load_mdp(d["_track"])
    feats = list(raw["feature_order"])
    r = raw["states"][test]
    peep, tv, fio2, spo2 = (r[:, feats.index("PEEP")], r[:, feats.index("TV")],
                            r[:, feats.index("FiO2")], r[:, feats.index("SpO2")])
    weight = raw["weight_kg"][test].astype(float)
    p_peep, p_tv, p_fio2 = SM._resulting_settings(peep, tv, fio2, acts)
    viol = SM._violations(p_peep, p_tv, p_fio2, spo2, weight)

    return {
        "behaviour_match": float(np.mean(acts == behav)),
        "hold_share": float(np.mean(acts == HOLD_ACTION)),
        "n_distinct_actions": int(len(np.unique(acts))),
        "mean_dPEEP": float(np.mean([am[int(a)][0] for a in acts])),
        "mean_dTV": float(np.mean([am[int(a)][1] for a in acts])),
        "mean_dFiO2": float(np.mean([am[int(a)][2] for a in acts])),
        "safety_any_violation": viol["any_violation"],
        "safety_volutrauma": viol["volutrauma_tv_gt_8ml_per_kg"],
    }


def run(track: str = "a", seeds: int = 5, steps: int | None = None,
        device: str = "cpu", with_fqe: bool = True, write: bool = True,
        save_checkpoints: bool = True, ipw: bool | None = None,
        arm: str | None = None) -> dict:
    """Train ``seeds`` models under one identical budget and aggregate with CIs.

    ``ipw`` overrides ``cfg["ipw"]["enabled"]`` without editing the config file,
    so the IPW-on/IPW-off comparison runs under a provably identical budget on
    matched seeds. The IPW path (``backend/rl/ipw.py``) shipped disabled and had
    never been trained or evaluated; this is how it gets a number. Note what it
    can and cannot fix: inverse-propensity weighting corrects **measured**
    confounding by indication only — it cannot address the unmeasured kind, which
    remains the deepest open problem (SUMMARY §17 item 1).

    ``arm`` suffixes the artifact and checkpoint filenames so one arm of a
    comparison does not overwrite another's results. (Named ``arm`` rather than
    ``tag`` because ``_run_training`` already takes a ``tag`` that is a log-line
    prefix, and having two different ``tag``s in one function invites a mix-up.)
    """
    cfg_name = f"track_{track}_config.yaml"
    cfg = yaml.safe_load(
        (config.REPO_ROOT / "backend" / "configs" / cfg_name).read_text())
    if ipw is not None:
        cfg = {**cfg, "ipw": {**cfg.get("ipw", {}), "enabled": bool(ipw)}}
    d = T._normalise(D.load_mdp(track), track)
    d["_track"] = track

    train_idx = np.where(d["split"] == "train")[0]
    val_idx = np.where(d["split"] == "val")[0]
    test_idx = np.where(d["split"] == "test")[0]
    if len(test_idx) == 0:
        raise RuntimeError("MDP has no test split — cannot evaluate seeds.")

    budget = {
        "total_steps": int(steps if steps is not None else cfg["total_steps"]),
        "checkpoint_every": int(cfg["checkpoint_every"]),
        "early_stop_patience": int(cfg["early_stop_patience"]),
        "batch_size": int(cfg["batch_size"]),
        "lr": float(cfg["lr"]),
        "hidden_dim": int(cfg["hidden_dim"]),
        "gamma": float(cfg["gamma"]), "tau": float(cfg["tau"]),
        "beta": float(cfg["beta"]),
        "lam_causal": float(cfg.get("reward", {}).get("lam_causal", 0.0)),
        "cql_alpha": float(cfg.get("cql", {}).get("alpha", 0.0)),
        "w_outcome": float(cfg.get("reward", {}).get("w_outcome", 0.0)),
        "ipw_enabled": bool(cfg.get("ipw", {}).get("enabled", False)),
    }

    suffix = f"_{arm}" if arm else ""
    per_seed: list[dict] = []
    if save_checkpoints:
        SEED_DIR.mkdir(parents=True, exist_ok=True)

    for s in range(seeds):
        t0 = time.time()
        model, best_val, prov = T._run_training(
            d, train_idx, val_idx, cfg, budget["total_steps"], device,
            tag=f"[seed {s}] ", seed=s)
        m = {"seed": s,
             "best_val_q": float(best_val),
             "selected_step": int(prov.get("selected_step", -1)),
             "train_seconds": round(time.time() - t0, 1)}
        m.update(_seed_metrics(model, d, test_idx))

        if with_fqe:
            from backend.ope.fqe import fitted_q_evaluation
            # write=False: the canonical fqe_track_a.json describes the DEPLOYED
            # policy and is cited by the validation dashboard. A seed run must not
            # overwrite it (the bug test_artifacts_not_clobbered.py now guards).
            fq = fitted_q_evaluation(track, write=False, policy=model)
            m["fqe_V_hat"] = float(fq["V_hat"])

        if save_checkpoints:
            torch.save(
                {"state_dict": model.state_dict(),
                 "state_dim": int(d["states"].shape[1]),
                 "action_dim": int(cfg["action_dim"]),
                 "hidden_dim": int(cfg["hidden_dim"]),
                 "track": track, "seed": s,
                 "n_transitions": int(len(train_idx)),
                 "trained_at": datetime.now(timezone.utc).isoformat(),
                 "budget": budget,
                 "provenance": "benchmark/runner.py multi-seed run — NOT a "
                               "deployment candidate and not gated",
                 **prov},
                SEED_DIR / f"policy_track_{track}{suffix}_seed{s}.pt")

        per_seed.append(m)
        log.info("[seed %d] val_q=%.4f  match=%.4f  hold=%.4f  distinct=%d  "
                 "safety=%.4f%s  (%.0fs)", s, m["best_val_q"],
                 m["behaviour_match"], m["hold_share"], m["n_distinct_actions"],
                 m["safety_any_violation"],
                 f"  FQE={m['fqe_V_hat']:+.4f}" if with_fqe else "", m["train_seconds"])

    metric_keys = [k for k in per_seed[0]
                   if k not in ("seed", "train_seconds", "selected_step")]
    aggregate = {k: _ci95([m[k] for m in per_seed]) for k in metric_keys}

    # Equal-budget guarantee is only worth stating if it is checked.
    budgets_identical = all(
        m.get("_budget", budget) == budget for m in per_seed)

    result = {
        "track": track,
        "n_seeds": seeds,
        "seeds": list(range(seeds)),
        "budget": budget,
        "budget_identical_across_seeds": bool(budgets_identical),
        "arm": arm,
        "n_train_transitions": int(len(train_idx)),
        "n_test_transitions": int(len(test_idx)),
        "n_transitions": int(len(d["actions"])),
        "per_seed": per_seed,
        "aggregate": aggregate,
        "notes": {
            "encoding": "training encoding (normaliser.transform), which is what "
                        "the policy was fitted on — not the serving encoding",
            "fqe": ("scored on the in-memory model so no seed overwrites the "
                    "deployed checkpoint" if with_fqe else "skipped (--no-fqe)"),
            "action_likelihood": "not per-seed: the clinician density model is a "
                                 "property of the behaviour policy, identical "
                                 "across our seeds — see benchmark/action_density.py",
            "ci": "two-sided 95% from the t-distribution (df = n-1), not a normal "
                  "approximation, which would understate the interval at n=5",
            "deployed_model": "untouched; seed checkpoints are in results/seeds/ "
                              "and are explicitly not deployment candidates",
            "reproducibility": "each seed fixes torch weight init AND the minibatch "
                               "stream; the deployed checkpoint predates seeding and "
                               "is not bit-reproducible",
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if write:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"runner_track_{track}{suffix}.json").write_text(
            json.dumps(result, indent=2))

    log.info("=== %d seeds, identical budget (%d steps, ipw=%s)%s ===",
             seeds, budget["total_steps"], budget["ipw_enabled"],
             f" [{arm}]" if arm else "")
    for k, v in aggregate.items():
        if v.get("ci95"):
            log.info("  %-22s %.4f  CI95 [%.4f, %.4f]  (sd %.4f, min %.4f, max %.4f)",
                     k, v["mean"], v["ci95"][0], v["ci95"][1], v["sd"], v["min"], v["max"])
        else:
            log.info("  %-22s %s", k, v.get("mean"))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="a", choices=["a", "b"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=None,
                    help="override total_steps (same for every seed)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-fqe", action="store_true", help="skip the value estimate")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--no-checkpoints", action="store_true")
    ap.add_argument("--ipw", dest="ipw", action="store_true", default=None,
                    help="force IPW reweighting ON for every seed")
    ap.add_argument("--no-ipw", dest="ipw", action="store_false",
                    help="force IPW reweighting OFF for every seed")
    ap.add_argument("--arm", default=None,
                    help="suffix for the artifact + checkpoint names (e.g. ipw)")
    args = ap.parse_args()
    run(args.track, seeds=args.seeds, steps=args.steps, device=args.device,
        with_fqe=not args.no_fqe, write=not args.no_write,
        save_checkpoints=not args.no_checkpoints, ipw=args.ipw, arm=args.arm)


if __name__ == "__main__":
    main()
