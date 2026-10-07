"""HybridIQL training loop with checkpointing + early stopping (Section 6.5).

Trains a track's policy on its offline MDP dataset and also fits the lightweight
baselines (BC, CQI). Models are saved per track: models/policy_track_<track>.pt.

Run:
    python -m backend.rl.trainer --track a
    python -m backend.rl.trainer --track b --steps 5000
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone

import numpy as np
import torch
import yaml

from backend.mdp import dataset as D
from backend.mdp import action_space
from backend.pipeline import config, splits
from backend.pipeline.logging_utils import get_logger
from backend.rl import behavior_clone, cqi
from backend.rl.hybrid_iql import Batch, HybridIQL

log = get_logger("rl_trainer")


def _snapshot(model: HybridIQL) -> dict:
    """A real copy of the weights, for best-checkpoint restoration.

    ``HybridIQL.state_dict()`` returns the live parameter tensors by reference (as
    ``nn.Module.state_dict()`` does). Holding that as "the best checkpoint" is a
    silent no-op: the optimiser mutates those same tensors in place, so by the end
    of training the "best" dict holds the FINAL weights and the closing
    ``load_state_dict(best_sd)`` restores nothing. Early stopping then reports a
    best validation score it did not actually keep, and a run whose validation loss
    diverges ships the diverged model.

    Deep-copying at each improvement is what makes early stopping real.
    """
    return copy.deepcopy(model.state_dict())


def _to_batch(d: dict, idx: np.ndarray, device: str) -> Batch:
    t = lambda a, dt: torch.as_tensor(a[idx], dtype=dt, device=device)
    weights = (torch.as_tensor(d["ipw"][idx], dtype=torch.float32, device=device)
               if "ipw" in d else None)
    return Batch(
        states=t(d["states"], torch.float32),
        actions=t(d["actions"], torch.long),
        rewards=t(d["rewards"], torch.float32),
        next_states=t(d["next_states"], torch.float32),
        dones=t(d["dones"].astype(np.float32), torch.float32),
        weights=weights,
    )


def _val_q_loss(model: HybridIQL, d: dict, val_idx: np.ndarray, device: str) -> float:
    if len(val_idx) == 0:
        return float("nan")
    b = _to_batch(d, val_idx, device)
    with torch.no_grad():
        s = model._phi(b.states)
        idx = torch.arange(len(b.actions), device=model.device)
        q_taken = model.Q(s)[idx, b.actions]
        v_next = model.V_target(model._phi(b.next_states))
        y = b.rewards + model.gamma * (1 - b.dones) * v_next
        return float(((q_taken - y) ** 2).mean())


def _normalise(d: dict, track: str) -> dict:
    """Z-score the stored raw states with the track's normaliser."""
    from backend.mdp import normaliser as N
    nf = "normaliser_stats.json" if track == "a" else "normaliser_stats_track_b.json"
    stats = N.load(config.MODEL_PATH / nf)
    order = d["feature_order"]
    d["states"] = N.transform(d["states"], stats, order).astype(np.float32)
    d["next_states"] = N.transform(d["next_states"], stats, order).astype(np.float32)
    return d


def _run_training(d: dict, train_idx: np.ndarray, val_idx: np.ndarray,
                  cfg: dict, steps: int | None, device: str,
                  tag: str = "") -> tuple[HybridIQL, float, dict]:
    """Core HybridIQL training loop with early stopping.

    Returns (model, best_val_q, provenance), where provenance is empty unless the
    config asked for a warm start.
    """
    total_steps = steps if steps is not None else cfg["total_steps"]
    ckpt_every = min(cfg["checkpoint_every"], max(500, total_steps // 10))
    if len(train_idx) == 0:
        train_idx = np.arange(len(d["actions"]))

    # Optional IPW reweighting for confounding-by-indication (§16 item 8). Fit the
    # behaviour propensity model on THIS split's train transitions only (no leak);
    # weights default to 1.0 outside the train split. Opt-in via config.
    if cfg.get("ipw", {}).get("enabled", False):
        from backend.rl.ipw import compute_ipw_weights
        w = compute_ipw_weights(d["states"][train_idx], d["actions"][train_idx])
        ipw = np.ones(len(d["actions"]), dtype=np.float32)
        ipw[train_idx] = w
        d = {**d, "ipw": ipw}
        log.info("%sIPW enabled: reweighted %d train transitions (mean=%.2f, max=%.2f)",
                 tag, len(train_idx), float(w.mean()), float(w.max()))
    batch_size = min(cfg["batch_size"], len(train_idx))
    model = HybridIQL(state_dim=d["states"].shape[1], action_dim=cfg["action_dim"],
                      hidden_dim=cfg["hidden_dim"], gamma=cfg["gamma"],
                      tau=cfg["tau"], beta=cfg["beta"], lr=cfg["lr"],
                      cql_alpha=cfg.get("cql", {}).get("alpha", 0.0), device=device)

    # Optional warm start from a lower-dimensional policy (config `init_from`).
    # Track A has no `init_from`, so its training path is bit-identical to before.
    provenance: dict = {}
    init_from = cfg.get("init_from")
    if init_from:
        src = config.MODEL_PATH / str(init_from)
        if not src.exists():
            raise FileNotFoundError(
                f"init_from={init_from} not found at {src}. Train the source policy "
                "first, or drop `init_from` from the config to train from scratch.")
        provenance = model.warm_start_from(
            torch.load(src, map_location=device, weights_only=False))
        provenance["init_from"] = str(init_from)
        log.info("%swarm-started from %s (%s-dim, n_train=%s) — adapter = [I | 0], "
                 "so training starts AT the source policy",
                 tag, init_from, provenance["init_from_state_dim"],
                 provenance["init_from_n_transitions"])

    best_val, best_sd, stale = float("inf"), _snapshot(model), 0
    best_step = 0
    # Score the INITIALISATION before any gradient step, so "do not fine-tune at
    # all" is a candidate that early stopping can actually select. For a random
    # init this is a formality (the score is terrible and the first real checkpoint
    # beats it); for a warm start it is the whole point — the source policy is a
    # legitimate answer, and without this a warm start could only ever be made
    # worse, never left alone. If the validation loss rises monotonically from
    # here, the honest result is that fine-tuning on this cohort does not help.
    if init_from and len(val_idx):
        best_val = _val_q_loss(model, d, val_idx, device)
        if best_val != best_val:                      # NaN → unusable
            best_val = float("inf")
        else:
            log.info("%sstep 0 (warm-start baseline, no fine-tuning) | val_q=%.4f",
                     tag, best_val)
    rng = np.random.default_rng(config.SPLIT_SEED)
    last: dict = {}
    for step in range(total_steps):
        bi = rng.choice(train_idx, size=batch_size, replace=len(train_idx) < batch_size)
        last = model.update(_to_batch(d, bi, device))
        if (step + 1) % ckpt_every == 0 or step == total_steps - 1:
            vl = _val_q_loss(model, d, val_idx, device)
            score = vl if vl == vl else last["q_loss"]
            log.info("%sstep %d | v=%.4f q=%.4f pi=%.4f | val_q=%.4f",
                     tag, step + 1, last["v_loss"], last["q_loss"],
                     last["pi_loss"], score)
            if score < best_val:
                best_val, best_sd, stale = score, _snapshot(model), 0
                best_step = step + 1
            else:
                stale += ckpt_every
                if stale >= cfg["early_stop_patience"]:
                    log.info("%searly stop @ %d", tag, step + 1)
                    break
    model.load_state_dict(best_sd)
    provenance["selected_step"] = best_step
    if init_from and best_step == 0:
        log.warning("%sfine-tuning did NOT improve validation loss — keeping the "
                    "warm-start initialisation unchanged (selected_step=0). On this "
                    "cohort the extra state dimensions earn no weight.", tag)
    return model, best_val, provenance


def _eval_on(model: HybridIQL, d: dict, idx: np.ndarray, device: str = "cpu") -> dict:
    """Held-out metrics for a fitted policy on transition indices `idx`."""
    if len(idx) == 0:
        return {}
    acts = model.act_batch(d["states"][idx])
    behav = d["actions"][idx]
    am = action_space.ACTION_MAP
    hold = action_space.encode_action(0, 0, 0.0)
    return {
        "n_test": int(len(idx)),
        "test_q_loss": _val_q_loss(model, d, idx, device),
        "behaviour_match": float(np.mean(acts == behav)),
        "no_change_share": float(np.mean(acts == hold)),
        "mean_dPEEP": float(np.mean([am[int(a)][0] for a in acts])),
        "mean_dTV": float(np.mean([am[int(a)][1] for a in acts])),
        "mean_dFiO2": float(np.mean([am[int(a)][2] for a in acts])),
    }


def train_kfold(track: str, k: int = 5, steps: int | None = None,
                device: str = "cpu") -> dict:
    """Patient-level k-fold cross-validation, then a final model on the full split.

    For each of k folds: hold the fold's stays out as test, carve 10% of the rest
    as a validation set for early stopping, train on the remainder, and evaluate
    on the held-out test stays. Per-fold metrics are aggregated to mean±std and
    written to logs/cv_track_<track>.json. The deployed model is then trained by
    ``train()`` on the canonical train/val split and saved as usual.
    """
    cfg_name = "track_a_config.yaml" if track == "a" else "track_b_config.yaml"
    cfg = yaml.safe_load((config.REPO_ROOT / "backend" / "configs" / cfg_name).read_text())
    d = _normalise(D.load_mdp(track), track)
    sid = d["stay_id"]
    uniq = np.unique(sid)
    folds = splits.make_kfold(uniq, k=k, name=f"kfold_split_track_{track}.json")
    log.info("Track %s: %d-fold CV over %d stays / %d transitions, state_dim=%d",
             track.upper(), k, len(uniq), len(sid), d["states"].shape[1])

    fold_metrics: list[dict] = []
    for fi, test_stays in enumerate(folds):
        test_set = set(test_stays)
        rest = np.array([s for s in uniq if s not in test_set])
        rng = np.random.default_rng(config.SPLIT_SEED + fi)
        rng.shuffle(rest)
        n_val = max(1, int(0.1 * len(rest)))
        val_stays = set(int(s) for s in rest[:n_val])
        train_stays = set(int(s) for s in rest[n_val:])
        train_idx = np.where(np.isin(sid, list(train_stays)))[0]
        val_idx = np.where(np.isin(sid, list(val_stays)))[0]
        test_idx = np.where(np.isin(sid, list(test_set)))[0]
        log.info("=== fold %d/%d: %d train / %d val / %d test transitions ===",
                 fi + 1, k, len(train_idx), len(val_idx), len(test_idx))
        model, best_val, _ = _run_training(d, train_idx, val_idx, cfg, steps, device,
                                        tag=f"[f{fi + 1}] ")
        m = _eval_on(model, d, test_idx, device)
        m.update({"fold": fi + 1, "val_q": best_val})
        fold_metrics.append(m)
        log.info("fold %d test: behaviour_match=%.4f no_change=%.4f test_q=%.4f",
                 fi + 1, m["behaviour_match"], m["no_change_share"], m["test_q_loss"])

    # aggregate mean±std across folds
    keys = ["behaviour_match", "no_change_share", "test_q_loss",
            "mean_dPEEP", "mean_dTV", "mean_dFiO2", "val_q"]
    summary = {key: {"mean": float(np.mean([fm[key] for fm in fold_metrics])),
                     "std": float(np.std([fm[key] for fm in fold_metrics]))}
               for key in keys}
    cv_out = {"k": k, "folds": fold_metrics, "summary": summary}
    (config.LOGS_PATH / f"cv_track_{track}.json").write_text(json.dumps(cv_out, indent=2))
    log.info("Track %s %d-fold CV summary: behaviour_match=%.4f±%.4f, "
             "test_q=%.4f±%.4f", track.upper(), k,
             summary["behaviour_match"]["mean"], summary["behaviour_match"]["std"],
             summary["test_q_loss"]["mean"], summary["test_q_loss"]["std"])

    # final deployed model on the canonical split
    log.info("=== training final deployed model (canonical split) ===")
    final = train(track, steps=steps, device=device)
    return {"cv_summary": summary, "final": final}


def train(track: str, steps: int | None = None, device: str = "cpu") -> dict:
    cfg_name = "track_a_config.yaml" if track == "a" else "track_b_config.yaml"
    cfg = yaml.safe_load((config.REPO_ROOT / "backend" / "configs" / cfg_name).read_text())
    d = _normalise(D.load_mdp(track), track)

    train_idx = np.where(d["split"] == "train")[0]
    val_idx = np.where(d["split"] == "val")[0]
    if len(train_idx) == 0:
        train_idx = np.arange(len(d["actions"]))
    log.info("Track %s: %d train / %d val transitions, state_dim=%d",
             track.upper(), len(train_idx), len(val_idx), d["states"].shape[1])

    model, best_val, provenance = _run_training(d, train_idx, val_idx, cfg,
                                                steps, device)
    out = config.MODEL_PATH / f"policy_track_{track}.pt"
    # Persist the training operating point IN the checkpoint so the deployed
    # artifact is self-describing (no re-deriving lam_causal/alpha from logs).
    # This is what §15.3's tuning saga needed: a checkpoint you can interrogate.
    lam_causal = float(cfg.get("reward", {}).get("lam_causal", 0.0))
    cql_alpha = float(cfg.get("cql", {}).get("alpha", 0.0))
    torch.save({"state_dict": model.state_dict(), "state_dim": d["states"].shape[1],
                "action_dim": cfg["action_dim"], "hidden_dim": cfg["hidden_dim"],
                "track": track,
                "lam_causal": lam_causal, "cql_alpha": cql_alpha,
                "n_transitions": int(len(train_idx)),
                "trained_at": datetime.now(timezone.utc).isoformat(),
                # Warm-start provenance (empty for a from-scratch fit). Downstream
                # readers need it to know this policy inherited a Q-function fitted
                # on far more data than `n_transitions` — the 12-vs-18 ablation uses
                # it to tell a nested comparison from a confounded one.
                **provenance,
                "config_snapshot": cfg}, out)
    log.info("Saved HybridIQL → %s (best val_q=%.4f, lam_causal=%.2f, cql_alpha=%.2f, "
             "n_train=%d)", out, best_val, lam_causal, cql_alpha, len(train_idx))

    # --- baselines (BC + CQI) on train split ---
    Xtr, Atr = d["states"][train_idx], d["actions"][train_idx]
    behavior_clone.train_bc(Xtr, Atr, track)
    qv = np.array([model.q_values(x) for x in Xtr])
    q_taken = qv[np.arange(len(Atr)), Atr]
    cqi.train_cqi(Xtr, Atr, q_taken, track)

    # action-distribution sanity on val/test
    metrics = _policy_metrics(model, d)
    (config.LOGS_PATH / f"train_track_{track}.json").write_text(json.dumps(
        {"best_val_q": best_val, **metrics}, indent=2))
    log.info("Track %s metrics: %s", track.upper(), metrics)
    return metrics


def _policy_metrics(model: HybridIQL, d: dict) -> dict:
    eval_idx = np.where(np.isin(d["split"], ["test", "val"]))[0]
    if len(eval_idx) == 0:
        eval_idx = np.arange(len(d["actions"]))
    acts = model.act_batch(d["states"][eval_idx])
    behav = d["actions"][eval_idx]
    am = action_space.ACTION_MAP
    hold = action_space.encode_action(0, 0, 0.0)   # the no-change action index
    return {
        "n_eval": int(len(eval_idx)),
        "behaviour_match": float(np.mean(acts == behav)),
        "no_change_share": float(np.mean(acts == hold)),
        "mean_dPEEP": float(np.mean([am[int(a)][0] for a in acts])),
        "mean_dTV": float(np.mean([am[int(a)][1] for a in acts])),
        "mean_dFiO2": float(np.mean([am[int(a)][2] for a in acts])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["a", "b"], required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--kfold", type=int, default=0,
                    help="If >1, run k-fold CV before training the final model.")
    args = ap.parse_args()
    if args.kfold and args.kfold > 1:
        train_kfold(args.track, k=args.kfold, steps=args.steps)
    else:
        train(args.track, steps=args.steps)


if __name__ == "__main__":
    main()
