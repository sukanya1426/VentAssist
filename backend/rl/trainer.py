"""HybridIQL training loop with checkpointing + early stopping (Section 6.5).

Trains a track's policy on its offline MDP dataset and also fits the lightweight
baselines (BC, CQI). Models are saved per track: models/policy_track_<track>.pt.

Run:
    python -m backend.rl.trainer --track a
    python -m backend.rl.trainer --track b --steps 5000
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import yaml

from backend.mdp import dataset as D
from backend.mdp import action_space
from backend.pipeline import config
from backend.pipeline.logging_utils import get_logger
from backend.rl import behavior_clone, cqi
from backend.rl.hybrid_iql import Batch, HybridIQL

log = get_logger("rl_trainer")


def _to_batch(d: dict, idx: np.ndarray, device: str) -> Batch:
    t = lambda a, dt: torch.as_tensor(a[idx], dtype=dt, device=device)
    return Batch(
        states=t(d["states"], torch.float32),
        actions=t(d["actions"], torch.long),
        rewards=t(d["rewards"], torch.float32),
        next_states=t(d["next_states"], torch.float32),
        dones=t(d["dones"].astype(np.float32), torch.float32),
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

    total_steps = steps if steps is not None else cfg["total_steps"]
    ckpt_every = min(cfg["checkpoint_every"], max(500, total_steps // 10))
    batch_size = min(cfg["batch_size"], len(train_idx))
    model = HybridIQL(state_dim=d["states"].shape[1], action_dim=cfg["action_dim"],
                      hidden_dim=cfg["hidden_dim"], gamma=cfg["gamma"],
                      tau=cfg["tau"], beta=cfg["beta"], lr=cfg["lr"], device=device)

    best_val, best_sd, stale = float("inf"), model.state_dict(), 0
    rng = np.random.default_rng(config.SPLIT_SEED)
    last = {}
    for step in range(total_steps):
        bi = rng.choice(train_idx, size=batch_size, replace=len(train_idx) < batch_size)
        last = model.update(_to_batch(d, bi, device))
        if (step + 1) % ckpt_every == 0 or step == total_steps - 1:
            vl = _val_q_loss(model, d, val_idx, device)
            score = vl if vl == vl else last["q_loss"]
            log.info("step %d | v=%.4f q=%.4f pi=%.4f | val_q=%.4f",
                     step + 1, last["v_loss"], last["q_loss"], last["pi_loss"], score)
            if score < best_val:
                best_val, best_sd, stale = score, model.state_dict(), 0
            else:
                stale += ckpt_every
                if stale >= cfg["early_stop_patience"]:
                    log.info("early stop @ %d", step + 1)
                    break

    model.load_state_dict(best_sd)
    out = config.MODEL_PATH / f"policy_track_{track}.pt"
    torch.save({"state_dict": best_sd, "state_dim": d["states"].shape[1],
                "action_dim": cfg["action_dim"], "hidden_dim": cfg["hidden_dim"],
                "track": track}, out)
    log.info("Saved HybridIQL → %s (best val_q=%.4f)", out, best_val)

    # --- baselines (BC + CQI) on train split ---
    Xtr, Atr = d["states"][train_idx], d["actions"][train_idx]
    behavior_clone.train_bc(Xtr, Atr, track)
    qv = np.array([model.q_values(x) for x in Xtr])
    q_taken = qv[np.arange(len(Atr)), Atr]
    cqi.train_cqi(Xtr, Atr, q_taken, track)

    # action-distribution sanity on val/test
    metrics = _policy_metrics(model, d)
    (config.LOGS_PATH / f"train_track_{track}.json").write_text(json.dumps(
        {"best_val_q": best_val, "steps": total_steps, **metrics}, indent=2))
    log.info("Track %s metrics: %s", track.upper(), metrics)
    return metrics


def _policy_metrics(model: HybridIQL, d: dict) -> dict:
    eval_idx = np.where(np.isin(d["split"], ["test", "val"]))[0]
    if len(eval_idx) == 0:
        eval_idx = np.arange(len(d["actions"]))
    acts = np.array([model.act(d["states"][i]) for i in eval_idx])
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
    args = ap.parse_args()
    train(args.track, steps=args.steps)


if __name__ == "__main__":
    main()
