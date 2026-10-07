"""Fitted Q-Evaluation (Section 7.1) — estimate a policy's value offline.

Iteratively fits Q^(k) to the Bellman target r + γ·Q^(k-1)(s', π(s')) on the test
split, then reports V̂ = E_{s0}[Q^(K)(s0, π(s0))] over episode-initial states.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F

from backend.mdp import dataset as D
from backend.mdp import normaliser as N
from backend.pipeline import config
from backend.pipeline.logging_utils import get_logger
from backend.rl.hybrid_iql import QNetwork

log = get_logger("ope_fqe")


def _load_policy(track: str):
    from backend.rl.hybrid_iql import HybridIQL
    ckpt = torch.load(config.MODEL_PATH / f"policy_track_{track}.pt", map_location="cpu")
    m = HybridIQL(state_dim=ckpt["state_dim"], action_dim=ckpt["action_dim"],
                  hidden_dim=ckpt["hidden_dim"])
    m.load_state_dict(ckpt["state_dict"])
    return m, ckpt


def fitted_q_evaluation(track: str = "a", K: int = 50, gamma: float = 0.99,
                        batch_size: int = 8192, steps_per_iter: int = 200,
                        device: str = "cpu", write: bool = True,
                        policy=None) -> dict:
    """Fitted-Q evaluation of a policy under the dataset's reward.

    ``policy`` lets a caller score a model held in memory instead of the
    deployed checkpoint. ``benchmark/runner.py`` needs this to put a confidence
    interval on the value estimate across training seeds: without it, scoring a
    seed would mean writing that seed's weights to
    ``backend/models/policy_track_a.pt`` — overwriting the deployed, gated
    policy — which the benchmark ground rules forbid. ``None`` keeps the
    historical behaviour of loading the deployed checkpoint.
    """
    d = D.load_mdp(track)
    feats = d["feature_order"]
    nf = "normaliser_stats.json" if track == "a" else "normaliser_stats_track_b.json"
    stats = N.load(config.MODEL_PATH / nf)
    S = N.transform(d["states"], stats, feats).astype(np.float32)
    NS = N.transform(d["next_states"], stats, feats).astype(np.float32)

    if policy is None:
        policy, ckpt = _load_policy(track)
    else:
        # The FQE critic below is sized from ckpt, so an in-memory policy has to
        # supply the same dims. Read them off the model rather than trusting a
        # caller-passed config, so the critic can never be built at a width the
        # policy does not actually have.
        ckpt = {"in_memory": True,
                "action_dim": int(policy.action_dim),
                "hidden_dim": int(policy.Q.net[0].out_features)}
    # policy's greedy next action (batched — was a per-row Python loop over ~1M)
    pi_next = policy.act_batch(NS)

    test = np.where(d["split"] == "test")[0]
    if len(test) == 0:
        test = np.arange(len(S))

    dev = torch.device(device)
    St = torch.as_tensor(S, device=dev)
    NSt = torch.as_tensor(NS, device=dev)
    A = torch.as_tensor(d["actions"], dtype=torch.long, device=dev)
    R = torch.as_tensor(d["rewards"], dtype=torch.float32, device=dev)
    Dn = torch.as_tensor(d["dones"].astype(np.float32), device=dev)
    PiN = torch.as_tensor(pi_next, dtype=torch.long, device=dev)

    q = QNetwork(S.shape[1], ckpt["action_dim"], ckpt["hidden_dim"]).to(dev)
    q_target = QNetwork(S.shape[1], ckpt["action_dim"], ckpt["hidden_dim"]).to(dev)
    q_target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    idx = torch.arange(len(A), device=dev)

    # Fitted-Q iteration with a target network. The inner optimisation is now
    # mini-batch SGD (was full-batch over all ~1M rows every epoch), which is the
    # standard FQE recipe and ~an order of magnitude less compute per outer step.
    Nrows = len(A)
    bs = min(batch_size, Nrows)
    gen = torch.Generator(device=dev).manual_seed(0)
    prev = None
    for k in range(K):
        with torch.no_grad():
            y = R + gamma * (1 - Dn) * q_target(NSt)[idx, PiN]     # fixed targets
        for _ in range(steps_per_iter):
            b = torch.randint(0, Nrows, (bs,), generator=gen, device=dev)
            qa = q(St[b])[torch.arange(bs, device=dev), A[b]]
            loss = F.mse_loss(qa, y[b])
            opt.zero_grad(); loss.backward(); opt.step()
        q_target.load_state_dict(q.state_dict())
        with torch.no_grad():
            cur = float(q(St)[idx, A].mean())
        if prev is not None and abs(cur - prev) < 1e-4:
            log.info("FQE converged at iter %d", k + 1)
            break
        prev = cur

    # V̂ over episode-initial states (hour==0 proxy: first transition per stay in test)
    with torch.no_grad():
        pi_test = torch.as_tensor(policy.act_batch(S[test]),
                                  dtype=torch.long, device=dev)
        v = q(St[torch.as_tensor(test, device=dev)])[torch.arange(len(test)), pi_test]
    v_hat = float(v.mean())
    v_std = float(v.std())
    result = {"track": track, "V_hat": round(v_hat, 4), "v_hat": round(v_hat, 4),
              "V_std": round(v_std, 4), "n_test": int(len(test)),
              "n_transitions": int(len(d["actions"])), "K": K, "gamma": gamma,
              "timestamp": datetime.now(timezone.utc).isoformat()}
    if write:
        (config.LOGS_PATH / f"fqe_track_{track}.json").write_text(
            json.dumps(result, indent=2))
    log.info("FQE Track %s: V̂=%.4f ± %.4f (n=%d)", track.upper(), v_hat, v_std, len(test))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="a", choices=["a", "b"])
    ap.add_argument("--K", type=int, default=50)
    args = ap.parse_args()
    fitted_q_evaluation(args.track, K=args.K)


if __name__ == "__main__":
    main()
