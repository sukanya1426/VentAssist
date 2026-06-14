"""Distributional FQE (Section 7.2) — conservative bounds via bootstrap.

Builds on FQE's fitted Q: bootstraps the per-state policy values over the test
set to produce a 95% CI on V̂ and a conservative lower-confidence bound (5th
percentile). A pragmatic stand-in for full quantile-regression DFQE.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from backend.mdp import dataset as D
from backend.mdp import normaliser as N
from backend.ope.fqe import _load_policy
from backend.pipeline import config
from backend.pipeline.logging_utils import get_logger
from backend.rl.hybrid_iql import QNetwork
import torch.nn.functional as F

log = get_logger("ope_dfqe")


def distributional_fqe(track: str = "a", K: int = 30, gamma: float = 0.99,
                       B: int = 100, alpha: float = 0.05) -> dict:
    d = D.load_mdp(track)
    feats = d["feature_order"]
    nf = "normaliser_stats.json" if track == "a" else "normaliser_stats_track_b.json"
    stats = N.load(config.MODEL_PATH / nf)
    S = N.transform(d["states"], stats, feats).astype(np.float32)
    NS = N.transform(d["next_states"], stats, feats).astype(np.float32)
    policy, ckpt = _load_policy(track)
    pi_next = np.array([policy.act(s) for s in NS], dtype=np.int64)

    St, NSt = torch.tensor(S), torch.tensor(NS)
    A = torch.tensor(d["actions"]).long()
    R = torch.tensor(d["rewards"]).float()
    Dn = torch.tensor(d["dones"].astype(np.float32))
    PiN = torch.tensor(pi_next).long()
    idx = torch.arange(len(A))

    q = QNetwork(S.shape[1], ckpt["action_dim"], ckpt["hidden_dim"])
    qt = QNetwork(S.shape[1], ckpt["action_dim"], ckpt["hidden_dim"])
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    for _ in range(K):
        with torch.no_grad():
            y = R + gamma * (1 - Dn) * qt(NSt)[idx, PiN]
        for _ in range(40):
            loss = F.mse_loss(q(St)[idx, A], y)
            opt.zero_grad(); loss.backward(); opt.step()
        qt.load_state_dict(q.state_dict())

    test = np.where(d["split"] == "test")[0]
    if len(test) == 0:
        test = np.arange(len(S))
    with torch.no_grad():
        pi_t = torch.tensor([policy.act(S[i]) for i in test]).long()
        vt = q(St[torch.tensor(test)])[torch.arange(len(test)), pi_t].numpy()

    rng = np.random.default_rng(42)
    boot = np.array([vt[rng.integers(0, len(vt), len(vt))].mean() for _ in range(B)])
    result = {
        "track": track,
        "V_hat": round(float(vt.mean()), 4),
        "CI_95": [round(float(np.percentile(boot, 2.5)), 4),
                  round(float(np.percentile(boot, 97.5)), 4)],
        "LCB_alpha": round(float(np.percentile(vt, 100 * alpha)), 4),
        "return_variance": round(float(vt.var()), 4),
        "n_test": int(len(test)), "B": B,
    }
    (config.LOGS_PATH / f"dfqe_track_{track}.json").write_text(json.dumps(result, indent=2))
    log.info("DFQE Track %s: V̂=%.3f CI95=%s LCB=%.3f",
             track.upper(), result["V_hat"], result["CI_95"], result["LCB_alpha"])
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="a", choices=["a", "b"])
    args = ap.parse_args()
    distributional_fqe(args.track)


if __name__ == "__main__":
    main()
