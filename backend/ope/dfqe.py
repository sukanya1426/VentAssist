"""Distributional FQE (Section 12.2) — the full return distribution Z^π(s,a).

Two methods share this module:

* ``method="qr"`` (DEFAULT) — a real **quantile-regression** DFQE. A Q-network
  emits ``n_quantiles`` outputs per action and is trained with the
  **quantile-Huber (pinball) loss** under a distributional Bellman backup with a
  target network (same mini-batch fitted-Q recipe as ``fqe.py``). This yields the
  return distribution, from which we report ``V_hat`` (mean over quantiles at the
  π-actions on episode starts), ``LCB_alpha`` (the α-quantile lower-confidence
  bound), ``return_variance``, and the policy coverage ``d^π`` (fraction of π's
  greedy actions that lie in the behaviour-supported action set).

* ``method="bootstrap"`` — the earlier stand-in: point FQE plus a bootstrap CI
  over per-state values. Kept for back-compat / cross-checking; not a real return
  distribution.

Quantile crossing (θ_τ non-monotone in τ) is removed by **rearrangement**
(sorting the predicted quantiles), the standard monotone-rearrangement fix
(Chernozhukov, Fernández-Val & Galichon, 2010) — so the reported quantile
function is weakly increasing by construction and the α-quantile LCB is a genuine
lower tail of the distribution.

Both methods write ``logs/dfqe_track_{track}.json`` with ``n_transitions`` + a
``timestamp`` so staleness is detectable.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.mdp import dataset as D
from backend.mdp import normaliser as N
from backend.ope.fqe import _load_policy
from backend.pipeline import config
from backend.pipeline.logging_utils import get_logger
from backend.rl.hybrid_iql import QNetwork

log = get_logger("ope_dfqe")


# --------------------------------------------------------------------------- #
# Quantile-regression network + loss
# --------------------------------------------------------------------------- #
class QRQNetwork(nn.Module):
    """Q-network with ``n_quantiles`` outputs per action → (B, action_dim, n_q)."""

    def __init__(self, state_dim: int, action_dim: int, n_quantiles: int,
                 hidden_dim: int = 256):
        super().__init__()
        self.action_dim = action_dim
        self.n_quantiles = n_quantiles
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * n_quantiles),
        )

    def forward(self, s):
        return self.net(s).view(-1, self.action_dim, self.n_quantiles)


def quantile_huber_loss(pred: torch.Tensor, target: torch.Tensor,
                        taus: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """Quantile-Huber (pinball) loss for QR-DQN-style distributional regression.

    ``pred`` are the predicted quantiles θ_{τ_i}(s,a) at levels ``taus`` (B, n_q);
    ``target`` are the distributional-Bellman target samples T_j (B, n_q). The
    pairwise TD error is u_ij = T_j − θ_i; the loss is
    |τ_i − 1{u_ij < 0}| · Huber_κ(u_ij), summed over i and averaged over j.
    """
    u = target.unsqueeze(1) - pred.unsqueeze(2)              # (B, n_q_i, n_q_j)
    abs_u = u.abs()
    huber = torch.where(abs_u <= kappa, 0.5 * u ** 2,
                        kappa * (abs_u - 0.5 * kappa))
    weight = (taus.view(1, -1, 1) - (u.detach() < 0).float()).abs()
    loss = (weight * huber / kappa)                          # (B, n_q_i, n_q_j)
    return loss.sum(dim=1).mean(dim=1).mean()


def _theta_at(net: QRQNetwork, S: torch.Tensor, actions: torch.Tensor,
              chunk: int = 65536) -> torch.Tensor:
    """θ(s, a, :) for given (state, action) pairs → (N, n_q), computed in chunks.

    Avoids materialising the full (N, action_dim, n_q) tensor (billions of floats
    over ~1M rows); only the selected action's quantiles are kept."""
    out = torch.empty(len(S), net.n_quantiles)
    for i in range(0, len(S), chunk):
        z = net(S[i:i + chunk])                              # (c, A, n_q)
        a = actions[i:i + chunk]
        out[i:i + chunk] = z[torch.arange(len(a)), a, :]
    return out


# --------------------------------------------------------------------------- #
# Shared data loading
# --------------------------------------------------------------------------- #
def _load(track: str):
    d = D.load_mdp(track)
    feats = d["feature_order"]
    nf = "normaliser_stats.json" if track == "a" else "normaliser_stats_track_b.json"
    stats = N.load(config.MODEL_PATH / nf)
    S = N.transform(d["states"], stats, feats).astype(np.float32)
    NS = N.transform(d["next_states"], stats, feats).astype(np.float32)
    policy, ckpt = _load_policy(track)
    return d, S, NS, policy, ckpt


def _episode_starts(d: dict) -> np.ndarray:
    """Indices of the first transition per stay within the test split (episode s0)."""
    test = np.where(d["split"] == "test")[0]
    if len(test) == 0:
        test = np.arange(len(d["actions"]))
    sid, hr = d["stay_id"][test], d["hour"][test]
    order = np.lexsort((hr, sid))
    test_sorted = test[order]
    sid_sorted = sid[order]
    first = np.ones(len(sid_sorted), dtype=bool)
    first[1:] = sid_sorted[1:] != sid_sorted[:-1]
    return test_sorted[first]


def _policy_coverage(d: dict, S: np.ndarray, policy, n: int = 2000,
                     seed: int = 0) -> float:
    """d^π = fraction of π's greedy actions (on a test sample) that are in the
    behaviour-supported action set (actions observed in the train split)."""
    train = np.where(d["split"] == "train")[0]
    test = np.where(d["split"] == "test")[0]
    if len(train) == 0 or len(test) == 0:
        return float("nan")
    valid = set(int(a) for a in np.unique(d["actions"][train]))
    rng = np.random.default_rng(seed)
    sample = rng.choice(test, size=min(n, len(test)), replace=False)
    acts = policy.act_batch(S[sample])
    return float(np.mean([int(a) in valid for a in acts]))


# --------------------------------------------------------------------------- #
# QR-DFQE (the real distributional estimator)
# --------------------------------------------------------------------------- #
def distributional_fqe_qr(track: str = "a", n_quantiles: int = 21, K: int = 25,
                          gamma: float = 0.99, alpha: float = 0.05,
                          batch_size: int = 4096, steps_per_iter: int = 120,
                          seed: int = 0, write: bool = True) -> dict:
    torch.manual_seed(seed)
    d, S, NS, policy, ckpt = _load(track)
    pi_next = policy.act_batch(NS)                            # batched greedy a'

    St, NSt = torch.tensor(S), torch.tensor(NS)
    A = torch.tensor(d["actions"]).long()
    R = torch.tensor(d["rewards"]).float()
    Dn = torch.tensor(d["dones"].astype(np.float32))
    PiN = torch.tensor(pi_next).long()
    taus = (torch.arange(n_quantiles).float() + 0.5) / n_quantiles

    q = QRQNetwork(S.shape[1], ckpt["action_dim"], n_quantiles, ckpt["hidden_dim"])
    qt = QRQNetwork(S.shape[1], ckpt["action_dim"], n_quantiles, ckpt["hidden_dim"])
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)

    Nrows = len(A)
    bs = min(batch_size, Nrows)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(K):
        # Distributional Bellman target: T = r + γ(1-done)·θ_target(s', π(s'), :)
        with torch.no_grad():
            theta_next = _theta_at(qt, NSt, PiN)             # (Nrows, n_q)
            target = R[:, None] + gamma * (1 - Dn)[:, None] * theta_next
        for _ in range(steps_per_iter):
            b = torch.randint(0, Nrows, (bs,), generator=gen)
            pred = q(St[b])[torch.arange(bs), A[b], :]       # (bs, n_q)
            loss = quantile_huber_loss(pred, target[b], taus)
            opt.zero_grad(); loss.backward(); opt.step()
        qt.load_state_dict(q.state_dict())

    # Return distribution at episode starts, for the policy's greedy action.
    starts = _episode_starts(d)
    with torch.no_grad():
        pi_s0 = torch.tensor(policy.act_batch(S[starts])).long()
        z0 = _theta_at(q, St[torch.tensor(starts)], pi_s0).numpy()  # (n_starts, n_q)
    z0 = np.sort(z0, axis=1)                                  # monotone rearrangement
    qbar = z0.mean(axis=0)                                    # mean quantile fn (n_q,)

    v_hat = float(z0.mean())
    lcb = float(np.interp(alpha, taus.numpy(), qbar))
    result = {
        "track": track, "method": "qr", "n_quantiles": n_quantiles,
        "V_hat": round(v_hat, 4), "v_hat": round(v_hat, 4),
        "LCB_alpha": round(lcb, 4), "alpha": alpha,
        "return_variance": round(float(z0.var()), 4),
        "policy_coverage": round(_policy_coverage(d, S, policy), 4),
        "quantile_levels": [round(float(t), 4) for t in taus.numpy()],
        "quantile_values": [round(float(v), 4) for v in qbar],
        # Back-compat: a 95% CI over the per-start mean returns (bootstrap).
        "CI_95": _mean_ci(z0.mean(axis=1)),
        "n_test": int(len(starts)), "n_transitions": int(len(d["actions"])),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if write:
        (config.LOGS_PATH / f"dfqe_track_{track}.json").write_text(
            json.dumps(result, indent=2))
    log.info("QR-DFQE Track %s: V̂=%.3f LCB(%.0f%%)=%.3f var=%.3f d^π=%.3f",
             track.upper(), v_hat, 100 * alpha, lcb,
             result["return_variance"], result["policy_coverage"])
    return result


def _mean_ci(x: np.ndarray, B: int = 100, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    boot = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)])
    return [round(float(np.percentile(boot, 2.5)), 4),
            round(float(np.percentile(boot, 97.5)), 4)]


# --------------------------------------------------------------------------- #
# Bootstrap DFQE (legacy stand-in — point FQE + bootstrap CI)
# --------------------------------------------------------------------------- #
def distributional_fqe_bootstrap(track: str = "a", K: int = 30, gamma: float = 0.99,
                                 B: int = 100, alpha: float = 0.05,
                                 batch_size: int = 8192, steps_per_iter: int = 200) -> dict:
    d = D.load_mdp(track)
    feats = d["feature_order"]
    nf = "normaliser_stats.json" if track == "a" else "normaliser_stats_track_b.json"
    stats = N.load(config.MODEL_PATH / nf)
    S = N.transform(d["states"], stats, feats).astype(np.float32)
    NS = N.transform(d["next_states"], stats, feats).astype(np.float32)
    policy, ckpt = _load_policy(track)
    pi_next = policy.act_batch(NS)               # batched (was per-row loop over ~1M)

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
    # Mini-batch fitted-Q iteration (was full-batch over ~1M rows every epoch).
    Nrows = len(A)
    bs = min(batch_size, Nrows)
    gen = torch.Generator().manual_seed(0)
    for _ in range(K):
        with torch.no_grad():
            y = R + gamma * (1 - Dn) * qt(NSt)[idx, PiN]
        for _ in range(steps_per_iter):
            b = torch.randint(0, Nrows, (bs,), generator=gen)
            loss = F.mse_loss(q(St[b])[torch.arange(bs), A[b]], y[b])
            opt.zero_grad(); loss.backward(); opt.step()
        qt.load_state_dict(q.state_dict())

    test = np.where(d["split"] == "test")[0]
    if len(test) == 0:
        test = np.arange(len(S))
    with torch.no_grad():
        pi_t = torch.tensor(policy.act_batch(S[test])).long()
        vt = q(St[torch.tensor(test)])[torch.arange(len(test)), pi_t].numpy()

    rng = np.random.default_rng(42)
    boot = np.array([vt[rng.integers(0, len(vt), len(vt))].mean() for _ in range(B)])
    result = {
        "track": track, "method": "bootstrap",
        "V_hat": round(float(vt.mean()), 4),
        "v_hat": round(float(vt.mean()), 4),
        "CI_95": [round(float(np.percentile(boot, 2.5)), 4),
                  round(float(np.percentile(boot, 97.5)), 4)],
        "LCB_alpha": round(float(np.percentile(vt, 100 * alpha)), 4),
        "return_variance": round(float(vt.var()), 4),
        "n_test": int(len(test)), "n_transitions": int(len(d["actions"])), "B": B,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if write:
        (config.LOGS_PATH / f"dfqe_track_{track}.json").write_text(
            json.dumps(result, indent=2))
    log.info("DFQE(bootstrap) Track %s: V̂=%.3f CI95=%s LCB=%.3f",
             track.upper(), result["V_hat"], result["CI_95"], result["LCB_alpha"])
    return result


def distributional_fqe(track: str = "a", method: str = "qr", **kw) -> dict:
    """Dispatch to the quantile-regression (default) or bootstrap DFQE."""
    if method == "qr":
        return distributional_fqe_qr(track, **kw)
    if method == "bootstrap":
        return distributional_fqe_bootstrap(track, **kw)
    raise ValueError(f"unknown DFQE method {method!r} (use 'qr' or 'bootstrap')")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="a", choices=["a", "b"])
    ap.add_argument("--method", default="qr", choices=["qr", "bootstrap"])
    args = ap.parse_args()
    distributional_fqe(args.track, method=args.method)


if __name__ == "__main__":
    main()
