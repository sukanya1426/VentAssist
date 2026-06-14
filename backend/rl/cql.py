"""Discrete Conservative Q-Learning baseline (Section 6.2).

Dependency-free torch implementation (the spec referenced d3rlpy, but a minimal
in-house DiscreteCQL avoids that heavy dependency and is sufficient as a
baseline). Minimises the standard TD error plus the CQL conservative penalty
    alpha * ( logsumexp_a Q(s,a) - Q(s, a_taken) ).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from backend.pipeline import config
from backend.pipeline.logging_utils import get_logger
from backend.rl.hybrid_iql import QNetwork

log = get_logger("rl_cql")


def train_cql(states, actions, rewards, next_states, dones, action_dim=125,
              alpha=0.25, gamma=0.99, hidden_dim=256, steps=20000, lr=3e-4,
              batch_size=256, track="a", device="cpu"):
    dev = torch.device(device)
    sd = states.shape[1]
    q = QNetwork(sd, action_dim, hidden_dim).to(dev)
    q_t = QNetwork(sd, action_dim, hidden_dim).to(dev)
    q_t.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)

    S = torch.as_tensor(states, dtype=torch.float32, device=dev)
    NS = torch.as_tensor(next_states, dtype=torch.float32, device=dev)
    A = torch.as_tensor(actions, dtype=torch.long, device=dev)
    R = torch.as_tensor(rewards, dtype=torch.float32, device=dev)
    Dn = torch.as_tensor(dones.astype(np.float32), dtype=torch.float32, device=dev)
    n = len(A)
    bs = min(batch_size, n)
    rng = np.random.default_rng(42)

    for step in range(steps):
        bi = rng.choice(n, size=bs, replace=n < bs)
        idx = torch.as_tensor(bi, device=dev)
        qa = q(S[idx])
        q_taken = qa[torch.arange(len(idx)), A[idx]]
        with torch.no_grad():
            y = R[idx] + gamma * (1 - Dn[idx]) * q_t(NS[idx]).max(dim=-1).values
        td = F.mse_loss(q_taken, y)
        cql_pen = (torch.logsumexp(qa, dim=-1) - q_taken).mean()
        loss = td + alpha * cql_pen
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % max(1, steps // 200) == 0:
            with torch.no_grad():
                for p, pt in zip(q.parameters(), q_t.parameters()):
                    pt.mul_(0.995).add_(0.005 * p)

    out = config.MODEL_PATH / f"cql_track_{track}.pt"
    torch.save({"state_dict": q.state_dict(), "state_dim": sd,
                "action_dim": action_dim, "hidden_dim": hidden_dim}, out)
    log.info("Saved DiscreteCQL → %s", out)
    return q
