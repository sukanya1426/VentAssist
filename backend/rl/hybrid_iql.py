"""HybridIQL — Implicit Q-Learning for discrete ventilator actions (Section 6.1).

Three coupled objectives (Kostrikov et al., 2022), all trained on offline data:
  * V via expectile regression of Q(s, a_taken)         (avoids OOD Q-queries)
  * Q via Bellman backup using a target V-network
  * π via advantage-weighted behaviour cloning

A FeatureAdapter optionally projects an 18-dim Track B state into a 12-dim latent
so the same network widths serve both tracks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class VNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = _mlp(state_dim, hidden_dim, 1)

    def forward(self, s):
        return self.net(s).squeeze(-1)


class QNetwork(nn.Module):
    """Outputs Q for all actions."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = _mlp(state_dim, hidden_dim, action_dim)

    def forward(self, s):
        return self.net(s)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = _mlp(state_dim, hidden_dim, action_dim)

    def forward(self, s):
        return self.net(s)            # logits


class FeatureAdapter(nn.Module):
    """Project an 18-dim state to a 12-dim latent (identity for 12-dim input)."""
    def __init__(self, in_dim: int, out_dim: int = 12):
        super().__init__()
        self.identity = in_dim == out_dim
        self.proj = None if self.identity else nn.Linear(in_dim, out_dim)

    def forward(self, s):
        return s if self.identity else self.proj(s)


def expectile_loss(delta: torch.Tensor, tau: float) -> torch.Tensor:
    weight = torch.where(delta < 0, torch.tensor(1 - tau, device=delta.device),
                         torch.tensor(tau, device=delta.device))
    return (weight * delta ** 2).mean()


@dataclass
class Batch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class HybridIQL:
    def __init__(self, state_dim: int, action_dim: int = 125, hidden_dim: int = 256,
                 gamma: float = 0.99, tau: float = 0.8, beta: float = 2.0,
                 lr: float = 3e-4, polyak: float = 0.005, latent_dim: int = 12,
                 device: str = "cpu"):
        self.device = torch.device(device)
        self.gamma, self.tau, self.beta, self.polyak = gamma, tau, beta, polyak
        self.action_dim = action_dim

        self.adapter = FeatureAdapter(state_dim, latent_dim).to(self.device)
        ld = latent_dim
        self.V = VNetwork(ld, hidden_dim).to(self.device)
        self.V_target = VNetwork(ld, hidden_dim).to(self.device)
        self.V_target.load_state_dict(self.V.state_dict())
        self.Q = QNetwork(ld, action_dim, hidden_dim).to(self.device)
        self.pi = PolicyNetwork(ld, action_dim, hidden_dim).to(self.device)

        params_v = list(self.V.parameters()) + list(self.adapter.parameters())
        self.opt_V = torch.optim.Adam(params_v, lr=lr)
        self.opt_Q = torch.optim.Adam(self.Q.parameters(), lr=lr)
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=lr)

    # --- losses ---
    def _phi(self, s):
        return self.adapter(s)

    def update(self, batch: Batch) -> dict:
        s = self._phi(batch.states)
        ns = self._phi(batch.next_states)
        a = batch.actions
        B = len(a)
        idx = torch.arange(B, device=self.device)

        # V loss (expectile regression toward Q(s, a_taken))
        with torch.no_grad():
            q_taken = self.Q(s)[idx, a]
        v = self.V(s)
        v_loss = expectile_loss(q_taken - v, self.tau)
        self.opt_V.zero_grad(); v_loss.backward(); self.opt_V.step()

        # Q loss (Bellman backup with target V)
        with torch.no_grad():
            v_next = self.V_target(self._phi(batch.next_states))
            y = batch.rewards + self.gamma * (1 - batch.dones) * v_next
        q_taken2 = self.Q(self._phi(batch.states))[idx, a]
        q_loss = F.mse_loss(q_taken2, y)
        self.opt_Q.zero_grad(); q_loss.backward(); self.opt_Q.step()

        # Policy loss (advantage-weighted BC)
        with torch.no_grad():
            sd = self._phi(batch.states)
            adv = self.Q(sd)[idx, a] - self.V(sd)
            w = torch.exp(adv / self.beta).clamp(max=100.0)
        logp = F.log_softmax(self.pi(self._phi(batch.states)), dim=-1)[idx, a]
        pi_loss = -(w * logp).mean()
        self.opt_pi.zero_grad(); pi_loss.backward(); self.opt_pi.step()

        # Polyak update of target V
        with torch.no_grad():
            for p, pt in zip(self.V.parameters(), self.V_target.parameters()):
                pt.mul_(1 - self.polyak).add_(self.polyak * p)

        return {"v_loss": float(v_loss), "q_loss": float(q_loss),
                "pi_loss": float(pi_loss)}

    # --- inference ---
    @torch.no_grad()
    def act(self, state: np.ndarray) -> int:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
        q = self.Q(self._phi(s))
        return int(q.argmax(dim=-1).item())

    @torch.no_grad()
    def q_values(self, state: np.ndarray) -> np.ndarray:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
        return self.Q(self._phi(s)).cpu().numpy().ravel()

    # --- persistence ---
    def state_dict(self) -> dict:
        return {"adapter": self.adapter.state_dict(), "V": self.V.state_dict(),
                "V_target": self.V_target.state_dict(), "Q": self.Q.state_dict(),
                "pi": self.pi.state_dict()}

    def load_state_dict(self, sd: dict):
        self.adapter.load_state_dict(sd["adapter"]); self.V.load_state_dict(sd["V"])
        self.V_target.load_state_dict(sd["V_target"]); self.Q.load_state_dict(sd["Q"])
        self.pi.load_state_dict(sd["pi"])
