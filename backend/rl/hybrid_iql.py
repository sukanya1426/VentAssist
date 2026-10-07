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


def expectile_loss(delta: torch.Tensor, tau: float,
                   sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    weight = torch.where(delta < 0, torch.tensor(1 - tau, device=delta.device),
                         torch.tensor(tau, device=delta.device))
    loss = weight * delta ** 2
    if sample_weight is not None:
        loss = loss * sample_weight
    return loss.mean()


@dataclass
class Batch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    weights: torch.Tensor | None = None      # per-transition IPW weights (§16 item 8)


class HybridIQL:
    def __init__(self, state_dim: int, action_dim: int = 125, hidden_dim: int = 256,
                 gamma: float = 0.99, tau: float = 0.8, beta: float = 2.0,
                 lr: float = 3e-4, polyak: float = 0.005, latent_dim: int = 12,
                 cql_alpha: float = 0.0, device: str = "cpu"):
        self.device = torch.device(device)
        self.gamma, self.tau, self.beta, self.polyak = gamma, tau, beta, polyak
        self.cql_alpha = cql_alpha
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

        w = batch.weights                       # IPW weights (mean≈1) or None

        # V loss (expectile regression toward Q(s, a_taken))
        with torch.no_grad():
            q_taken = self.Q(s)[idx, a]
        v = self.V(s)
        v_loss = expectile_loss(q_taken - v, self.tau, w)
        self.opt_V.zero_grad(); v_loss.backward(); self.opt_V.step()

        # Q loss (Bellman backup with target V) + optional CQL conservatism.
        # The CQL term pushes down Q(s, ·) for OOD actions relative to the taken
        # action — logsumexp_a Q(s,a) - Q(s, a_taken) — so the greedy argmax stays
        # within the data support instead of extrapolating to unseen actions.
        with torch.no_grad():
            v_next = self.V_target(self._phi(batch.next_states))
            y = batch.rewards + self.gamma * (1 - batch.dones) * v_next
        q_all = self.Q(self._phi(batch.states))
        q_taken2 = q_all[idx, a]
        # IPW-weighted Bellman regression (§16 item 8): down-weights the
        # frequently-taken (confounded-by-indication) transitions. The CQL
        # conservatism term stays unweighted — it is a support regulariser, not a
        # data-distribution term.
        if w is None:
            q_loss = F.mse_loss(q_taken2, y)
        else:
            q_loss = (w * (q_taken2 - y) ** 2).mean()
        if self.cql_alpha > 0:
            cql_pen = (torch.logsumexp(q_all, dim=-1) - q_taken2).mean()
            q_loss = q_loss + self.cql_alpha * cql_pen
        self.opt_Q.zero_grad(); q_loss.backward(); self.opt_Q.step()

        # Policy loss (advantage-weighted BC), optionally × IPW weights
        with torch.no_grad():
            sd = self._phi(batch.states)
            adv = self.Q(sd)[idx, a] - self.V(sd)
            aw = torch.exp(adv / self.beta).clamp(max=100.0)
            if w is not None:
                aw = aw * w
        logp = F.log_softmax(self.pi(self._phi(batch.states)), dim=-1)[idx, a]
        pi_loss = -(aw * logp).mean()
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

    @torch.no_grad()
    def act_batch(self, states: np.ndarray, chunk: int = 65536) -> np.ndarray:
        """Vectorised greedy action for a batch of states → int64 array (N,).

        Equivalent to ``[self.act(s) for s in states]`` but as chunked batched
        forward passes instead of one per row (the OPE/eval hot path — §16 item 9
        vectorisation). Chunking bounds peak memory (Q is N×action_dim)."""
        arr = np.asarray(states, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        out = np.empty(len(arr), dtype=np.int64)
        for i in range(0, len(arr), chunk):
            s = torch.as_tensor(arr[i:i + chunk], device=self.device)
            out[i:i + chunk] = self.Q(self._phi(s)).argmax(dim=-1).cpu().numpy()
        return out

    # --- transfer ---
    def warm_start_from(self, ckpt: dict) -> dict:
        """Initialise this model from a LOWER-dimensional policy's checkpoint.

        Why this exists: V/Q/π all operate on the ``latent_dim`` output of
        ``FeatureAdapter``, so a 12-dim policy and an 18-dim policy have
        *identical* V/Q/π shapes — only the adapter differs. That makes the 18-dim
        waveform policy a strict extension of the 12-dim clinical one rather than
        a separate model that happens to share an architecture.

        Training the 18-dim policy from scratch cannot work at the waveform
        cohort's size: ~600 transitions against 125 actions and an 18-dim state
        diverges immediately (validation Q-loss minimised at the first checkpoint,
        then rising). Here V/Q/π are copied from the source policy — which was fit
        on the full ~694k-transition cohort — and the adapter is initialised to

            W = [I_latent | 0],   b = 0

        so ``forward(s) == s[:latent_dim]`` exactly. The 18-dim model therefore
        *starts numerically identical to the source policy* and fine-tuning only
        has to learn how much the extra dimensions should perturb the latent. The
        waveform columns begin with zero influence and have to earn it.

        It also makes the 12-vs-18 ablation a NESTED comparison: both arms share a
        Q-function fitted on the same data, so ΔV̂ reflects the extra state
        dimensions rather than a difference in training-set size.

        Returns provenance to record in the trained checkpoint.
        """
        sd = ckpt["state_dict"]
        src_dim = int(ckpt["state_dim"])
        if self.adapter.identity:
            raise ValueError(
                "warm_start_from needs a projecting adapter (state_dim > latent_dim); "
                f"this model is {src_dim}-dim identity, so there is nothing to extend.")
        proj = self.adapter.proj
        latent, in_dim = proj.weight.shape
        if src_dim != latent:
            raise ValueError(
                f"source policy is {src_dim}-dim but this model's latent is {latent}-dim; "
                "warm-starting requires the source to match the latent width exactly.")
        if in_dim < src_dim:
            raise ValueError(f"target state_dim {in_dim} < source state_dim {src_dim}")

        # V/Q/π are latent-space modules → shapes match, load them verbatim.
        self.V.load_state_dict(sd["V"])
        self.V_target.load_state_dict(sd["V_target"])
        self.Q.load_state_dict(sd["Q"])
        self.pi.load_state_dict(sd["pi"])

        # Adapter = [I | 0] so the model reproduces the source policy exactly on
        # the shared leading dimensions, with the new dimensions contributing zero.
        with torch.no_grad():
            proj.weight.zero_()
            proj.weight[:, :src_dim] = torch.eye(src_dim, device=proj.weight.device)
            if proj.bias is not None:
                proj.bias.zero_()

        return {"init_from_state_dim": src_dim,
                "init_from_n_transitions": ckpt.get("n_transitions"),
                "init_from_trained_at": ckpt.get("trained_at")}

    # --- persistence ---
    def state_dict(self) -> dict:
        return {"adapter": self.adapter.state_dict(), "V": self.V.state_dict(),
                "V_target": self.V_target.state_dict(), "Q": self.Q.state_dict(),
                "pi": self.pi.state_dict()}

    def load_state_dict(self, sd: dict):
        self.adapter.load_state_dict(sd["adapter"]); self.V.load_state_dict(sd["V"])
        self.V_target.load_state_dict(sd["V_target"]); self.Q.load_state_dict(sd["Q"])
        self.pi.load_state_dict(sd["pi"])
