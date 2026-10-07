"""Tests for warm-starting an 18-dim policy from the 12-dim one, and for the
best-checkpoint restoration that early stopping depends on.

Both pin bugs that are invisible from the outside:

  * ``HybridIQL.state_dict()`` returns the live parameter tensors by reference, so
    holding it as "the best checkpoint" is a no-op and early stopping ships the
    FINAL weights while reporting the BEST score. See ``trainer._snapshot``.
  * ``warm_start_from`` must reproduce the source policy exactly, including under
    the source's normalisation — otherwise the inherited Q is evaluated on a
    different z-scale and the 12-vs-18 ablation measures normalisation drift
    rather than the waveform features.

Run:  pytest backend/tests/test_warm_start.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from backend.pipeline import config
from backend.rl import trainer as T
from backend.rl.hybrid_iql import Batch, HybridIQL

TABULAR = config.TABULAR_FEATURES
LATENT = 12
N_WAVE = 6


class Skip(Exception):
    pass


def _batch(dim: int, n: int = 64, seed: int = 0) -> Batch:
    g = torch.Generator().manual_seed(seed)
    return Batch(
        states=torch.randn(n, dim, generator=g),
        actions=torch.randint(0, 125, (n,), generator=g),
        rewards=torch.randn(n, generator=g),
        next_states=torch.randn(n, dim, generator=g),
        dones=torch.zeros(n),
    )


def _source_ckpt() -> dict:
    path = config.MODEL_PATH / "policy_track_a.pt"
    if not path.exists():
        pytest.skip("policy_track_a.pt not present — train Track A first.")
    return torch.load(path, map_location="cpu", weights_only=False)


# --------------------------------------------------------------------------- #
# _snapshot — early stopping actually keeps what it says it keeps
# --------------------------------------------------------------------------- #
def test_state_dict_alone_does_not_snapshot():
    """Documents the trap: the naive idiom aliases the live weights.

    If this ever starts passing, PyTorch changed its semantics and ``_snapshot``
    could be simplified — but until then the deep copy is load-bearing.
    """
    m = HybridIQL(state_dim=LATENT, action_dim=125, hidden_dim=256)
    aliased = m.state_dict()
    before = aliased["Q"]["net.0.weight"].clone()
    b = _batch(LATENT)
    for _ in range(25):
        m.update(b)
    assert not torch.allclose(before, aliased["Q"]["net.0.weight"]), (
        "state_dict() no longer aliases live tensors — _snapshot's deep copy may "
        "be redundant, but verify before removing it")


def test_snapshot_survives_further_training():
    m = HybridIQL(state_dim=LATENT, action_dim=125, hidden_dim=256)
    snap = T._snapshot(m)
    before = snap["Q"]["net.0.weight"].clone()
    b = _batch(LATENT)
    for _ in range(25):
        m.update(b)
    assert torch.allclose(before, snap["Q"]["net.0.weight"]), \
        "_snapshot must not change when the model trains on"


def test_restoring_a_snapshot_rewinds_the_model():
    """The property early stopping relies on: restore → the old predictions."""
    m = HybridIQL(state_dim=LATENT, action_dim=125, hidden_dim=256)
    s = np.random.default_rng(0).normal(size=LATENT).astype(np.float32)
    q_before = m.q_values(s).copy()
    snap = T._snapshot(m)
    b = _batch(LATENT)
    for _ in range(50):
        m.update(b)
    assert not np.allclose(q_before, m.q_values(s)), "training changed nothing — bad test"
    m.load_state_dict(snap)
    assert np.allclose(q_before, m.q_values(s), atol=1e-6), \
        "restoring a snapshot did not rewind the model"


# --------------------------------------------------------------------------- #
# warm_start_from — the 18-dim model starts AT the 12-dim one
# --------------------------------------------------------------------------- #
def test_warm_start_reproduces_the_source_exactly():
    ck = _source_ckpt()
    src = HybridIQL(state_dim=ck["state_dim"], action_dim=ck["action_dim"],
                    hidden_dim=ck["hidden_dim"])
    src.load_state_dict(ck["state_dict"])
    tgt = HybridIQL(state_dim=ck["state_dim"] + N_WAVE,
                    action_dim=ck["action_dim"], hidden_dim=ck["hidden_dim"])
    tgt.warm_start_from(ck)

    rng = np.random.default_rng(0)
    S = rng.normal(size=(200, ck["state_dim"])).astype(np.float32)
    # The waveform dims must have ZERO influence at initialisation, so any filling
    # of them — including wild values — leaves the recommendation untouched.
    for fill in (0.0, 7.5, -4.0, 1e3):
        S18 = np.concatenate([S, np.full((len(S), N_WAVE), fill, np.float32)], axis=1)
        q_src = np.array([src.q_values(s) for s in S])
        q_tgt = np.array([tgt.q_values(s) for s in S18])
        assert np.allclose(q_src, q_tgt, atol=1e-6), \
            f"warm-started Q differs from the source with waveform dims = {fill}"
        assert (q_src.argmax(1) == q_tgt.argmax(1)).all(), \
            f"warm-started action differs from the source with waveform dims = {fill}"


def test_warm_start_adapter_is_identity_then_zero():
    ck = _source_ckpt()
    tgt = HybridIQL(state_dim=ck["state_dim"] + N_WAVE,
                    action_dim=ck["action_dim"], hidden_dim=ck["hidden_dim"])
    tgt.warm_start_from(ck)
    W = tgt.adapter.proj.weight.detach()
    d = int(ck["state_dim"])
    assert torch.allclose(W[:, :d], torch.eye(d)), "shared block must be the identity"
    assert torch.allclose(W[:, d:], torch.zeros(d, N_WAVE)), "new block must be zero"
    assert torch.allclose(tgt.adapter.proj.bias.detach(), torch.zeros(d))


def test_warm_start_reports_source_provenance():
    ck = _source_ckpt()
    tgt = HybridIQL(state_dim=ck["state_dim"] + N_WAVE,
                    action_dim=ck["action_dim"], hidden_dim=ck["hidden_dim"])
    prov = tgt.warm_start_from(ck)
    assert prov["init_from_state_dim"] == ck["state_dim"]
    # The inherited training size is what tells the ablation this is a nested
    # comparison rather than one confounded by training-set size.
    assert prov["init_from_n_transitions"] == ck.get("n_transitions")


def test_warm_start_rejects_a_same_width_target():
    """An identity adapter has nothing to extend — fail loudly, not silently."""
    ck = _source_ckpt()
    same = HybridIQL(state_dim=ck["state_dim"], action_dim=ck["action_dim"],
                     hidden_dim=ck["hidden_dim"])
    with pytest.raises(ValueError):
        same.warm_start_from(ck)


def test_warm_start_rejects_a_latent_mismatch():
    ck = dict(_source_ckpt())
    ck["state_dim"] = 11                      # no longer matches the 12-dim latent
    tgt = HybridIQL(state_dim=18, action_dim=125, hidden_dim=256)
    with pytest.raises(ValueError):
        tgt.warm_start_from(ck)


# --------------------------------------------------------------------------- #
# Normalisation inheritance — the warm start must be exact on RAW inputs too
# --------------------------------------------------------------------------- #
def test_track_b_normaliser_inherits_the_tabular_stats():
    """Shared features must keep the source's mean/std.

    Without this the [I | 0] identity holds in latent space but breaks on raw
    inputs: the waveform cohort's own statistics drift up to ~0.5 sigma in the
    mean and ~25% in the std, so the inherited Q would see a z-scale it was never
    fitted on, and ΔV̂ would measure normalisation drift instead of waveforms.
    """
    from backend.mdp import normaliser as N
    a_path = config.MODEL_PATH / "normaliser_stats.json"
    b_path = config.MODEL_PATH / "normaliser_stats_track_b.json"
    if not a_path.exists() or not b_path.exists():
        pytest.skip("normaliser stats not built — run the pipelines first.")
    from backend.mdp import dataset as D
    if not D._track_cfg("b").get("init_from"):
        pytest.skip("Track B is not configured to warm-start.")
    a, b = N.load(a_path), N.load(b_path)
    for f in TABULAR:
        assert f in b, f"{f} missing from the Track B normaliser"
        assert a[f]["mean"] == pytest.approx(b[f]["mean"]), \
            f"{f} mean not inherited from Track A ({a[f]['mean']} vs {b[f]['mean']})"
        assert a[f]["std"] == pytest.approx(b[f]["std"]), \
            f"{f} std not inherited from Track A"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except (Skip, pytest.skip.Exception) as e:
                print(f"SKIP {name}: {e}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    print("ALL TESTS PASSED" if failures == 0 else f"{failures} TEST(S) FAILED")
    raise SystemExit(1 if failures else 0)
