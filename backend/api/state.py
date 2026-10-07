"""Lazily-loaded singleton holding the inference services (models, trees)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from backend.explainability.tree_distillation import DistilledTree
from backend.mdp import normaliser
from backend.pipeline import config
from backend.router import safety_filter
from backend.router.policy_router import PolicyRouter, TABULAR, WAVEFORM


@dataclass
class Services:
    router: PolicyRouter
    safety = safety_filter
    tree_a: Optional[DistilledTree] = None
    tree_b: Optional[DistilledTree] = None


_services: Optional[Services] = None


def _distil(track: str, router: PolicyRouter):
    # Full MDP parquet when present (local dev), else the 56 KB pre-subsampled
    # one. A deployment ships only the sample: the full track-A parquet is 42 MB
    # and reading it costs a few hundred MB of RSS — enough to OOM a small
    # container — while this function immediately throws 99.8% of it away. The
    # sample was drawn with the same seed used below, so the deployed decision
    # tree is identical to the local one.
    path = config.PROCESSED_PATH / f"mdp_track_{track}.parquet"
    if not path.exists():
        path = config.PROCESSED_PATH / f"mdp_track_{track}_treesample.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    feats = TABULAR if track == "a" else TABULAR + WAVEFORM
    raw = df[[f"s_{f}" for f in feats]].to_numpy(float)
    if len(raw) > 2000:
        raw = raw[np.random.default_rng(0).choice(len(raw), 2000, replace=False)]
    stats = router.norm_a if track == "a" else router.norm_b
    z = normaliser.transform(raw, stats, feats)
    model = router.track_a if track == "a" else router.track_b
    if model is None:
        return None
    return DistilledTree.fit(model, z, feats, norm_stats=stats)


def get_services() -> Services:
    global _services
    if _services is None:
        router = PolicyRouter()
        svc = Services(router=router)
        svc.tree_a = _distil("a", router)
        if router.track_b is not None:
            svc.tree_b = _distil("b", router)
        _services = svc
    return _services
