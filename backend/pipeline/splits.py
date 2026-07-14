"""Deterministic patient-level train/val/test split (no leakage across splits).

The split is by ``stay_id`` and fully reproducible: stay_ids are shuffled with a
fixed seed, then partitioned by the configured fractions. The result is persisted
to ``data/processed/train_val_test_split.json`` so every downstream stage
(state normalisation, MDP assembly, OPE) uses the identical partition.
"""

from __future__ import annotations

import json
from typing import Iterable

import numpy as np

from backend.pipeline import config

_DEFAULT_SPLIT = "train_val_test_split.json"


def _path(name: str):
    return config.PROCESSED_PATH / name


def make_split(stay_ids: Iterable[int], name: str = _DEFAULT_SPLIT) -> dict[str, list[int]]:
    """Partition stay_ids into train/val/test deterministically and persist it."""
    ids = sorted({int(s) for s in stay_ids})
    rng = np.random.default_rng(config.SPLIT_SEED)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * config.SPLIT_FRACTIONS["train"])
    n_val = int(n * config.SPLIT_FRACTIONS["val"])
    split = {
        "train": sorted(ids[:n_train]),
        "val": sorted(ids[n_train:n_train + n_val]),
        "test": sorted(ids[n_train + n_val:]),
    }
    config.ensure_output_dirs()
    _path(name).write_text(json.dumps(split, indent=2))
    return split


def make_kfold(stay_ids: Iterable[int], k: int = 5,
               name: str = "kfold_split.json") -> list[list[int]]:
    """Partition stay_ids into k disjoint patient-level folds (deterministic).

    Returns a list of k lists of stay_ids (the held-out test stays for each
    fold). Persisted so the CV partition is reproducible across runs.
    """
    ids = sorted({int(s) for s in stay_ids})
    rng = np.random.default_rng(config.SPLIT_SEED)
    rng.shuffle(ids)
    folds = [sorted(int(s) for s in part) for part in np.array_split(ids, k)]
    config.ensure_output_dirs()
    _path(name).write_text(json.dumps({"k": k, "folds": folds}, indent=2))
    return folds


def load_split(name: str = _DEFAULT_SPLIT) -> dict[str, list[int]]:
    """Load the persisted split, creating nothing — raises if absent."""
    p = _path(name)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found — run state_builder (or make_split) first.")
    return json.loads(p.read_text())


def get_or_make_split(stay_ids: Iterable[int], name: str = _DEFAULT_SPLIT) -> dict[str, list[int]]:
    """Return the existing split if present, otherwise create it."""
    if _path(name).exists():
        return load_split(name)
    return make_split(stay_ids, name)
