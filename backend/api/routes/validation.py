"""GET /api/validation — the offline-evaluation record behind the deployed policy.

Nothing is computed here. The OPE modules (``backend/ope/*``) already write their
results to ``backend/logs/*.json``; this endpoint reads those files plus the
deployed checkpoint's own metadata so the UI can show *how the policy was
validated* alongside the recommendation it produces.

Staleness is the point of the endpoint as much as the numbers are: an evaluation
whose ``timestamp`` predates the checkpoint's ``trained_at`` describes a model
that is no longer being served. Each estimator therefore carries ``stale``, and a
stale entry must be rendered as a warning rather than as evidence.
"""

from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache

import torch
from fastapi import APIRouter

from backend.api import models as M
from backend.pipeline import config

router = APIRouter()

# Reward-model-relative: every V̂ below is the discounted return under *our*
# reward function, not a patient-outcome measure. The supported claim is
# "no worse than the logged clinician", and the UI must carry that wording.
CAVEAT = (
    "All values are discounted returns under VentAssist's own reward function "
    "(γ = 0.99), estimated offline on a held-out split — not measured patient "
    "outcomes. They support the claim “no worse than the logged clinician”, not "
    "“better for patients”. NWE is the conservative estimator; where it sits near "
    "the clinician baseline, read the evidence as parity rather than improvement."
)


def _read(name: str) -> dict | None:
    path = config.LOGS_PATH / name
    if not path.exists():
        return None
    try:
        with path.open() as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


@lru_cache(maxsize=2)
def _checkpoint_meta(track: str) -> dict:
    """The deployed checkpoint's self-description (§15.6.2 self-describing ckpts)."""
    path = config.MODEL_PATH / f"policy_track_{track}.pt"
    if not path.exists():
        return {}
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt.get("config_snapshot", {}) or {}
    return {
        "trained_at": ckpt.get("trained_at"),
        "n_transitions": ckpt.get("n_transitions"),
        "lam_causal": ckpt.get("lam_causal"),
        "cql_alpha": ckpt.get("cql_alpha"),
        "w_outcome": (cfg.get("reward") or {}).get("w_outcome"),
        "gamma": cfg.get("gamma"),
    }


def _estimator(key: str, label: str, log: dict | None, blurb: str,
               trained_at: datetime | None, *, n_key: str,
               n_unit: str) -> M.OPEEstimator | None:
    """One estimator row, or None when its log is missing."""
    if log is None:
        return None
    ci = log.get("CI_95") or [None, None]
    ts = log.get("timestamp")
    parsed = _parse_ts(ts)
    # An evaluation older than the checkpoint it claims to describe is evidence
    # about a retired model. Unknown timestamps are treated as stale, not as fine.
    stale = True
    if parsed is not None and trained_at is not None:
        stale = parsed < trained_at
    elif parsed is not None and trained_at is None:
        stale = False
    return M.OPEEstimator(
        key=key, label=label, blurb=blurb,
        v_hat=log.get("V_hat"),
        ci_low=ci[0], ci_high=ci[1],
        lcb=log.get("LCB_alpha"),
        n=log.get(n_key), n_unit=n_unit,
        timestamp=ts, stale=stale,
        n_transitions=log.get("n_transitions"),
    )


def _safety_rows(nwe: dict | None) -> list[M.SafetyComparison]:
    """Policy-vs-clinician safety metrics from the NWE rollout.

    These are far more legible to a clinician than any V̂ — same episodes, same
    starting states, directly comparable. ``n`` is carried through because the
    hypoxaemic-start subgroup is small enough that the UI must show it.
    """
    metrics = (nwe or {}).get("safety_metrics") or {}
    pol, clin = metrics.get("hybrid_iql"), metrics.get("clinician")
    if not pol or not clin:
        return []
    n_hypox = pol.get("n_episodes_start_lt95")
    specs = [
        ("pct_terminal_spo2_ge95", "Episodes ending at SpO₂ ≥ 95%",
         "percent", True, (nwe or {}).get("n_starts")),
        ("mean_delta_spo2_start_lt95", "Mean SpO₂ gain (hypoxaemic starts)",
         "points", True, n_hypox),
        ("pct_aggressive_steps", "Aggressive steps",
         "percent", False, (nwe or {}).get("n_starts")),
    ]
    rows: list[M.SafetyComparison] = []
    for key, label, unit, higher_better, n in specs:
        if pol.get(key) is None or clin.get(key) is None:
            continue
        rows.append(M.SafetyComparison(
            key=key, label=label, unit=unit, higher_is_better=higher_better,
            policy=pol[key], clinician=clin[key], n=n))
    return rows


@router.get("/validation", response_model=M.ValidationResponse)
def validation(track: str = "a") -> M.ValidationResponse:
    track = "b" if track in ("b", "track_b") else "a"
    meta = _checkpoint_meta(track)
    trained_at = _parse_ts(meta.get("trained_at"))

    beh = _read(f"behaviour_track_{track}.json")
    baseline = None
    if beh is not None:
        baseline = M.OPEBaseline(
            label="Clinician (logged behaviour)",
            v_hat=beh.get("V_hat"),
            ci_low=beh.get("ci95_low"), ci_high=beh.get("ci95_high"),
            n=beh.get("n_test_episodes"), timestamp=beh.get("timestamp"))

    candidates = [
        _estimator("fqe", "FQE", _read(f"fqe_track_{track}.json"),
                   "Fitted-Q evaluation on the held-out split.",
                   trained_at, n_key="n_test", n_unit="transitions"),
        _estimator("dfqe", "DFQE", _read(f"dfqe_track_{track}.json"),
                   "Distributional fitted-Q — also yields a 5% lower bound.",
                   trained_at, n_key="n_test", n_unit="episodes"),
        _estimator("nwe", "NWE", _read(f"nwe_track_{track}.json"),
                   "Model-based rollout; the conservative estimator.",
                   trained_at, n_key="n_starts", n_unit="episodes"),
    ]
    estimators = [e for e in candidates if e is not None]

    cross = _read("ventilation_crosscheck.json") or {}
    return M.ValidationResponse(
        track=track,
        model=M.DeployedModel(**meta) if meta else None,
        baseline=baseline,
        estimators=estimators,
        safety=_safety_rows(_read(f"nwe_track_{track}.json")),
        cohort_stays=cross.get("final_cohort_stays"),
        any_stale=any(e.stale for e in estimators),
        caveat=CAVEAT,
    )
