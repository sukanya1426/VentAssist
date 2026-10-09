"""Model selection on clinical reflexes rather than TD-error (the §17 open problem).

THE PROBLEM THIS SOLVES. Three criteria pick three different checkpoints of the
same training run, and the one the trainer actually early-stops on is the worst of
the three clinically:

  * **validation Q-loss** (what ``trainer._run_training`` selects on) picks step
    ~5,750 and scores 6.60/8 on the clinical battery;
  * **FQE value** peaks near step 10,000 and is non-monotonic in step;
  * **the 8-case clinical battery** keeps improving out to the deployed artifact's
    step ~27,000, which scores 8.00/8.

``benchmark/results/checkpoint_battery_track_a.json`` localises the whole
disagreement to two cases. Six of the eight pass in every family; the gap is
entirely ``high_peep`` (pass rate 0.0 → 0.4 → 1.0 across steps 5,750 → 10,000 →
27,000) and ``hypoxaemic`` (0.6 → 0.8 → 1.0). Both are *reflexes acquired late* —
long after the Q-function's held-out TD-error has stopped improving. Validation
Q-loss cannot see them because a policy that never raises PEEP for a hypoxaemic
patient is not thereby worse at predicting returns; it is worse at being a
ventilator.

WHY NOT JUST SELECT ON THE BATTERY. Because the battery *is* the deploy gate
(``backend/scripts/verify_before_deploy.py``). Selecting on it would collapse the
distinction between the thing being chosen and the thing that certifies the
choice: every candidate would pass the gate by construction, and the gate would
stop carrying information. It is also eight hand-written states — selecting on
eight points invites fitting them rather than the physiology behind them.

WHAT THIS MODULE DOES INSTEAD. It measures the same clinical reflexes on **real
held-out cohort states**. Each battery case names a physiological situation; that
situation occurs thousands of times in the validation split, and whether the
policy responds to it in the clinically-indicated *direction* is a dense,
held-out, non-TD-error quantity. The eight strata below are the real-data
analogues of the eight probes, and the sets are **disjoint by construction**: the
gate scores eight synthetic states, this scores rows of the validation split, and
nothing is shared but the physiological reasoning. ``test_selection.py`` pins that
disjointness so a future edit cannot quietly turn the criterion into the gate.

THE SCORE IS MACRO-AVERAGED, AND THAT IS THE POINT. Clinicians hold 77.65% of the
time, so a micro-average over transitions is dominated by the one stratum where
holding is correct — and would hand the trap a prize (SUMMARY §15.x: *"clinician
agreement rises as the policy gets WORSE"*). Macro-averaging over strata gives the
rare hypoxaemic hour the same weight as the common at-target hour. Every non-target
stratum demands a STRICT direction (``dp < 0``, not ``dp <= 0``), so the
hold-everything policy passes exactly one stratum of eight and scores 0.125, which
is the floor and not an artifact.

TWO GUARDS, NOT A WEIGHTED BLEND. A candidate is disqualified — not merely
penalised — if it collapses its repertoire below ``min_distinct_actions`` or if its
recommended settings violate lung-protective rules more often than the clinician's
do on the same states. Disqualification keeps the criterion readable: the score
answers "does it have the reflexes", and the guards answer "is it a policy at all".
Blending the three into one number would hide which of them a rejection came from.

WHAT THIS IS NOT. It is not a value estimate and must never be reported as one;
it says nothing about whether the policy helps patients. It is a *selection*
criterion — a way of choosing between checkpoints of one training run that is
sensitive to behaviour the TD-error is blind to. The claim boundary in SUMMARY
§15.15 is untouched by it.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

# Minimum rows a stratum needs before it is allowed into the macro-average. Below
# this the per-stratum rate is too noisy to select on, and the stratum is reported
# with its n but excluded from the score (rather than silently averaged in).
MIN_STRATUM_N = 50

# A policy using fewer than this many of the 125 actions on the whole validation
# split has collapsed. 10 is deliberately lenient: the observed range is 11.8
# (IPW, degenerate) to 34.8 (healthy), so this rejects only genuine collapse and
# does not quietly prefer diversity for its own sake.
MIN_DISTINCT_ACTIONS = 10


# --------------------------------------------------------------------------- #
# The eight strata: real-cohort analogues of the eight battery probes.
#
# `where` selects the rows of the validation split in that physiological
# situation; `ok` is the clinically-indicated direction for the recommended
# (ΔPEEP, ΔTV, ΔFiO₂). Thresholds are the ordinary clinical ones (ARDSnet tidal
# volume, 92–96% SpO₂ target, 35–45 mmHg PaCO₂, pH 7.35–7.45) — not tuned to make
# any checkpoint look good.
#
# The non-target strata are conditioned so the indicated direction is unambiguous:
# `high_peep` requires adequate oxygenation (lowering PEEP on a hypoxaemic patient
# is wrong, so those rows are excluded rather than scored), `hypercapnia` requires
# tidal volume with room to rise, and `low_peep` requires a FiO₂ high enough that
# recruitment is the indicated answer.
# --------------------------------------------------------------------------- #
Strata = dict[str, tuple[Callable[[dict], np.ndarray], Callable[..., np.ndarray], str]]

STRATA: Strata = {
    "hypoxaemia": (
        lambda v: (v["SpO2"] < 90) | (v["PaO2"] < 60),
        lambda dp, dt, df: (dp > 0) | (df > 0),
        "SpO2<90 or PaO2<60 → raise PEEP or FiO2",
    ),
    "hyperoxia": (
        lambda v: (v["SpO2"] >= 98) & (v["FiO2"] > 0.5),
        lambda dp, dt, df: df < 0,
        "SpO2>=98 on FiO2>0.5 → lower FiO2",
    ),
    "hypercapnia": (
        lambda v: (v["PaCO2"] > 55) & (v["pH"] < 7.32) & (v["tv_per_kg"] < 8.0),
        lambda dp, dt, df: dt > 0,
        "PaCO2>55 with pH<7.32 and room under 8 mL/kg → raise TV",
    ),
    "hypocapnia": (
        lambda v: (v["PaCO2"] < 32) & (v["pH"] > 7.45),
        lambda dp, dt, df: dt < 0,
        "PaCO2<32 with pH>7.45 → lower TV",
    ),
    "high_peep": (
        lambda v: (v["PEEP"] >= 14) & (v["SpO2"] >= 92),
        lambda dp, dt, df: dp < 0,
        "PEEP>=14 while oxygenating adequately → lower PEEP",
    ),
    "low_peep": (
        lambda v: (v["PEEP"] <= 5) & (v["FiO2"] >= 0.4),
        lambda dp, dt, df: dp > 0,
        "PEEP<=5 on FiO2>=0.4 → raise PEEP rather than buy oxygenation with FiO2",
    ),
    "volutrauma": (
        lambda v: v["tv_per_kg"] > 8.0,
        lambda dp, dt, df: dt < 0,
        "TV>8 mL/kg PBW → cut TV",
    ),
    "at_target": (
        lambda v: ((v["SpO2"] >= 92) & (v["SpO2"] <= 96)
                   & (v["PaCO2"] >= 35) & (v["PaCO2"] <= 45)
                   & (v["pH"] >= 7.35) & (v["pH"] <= 7.45)
                   & (v["tv_per_kg"] >= 6.0) & (v["tv_per_kg"] <= 8.0)
                   & (v["PEEP"] >= 5) & (v["PEEP"] <= 12) & (v["FiO2"] <= 0.5)),
        lambda dp, dt, df: (dp == 0) & (dt == 0) & (df == 0),
        "every target met → hold",
    ),
}


# Plausible ranges for the median of each column, in clinical units. These are not
# thresholds for anything clinical — they exist only to catch an encoding mistake.
_PLAUSIBLE_MEDIAN = {
    "PEEP": (0.0, 30.0), "TV": (100.0, 1200.0), "FiO2": (0.15, 1.0),
    "SpO2": (50.0, 100.0), "PaO2": (20.0, 500.0), "PaCO2": (10.0, 120.0),
    "pH": (6.6, 7.9),
}


def _assert_raw_units(v: dict[str, np.ndarray]) -> None:
    """Refuse z-scored input loudly instead of scoring it.

    Every threshold in ``STRATA`` is in clinical units, so normalised states must
    never reach them. The failure mode if they do is not an empty score — it is a
    PLAUSIBLE-LOOKING one. A z-scored column has median ~0, which satisfies
    ``PEEP <= 5`` and ``PaCO2 < 32`` for most rows, so the ``low_peep`` stratum
    fills up with nonsense and the macro-average returns a number nobody can tell
    is wrong. That is the same class of error as the serving/training encoding
    mix-up recorded in SUMMARY §15.x, and it is worth a hard failure: a selection
    criterion that silently scores the wrong encoding would pick a checkpoint for
    no reason at all.
    """
    bad = []
    for name, (lo, hi) in _PLAUSIBLE_MEDIAN.items():
        med = float(np.median(v[name]))
        if not (lo <= med <= hi):
            bad.append(f"{name} median {med:.4g} outside [{lo:g}, {hi:g}]")
    if bad:
        raise ValueError(
            "clinical_reflex_score needs RAW physiological states, but these columns "
            "are not in clinical units: " + "; ".join(bad) + ". The normalised "
            "training/serving encodings must be passed as `encoded_states` (for the "
            "policy) while `raw_states` stays in clinical units (for the strata).")


def _views(raw_states: np.ndarray, feature_order: Sequence[str],
           weight_kg: np.ndarray) -> dict[str, np.ndarray]:
    """Named raw-unit columns the strata predicates read.

    ``raw_states`` must be RAW physiological values, not z-scores — see
    ``_assert_raw_units``, which enforces it.
    """
    f = list(feature_order)
    v = {name: raw_states[:, f.index(name)].astype(float)
         for name in ("PEEP", "TV", "FiO2", "SpO2", "PaO2", "PaCO2", "pH")}
    v["tv_per_kg"] = v["TV"] / np.maximum(np.asarray(weight_kg, dtype=float), 1.0)
    _assert_raw_units(v)
    return v


def strata_masks(raw_states: np.ndarray, feature_order: Sequence[str],
                 weight_kg: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean row mask per stratum. Strata overlap; that is intended.

    A patient can be simultaneously hypoxaemic and on a high PEEP, and both
    reflexes are worth measuring. Overlap costs nothing because the score is a
    macro-average of per-stratum rates, not a partition of the split.
    """
    v = _views(raw_states, feature_order, weight_kg)
    return {name: np.asarray(where(v), dtype=bool) for name, (where, _, _) in STRATA.items()}


def clinical_reflex_score(actions: np.ndarray, raw_states: np.ndarray,
                          feature_order: Sequence[str], weight_kg: np.ndarray,
                          min_stratum_n: int = MIN_STRATUM_N) -> dict:
    """Macro-averaged direction-compliance of ``actions`` over the eight strata.

    ``actions`` are action-space indices, one per row of ``raw_states`` — the
    policy's recommendation for that state, or the clinician's actual action when
    scoring the behaviour baseline for reference.
    """
    from backend.mdp import action_space

    am = action_space.ACTION_MAP
    a = np.asarray(actions, dtype=int)
    dp = np.array([am[int(x)][0] for x in a], dtype=float)
    dt = np.array([am[int(x)][1] for x in a], dtype=float)
    df = np.array([am[int(x)][2] for x in a], dtype=float)

    masks = strata_masks(raw_states, feature_order, weight_kg)
    per: dict[str, dict] = {}
    for name, (_, ok, description) in STRATA.items():
        m = masks[name]
        n = int(m.sum())
        rate = (float(np.mean(np.asarray(ok(dp[m], dt[m], df[m]), dtype=bool)))
                if n else None)
        per[name] = {"n": n, "compliance": None if rate is None else round(rate, 4),
                     "counted": bool(n >= min_stratum_n), "indicated": description}

    counted = [s["compliance"] for s in per.values() if s["counted"]]
    return {
        "crs": None if not counted else round(float(np.mean(counted)), 4),
        "n_strata_counted": len(counted),
        "n_strata": len(STRATA),
        "min_stratum_n": min_stratum_n,
        "per_stratum": per,
    }


def score(model, raw_states: np.ndarray, encoded_states: np.ndarray,
          feature_order: Sequence[str], weight_kg: np.ndarray,
          clinician_actions: np.ndarray | None = None,
          val_q_loss: float | None = None,
          min_distinct_actions: int = MIN_DISTINCT_ACTIONS,
          min_stratum_n: int = MIN_STRATUM_N) -> dict:
    """Score one checkpoint: the reflex score, the two guards, and the verdict.

    ``encoded_states`` is ``raw_states`` under whatever encoding the model was
    fitted with — the caller owns that choice, because the training encoding
    (``normaliser.transform``) and the serving encoding
    (``normaliser.transform_inference``, which winsorises) are different
    quantities and mixing them is the mistake SUMMARY §15.x records.

    ``val_q_loss`` is carried through for tie-breaking and for reporting the
    disagreement; it is never part of ``crs``.
    """
    from backend.mdp import action_space
    from backend.router import safety_rules as SR

    acts = model.act_batch(np.asarray(encoded_states, dtype=np.float32))
    reflex = clinical_reflex_score(acts, raw_states, feature_order, weight_kg,
                                   min_stratum_n=min_stratum_n)

    f = list(feature_order)
    peep, tv, fio2, spo2 = (raw_states[:, f.index("PEEP")], raw_states[:, f.index("TV")],
                            raw_states[:, f.index("FiO2")], raw_states[:, f.index("SpO2")])
    w = np.asarray(weight_kg, dtype=float)
    p_peep, p_tv, p_fio2 = SR.resulting_settings(peep, tv, fio2, acts)
    policy_viol = SR.violations(p_peep, p_tv, p_fio2, spo2, w)

    clinician_viol = None
    if clinician_actions is not None:
        c_peep, c_tv, c_fio2 = SR.resulting_settings(peep, tv, fio2,
                                                     np.asarray(clinician_actions, int))
        clinician_viol = SR.violations(c_peep, c_tv, c_fio2, spo2, w)

    n_distinct = int(len(np.unique(acts)))
    hold = action_space.encode_action(0, 0, 0.0)
    guards = {
        "repertoire": {
            "n_distinct_actions": n_distinct, "minimum": min_distinct_actions,
            "pass": bool(n_distinct >= min_distinct_actions),
            "why": "a collapsed repertoire is not a policy, whatever its TD-error",
        },
        "no_worse_than_clinician_on_safety": {
            "policy_any_violation": policy_viol["any_violation"],
            "clinician_any_violation": (None if clinician_viol is None
                                        else clinician_viol["any_violation"]),
            "pass": (True if clinician_viol is None
                     else bool(policy_viol["any_violation"]
                               <= clinician_viol["any_violation"])),
            "why": ("the claim boundary is 'no worse than the clinician' — a "
                    "candidate that violates lung-protective rules more often than "
                    "the clinician does on the same states is not selectable, "
                    "however good its reflexes"),
        },
    }
    selectable = bool(reflex["crs"] is not None
                      and all(g["pass"] for g in guards.values()))

    return {
        "crs": reflex["crs"],
        "selectable": selectable,
        "val_q_loss": None if val_q_loss is None else round(float(val_q_loss), 6),
        "hold_share": round(float(np.mean(acts == hold)), 4),
        "n_distinct_actions": n_distinct,
        "guards": guards,
        "reflex": reflex,
        "safety": {"policy": policy_viol, "clinician": clinician_viol},
    }


def best(candidates: dict[str, dict]) -> tuple[str | None, dict]:
    """Pick the highest-CRS selectable candidate; tie-break on lower val Q-loss.

    ``candidates`` maps a label (a step, a seed, a path) to a ``score`` result.
    Returns ``(label, rationale)``; the label is ``None`` when every candidate is
    disqualified, which is a real answer — it means the run produced nothing
    deployable and the honest response is to train differently, not to promote the
    least-bad checkpoint.

    Ties within ``tol`` of the best CRS go to the lower validation Q-loss. TD-error
    is thus demoted to a tie-breaker rather than removed: among checkpoints whose
    clinical behaviour is indistinguishable, the better-fitted Q-function is the
    reasonable choice.
    """
    tol = 1e-9
    ok = {k: v for k, v in candidates.items() if v.get("selectable")}
    if not ok:
        return None, {"reason": "no candidate passed the guards",
                      "disqualified": {k: [n for n, g in v.get("guards", {}).items()
                                           if not g["pass"]]
                                       for k, v in candidates.items()}}
    top = max(v["crs"] for v in ok.values())
    tied = {k: v for k, v in ok.items() if v["crs"] >= top - tol}
    if len(tied) > 1:
        label = min(tied, key=lambda k: (tied[k]["val_q_loss"]
                                         if tied[k]["val_q_loss"] is not None
                                         else float("inf")))
        return label, {"crs": top, "tie_broken_on": "lower validation Q-loss",
                       "tied_candidates": sorted(tied)}
    label = next(iter(tied))
    return label, {"crs": top, "tie_broken_on": None}
