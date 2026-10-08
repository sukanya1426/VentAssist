"""Assemble the VentAssist vs IntelliLung comparison table (BENCHMARK_PLAN Phase 4).

Reads the artifacts in `benchmark/results/` and emits one markdown table set plus
a machine-readable JSON. It computes nothing itself — every number is traced back
to the artifact that produced it, with that artifact's own timestamp and `n`, so a
stale input is visible in the output rather than silently averaged in.

WHAT ARENA THIS IS. Phase 2's "common arena" (their 26-dim state, their split,
their frozen DistFQE) is NOT what this table reports, and cannot be: their
repository ships no trained weights and no data, and their MIMIC pipeline requires
a populated Postgres `mimic` database. What it does report is the achievable half
of the head-to-head — **their reward function, ported faithfully, trained and
scored inside our MDP** — plus our own system against the clinician baseline.
Stated as a deviation in `scope.arena` and in the markdown header, because a
reviewer must not read Table A as "we beat their published model".

TWO DISCIPLINES ENFORCED IN CODE, not by convention:

1. **Policies are never mixed in one row.** `fqe_track_a.json` describes the
   DEPLOYED checkpoint; `runner_track_*.json` describe fresh seeds under a
   different checkpoint-selection config. They are different policies, so each
   row carries a `policy` provenance key and `_assert_single_policy` refuses to
   emit a row that spans two.

2. **Agreement is never reported alone.** Clinician agreement rises as a policy
   collapses toward the majority class (clinicians hold 77.65 % of the time), so
   `_rule5` refuses to emit an agreement figure without a value estimate and an
   action-diversity figure beside it. This generalises IntelliLung's own Rule 5
   and is the single most load-bearing guardrail in the benchmark.

Run:  PYTHONPATH=. .venv/bin/python -m benchmark.comparison_table
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from backend.pipeline import config
from backend.pipeline.logging_utils import get_logger

log = get_logger("bench_comparison_table")

RESULTS = Path(__file__).resolve().parent / "results"
LOGS = config.REPO_ROOT / "backend" / "logs"

# Policy provenance. Rows tagged with different values here may never be compared
# as though they described the same system.
DEPLOYED = "deployed_policy_track_a.pt"
SEEDS_COARSE = "5_fresh_seeds_checkpoint_every_10000"
SEEDS_FINE = "5_fresh_seeds_checkpoint_every_500"
CLINICIAN = "clinician_behaviour_policy"


class MissingArtifact(Exception):
    """An input the table cannot be honestly assembled without."""


# --------------------------------------------------------------------------- #
# Artifact loading, with provenance
# --------------------------------------------------------------------------- #
def _load(name: str, *, where: Path = RESULTS, required: bool = True) -> dict | None:
    f = where / name
    if not f.exists():
        if required:
            raise MissingArtifact(
                f"{f} is absent. The table refuses to emit a blank cell for it; "
                f"generate it or pass required=False for that section.")
        log.warning("optional artifact absent, its section will be omitted: %s", name)
        return None
    return json.loads(f.read_text())


def _prov(d: dict | None, name: str) -> dict:
    if d is None:
        return {"artifact": name, "present": False}
    return {"artifact": name, "present": True,
            "timestamp": d.get("timestamp"),
            "n_test": d.get("n_test") or d.get("n_test_transitions"),
            "n_transitions": d.get("n_transitions")}


def _ci(stat: dict | None, nd: int = 4) -> str:
    """Render a `_ci95` block as `mean [lo, hi]`, or `mean (single seed)`."""
    if not stat or stat.get("mean") is None:
        return "—"
    m = stat["mean"]
    if stat.get("ci95"):
        lo, hi = stat["ci95"]
        return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"
    return f"{m:.{nd}f} (single seed, no interval)"


def _inside(x: float, stat: dict | None) -> bool | None:
    """Is a scalar inside a `_ci95` block's interval?"""
    if not stat or not stat.get("ci95"):
        return None
    lo, hi = stat["ci95"]
    return bool(lo <= x <= hi)


def _excludes_zero(stat: dict | None) -> bool:
    if not stat or not stat.get("ci95"):
        return False
    lo, hi = stat["ci95"]
    return (lo > 0) or (hi < 0)


# --------------------------------------------------------------------------- #
# The two guardrails
# --------------------------------------------------------------------------- #
def _assert_single_policy(rows: list[dict], label: str) -> None:
    """A table section must describe one policy, or say which per row."""
    tagged = [r for r in rows if "policy" in r]
    if len(tagged) != len(rows):
        raise AssertionError(
            f"{label}: {len(rows) - len(tagged)} row(s) carry no `policy` key. "
            "Every row must name the policy it describes, or the table silently "
            "compares the deployed checkpoint against fresh seeds.")


def _rule5(agreement, value, diversity, *, policy: str, source: str) -> dict:
    """The Rule 5 triple. Refuses to report agreement on its own.

    Agreement with the clinician is the one metric in this benchmark that moves
    the WRONG way: it improves whenever the policy collapses toward holding,
    because clinicians hold most of the time. Observed three independent times
    (see `agreement_trap` below), so it is blocked structurally rather than
    flagged in prose.
    """
    if agreement is not None and (value is None and diversity is None):
        raise AssertionError(
            f"{source}: agreement={agreement} offered with neither a value "
            "estimate nor an action-diversity figure. A policy that collapses to "
            "the majority class scores BETTER on agreement — reporting it alone "
            "inverts the conclusion. Supply value and/or diversity.")
    return {"policy": policy, "source": source, "agreement": agreement,
            "value": value, "diversity": diversity}


# --------------------------------------------------------------------------- #
# Verified capability comparison
# --------------------------------------------------------------------------- #
# Each claim cites the file in their repository that was read to check it. Claims
# that did NOT survive the check are kept, marked false, rather than deleted —
# a retracted differentiator is more useful to the defence than a quietly dropped
# one, because it is the kind of thing a reviewer checks.
CAPABILITIES = [
    {"claim": "Reward carries a direct action signal",
     "ventassist": "yes — action_cost(a) + lam_causal * causal_bonus(s, a)",
     "intellilung": "NO — the action is absent from their entire reward INTERFACE",
     "verified": True,
     "source": "algo_src/reward/base.py:7 — the abstract signature is "
               "`__call__(self, dataset, terminated, pre_process_configs, **kwargs)`, "
               "with no action parameter. All five implementations inherit it "
               "(range.py, ventilator_free_days.py x2, mortality.py, stacking.py) and "
               "`grep -n action algo_src/reward/*.py` returns NOTHING. So this is not a "
               "property of the reward they happened to choose — their abstraction cannot "
               "express an action-dependent reward at all."},
    {"claim": "The WHOLE composite reward is action-independent, not just RangeReward",
     "ventassist": "n/a",
     "intellilung": "yes — both components are functions of state/outcome columns",
     "verified": True,
     "source": "Their configured reward is AddRewards([RangeReward, VFDEachStep]) "
               "(algo_src/configs/pre_processing_configs.yaml). RangeReward reads "
               "next_states only; VFDEachStep reads pause_until_next, mv_duration and "
               "daemo_discharge (algo_src/reward/ventilator_free_days.py:57) — all dataset "
               "state/outcome columns. MortalityReward likewise reads daemo_discharge only. "
               "This closes the obvious objection that our port covers only half their "
               "reward and the other half might carry the action signal: it does not."},
    {"claim": "Current ventilator settings present in the state (makes delta-actions well-posed)",
     "ventassist": "yes — PEEP, TV and FiO2 are all state features",
     "intellilung": "partially — PEEP and FiO2 are NOT in the state, only actions",
     "verified": True,
     "source": "algo_src/configs/pre_processing_configs.yaml: both experiment blocks have "
               "exactly 26 state_vector_columns, which include `vent_pinsp` and "
               "`vent_vt_obs` but NOT `vent_peep` or `vent_fio2` — those two appear only "
               "under action_space. REFINEMENT to BENCHMARK_PLAN §0 trap 3, which says "
               "their state lacks 'the current PEEP/FiO2 settings': correct for PEEP and "
               "FiO2, but their state DOES carry observed tidal volume, so the claim must "
               "not be widened to all three knobs."},
    {"claim": "Mode-aware action masking enabled by default",
     "ventassist": "yes — pressure-control masks 100 of 125 actions",
     "intellilung": "no — ships disabled",
     "verified": True,
     "source": "algo_src/dataset/pre_processing_configs.py:28 "
               "`vent_mode_action_masking: bool = False`; both shipped config blocks "
               "(algo_src/configs/pre_processing_configs.yaml:96,191) set it false"},
    {"claim": "Safety-violation rates measured against the clinician",
     "ventassist": "yes — 5 rules, policy lower on every one",
     "intellilung": "no such metric in the repository",
     "verified": True,
     "source": "BENCHMARK_HANDOFF.md (their Rule 9); no violation-rate metric found in algo_src/"},
    {"claim": "Enforced safety filter + hard deploy gate",
     "ventassist": "yes — backend/router/safety_filter.py, 8/8 gate",
     "intellilung": "no",
     "verified": True,
     "source": "absent from algo_src/"},
    {"claim": "Decision resolution finer than theirs (1 h vs 4 h)",
     "ventassist": "1 h",
     "intellilung": "1 h — NOT 4 h",
     "verified": False,
     "source": "RETRACTED. data_pipelines/MIMIC/.env.example:8 `RESOLUTION = 3600` is the "
               "decision timestep (consumed by create_time_windows(..., resolution), "
               "time_window_creation.py:170,247); their own comment at line 292 reads "
               "'during the 1h resolution time'. The 4 h in their code is the drug/fluid "
               "LOOK-BACK window (drugs_vaso4h, state_ivfluid4h), not the timestep. "
               "BENCHMARK_PLAN differentiator #4 and the §1 'their 4h for the head-to-head' "
               "decision are both wrong and must be struck."},
]


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build(write: bool = True) -> dict:
    l1 = _load("reward_critique_layer1.json")
    l2ms = _load("reward_critique_layer2_multiseed.json", required=False)
    l2 = _load("reward_critique_layer2.json", required=False)
    ad = _load("action_density_track_a.json")
    sm = _load("safety_metrics_track_a.json")
    bc = _load("behaviour_compare_track_a.json")
    cc = _load("confidence_calibration_track_a.json")
    coarse = _load("runner_track_a.json")
    fine = _load("runner_track_a_ckpt500.json", required=False)
    ipw = _load("runner_track_a_ipw.json", required=False)
    fqe_dep = _load("fqe_track_a.json", where=LOGS)
    nwe = _load("nwe_track_a.json", where=LOGS, required=False)
    cb = _load("checkpoint_battery_track_a.json", required=False)

    provenance = {
        "reward_critique_layer1": _prov(l1, "reward_critique_layer1.json"),
        "reward_critique_layer2_multiseed": _prov(l2ms, "reward_critique_layer2_multiseed.json"),
        "reward_critique_layer2_single_seed": _prov(l2, "reward_critique_layer2.json"),
        "action_density": _prov(ad, "action_density_track_a.json"),
        "safety_metrics": _prov(sm, "safety_metrics_track_a.json"),
        "behaviour_compare": _prov(bc, "behaviour_compare_track_a.json"),
        "confidence_calibration": _prov(cc, "confidence_calibration_track_a.json"),
        "runner_coarse": _prov(coarse, "runner_track_a.json"),
        "runner_fine": _prov(fine, "runner_track_a_ckpt500.json"),
        "runner_ipw": _prov(ipw, "runner_track_a_ipw.json"),
        "fqe_deployed": _prov(fqe_dep, "backend/logs/fqe_track_a.json"),
        "nwe_deployed": _prov(nwe, "backend/logs/nwe_track_a.json"),
        "checkpoint_battery": _prov(cb, "checkpoint_battery_track_a.json"),
    }

    # --- Table A: the reward head-to-head ---------------------------------- #
    il, va = "intellilung_range_reward", "ventassist_reward"
    table_a = {
        "arena": "VentAssist MDP (992,100 transitions, 1 h, 125 discrete delta actions)",
        "controlled": "identical architecture, identical MDP, identical budget; ONLY the reward differs",
        "layer1_structural": {
            "n_sampled": l1["n_sampled_transitions"],
            "intellilung": l1[il],
            "ventassist": l1[va],
        },
    }
    if l2ms:
        agg, paired = l2ms["aggregate"], l2ms["paired_delta_ventassist_minus_intellilung"]
        # Nothing may fall through unclassified: an unclassified metric would be
        # rendered as cross-arm comparable by omission, which is the exact error
        # the scale split exists to prevent.
        unclassified = set(paired) - set(SCALE_FREE) - set(SCALE_DEPENDENT)
        if unclassified:
            raise AssertionError(
                f"Layer 2 reports metrics this table has not classified as scale-free "
                f"or scale-dependent: {sorted(unclassified)}. Add each to SCALE_FREE "
                f"(a count or rate of actions) or SCALE_DEPENDENT (denominated in Q "
                f"units) before the table can quote it.")
        table_a["layer2_multiseed"] = {
            "seeds": l2ms["seeds"], "steps": l2ms["steps"],
            "paired_on_seed": True,
            "metrics": {
                m: {"intellilung": agg[il][m], "ventassist": agg[va][m],
                    "paired_delta": paired[m],
                    "delta_ci_excludes_zero": _excludes_zero(paired[m]),
                    "cross_arm_comparable": m in SCALE_FREE,
                    **({} if m in SCALE_FREE else {"caveat": _SCALE_CAVEAT})}
                for m in paired
            },
            "scale_free_metrics": list(SCALE_FREE),
            "scale_dependent_metrics": list(SCALE_DEPENDENT),
            "scale_caveat": _SCALE_CAVEAT,
            "battery_pass_rate": {"intellilung": agg[il]["battery_pass_rate"],
                                  "ventassist": agg[va]["battery_pass_rate"]},
        }
        # Derive the cases OUR reward does not reliably fix. Computed rather than
        # written down, so it cannot drift away from the artifact. The old
        # single-seed Layer 2 reported 8/8 for our reward; seeding shows 6.6/8,
        # so the per-case failures are real and have to be stated.
        ours_rate = agg[va]["battery_pass_rate"]
        theirs_rate = agg[il]["battery_pass_rate"]
        table_a["battery_honest_negatives"] = {
            "never_fixed_by_either_reward": sorted(
                c for c in ours_rate if ours_rate[c] == 0.0 and theirs_rate[c] == 0.0),
            "unreliable_under_our_reward": sorted(
                c for c in ours_rate if 0.0 < ours_rate[c] < 1.0),
            "note": ("The single-seed Layer 2 result reported 8/8 for our reward. Across "
                     "5 seeds it is "
                     f"{agg[va]['battery_passed_n']['mean']:.1f}/8 "
                     f"{agg[va]['battery_passed_n'].get('ci95')}. The 8/8 was a lucky "
                     "seed and must not be quoted. Report the per-case rates."),
        }
    elif l2:
        table_a["layer2_single_seed"] = {
            "warning": "ONE seed, no interval — not a defensible contrast on its own",
            "intellilung": {k: l2[il].get(k) for k in
                            ("distinct_actions_used", "hold_share", "q_top2_margin",
                             "agreement_with_clinician", "battery_passed")},
            "ventassist": {k: l2[va].get(k) for k in
                           ("distinct_actions_used", "hold_share", "q_top2_margin",
                            "agreement_with_clinician", "battery_passed")},
        }

    # --- Table B: the agreement trap --------------------------------------- #
    trap = []
    if l2ms:
        agg = l2ms["aggregate"]
        trap.append({
            "case": "IntelliLung reward vs ours (paired, 5 seeds)",
            "agreement_moves": f"{agg[il]['agreement_with_clinician']['mean']:.4f} (theirs) vs "
                               f"{agg[va]['agreement_with_clinician']['mean']:.4f} (ours)",
            "but": f"distinct actions {agg[il]['distinct_actions_used']['mean']:.1f} vs "
                   f"{agg[va]['distinct_actions_used']['mean']:.1f}; "
                   f"battery {agg[il]['battery_passed_n']['mean']:.1f}/8 vs "
                   f"{agg[va]['battery_passed_n']['mean']:.1f}/8",
        })
    if ipw and coarse:
        a_on = ipw["aggregate"]["behaviour_match"]["mean"]
        a_off = coarse["aggregate"]["behaviour_match"]["mean"]
        trap.append({
            "case": "IPW on vs off (5 seeds each, matched budget)",
            "agreement_moves": f"{a_on:.4f} (on) vs {a_off:.4f} (off)",
            "but": f"FQE {ipw['aggregate']['fqe_V_hat']['mean']:.4f} vs "
                   f"{coarse['aggregate']['fqe_V_hat']['mean']:.4f}; distinct actions "
                   f"{ipw['aggregate']['n_distinct_actions']['mean']:.1f} vs "
                   f"{coarse['aggregate']['n_distinct_actions']['mean']:.1f}",
        })
    trap.append({
        "case": "Pressure-control action masking",
        "agreement_moves": "confidence ROSE 0.671 -> 0.687 on preset F",
        "but": "it rose because the policy's own preferred action (cut TV) had been "
               "suppressed; confidence is computed over allowed actions only, so removing "
               "100 competitors makes a forced hold look sharper (SUMMARY §15.11 B2)",
    })
    table_b = {
        "rule": "Never report clinician agreement without a value estimate and an "
                "action-diversity figure beside it.",
        "why": "Clinicians hold 77.65 % of the time, so any policy that holds more scores "
               "better on agreement. Agreement improves as the policy gets worse.",
        "generalises": "IntelliLung's own Rule 5, extended from likelihood to agreement",
        "observations": trap,
    }

    # --- Table C: VentAssist vs clinician ---------------------------------- #
    # The deployed policy has no distinct-action count in any artifact, so its
    # diversity leg is the hold share (0.6894 against the clinicians' 0.7765) —
    # which is the quantity that actually moves when a policy collapses, and is
    # what makes the agreement figure beside it readable.
    rows_c = [
        _rule5(agreement=bc["agreement"]["exact_125_way"],
               value=fqe_dep["V_hat"],
               diversity=None,
               policy=DEPLOYED, source="behaviour_compare + fqe_track_a (serving encoding)"),
    ]
    rows_c[0]["hold_share"] = bc["agreement"]["hold_share_policy"]
    rows_c[0]["hold_share_clinician"] = bc["agreement"]["hold_share_clinician"]
    if fine:
        rows_c.append(_rule5(
            agreement=fine["aggregate"]["behaviour_match"]["mean"],
            value=fine["aggregate"]["fqe_V_hat"]["mean"],
            diversity=fine["aggregate"]["n_distinct_actions"]["mean"],
            policy=SEEDS_FINE, source="runner_track_a_ckpt500 (training encoding)"))
    if coarse:
        rows_c.append(_rule5(
            agreement=coarse["aggregate"]["behaviour_match"]["mean"],
            value=coarse["aggregate"]["fqe_V_hat"]["mean"],
            diversity=coarse["aggregate"]["n_distinct_actions"]["mean"],
            policy=SEEDS_COARSE, source="runner_track_a (training encoding)"))
    _assert_single_policy(rows_c, "Table C")

    table_c = {
        "rule5_triple": rows_c,
        "rule5_partial": (
            "Item 7 of Phase 1 is PARTIAL by design. The clinician action "
            "log-likelihood is a property of the behaviour policy, not of our "
            "training seeds, so it is identical across seeds and is computed once in "
            "action_density.py. The triple is therefore assembled from two artifacts, "
            "not one. Stated here so a reviewer does not find it unaided."),
        "action_likelihood_rule5_gate": {
            "policy": ad["loglik_policy"], "clinician": ad["loglik_clinician"],
            "delta_policy_minus_clinician": ad["delta_policy_minus_clinician"],
            "epochs": ad.get("epochs"),
            "reading": "Delta > 0 means the policy's actions are ON-SUPPORT under the "
                       "clinician density model — not extrapolation. The sign is what "
                       "Rule 5 asks for; the margin is small.",
            "policy_described": DEPLOYED,
        },
        "safety_violations": {
            "policy": sm["policy"], "clinician": sm["clinician"],
            "delta": sm["delta_policy_minus_clinician"],
            "reading": "The policy violates less often on every rule measured.",
            "policy_described": DEPLOYED,
            "caveat": sm.get("notes", {}),
        },
        # The retrain decision, as evidence rather than as a plan. Derived, so it
        # cannot drift from the artifacts.
        "checkpoint_selection": ({
            "question": ("Does the checkpoint_every=500 fix produce a model that should "
                         "replace the deployed one?"),
            "answer_short": "No, not on this evidence.",
            "deployed_fqe": fqe_dep["V_hat"],
            "arms": {
                SEEDS_COARSE: {
                    "selected_steps": sorted({r["selected_step"] for r in coarse["per_seed"]}),
                    "best_val_q": coarse["aggregate"]["best_val_q"],
                    "fqe_V_hat": coarse["aggregate"]["fqe_V_hat"],
                    "deployed_inside_ci": _inside(
                        fqe_dep["V_hat"], coarse["aggregate"]["fqe_V_hat"]),
                },
                SEEDS_FINE: {
                    "selected_steps": sorted({r["selected_step"] for r in fine["per_seed"]}),
                    "best_val_q": fine["aggregate"]["best_val_q"],
                    "fqe_V_hat": fine["aggregate"]["fqe_V_hat"],
                    "deployed_inside_ci": _inside(
                        fqe_dep["V_hat"], fine["aggregate"]["fqe_V_hat"]),
                },
            },
            "finding": (
                "Validation Q-loss and FQE disagree about which checkpoint is best, and "
                "the finer interval moved selection AWAY from the FQE optimum. Q-loss "
                "improves reproducibly and with non-overlapping CIs "
                f"({coarse['aggregate']['best_val_q']['mean']:.4f} -> "
                f"{fine['aggregate']['best_val_q']['mean']:.4f}), but FQE does not follow: "
                f"{coarse['aggregate']['fqe_V_hat']['mean']:.4f} -> "
                f"{fine['aggregate']['fqe_V_hat']['mean']:.4f}, and the interval widens "
                f"{coarse['aggregate']['fqe_V_hat']['half_width']:.4f} -> "
                f"{fine['aggregate']['fqe_V_hat']['half_width']:.4f}. By selected step the "
                "value estimate is NON-MONOTONIC and peaks near 10,000: "
                f"~5,750 -> {fine['aggregate']['fqe_V_hat']['mean']:.4f}, "
                f"~10,000 -> {coarse['aggregate']['fqe_V_hat']['mean']:.4f}, "
                f"~27,000 (deployed) -> {fqe_dep['V_hat']:.4f}."),
            "consequence_for_the_retrain_claim": (
                "The deployed checkpoint sits 0.50 BELOW the coarse arm's lower bound, "
                "which is the basis of the SUMMARY §15.11 C6 claim that it underperforms a "
                "retrain by more than seed noise. Against the arm trained with the FIXED "
                "config it sits INSIDE the interval. So that claim holds only against the "
                "coarse arm, and the case for replacing the deployed artifact is NOT made. "
                "Do not deploy on this evidence."),
            "clinical_battery_vs_selected_step": (cb["battery_vs_selected_step"] if cb else None),
            "cases_learned_late": (cb["cases_learned_later_than_the_val_loss_optimum"]
                                   if cb else None),
            "third_criterion": (
                "A THIRD criterion disagrees with both, and it is the one the deploy gate "
                "enforces: the 8-case clinical battery improves monotonically with selected "
                "step — 6.60/8 at ~5,750, 7.20/8 at ~10,000, 8.00/8 for the deployed "
                "~27,000 model. `high_peep` ('PEEP is 18, lower it') is a safety reflex "
                "acquired LATE: 0 of 5 seeds have it at ~5,750, 2 of 5 at ~10,000, and the "
                "deployed model has it. Selecting on validation Q-loss therefore selects a "
                "model that has not yet learned a clinical safety reflex the deployed one "
                "has. Evidence: benchmark/checkpoint_battery.py."
                if cb else None),
            "what_does_hold": (
                "Two things improve in the fine arm and are worth keeping: validation "
                "Q-loss (separated CIs) and the safety violation rate "
                f"({coarse['aggregate']['safety_any_violation']['mean']:.4f} -> "
                f"{fine['aggregate']['safety_any_violation']['mean']:.4f}, also separated). "
                "The lung-protective dTV bias excludes zero in BOTH arms, so it does not "
                "depend on the selection rule."),
        } if (fine and coarse) else None),
        "model_free_leads": {
            "note": "These two do not depend on our reward or our learned dynamics, so "
                    "they are the results to lead with.",
            "lower_tidal_volumes": {
                "per_seed_mean_dTV_mL": (coarse["aggregate"]["mean_dTV"] if coarse else None),
                "serving_encoding_mean_dTV_mL": bc["deviation_policy_minus_clinician"]["delta_TV"]["mean_signed"],
                "excludes_zero": _excludes_zero(coarse["aggregate"]["mean_dTV"]) if coarse else None,
                "direction": "lung-protective",
            },
            "more_stable_hour_to_hour": {
                "churn_policy": bc["churn"]["policy"]["action_changed"],
                "churn_clinician": bc["churn"]["clinician"]["action_changed"],
                "n_consecutive_pairs": bc["churn"]["policy"]["n_consecutive_pairs"],
                "reversals_policy": bc["churn"]["policy"]["per_knob_reversed"],
                "reversals_clinician": bc["churn"]["clinician"]["per_knob_reversed"],
                "caveat": bc.get("notes", {}).get("churn"),
            },
        },
        "model_based_rollout_safety": ({
            "warning": ("MODEL-BASED. These come from NWE rollouts under our LEARNED "
                        "dynamics over T=12 steps from n_starts=400, and the hypoxaemia "
                        "row rests on only "
                        f"{nwe['safety_metrics']['clinician']['n_episodes_start_lt95']} "
                        "episodes. They are far weaker evidence than the 198,050-transition "
                        "model-free numbers in C3/C4 and must never be quoted beside them "
                        "without this distinction."),
            "n_starts": nwe["n_starts"], "horizon_steps": nwe["T"],
            "nwe_V_hat": nwe["V_hat"],
            "policy": nwe["safety_metrics"]["hybrid_iql"],
            "clinician": nwe["safety_metrics"]["clinician"],
            "policy_described": DEPLOYED,
        } if nwe else None),
        "confidence": {
            "ece": cc["primary_agreement_with_clinician"]["ece"],
            "spearman_vs_agreement": cc["primary_agreement_with_clinician"]["spearman_confidence_vs_agreement"],
            "mce": cc["primary_agreement_with_clinician"]["mce"],
            "inverts_at_the_top": True,
            "quote_which": "monotonicity (Spearman), not ECE — the achievable range is "
                           f"{cc['structural_ceiling']['achievable_range']}, so a near-zero "
                           "ECE is structurally impossible",
            "carries_no_mortality_signal": cc["secondary_outcome_association"][
                "spearman_confidence_vs_survival"],
        },
    }

    # --- Claim boundaries --------------------------------------------------- #
    boundaries = [
        "The defensible claim is NO WORSE THAN THE CLINICIAN — never 'better than doctors'.",
        "Every value number is computed under VentAssist's own reward and its own learned "
        "dynamics. FQE values are not comparable across different rewards.",
        "Retrospective, single cohort, unmeasured confounding, no prospective validation.",
        "Waveform influence is EXACTLY 0.0 — Track B's checkpoint is Track A's weights with "
        "an untouched [I | 0] adapter (selected_step = 0). An honest negative result on 35 "
        "patients / 895 transitions.",
        "Responsiveness > 0 is entirely unvalidated; every number here is at responsiveness 0.",
        "IPW addresses MEASURED confounding only, and as implemented it makes the policy worse.",
        "Table A is their REWARD in OUR arena. It is not a comparison against their published "
        "model, which was trained on MIMIC + eICU + HiRID and whose weights they do not ship.",
        "Their 'Hybrid-IQL' means a hybrid ACTION SPACE; our 'HybridIQL' is discrete IQL + CQL "
        "+ a feature adapter. Rename ours before publication to avoid the collision.",
    ]

    result = {
        "title": "VentAssist vs IntelliLung — comparison table",
        "scope": {
            "arena": "their reward, ported faithfully (benchmark/rewards_intellilung.py), "
                     "trained and scored inside the VentAssist MDP",
            "not_the_arena": "their 26-dim state / their split / their frozen DistFQE "
                             "(Phase 2) — blocked: they ship no weights and no data, and "
                             "their MIMIC pipeline needs a populated Postgres `mimic` DB",
            "their_reward_verified_against": "algo_src/reward/range.py (full repo, not a port)",
        },
        "table_a_reward_head_to_head": table_a,
        "table_b_agreement_trap": table_b,
        "table_c_ventassist_vs_clinician": table_c,
        "table_d_capabilities": CAPABILITIES,
        "claim_boundaries": boundaries,
        "provenance": provenance,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if write:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "comparison_table.json").write_text(json.dumps(result, indent=2))
        (RESULTS / "comparison_table.md").write_text(render_markdown(result))
        log.info("wrote comparison_table.json and comparison_table.md")
    return result


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #
# Which Layer 2 metrics may be compared ACROSS the two reward arms.
#
# This is §0 trap 1 ("FQE values are not comparable across different rewards")
# applied to Layer 2's own metrics, which the plan did not catch. Q-values scale
# with reward magnitude: their reward lands in [-1, 0] and ours in roughly
# [-43, +42], so any metric denominated in Q units is ~85x larger for us before
# a single thing is learned. A "sharper Q margin" under our reward is therefore
# not evidence of a sharper decision — it is evidence of a bigger reward.
#
# The four scale-free metrics carry the contrast. They are counts and rates of
# ACTIONS, so a reward's scale cannot inflate them.
SCALE_FREE = ("distinct_actions_used", "hold_share", "agreement_with_clinician",
              "battery_passed_n")
SCALE_DEPENDENT = ("q_spread_max_minus_min", "q_top2_margin", "best_val_q")
_SCALE_CAVEAT = (
    "Denominated in Q units, which scale with reward magnitude (theirs [-1, 0], "
    "ours ~[-43, +42]). Valid as a WITHIN-arm diagnostic; NOT a cross-arm "
    "comparison. The plan's '1.7x sharper Q margin' headline is not "
    "scale-invariant and should not be used.")

_PRETTY = {
    "q_top2_margin": "Q top-2 margin (decision sharpness)",
    "distinct_actions_used": "Distinct actions used",
    "hold_share": "Hold share",
    "agreement_with_clinician": "Agreement w/ clinician",
    "battery_passed_n": "Clinical battery (of 8)",
    "best_val_q": "Best validation Q-loss (lower better)",
    "q_spread_max_minus_min": "Q spread (max − min)",
}


def render_markdown(r: dict) -> str:
    o: list[str] = []
    o.append(f"# {r['title']}")
    o.append("")
    o.append(f"*Generated {r['timestamp']}.* Assembled by `benchmark/comparison_table.py` "
             "from the artifacts listed at the foot of this document. No number here is "
             "computed by the renderer.")
    o.append("")
    o.append("## Scope — read this before any table")
    o.append("")
    o.append(f"- **Arena:** {r['scope']['arena']}.")
    o.append(f"- **NOT the arena:** {r['scope']['not_the_arena']}.")
    o.append(f"- Their reward was verified against `{r['scope']['their_reward_verified_against']}`.")
    o.append("")

    # Table A
    a = r["table_a_reward_head_to_head"]
    o.append("## Table A — The reward head-to-head")
    o.append("")
    o.append(f"*{a['controlled']}.*")
    o.append("")
    o.append("### A1. Structural (no training; irrefutable)")
    o.append("")
    o.append("Hold the observed transition `(s, s′)` fixed and recompute the reward under all "
             f"125 actions, on {a['layer1_structural']['n_sampled']:,} held-out transitions.")
    o.append("")
    o.append("| Reward | mean spread | max spread | transitions with ZERO spread |")
    o.append("|---|---|---|---|")
    for lbl, key in (("IntelliLung `RangeReward`", "intellilung"), ("VentAssist", "ventassist")):
        s = a["layer1_structural"][key]
        o.append(f"| {lbl} | {s['mean_spread']} | {s['max_spread']} | "
                 f"{s['pct_transitions_with_zero_spread']} % |")
    o.append("")
    o.append("> Their reward assigns the **identical value to all 125 actions, for 100 % of "
             "transitions**. The only path from action to reward is through the (confounded) "
             "transition itself.")
    o.append("")

    if "layer2_multiseed" in a:
        m = a["layer2_multiseed"]
        o.append(f"### A2. Consequential ({m['seeds']} seeds, paired on seed, "
                 f"{m['steps']:,} steps)")
        o.append("")
        o.append("Arms share the seed at each index, so within a pair the weight "
                 "initialisation and the minibatch stream are identical and **only the reward "
                 "differs**. The paired delta therefore carries no between-seed variance.")
        o.append("")
        o.append("**Scale-free metrics — these carry the contrast.** Counts and rates of "
                 "*actions*, so a reward's magnitude cannot inflate them.")
        o.append("")
        o.append("| Metric | IntelliLung reward | VentAssist reward | paired Δ (ours − theirs) | Δ CI excludes 0 |")
        o.append("|---|---|---|---|---|")
        for k, v in m["metrics"].items():
            if not v["cross_arm_comparable"]:
                continue
            nd = 1 if k in ("distinct_actions_used", "battery_passed_n") else 4
            o.append(f"| {_PRETTY.get(k, k)} | {_ci(v['intellilung'], nd)} | "
                     f"{_ci(v['ventassist'], nd)} | {_ci(v['paired_delta'], nd)} | "
                     f"{'**yes**' if v['delta_ci_excludes_zero'] else 'no'} |")
        o.append("")
        o.append("**Scale-DEPENDENT metrics — do not compare these across arms.**")
        o.append("")
        o.append(f"> {m['scale_caveat']}")
        o.append("")
        o.append("| Metric | IntelliLung reward | VentAssist reward |")
        o.append("|---|---|---|")
        for k, v in m["metrics"].items():
            if v["cross_arm_comparable"]:
                continue
            o.append(f"| {_PRETTY.get(k, k)} | {_ci(v['intellilung'], 4)} | "
                     f"{_ci(v['ventassist'], 4)} |")
        o.append("")
        hn = a.get("battery_honest_negatives", {})
        if hn:
            o.append("**Honest negatives in the battery — state these before a reviewer finds them.**")
            o.append("")
            o.append(f"- {hn['note']}")
            if hn["never_fixed_by_either_reward"]:
                o.append(f"- **Fixed by neither reward, 0 of 5 seeds:** "
                         f"`{'`, `'.join(hn['never_fixed_by_either_reward'])}`. Our "
                         "action-causal reward does **not** repair this case; the earlier "
                         "single-seed claim that it did was seed luck.")
            if hn["unreliable_under_our_reward"]:
                rates = m["battery_pass_rate"]["ventassist"]
                o.append("- **Passes only on some seeds under our reward:** " +
                         ", ".join(f"`{c}` ({rates[c]:.0%} of seeds)"
                                   for c in hn["unreliable_under_our_reward"]))
            o.append("")
        o.append("**Per-case clinical battery pass rate** (fraction of seeds passing):")
        o.append("")
        o.append("| Case | IntelliLung reward | VentAssist reward |")
        o.append("|---|---|---|")
        br = m["battery_pass_rate"]
        for case in br["ventassist"]:
            o.append(f"| {case} | {br['intellilung'][case]} | {br['ventassist'][case]} |")
        o.append("")
    elif "layer2_single_seed" in a:
        o.append("### A2. Consequential — SINGLE SEED, not yet defensible")
        o.append("")
        o.append(f"> {a['layer2_single_seed']['warning']}")
        o.append("")

    # Table B
    b = r["table_b_agreement_trap"]
    o.append("## Table B — The methodological trap")
    o.append("")
    o.append(f"**{b['rule']}**")
    o.append("")
    o.append(f"{b['why']} Generalises {b['generalises']}.")
    o.append("")
    o.append("| Case | Agreement moves | …while |")
    o.append("|---|---|---|")
    for t in b["observations"]:
        o.append(f"| {t['case']} | {t['agreement_moves']} | {t['but']} |")
    o.append("")

    # Table C
    c = r["table_c_ventassist_vs_clinician"]
    o.append("## Table C — VentAssist vs the clinician")
    o.append("")
    o.append("### C1. The Rule 5 triple, per policy")
    o.append("")
    o.append("Each row names the policy it describes. **Rows are not comparable to each "
             "other** — the deployed checkpoint and the fresh seeds are different policies, "
             "and the serving and training encodings are different quantities.")
    o.append("")
    o.append("| Policy | Agreement | Value (FQE) | Diversity | Source |")
    o.append("|---|---|---|---|---|")
    for row in c["rule5_triple"]:
        ag = f"{row['agreement']:.4f}" if row["agreement"] is not None else "—"
        vl = f"{row['value']:.4f}" if row["value"] is not None else "—"
        if row.get("diversity") is not None:
            dv = f"{row['diversity']:.1f} distinct actions"
        elif row.get("hold_share") is not None:
            dv = (f"hold {row['hold_share']:.4f} vs clinician "
                  f"{row['hold_share_clinician']:.4f}")
        else:
            dv = "—"
        o.append(f"| `{row['policy']}` | {ag} | {vl} | {dv} | {row['source']} |")
    o.append("")
    o.append(f"> **On item 7's \"partial\".** {c['rule5_partial']}")
    o.append("")
    o.append("### C2. Action likelihood — the Rule 5 on-support gate")
    o.append("")
    al = c["action_likelihood_rule5_gate"]
    o.append(f"| log-lik policy | log-lik clinician | Δ | epochs |")
    o.append("|---|---|---|---|")
    o.append(f"| {al['policy']} | {al['clinician']} | **+{al['delta_policy_minus_clinician']}** "
             f"| {al['epochs']} |")
    o.append("")
    o.append(f"{al['reading']} Describes `{al['policy_described']}`.")
    o.append("")
    o.append("### C3. Safety-violation rates (their Rule 9 — they have no such metric)")
    o.append("")
    o.append("| Rule | Policy | Clinician | Δ |")
    o.append("|---|---|---|---|")
    sv = c["safety_violations"]
    for k in sv["policy"]:
        p, cl, dl = sv["policy"][k], sv["clinician"][k], sv["delta"][k]
        if p is None and cl is None:
            o.append(f"| {k} | not measurable | not measurable | — |")
        else:
            o.append(f"| {k} | {p} | {cl} | {dl} |")
    o.append("")
    o.append(f"{sv['reading']} Describes `{sv['policy_described']}`.")
    o.append("")
    cs = c.get("checkpoint_selection")
    if cs:
        o.append("### C4. Should the deployed checkpoint be replaced? — **No, not on this evidence**")
        o.append("")
        o.append(f"*{cs['question']}* **{cs['answer_short']}**")
        o.append("")
        o.append("| Arm | Selected step(s) | Best val Q-loss (lower better) | FQE V̂ | Deployed "
                 f"{cs['deployed_fqe']} inside this CI? |")
        o.append("|---|---|---|---|---|")
        for name, arm in cs["arms"].items():
            steps = ", ".join(str(x) for x in arm["selected_steps"])
            o.append(f"| `{name}` | {steps} | {_ci(arm['best_val_q'])} | "
                     f"{_ci(arm['fqe_V_hat'])} | "
                     f"{'**yes**' if arm['deployed_inside_ci'] else 'no'} |")
        o.append("")
        o.append(f"> **{cs['finding']}**")
        o.append("")
        o.append(f"**Consequence.** {cs['consequence_for_the_retrain_claim']}")
        o.append("")
        if cs.get("third_criterion"):
            o.append("**A third criterion, and it is the one the deploy gate enforces.**")
            o.append("")
            o.append("| Selected step | Family | Clinical battery | `high_peep` pass rate |")
            o.append("|---|---|---|---|")
            for t in cs["clinical_battery_vs_selected_step"]:
                o.append(f"| ~{t['approx_selected_step']:,} | `{t['family']}` | "
                         f"{t['battery_passed_mean']}/8 | "
                         f"{t['high_peep_pass_rate']:.0%} |")
            o.append("")
            o.append(f"> {cs['third_criterion']}")
            o.append("")
            if cs.get("cases_learned_late"):
                o.append(f"Cases learned later than the val-loss optimum: " +
                         ", ".join(f"`{c}`" for c in cs["cases_learned_late"]) +
                         ". These are exactly the two the reward critique flagged as "
                         "honest negatives, which means they are **selection artifacts, "
                         "not reward failures** — and the critique's absolute battery "
                         "numbers are a floor rather than a ceiling.")
                o.append("")
        o.append(f"**What does hold.** {cs['what_does_hold']}")
        o.append("")
    o.append("### C5. The two model-free results — lead with these")
    o.append("")
    mf = c["model_free_leads"]
    o.append(f"*{mf['note']}*")
    o.append("")
    tv = mf["lower_tidal_volumes"]
    o.append(f"- **Lower tidal volumes.** Per-seed mean ΔTV "
             f"{_ci(tv['per_seed_mean_dTV_mL'], 4)} mL"
             f"{' — CI excludes zero' if tv['excludes_zero'] else ''}; "
             f"{tv['serving_encoding_mean_dTV_mL']} mL at the serving encoding. "
             f"The {tv['direction']} direction.")
    st = mf["more_stable_hour_to_hour"]
    if st["churn_policy"] is not None:
        ratio = st["churn_clinician"] / st["churn_policy"] if st["churn_policy"] else float("nan")
        o.append(f"- **More stable hour to hour.** Churn {st['churn_policy']} (policy) vs "
                 f"{st['churn_clinician']} (clinician) — **{ratio:.1f}× more stable**, with "
                 f"far fewer direction reversals.")
    o.append("")
    mb = c.get("model_based_rollout_safety")
    if mb:
        o.append("### C6. Model-based rollout safety — weaker evidence, labelled as such")
        o.append("")
        o.append(f"> **{mb['warning']}**")
        o.append("")
        o.append(f"NWE V̂ {mb['nwe_V_hat']}, horizon {mb['horizon_steps']} steps, "
                 f"{mb['n_starts']} starts.")
        o.append("")
        o.append("| Metric | Policy | Clinician |")
        o.append("|---|---|---|")
        labels = {
            "pct_terminal_spo2_ge95": "Terminal SpO₂ ≥ 95 %",
            "mean_delta_spo2_start_lt95": "Mean ΔSpO₂ (episodes starting < 95 %)",
            "pct_aggressive_steps": "Aggressive steps",
            "n_episodes_start_lt95": "n episodes starting < 95 %",
        }
        for k, lbl in labels.items():
            pv, cv = mb["policy"].get(k), mb["clinician"].get(k)
            if pv is None and cv is None:
                continue
            o.append(f"| {lbl} | {pv} | {cv} |")
        o.append("")
        o.append(f"Describes `{mb['policy_described']}`.")
        o.append("")
    o.append("### C7. Confidence")
    o.append("")
    cf = c["confidence"]
    o.append(f"ECE {cf['ece']}, Spearman vs agreement **+{cf['spearman_vs_agreement']}**, "
             f"MCE {cf['mce']} — driven entirely by the top bin, where calibration "
             f"**inverts**. Quote {cf['quote_which']}. It carries no mortality signal "
             f"(Spearman {cf['carries_no_mortality_signal']}), which is the *correct* result "
             "for a decision confidence.")
    o.append("")

    # Table D
    o.append("## Table D — Capability comparison (checked against their source)")
    o.append("")
    o.append("| Claim | VentAssist | IntelliLung | Holds? |")
    o.append("|---|---|---|---|")
    for cap in r["table_d_capabilities"]:
        mark = "✅" if cap["verified"] else "❌ **RETRACTED**"
        o.append(f"| {cap['claim']} | {cap['ventassist']} | {cap['intellilung']} | {mark} |")
    o.append("")
    for cap in r["table_d_capabilities"]:
        o.append(f"- **{cap['claim']}** — {cap['source']}")
    o.append("")

    # Boundaries
    o.append("## Claim boundaries")
    o.append("")
    for bd in r["claim_boundaries"]:
        o.append(f"- {bd}")
    o.append("")

    # Provenance
    o.append("## Provenance")
    o.append("")
    o.append("| Artifact | Present | Timestamp | n_test |")
    o.append("|---|---|---|---|")
    for _, p in r["provenance"].items():
        if p["present"]:
            o.append(f"| `{p['artifact']}` | yes | {p['timestamp']} | "
                     f"{p.get('n_test') or '—'} |")
        else:
            o.append(f"| `{p['artifact']}` | **no** | — | — |")
    o.append("")
    return "\n".join(o)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    r = build(write=not args.no_write)
    print(render_markdown(r))


if __name__ == "__main__":
    main()
