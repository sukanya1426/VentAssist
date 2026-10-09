"""The clinical-reflex selection criterion, and its independence from the deploy gate.

``backend/rl/selection.py`` exists to fix §17's model-selection problem: validation
Q-loss selects a checkpoint that has not yet acquired two clinical reflexes the
deployed model has. The criterion is only worth anything if two things hold, and
this module pins both.

1. IT MEASURES WHAT IT CLAIMS TO. A hold-everything policy must score the floor,
   not a good score — the trap recorded in SUMMARY (*"clinician agreement rises as
   the policy gets WORSE"*) is exactly what a micro-averaged score would walk into,
   since clinicians hold 77.65% of the time.

2. IT IS NOT THE DEPLOY GATE. If the criterion could see the gate's 8-case battery,
   a checkpoint selected by it would pass the gate by construction and the gate
   would certify nothing. The last two tests make that structural rather than
   a matter of trust.

Run:  PYTHONPATH=. .venv/bin/python -m backend.tests.test_selection
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from backend.mdp import action_space
from backend.rl import selection as SEL

FEATS = ["PEEP", "TV", "FiO2", "SpO2", "PaO2", "PaCO2", "pH",
         "HR", "SBP", "RR", "RASS", "Temp"]
WEIGHT = 70.0

# One state per stratum, each built to fall in EXACTLY one stratum so a
# per-stratum compliance rate is unambiguous. Values differ from the battery's
# eight probes on purpose: the strata are ranges, and a fixture that reused the
# gate's points would test the points rather than the ranges.
_BASE = dict(PEEP=8, TV=455, FiO2=0.40, SpO2=94, PaO2=86, PaCO2=41, pH=7.39,
             HR=82, SBP=118, RR=15, RASS=-1, Temp=36.9)
CASES: dict[str, dict] = {
    "at_target":   {},
    "hypoxaemia":  dict(SpO2=85, PaO2=54, PaCO2=46, pH=7.33),
    "hyperoxia":   dict(SpO2=99, FiO2=0.65, PaO2=140),
    "hypercapnia": dict(PaCO2=61, pH=7.29, TV=305, SpO2=93, PaO2=71),
    "hypocapnia":  dict(PaCO2=29, pH=7.49, TV=540, SpO2=97, PaO2=115),
    "high_peep":   dict(PEEP=16, SpO2=93, PaO2=78),
    "low_peep":    dict(PEEP=4, FiO2=0.45, SpO2=93, PaO2=68),
    "volutrauma":  dict(TV=720, SpO2=95, PaO2=92, PaCO2=37, pH=7.43),
}
# The clinically-indicated action for each stratum, as (ΔPEEP, ΔTV, ΔFiO₂).
CORRECT = {
    "at_target": (0, 0, 0.0), "hypoxaemia": (2, 0, 0.0), "hyperoxia": (0, 0, -0.10),
    "hypercapnia": (0, 50, 0.0), "hypocapnia": (0, -50, 0.0), "high_peep": (-2, 0, 0.0),
    "low_peep": (2, 0, 0.0), "volutrauma": (0, -50, 0.0),
}
HOLD = action_space.encode_action(0, 0, 0.0)
PER_CASE = 60          # above MIN_STRATUM_N so every stratum counts


def _states(order: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """``PER_CASE`` rows per stratum, in ``CASES`` order. Returns (raw, weights, labels)."""
    order = order or list(CASES)
    rows, labels = [], []
    for name in order:
        st = {**_BASE, **CASES[name]}
        for _ in range(PER_CASE):
            rows.append([float(st[f]) for f in FEATS])
            labels.append(name)
    return np.array(rows), np.full(len(rows), WEIGHT), labels


class _FixedPolicy:
    """A stand-in for HybridIQL that recommends a prescribed action per row."""

    def __init__(self, actions: np.ndarray):
        self._actions = np.asarray(actions, dtype=int)

    def act_batch(self, states: np.ndarray) -> np.ndarray:
        assert len(states) == len(self._actions)
        return self._actions


def _actions_for(labels: list[str], mapping: dict[str, tuple]) -> np.ndarray:
    return np.array([action_space.encode_action(*mapping[l]) for l in labels], dtype=int)


# --------------------------------------------------------------------------- #
# 1. the score measures reflexes
# --------------------------------------------------------------------------- #
def test_each_stratum_is_populated_and_mutually_exclusive_in_the_fixture():
    raw, w, labels = _states()
    masks = SEL.strata_masks(raw, FEATS, w)
    lab = np.array(labels)
    for name in CASES:
        assert masks[name].sum() == PER_CASE, (
            f"stratum {name} matched {int(masks[name].sum())} rows, expected {PER_CASE}")
        assert set(lab[masks[name]]) == {name}, (
            f"stratum {name} also matched rows built for {set(lab[masks[name]]) - {name}}")
    print("PASS test_each_stratum_is_populated_and_mutually_exclusive_in_the_fixture")


def test_a_policy_with_every_reflex_scores_one():
    raw, w, labels = _states()
    r = SEL.clinical_reflex_score(_actions_for(labels, CORRECT), raw, FEATS, w)
    assert r["n_strata_counted"] == 8, r
    assert r["crs"] == 1.0, r["per_stratum"]
    print("PASS test_a_policy_with_every_reflex_scores_one")


def test_hold_everything_scores_the_floor_not_a_good_score():
    """The trap: holding is right 77.65% of the time, so it must NOT score well.

    Only ``at_target`` is satisfied by holding — every other stratum demands a
    strict direction — so the floor is exactly 1/8.
    """
    raw, w, labels = _states()
    r = SEL.clinical_reflex_score(np.full(len(labels), HOLD), raw, FEATS, w)
    assert r["crs"] == 0.125, r["per_stratum"]
    assert r["per_stratum"]["at_target"]["compliance"] == 1.0
    for name in CASES:
        if name != "at_target":
            assert r["per_stratum"][name]["compliance"] == 0.0, name
    print("PASS test_hold_everything_scores_the_floor_not_a_good_score")


def test_macro_average_does_not_let_one_large_stratum_dominate():
    """A policy right only on the biggest stratum must not out-score a balanced one.

    Mirrors the real split, where ``low_peep`` is ~50% of validation rows: a
    micro-average over transitions would hand that stratum the whole score.
    """
    order = ["low_peep"] * 20 + ["hypoxaemia", "high_peep", "volutrauma", "hyperoxia",
                                 "hypercapnia", "hypocapnia", "at_target"]
    raw, w, labels = _states(order)
    only_big = {k: (CORRECT[k] if k == "low_peep" else (0, 0, 0.0)) for k in CASES}
    r_big = SEL.clinical_reflex_score(_actions_for(labels, only_big), raw, FEATS, w)
    r_all = SEL.clinical_reflex_score(_actions_for(labels, CORRECT), raw, FEATS, w)
    assert r_big["crs"] < r_all["crs"], (r_big["crs"], r_all["crs"])
    # low_peep is 20/27 of the ROWS but only 1/8 of the SCORE, alongside at_target
    assert r_big["crs"] == 0.25, r_big["per_stratum"]
    print("PASS test_macro_average_does_not_let_one_large_stratum_dominate")


def test_a_stratum_below_the_minimum_n_is_reported_but_not_scored():
    raw, w, labels = _states(["at_target"] * 1 + ["hypoxaemia"] * 20)
    r = SEL.clinical_reflex_score(_actions_for(labels, CORRECT), raw, FEATS, w)
    assert r["per_stratum"]["at_target"]["n"] == PER_CASE
    assert r["per_stratum"]["at_target"]["counted"] is True
    # strata with no rows at all are present, uncounted, and have no compliance
    assert r["per_stratum"]["volutrauma"]["n"] == 0
    assert r["per_stratum"]["volutrauma"]["counted"] is False
    assert r["per_stratum"]["volutrauma"]["compliance"] is None
    assert r["n_strata_counted"] == 2, r["n_strata_counted"]
    print("PASS test_a_stratum_below_the_minimum_n_is_reported_but_not_scored")


def test_normalised_states_are_refused_rather_than_silently_scored():
    """Every threshold is in clinical units; z-scores must raise, not score.

    This is the dangerous case, and it was initially mis-specified in this file:
    z-scored states do NOT empty the strata. A normalised column has median ~0,
    which satisfies ``PEEP <= 5`` and ``PaCO2 < 32``, so ``low_peep`` and
    ``hypocapnia`` fill with nonsense and the macro-average returns a plausible
    0.125 that nothing downstream can tell is wrong. Hence a hard failure.
    """
    raw, w, labels = _states()
    z = (raw - raw.mean(axis=0)) / np.maximum(raw.std(axis=0), 1e-6)
    try:
        SEL.clinical_reflex_score(_actions_for(labels, CORRECT), z, FEATS, w)
    except ValueError as e:
        assert "RAW physiological states" in str(e), str(e)
        assert "SpO2" in str(e), "the message must name the offending column"
    else:
        raise AssertionError("z-scored states were scored instead of refused")
    print("PASS test_normalised_states_are_refused_rather_than_silently_scored")


# --------------------------------------------------------------------------- #
# 2. the guards disqualify rather than penalise
# --------------------------------------------------------------------------- #
def test_repertoire_collapse_disqualifies_however_good_the_reflexes():
    raw, w, labels = _states()
    acts = _actions_for(labels, CORRECT)
    r = SEL.score(_FixedPolicy(acts), raw, raw, FEATS, w, clinician_actions=acts)
    assert r["crs"] == 1.0
    assert r["n_distinct_actions"] == 6          # 8 strata, 6 distinct actions
    assert r["guards"]["repertoire"]["pass"] is False
    assert r["selectable"] is False, "a 6-action policy must not be selectable"
    print("PASS test_repertoire_collapse_disqualifies_however_good_the_reflexes")


def test_violating_safety_more_often_than_the_clinician_disqualifies():
    """The claim boundary is 'no worse than the clinician' — so this is a hard gate.

    The states sit at PEEP 14, where holding is safe and a +2 step crosses the
    15 cmH₂O ceiling. The policy keeps a wide repertoire (25 distinct actions, all
    with ΔPEEP +2) so the ONLY guard it fails is the safety one — otherwise this
    would not distinguish the safety guard from the repertoire guard.
    """
    n = 200
    st = {**_BASE, "PEEP": 14, "FiO2": 0.40, "TV": 455, "SpO2": 93, "PaO2": 78}
    raw = np.array([[float(st[f]) for f in FEATS]] * n)
    w = np.full(n, WEIGHT)
    reckless = np.array([action_space.encode_action(2, dt, df)
                         for dt, df in [(dt, df) for dt in (-50, -25, 0, 25, 50)
                                        for df in (-0.10, -0.05, 0.0, 0.05, 0.10)]]
                        * (n // 25), dtype=int)
    r = SEL.score(_FixedPolicy(reckless), raw, raw, FEATS, w,
                  clinician_actions=np.full(n, HOLD))
    g = r["guards"]["no_worse_than_clinician_on_safety"]
    assert g["clinician_any_violation"] == 0.0, g
    assert g["policy_any_violation"] == 1.0, g
    assert g["pass"] is False
    assert r["guards"]["repertoire"]["pass"] is True, r["guards"]["repertoire"]
    assert r["selectable"] is False
    print("PASS test_violating_safety_more_often_than_the_clinician_disqualifies")


def test_best_returns_none_when_every_candidate_is_disqualified():
    raw, w, labels = _states()
    acts = _actions_for(labels, CORRECT)
    cands = {"a": SEL.score(_FixedPolicy(acts), raw, raw, FEATS, w,
                            clinician_actions=acts)}
    label, why = SEL.best(cands)
    assert label is None
    assert "no candidate passed the guards" in why["reason"]
    assert why["disqualified"]["a"] == ["repertoire"]
    print("PASS test_best_returns_none_when_every_candidate_is_disqualified")


def test_best_prefers_higher_crs_and_breaks_ties_on_validation_loss():
    hi = {"crs": 0.52, "selectable": True, "val_q_loss": 1.90, "guards": {}}
    lo = {"crs": 0.40, "selectable": True, "val_q_loss": 1.40, "guards": {}}
    label, why = SEL.best({"hi": hi, "lo": lo})
    assert label == "hi", "CRS must outrank validation Q-loss, not the reverse"
    assert why["tie_broken_on"] is None

    tie_a = {"crs": 0.52, "selectable": True, "val_q_loss": 1.90, "guards": {}}
    tie_b = {"crs": 0.52, "selectable": True, "val_q_loss": 1.48, "guards": {}}
    label, why = SEL.best({"a": tie_a, "b": tie_b})
    assert label == "b", "among clinically indistinguishable checkpoints, take the better fit"
    assert why["tie_broken_on"] == "lower validation Q-loss"
    print("PASS test_best_prefers_higher_crs_and_breaks_ties_on_validation_loss")


# --------------------------------------------------------------------------- #
# 3. the criterion is NOT the deploy gate
# --------------------------------------------------------------------------- #
def test_the_criterion_cannot_see_the_deploy_gate():
    """Structural separation: no import of the gate, no battery in the signature.

    Selecting on the gate would make every selected checkpoint pass it by
    construction, and the gate would stop carrying information. There must be no
    channel at all, not merely an unused one.
    """
    src = Path(SEL.__file__).read_text()
    forbidden = ("verify_before_deploy", "BATTERY", "_battery_verdict",
                 "checkpoint_battery", "reward_critique")
    hits = [tok for tok in forbidden
            if any(tok in line for line in src.splitlines()
                   if line.strip().startswith(("import ", "from ")))]
    assert not hits, f"the criterion imports the deploy gate: {hits}"

    for fn in (SEL.score, SEL.clinical_reflex_score, SEL.best, SEL.strata_masks):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"battery", "gate", "battery_score"}, (
            f"{fn.__name__} takes a gate score: {params}")
    print("PASS test_the_criterion_cannot_see_the_deploy_gate")


def test_the_criterion_scores_only_states_its_caller_supplies():
    """No hidden data source: the score is a pure function of the arrays passed in.

    If the criterion loaded anything itself — the MDP, the gate's probes, a cached
    artifact — the trace in benchmark/selection_criterion.py could not be trusted
    to describe the validation split it was handed.
    """
    raw, w, labels = _states()
    acts = _actions_for(labels, CORRECT)
    full = SEL.clinical_reflex_score(acts, raw, FEATS, w)
    half = SEL.clinical_reflex_score(acts[: len(acts) // 2], raw[: len(raw) // 2],
                                     FEATS, w[: len(w) // 2])
    assert full["per_stratum"]["at_target"]["n"] == PER_CASE
    assert half["n_strata_counted"] < full["n_strata_counted"], (
        "halving the input did not change the strata — the score is reading "
        "something other than its arguments")
    print("PASS test_the_criterion_scores_only_states_its_caller_supplies")


# --------------------------------------------------------------------------- #
# 4. trainer integration: the default path must be untouched
# --------------------------------------------------------------------------- #
def test_the_default_criterion_is_still_validation_q_loss():
    """Every existing checkpoint, the deployed one included, was chosen this way.

    If this flips, training changes silently and the deployed artifact's provenance
    stops describing how a fresh run would behave.
    """
    from backend.rl import trainer as T

    assert T._selection_criterion({}) == "val_q_loss"
    assert T._selection_criterion({"selection": {}}) == "val_q_loss"
    assert T._selection_criterion({"selection": None}) == "val_q_loss"
    assert T._selection_criterion(
        {"selection": {"criterion": "clinical_reflex"}}) == "clinical_reflex"
    print("PASS test_the_default_criterion_is_still_validation_q_loss")


def test_an_unknown_criterion_is_refused_rather_than_ignored():
    from backend.rl import trainer as T

    for bad in ("battery", "fqe", "", "clinical-reflex"):
        try:
            T._selection_criterion({"selection": {"criterion": bad}})
        except ValueError as e:
            assert "unknown selection.criterion" in str(e)
        else:
            raise AssertionError(f"criterion {bad!r} was accepted")
    print("PASS test_an_unknown_criterion_is_refused_rather_than_ignored")


def test_the_gate_battery_is_not_an_allowed_criterion():
    """Selecting on the deploy gate would have it certify its own choice."""
    from backend.rl import trainer as T

    try:
        T._selection_criterion({"selection": {"criterion": "clinical_battery"}})
    except ValueError:
        pass
    else:
        raise AssertionError("the deploy gate was accepted as a selection criterion")
    print("PASS test_the_gate_battery_is_not_an_allowed_criterion")


def _tiny_dataset() -> dict:
    raw, w, _ = _states()
    return {"states": raw.astype(np.float32), "next_states": raw.astype(np.float32),
            "feature_order": list(FEATS), "weight_kg": w.astype(np.float32),
            "actions": np.full(len(raw), HOLD, dtype=np.int64),
            "rewards": np.zeros(len(raw), dtype=np.float32),
            "dones": np.zeros(len(raw), dtype=np.float32)}


def test_normalise_keeps_the_raw_states_only_when_asked():
    from backend.rl import trainer as T

    plain = T._normalise(_tiny_dataset(), "a")
    assert "raw_states" not in plain, (
        "the default path now allocates a second copy of the states it never uses")

    kept = T._normalise(_tiny_dataset(), "a", keep_raw=True)
    assert "raw_states" in kept
    # the stash is the CLINICAL-unit array, not another copy of the z-scores
    assert kept["raw_states"][0, FEATS.index("SpO2")] == _BASE["SpO2"]
    assert not np.allclose(kept["raw_states"], kept["states"])
    print("PASS test_normalise_keeps_the_raw_states_only_when_asked")


def test_the_clinical_criterion_refuses_to_run_without_raw_states():
    """The failure must be loud at setup, not a wrong selection 30,000 steps later."""
    from backend.rl import trainer as T

    d = T._normalise(_tiny_dataset(), "a")          # deliberately no keep_raw
    cfg = {"action_dim": 125, "hidden_dim": 16, "gamma": 0.99, "tau": 0.8, "beta": 2.0,
           "lr": 3e-4, "batch_size": 8, "total_steps": 1, "checkpoint_every": 500,
           "early_stop_patience": 500, "cql": {"alpha": 0.5},
           "selection": {"criterion": "clinical_reflex"}}
    idx = np.arange(len(d["states"]))
    try:
        T._run_training(d, idx, idx, cfg, 1, "cpu")
    except RuntimeError as e:
        assert "raw_states" in str(e) and "clinical units" in str(e), str(e)
    else:
        raise AssertionError("training ran with the wrong encoding for the criterion")
    print("PASS test_the_clinical_criterion_refuses_to_run_without_raw_states")


def test_a_selected_checkpoint_records_which_criterion_chose_it():
    """Self-describing checkpoints: a reader must never infer this from logs."""
    from backend.rl import trainer as T

    d = T._normalise(_tiny_dataset(), "a", keep_raw=True)
    idx = np.arange(len(d["states"]))
    cfg = {"action_dim": 125, "hidden_dim": 16, "gamma": 0.99, "tau": 0.8, "beta": 2.0,
           "lr": 3e-4, "batch_size": 8, "total_steps": 2, "checkpoint_every": 1,
           "early_stop_patience": 500, "cql": {"alpha": 0.5}}
    for criterion in ("val_q_loss", "clinical_reflex"):
        _, _, prov = T._run_training(d, idx, idx, {**cfg, "selection": {
            "criterion": criterion, "min_stratum_n": 1}}, 2, "cpu", seed=0)
        assert prov["selection_criterion"] == criterion, prov
        assert "selected_step" in prov
    print("PASS test_a_selected_checkpoint_records_which_criterion_chose_it")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    print("ALL TESTS PASSED" if failures == 0 else f"{failures} TEST(S) FAILED")
    raise SystemExit(1 if failures else 0)
