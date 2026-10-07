# VentAssist vs IntelliLung — Benchmarking Plan

Companion to `BENCHMARK_HANDOFF.md` (which documents the reference system:
*Advancing Safe Mechanical Ventilation Using Offline RL With Hybrid Actions and Clinically
Aligned Rewards*, arXiv:2506.14375).

**All benchmark code lives in the top-level `benchmark/` package** — never in `backend/` or
`frontend/`. It imports from `backend/` read-only and writes to `benchmark/results/`. The deployed
system, its models, and its deploy gate are never modified by benchmark work.

---

## 0. Five traps that would silently invalidate the comparison

**1. FQE values are NOT comparable across different rewards.** V̂ is reward-relative. Their reward
is a priority-weighted in-range indicator + VFD (per-step, ≈[−1, +0.5]); ours is
Δoxygenation + Δventilation − action-cost − safety + causal bonus + terminal outcome (≈[−43, +42]).
"Our V̂ = 3.19 vs their V̂ = X" is meaningless.

> **The core design move: freeze ONE evaluation reward and score every policy under it, regardless
> of which reward it was TRAINED with.** Training reward is a *method choice*; evaluation reward is
> a *measuring stick*. This also yields a clean ablation isolating our reward's contribution.

**2. Their reward is action-independent — this is our biggest opening.** `reward/range.py` computes
`reward += in_range(next_state) * priority` then `− 1`. It is a function of **s′ only** and carries
no signal about *which action was taken*. That is precisely the pathology VentAssist diagnosed
(SUMMARY §14.3: action-independent reward → flat Q across actions → policy collapse), and our
`action_causal_bonus` is the documented fix. **Demonstrable, not rhetorical.**

**3. The action spaces don't line up.** They emit **absolute** settings (VT, RR, PEEP, FiO₂, driving
pressure + discrete mode; hybrid). We emit **125 discrete deltas** over 3 knobs. Their policy
interface (`select_action(obs, deterministic=True)`) cannot wrap ours without a bridge.
*(Asymmetry in our favour: their 26-dim state doesn't even contain the current PEEP/FiO₂ settings —
ours does, which is what makes delta-actions well-posed.)*

**4. Naming collision.** Their **"Hybrid-IQL"** = hybrid *action space* (continuous + discrete). Our
**"HybridIQL"** = discrete IQL + CQL + a feature adapter. A reviewer will assume we reimplemented
theirs. **Rename ours before publication** (e.g. `VA-IQL` / `CQL-IQL`).

**5. Gaps on OUR side that block their headline metric.**
- **No action-conditional density `p(a|s)`** — we only have a *state* OOD autoencoder
  (`backend/rl/ood_autoencoder.py`). Their Rule-5 metric pair *(FQE value, action log-likelihood)*
  is currently **not computable for us**. → Phase 1.
- **Split is not stratified** (theirs: episode-length quartile × discharge outcome, stay-exclusive,
  seed 42). Ours is patient-level but unstratified → different test difficulty.
- **Single runs, no seeds/CI.** Single-run offline-RL numbers are noise.
- Safety filter emits per-request *flags*, but **no violation-rate metric**.

---

## 1. Decisions (CONFIRMED 2026-07-15)

| Decision | Choice | Rationale |
|---|---|---|
| **Arena** | ✅ **Their contract** — retrain our method inside IntelliLung's state/action/reward/split and score with their frozen DistFQE | Most reviewer-proof: "we win in their arena." Requires their data pipeline + the action bridge (§2, Phase 2). |
| **Data scope** | ✅ **MIMIC-IV + eICU** (HiRID unavailable) | **Verified on disk:** MIMIC-IV (`files/mimiciv`, 80 GB) and **eICU v2.0** (`eicu-collaborative-research-database-2.0/`, 5.1 GB, all 31 tables). **HiRID is NOT present** — it is an ETH Zurich dataset requiring a separate application, not PhysioNet. |
| **Timestep** | **Their 4 h for the head-to-head** | Must match. Our 1 h resolution is reported separately as an advantage, not smuggled into the comparison. |
| **Next work** | ✅ **Prove the reward critique** (§0 trap 2) | Headline scientific claim; needs no new data; de-risks the thesis early. |

> ### ⚠️ Consequence of the missing HiRID source
> Their published numbers come from a **MIMIC + eICU + HiRID** model. We can build only
> **MIMIC + eICU**. Therefore we **do not compare against their published table** — we
> **re-train their baselines ourselves on our exact MIMIC+eICU contract** and compare there.
> This is what Rules 1 and 7 demand anyway (same frozen split, equal hyperparameter budget), and
> `run_e2e_experiment.py` is designed for exactly this. State the deviation explicitly in the paper:
> *"all baselines re-trained on a MIMIC-IV + eICU build of the reference contract; HiRID
> unavailable, so numbers are internally comparable but not directly comparable to the published
> table."*

---

## 1b. RESULT — the reward critique is PROVEN (2026-07-15)

`benchmark/reward_critique.py`, on our 992,100-transition MIMIC-IV MDP. Two layers.

### Layer 1 — structural (irrefutable, no training)
Hold the observed transition `(s, s′)` fixed; recompute the reward under all 125 actions.

| Reward | mean spread across actions | max spread | transitions with **zero** spread |
|---|---|---|---|
| **IntelliLung `RangeReward`** | **0.000000** | **0.000000** | **100.0 %** |
| VentAssist reward | 2.0250 | 4.4000 | 0.0 % |

> **IntelliLung's reward assigns the identical value to all 125 actions, for 100 % of transitions.**
> It carries *zero* action signal — the only path from action to reward is through the (confounded)
> transition itself. This is structural, not incidental: `reward/range.py` is literally a function
> of `next_states` alone.

### Layer 2 — consequential (same architecture, same MDP, ONLY the reward differs)

| | IntelliLung reward | VentAssist reward |
|---|---|---|
| Q top-2 margin (decision sharpness) | 0.268 | **0.453** (1.7× sharper) |
| Distinct actions used | 23 | **49** |
| Hold-share | 0.86 (near-degenerate) | 0.69 |
| Agreement w/ clinician | — | — |
| **8-case clinical battery** | **3 / 8** | **8 / 8** |

Under their reward the policy **does nothing** in the cases that matter most:

| Case | Expected | IntelliLung reward | VentAssist reward |
|---|---|---|---|
| hypoxaemic (SpO₂ 84) | raise PEEP/FiO₂ | **HOLD (0,0,0)** ❌ | ↑PEEP +2 ✅ |
| high PEEP 18 (safety) | lower PEEP | **HOLD (0,0,0)** ❌ | ↓PEEP −2 ✅ |
| hypercapnic (PaCO₂ 65) | raise TV | **HOLD (0,0,0)** ❌ | ↑TV +25 ✅ |
| hypocapnic | lower TV | ↓PEEP −2 ❌ | ↓TV −50 ✅ |
| low PEEP 2 | raise PEEP | ↑FiO₂ +0.10 ❌ | ↑PEEP +2 ✅ |

**A hypoxaemic patient at SpO₂ 84 gets no intervention, and an over-distending PEEP of 18 is left
untouched.** This is precisely the collapse VentAssist diagnosed in SUMMARY §14.3, reproduced by
swapping in their reward and changing nothing else.

*Pinned by `benchmark/tests/test_reward_critique.py`; raw output in
`benchmark/results/reward_critique_layer{1,2}.json`.*

---

## 2. Phases

### Phase 0 — Scope & de-risk *(blocking, external)*
- Build **their** MIMIC-IV dataset with `data_pipelines/MIMIC/` (their contract, their 26-dim state).
- Patch their shipped bugs first: PART 3 #2 (`hyper_param_tune/run_experiment.py` imports trainers
  that don't exist → `ImportError`), #3 (stale `factored_cql_config.yml` path in `examples.env`).
- Note PART 3 #4: `state_encoder_path` is **disabled inside FQE** — keep it disabled for us too, or
  values aren't comparable to published ones.

### Phase 1 — Adopt their evaluation rigor *(in `benchmark/`, do first)*

**STATUS: all 7 items implemented (2026-10-07).** Items 1–2 landed earlier; 3–6 and
the item-7 reporting discipline landed together. Results in `benchmark/results/`.

| # | Item | Why | Module | Status |
|---|---|---|---|---|
| 1 | **Action-conditional density `p(a|s)`** + action log-likelihood | Their Rule 5 gate; blocks their headline metric pair. *Easier for us*: our action space is fully discrete → a 125-way categorical, not Gaussian+categorical | `benchmark/action_density.py` | ✅ |
| 2 | **Safety violation rates vs clinician** | Their Rule 9 — **the reference repo does NOT have these**, and they call it "the strongest differentiator a successor system can claim." We already own the logic in `backend/router/safety_filter.py` | `benchmark/safety_metrics.py` | ✅ |
| 3 | **Stratified + frozen split** (episode-length quartile × outcome, persisted CSVs) | Test-set difficulty must match | `benchmark/split_compat.py` | ✅ |
| 4 | **Seeds + CI** (N=5, mean ± CI on every headline number) | Single-run numbers are noise | `benchmark/runner.py` | ✅ |
| 5 | **Equal hyperparameter budget** for every method | Otherwise the comparison is rigged | `benchmark/runner.py` | ✅ |
| 6 | **AI-vs-clinician behavioural analyses**: agreement, per-setting deviation, **churn** | Clinically legible secondary metrics | `benchmark/behaviour_compare.py` | ✅ |
| 7 | **Never report FQE value alone** — always the triple `(value, action-likelihood, safety-violation rate)` | Their Rule 5: an FQE gain with a likelihood drop is *extrapolation*, not improvement | reporting | ✅ partial |

**On item 7's "partial".** `runner.py` reports value and the safety rate per seed,
so two legs of the triple carry a CI. The clinician **action log-likelihood** is
not per-seed by design: it is a property of the *behaviour* policy, so it is
identical across our training seeds and is computed once in
`benchmark/action_density.py`. The triple is therefore assembled from two
artifacts rather than one — stated here so nobody reads the omission as an
oversight.

**What Phase 1 turned up that was not on the list.**

* `backend/rl/trainer.py` pinned the minibatch stream to `config.SPLIT_SEED` but
  never called `torch.manual_seed`, so weight initialisation varied run to run.
  Training looked deterministic and was not: **the deployed `policy_track_a.pt`
  is not bit-reproducible from the code that made it.** The trainer now takes an
  explicit `seed` (default `None` = the historical path, so the deploy gate is
  unaffected), and the checkpoint records it — `seed: null` marks a checkpoint
  from before seeding existed.
* The shipped hash split is **already balanced**: its test fold matches the
  population within 0.75pp in every one of the 8 strata, and its mortality rate
  is 0.2191 against a population 0.2130. The single-draw objection is therefore
  answered with evidence rather than by appealing to the hash, and the stratified
  split (max deviation 0.0001) is a belt-and-braces artifact, not a correction.
* Confidence is **well calibrated against clinician agreement** (ECE 0.041,
  Spearman +0.289) up to about 0.74, then **inverts sharply**: the top bin
  (confidence ~0.81, n=192 of 198,050) agrees with the clinician only 11% of the
  time. It carries **no mortality signal** (Spearman +0.040), which is the
  correct result for a decision confidence and settles how the SRS must
  describe it.

### Phase 2 — Build the common arena
Freeze: their split · their 26-dim state · **their evaluation reward** · their `dist_fqe_config.yml`
(50 atoms) · their action-density model.

**Action bridge (shared-action-subspace).** Compare only on the **3 shared knobs (PEEP, VT, FiO₂)**.
Convert our deltas → absolute settings (`current + Δ`). For **every** policy, pin the non-shared
knobs (RR, driving pressure, mode) to the **clinician's recorded value**, so the action vector fed to
FQE is complete and the *only* difference is the shared three. Document this explicitly in the paper.

### Phase 3 — Head-to-head runs
All scored with the **same** DistFQE under the **same** evaluation reward, on the same frozen split:

| Policy | Trained with | Isolates |
|---|---|---|
| **Clinician** | — | the denominator (FQE **and** empirical MC return — their Rule 4 sanity check) |
| IntelliLung Hybrid-IQL / Discrete-IQL / CQL / CF-CQL | their reward | reference baselines |
| **VentAssist method, *their* reward** | their reward | **our algorithm** |
| **VentAssist method, *our* reward** (causal + outcome) | our reward | **our reward contribution** |

The last pair is the scientifically valuable ablation — only possible because of the frozen evaluator.

### Phase 4 — The table & the claim
Their PART 2 table **plus our safety rows**. The claim to defend is *not* "higher FQE value":

> **Equal-or-higher FQE value, at equal-or-better action likelihood under the clinician density
> model, with lower safety-violation rates — replicated across seeds.**

### Phase 5 *(stretch)* — eICU + HiRID for external generalization (needs credentialing).

---

## 3. Our differentiators (what we actually claim)

1. **Action-causal reward** fixing their action-independent `RangeReward` *(demonstrable)*
2. **Safety violation metrics + enforced safety filter + hard deploy gate** — they have none
3. **Mode-aware action masking actually ON** (theirs ships `vent_mode_action_masking: false`)
4. **1-hour decision resolution** vs their 4-hour
5. **Learned OOD gate + confidence tempering**, with an honest calibration caveat
6. **Interpretability**: SHAP + decision-tree distillation
7. **GP (LMC) blood-gas imputation**

---

## 4. Ground rules for benchmark code

- Lives in **`benchmark/`** only. Imports `backend/` read-only; **never** modifies the deployed
  model, `backend/models/policy_track_a.pt`, or the deploy gate.
- Results → `benchmark/results/*.json`, each carrying `n_transitions` + `timestamp` so staleness is
  detectable (same convention as `backend/logs/`).
- Tests → `benchmark/tests/test_*.py`, run as `python -m benchmark.tests.<name>` (no pytest, `Skip`
  pattern for model-dependent tests — same convention as `backend/tests/`).
- Run with `PYTHONPATH=. .venv/bin/python -m benchmark.<module>`.
