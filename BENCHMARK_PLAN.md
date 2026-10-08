# VentAssist vs IntelliLung — Benchmarking Plan

Companion to `BENCHMARK_HANDOFF.md` (which documents the reference system:
*Advancing Safe Mechanical Ventilation Using Offline RL With Hybrid Actions and Clinically
Aligned Rewards*, arXiv:2506.14375).

**All benchmark code lives in the top-level `benchmark/` package** — never in `backend/` or
`frontend/`. It imports from `backend/` read-only and writes to `benchmark/results/`. The deployed
system, its models, and its deploy gate are never modified by benchmark work.

---

## 0. Six traps that would silently invalidate the comparison

**1. FQE values are NOT comparable across different rewards.** V̂ is reward-relative. Their reward
is a priority-weighted in-range indicator + VFD (per-step, ≈[−1, +0.5]); ours is
Δoxygenation + Δventilation − action-cost − safety + causal bonus + terminal outcome (≈[−43, +42]).
"Our V̂ = 3.19 vs their V̂ = X" is meaningless.

> **The core design move: freeze ONE evaluation reward and score every policy under it, regardless
> of which reward it was TRAINED with.** Training reward is a *method choice*; evaluation reward is
> a *measuring stick*. This also yields a clean ablation isolating our reward's contribution.

**2. Their reward is action-independent — this is our biggest opening.**
`algo_src/reward/range.py` computes `reward += in_range(next_states) * priority` then `− 1`, where
`next_states` comes from `groupby(id).shift(-1)`. It is a function of **s′ only** and carries no
signal about *which action was taken*.

**Verified against their full repository on 2026-10-08, and it is stronger than first stated.** The
action is absent from their reward *interface*, not merely unused by the reward they chose:

- `algo_src/reward/base.py:7` — `def __call__(self, dataset, terminated, pre_process_configs,
  **kwargs)`. **No action parameter.** All five implementations inherit it unchanged.
- `grep -n action algo_src/reward/*.py` returns **nothing** — across `range.py`,
  `ventilator_free_days.py`, `mortality.py`, `stacking.py` and `base.py`.
- Their *configured* reward is the composite `AddRewards([RangeReward, VFDEachStep])`, so porting
  only `RangeReward` is not cherry-picking: `VFDEachStep` reads `pause_until_next`, `mv_duration`
  and `daemo_discharge` (all state/outcome columns), and `MortalityReward` reads
  `daemo_discharge` alone. **Both halves are action-blind.**

So their abstraction *cannot express* an action-dependent reward. That is a design-level claim, not
a parameter-choice one, and it is the form to use in the report. It is precisely the pathology
VentAssist diagnosed (SUMMARY §14.3: action-independent reward → flat Q across actions → policy
collapse), and our `action_causal_bonus` is the documented fix. **Demonstrable, not rhetorical** —
and now re-runnable: `benchmark/tests/test_reward_critique.py` asserts all of the above against
their source and skips when the repo is absent.

**3. The action spaces don't line up.** They emit **absolute** settings (VT, RR, PEEP, FiO₂, driving
pressure + discrete mode; hybrid). We emit **125 discrete deltas** over 3 knobs. Their policy
interface (`select_action(obs, deterministic=True)`) cannot wrap ours without a bridge.

*Asymmetry in our favour, verified 2026-10-08 and narrowed:* both experiment blocks in
`algo_src/configs/pre_processing_configs.yaml` carry exactly **26** `state_vector_columns`, and
those include `vent_pinsp` and `vent_vt_obs` but **not** `vent_peep` and **not** `vent_fio2` —
those two appear only under `action_space`. So the claim holds **for PEEP and FiO₂**, which is what
makes delta-actions on those two knobs ill-posed for them and well-posed for us. **Do not widen it
to all three knobs**: their state does carry observed tidal volume.

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

**6. Clinician agreement rises as the policy gets WORSE.** The deepest reporting trap we have
found, and it is ours rather than theirs. Clinicians hold **77.65 %** of the time, so any policy
that collapses toward the majority class scores *better* on agreement. Observed three independent
times:

| Case | Agreement moves | …while |
|---|---|---|
| IntelliLung reward vs ours | rises under *theirs* | action repertoire and clinical battery both collapse |
| IPW on vs off (5 seeds each) | 0.5322 → 0.5867 | FQE **2.4823 → 0.2382**, distinct actions **34.8 → 11.8** |
| Pressure-control action masking | confidence *rose* 0.671 → 0.687 | it rose *because* the policy's preferred action had been suppressed |

> **Never report agreement, likelihood or confidence without a value estimate and an
> action-diversity figure beside it.** This generalises IntelliLung's own Rule 5 from likelihood to
> agreement, and it is enforced in code: `benchmark/comparison_table._rule5` raises on a bare
> agreement figure, pinned by `benchmark/tests/test_comparison_table.py`.

---

## 1. Decisions (CONFIRMED 2026-07-15)

| Decision | Choice | Rationale |
|---|---|---|
| **Arena** | ✅ **Their contract** — retrain our method inside IntelliLung's state/action/reward/split and score with their frozen DistFQE | Most reviewer-proof: "we win in their arena." Requires their data pipeline + the action bridge (§2, Phase 2). |
| **Data scope** | ✅ **MIMIC-IV + eICU** (HiRID unavailable) | **Verified on disk:** MIMIC-IV (`files/mimiciv`, 80 GB) and **eICU v2.0** (`eicu-collaborative-research-database-2.0/`, 5.1 GB, all 31 tables). **HiRID is NOT present** — it is an ETH Zurich dataset requiring a separate application, not PhysioNet. |
| **Timestep** | ~~Their 4 h for the head-to-head~~ → **both are 1 h; no bridge needed** | **CORRECTED 2026-10-08.** Their MIMIC pipeline ships `RESOLUTION = 3600` (1 h) and that value *is* the decision timestep — it is consumed by `create_time_windows(..., resolution)` (`time_window_creation.py:170,247`) and their own comment at line 292 reads "during the 1h resolution time". The 4 h figure in their code is the drug/fluid **look-back** window (`drugs_vaso4h`, `state_ivfluid4h`), not the timestep. There is no resolution mismatch to reconcile, and no resolution advantage to claim. |
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

## 1b. RESULT — the reward critique is PROVEN (2026-07-15; multi-seeded 2026-10-08)

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

### Layer 2 — consequential, **5 SEEDS, PAIRED** (2026-10-08 — supersedes the single-seed table)

> **Why this was redone.** The original Layer 2 was one run, and worse: `layer2` accepted a `seed`,
> recorded `"seed": 0` in the artifact, and **never passed it to the trainer**, so the committed
> result came from the unseeded (non-reproducible) path and the recorded seed was false. The module
> had also been broken outright by a trainer signature change — it unpacked two values from
> `_run_training`, which returns three — so it could not have been re-run as it stood. Both fixed;
> arms are now **paired on seed**, meaning within a pair the weight initialisation and the
> minibatch stream are identical and *only the reward differs*. The paired delta therefore carries
> no between-seed variance.

**Scale-free metrics — these carry the contrast.** Counts and rates of *actions*, so a reward's
magnitude cannot inflate them. All four paired deltas exclude zero.

| | IntelliLung reward | VentAssist reward | paired Δ (ours − theirs) | Δ CI excl. 0 |
|---|---|---|---|---|
| Distinct actions used | 14.6 [11.4, 17.8] | **29.8 [27.4, 32.2]** | **15.2 [12.5, 17.9]** | ✅ |
| Hold share | 0.9467 [0.9402, 0.9531] | **0.6067 [0.5919, 0.6215]** | **-0.3400 [-0.3600, -0.3200]** | ✅ |
| Agreement w/ clinician | 0.7476 [0.7437, 0.7514] | **0.5138 [0.5022, 0.5255]** | **-0.2337 [-0.2492, -0.2183]** | ✅ |
| 8-case clinical battery (of 8) | 3.2 [1.8, 4.6] | **6.6 [5.9, 7.3]** | **3.4 [1.5, 5.3]** | ✅ |

**Scale-DEPENDENT metrics — NOT comparable across arms.** This is §0 trap 1 applied to Layer 2's
own metrics, which earlier drafts missed. Q-values scale with reward magnitude — theirs lands in
`[−1, 0]`, ours in roughly `[−43, +42]`, so anything denominated in Q units is ~85× larger for us
before a single thing is learned. **The plan's old "Q top-2 margin 0.268 vs 0.453, 1.7× sharper"
headline is not scale-invariant and must not be used.**

| | IntelliLung reward | VentAssist reward |
|---|---|---|
| Q top-2 margin | 0.3538 [0.3112, 0.3965] | 0.4115 [0.3836, 0.4394] |
| Q spread (max − min) | 9.1157 [8.2093, 10.0222] | 14.9662 [14.6145, 15.3179] |
| Best validation Q-loss | 0.3394 [0.3155, 0.3632] | 1.4802 [1.4740, 1.4864] |

### The clinical battery, per case (fraction of 5 seeds passing)

| Case | IntelliLung reward | VentAssist reward |
|---|---|---|
| `stable` | 100% | 100% |
| `hypoxaemic` | 0% | 60% |
| `hyperoxic` | 80% | 100% |
| `hypercapnic` | 20% | 100% |
| `hypocapnic` | 0% | 100% |
| `high_peep` | 0% | 0% |
| `low_peep` | 80% | 100% |
| `volutrauma` | 40% | 100% |

### Honest negatives — state these before a reviewer finds them

- **"8/8" does not survive seeding.** Our reward scores **6.6/8**
  [5.9, 7.3]. The old 8/8 was a lucky seed. Quote the per-case rates.
- **`high_peep` is fixed by neither reward — 0 of 5 seeds, both arms.** An over-distending PEEP of
  18 is left untouched *under our reward too*. The old single-seed table claimed ↓PEEP −2 ✅ here;
  that was seed luck, and the claim must be struck. This is the most important correction in this
  section: it was being used as a headline example.
- **`hypoxaemic` passes on only 3 of 5 seeds (60 %)** under our reward. Also previously claimed as
  a clean ✅.
- Four cases *are* clean and reproducible under our reward and broken under theirs:
  `hypercapnic` (100 % vs 20 %), `hypocapnic` (100 % vs 0 %), `volutrauma` (100 % vs 40 %),
  `low_peep` (100 % vs 80 %).

### What survives, stated at full strength

The **structural** result (Layer 1) is untouched and remains the strongest claim: their reward is
action-invariant for 100 % of transitions, by construction. The **consequential** result is now
*better* evidenced than before, because it has intervals: under their reward the policy collapses
to holding **94.7%** of the time using **14.6**
of 125 actions, against **60.7%** and
**29.8** under ours.

And it reproduces §0 trap 6 cleanly: **their reward scores
0.7476 on clinician agreement against our
0.5138** — a 0.234
*advantage* for the reward that uses half the action repertoire and fails twice as many clinical
cases. Agreement, read alone, would have picked the collapsed policy.

*Pinned by `benchmark/tests/test_reward_critique.py` (which now also checks the claims
against **their** source and skips when the reference repo is absent); raw output in
`benchmark/results/reward_critique_layer1.json` and
`benchmark/results/reward_critique_layer2_multiseed.json`. The single-seed
`reward_critique_layer2.json` is retained but **superseded**.*

---

## 2. Phases

### Phase 0 — Scope & de-risk *(STILL BLOCKED — reassessed 2026-10-08 with the full repo on disk)*

Their complete repository is now present at `intellilung-advancing-mechanical-ventilation/`
(17 MB, untracked). **Everything lives under `algo_src/`**, so every path quoted in older drafts
of this document needs that prefix:

| Older draft says | Actually at |
|---|---|
| `reward/range.py` | `algo_src/reward/range.py` |
| `actions/masking.py` | `algo_src/actions/masking.py` |
| `hyper_param_tune/run_experiment.py` | `algo_src/hyper_param_tune/run_experiment.py` |
| `configs/pre_processing_configs.yaml` | `algo_src/configs/pre_processing_configs.yaml` |
| `configs/dist_fqe_config.yml` | `algo_src/configs/mimiciv_7_action_setup_w_hist/dist_fqe_config.yml` |

Their imports are bare (`from agents.configs import ...`), so `algo_src/` must be on `PYTHONPATH`.
Their MIMIC pipeline is `data_pipelines/MIMIC/` (`sql/`, `vectorisation/`,
`data_extraction_mimic_iv.py`, `data_processing_mimic_iv.py`); HiRID and eICU pipelines also ship.

**Having the repository did NOT unblock Phase 0.** Three findings, all verified on disk:

1. **They ship no weights and no data.** A search for `*.pt`, `*.pth`, `*.ckpt`, `*.pkl`,
   `*.parquet`, `*.csv` across the whole repository returns **zero files**. So *"score with their
   frozen DistFQE"* — the move that made the arena reviewer-proof — **is not available**. Their
   DistFQE is a config (`dist_fqe_config.yml`: 80,000 steps, 50 atoms, `[256,256,256]`), not an
   artifact. A DistFQE *we* train is our evaluator with their hyperparameters, which is a
   materially weaker claim and must not be described as theirs.
2. **Their pipeline needs a populated Postgres database.** `data_pipelines/MIMIC/.env.example`
   wants `MIMIC_DBNAME = "mimic"`, `POSTGRES_USER`, `POSTGRES_PASS`, and the SQL under
   `sql/base/` and `sql/templates/` queries `mimiciv_chartevents` / `inputevents` /
   `outputevents` / `datetimeevents` / `ingredientevents`. Postgres 15 is installed and running
   locally but **has no `mimic` database**, and loading MIMIC-IV 3.1 needs ~100+ GB on top of the
   80 GB of CSVs already in `files/mimiciv/3.1/`. **There is 7.4 GiB free on a 460 GiB volume at
   99 % capacity.** This is a hard stop, not a scheduling problem.
3. **Their environment is CUDA-shaped but survivably so.** `environment.yml` pins
   `cuda-version=11.8` and `cuml=25.06` (RAPIDS, NVIDIA-only) and every `examples.env` block sets
   `DEVICE=cuda`. However `cuml`/`cupy` appear in exactly **one** file,
   `algo_src/hyper_param_tune/hparam_submit.py` (a SLURM submitter), so the core algorithm code
   honours `DEVICE=cpu`. `lejepa` installs from a git URL. Setup is a day's work, not a blocker.

**To unblock, in order:** free ~150 GB → load MIMIC-IV into Postgres → run their four-stage
pipeline → train their four baselines → train a DistFQE. That is multi-session work, and the
headline claim it buys is weaker than originally planned because the evaluator cannot be theirs.

Still to do if it is ever attempted — patch their shipped bugs first: PART 3 #2
(`algo_src/hyper_param_tune/run_experiment.py` imports trainers that don't exist → `ImportError`),
#3 (stale `factored_cql_config.yml` path in `algo_src/examples.env`). And note PART 3 #4:
`state_encoder_path` is **disabled inside FQE** — keep it disabled for us too, or values aren't
comparable to published ones.

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
* **The deployed checkpoint underperforms a retrain by more than seed noise.**
  Deployed `policy_track_a.pt` scores FQE **1.8787**; five fresh seeds under an
  identical budget score **2.4823, CI95 [2.3813, 2.5833]** — the deployed value
  sits 0.50 *below* the lower bound. Same encoding, same 198,050 test
  transitions, so this is a valid comparison. It is the measured cost of the
  `state_dict()` aliasing bug (`best_sd = model.state_dict()` returned
  references to the live weights, so early stopping could never restore the best
  checkpoint, and the deployed artifact is the step-27,000 weights). Evidence for
  retraining — **not** a claim the deployed model is unsafe; it still passes 8/8.

  > **QUALIFIED 2026-10-08.** This holds against the **coarse** arm only. Retrained
  > with the corrected `checkpoint_every: 500`, the 5-seed FQE is **2.0016
  > [1.3593, 2.6439]** and the deployed **1.8787 sits INSIDE** that interval. The
  > clinical battery runs the other way entirely — 6.60/8 at the fine arm's step
  > ~5,750 against **8.00/8** for the deployed model, because `high_peep` is a
  > safety reflex acquired late. **The case for replacing the deployed artifact is
  > not made; do not deploy.** See SUMMARY §15.12 F and
  > `benchmark/results/checkpoint_battery_track_a.json`.

* **The checkpoint interval WAS hiding a better model — confirmed and fixed.**
  All five seeds selected step **10,000**, the *first* checkpoint evaluated, then
  early-stopped near 30,000: the signature of an interval too coarse to locate
  the minimum rather than of a genuine optimum. Re-running all five seeds at
  `checkpoint_every: 500` puts every one of them at step **5,500–6,000**, and
  lowers best validation Q-loss from **1.5340 [1.5135, 1.5545]** to **1.4802
  [1.4740, 1.4864]** — non-overlapping CIs. Step ~6,000 is exactly the
  best-validation step the handoff reported for the deployed model, reached here
  independently. `backend/configs/track_a_config.yaml` is now
  `checkpoint_every: 500`; `early_stop_patience` needed no change because
  patience is counted in steps (`stale += ckpt_every`), not in checkpoints.
  Track B already used 250 and was never affected. Evidence:
  `benchmark/results/checkpoint_granularity_track_a.json`.
* **The canonical `action_density` artifact was a reduced test run.** Both
  `action_density.evaluate` and `safety_metrics.evaluate` wrote unconditionally
  and `test_action_density` called the first with `epochs=8` against a default of
  15, so the artifact the Rule 5 likelihood pair is quoted from was last written
  by the test suite. Regenerated at 15 epochs the Δ is **+0.0226**, not the
  +0.0519 that was committed — same sign (the policy is on-support, which is what
  Rule 5 asks) but **the margin was overstated by more than 2×**. Both entry
  points now take `write`, their tests pass `write=False`, `action_density`
  records `epochs`, and `benchmark/tests/test_artifacts_not_clobbered.py` guards
  it. Ground rule 4 of this document should be read as requiring a `write` flag
  on every entry point that persists a result.
* **The lung-protective deviation survives seeds.** Mean ΔTV **−7.45 mL**, CI95
  **[−8.40, −6.50]** — entirely below zero. The policy systematically asks for
  lower tidal volumes than the clinician chose, and that is not seed noise.
* **IPW makes the policy worse, so `enabled: false` is now an evidenced
  decision.** Five seeds with IPW on, same seeds and same budget: FQE collapses
  **2.4823 [2.3813, 2.5833] → 0.2382 [0.0696, 0.4069]**, the action repertoire
  shrinks **34.8 → 11.8** distinct actions, and validation loss rises **1.5340 →
  1.8761** — all three separated. The one metric that *improves*,
  `behaviour_match` 0.5322 → 0.5867, is a trap: hold share rises in step
  (0.6309 → 0.7027, toward the clinicians' 0.7765) while diversity collapses, so
  the policy agrees more by mimicking the majority class. Agreement must never be
  read without value and diversity beside it. Note the scope: this is IPW *as
  implemented* (weights clipped to [0.1, 8.63], mean 1.000, p99 1.657 — a mild
  reweighting with a severe effect), and it addresses **measured** confounding
  only, so it leaves the unmeasured problem untouched either way.

### Phase 2 — Build the common arena *(BLOCKED — see Phase 0, reassessed 2026-10-08)*

> **Not attemptable as written.** "Their frozen DistFQE" does not exist as an artifact — they ship
> no weights at all — so the evaluator would have to be *ours, with their hyperparameters*. That is
> a materially weaker claim and must never be described as theirs. Combined with the ~150 GB of
> disk needed to stand up their Postgres MIMIC build, this phase is multi-session work whose
> headline payoff has shrunk. **The achievable substitute, already done, is Phase 4's Table A:
> their reward, ported faithfully and verified against their source, trained and scored inside our
> MDP.** That isolates the reward contribution — the scientifically valuable half of Phase 3's
> ablation — without needing their data or their weights.

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

### Phase 4 — The table & the claim *(BUILT 2026-10-08 — `benchmark/comparison_table.py`)*
Their PART 2 table **plus our safety rows**. The claim to defend is *not* "higher FQE value":

> **Equal-or-higher FQE value, at equal-or-better action likelihood under the clinician density
> model, with lower safety-violation rates — replicated across seeds.**

`benchmark/comparison_table.py` assembles it from the artifacts in `benchmark/results/` and
computes nothing itself; every cell carries the artifact and timestamp it came from, so a stale
input shows up in the output. Outputs `comparison_table.md` and `comparison_table.json`. Four
tables: **A** the reward head-to-head, **B** the agreement trap (§0 trap 6), **C** VentAssist vs the
clinician (the Rule 5 triple), **D** the capability comparison, each claim citing the file in their
repository that was read to check it — including the one that **failed** its check and is marked
retracted rather than deleted.

Two disciplines are enforced in code rather than by convention, because both are mistakes a careful
person still makes under deadline:

1. `_assert_single_policy` rejects any row that does not name the policy it describes, so the
   **deployed** checkpoint (`backend/logs/fqe_track_a.json`) can never be tabulated alongside
   **fresh seeds** (`runner_track_*.json`) as though they were one system.
2. `_rule5` raises on an agreement figure offered without a value estimate or an action-diversity
   figure (§0 trap 6).

### Phase 5 *(stretch)* — eICU + HiRID for external generalization (needs credentialing).

---

## 3. Our differentiators (what we actually claim)

1. **Action-causal reward** fixing their action-independent `RangeReward` *(demonstrable)*
2. **Safety violation metrics + enforced safety filter + hard deploy gate** — they have none
3. **Mode-aware action masking actually ON** (theirs ships `vent_mode_action_masking: false`)
4. ~~**1-hour decision resolution** vs their 4-hour~~ — **RETRACTED 2026-10-08.** Their
   MIMIC build is also 1 h (`RESOLUTION = 3600`); see the corrected §1 row. Kept here,
   struck, so the claim is not re-made from an older draft.
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
