# VentAssist

Clinical decision support for mechanical ventilation. An offline reinforcement learning policy
trained on MIMIC-IV reads a ventilated patient's hourly state and recommends a change to three
settings: PEEP, tidal volume, and FiO₂.

## What it does

Every hour of a ventilation course is one decision point. The policy sees 12 clinical values and
picks one of 125 discrete actions, each a triple of deltas:

| Setting | Options |
| --- | --- |
| ΔPEEP | −2, −1, 0, +1, +2 cmH₂O |
| ΔTidal volume | −50, −25, 0, +25, +50 mL |
| ΔFiO₂ | −0.10, −0.05, 0, +0.05, +0.10 |

Every recommendation also carries a confidence derived from the policy's Q-value margin, the top-3
alternative actions with their margins, a safety check on the resulting settings, and the three
features that mattered most.

## Two tracks

**Track A** is the system. 12-dim clinical state, 992,100 transitions from 23,955 ICU stays.
This is what is trained, validated, gated and served.

**Track B** adds 6 waveform features (HRV, arrhythmia rate, perfusion index, respiratory-rate
variability, breathing regularity, asynchrony) extracted from ECG, Pleth and Resp signals. It is a
proof of concept: 895 transitions from 35 stays, warm-started from Track A.


## What the evidence says

```text
VALUE (reward-relative, gamma = 0.99)
  clinician (empirical MC)    -1.3034   CI95 [-1.5077, -1.0986]   n = 4,792 episodes
  policy NWE (conservative)   -0.4141
  policy FQE                  +1.8787
  policy DFQE                 +3.1886   CI95 [3.0490, 3.3007]     LCB(5%) +1.0988

SAFETY (400 starts x T=12, same held-out states)
  % ending SpO2 >= 95          0.99 vs 0.87
  mean dSpO2, hypoxaemic      +5.31 vs +3.30   (n = 44)
  % aggressive settings        0.011 vs 0.065

AGREEMENT (198,050 held-out transitions)
  exact 125-way                0.5699
  per-knob                     PEEP 0.897   FiO2 0.881   TV 0.695
  hold share                   0.689 vs clinicians' 0.777
  churn                        0.147 vs clinicians' 0.407
  dTV bias                     -6.05 mL   (deployed model, serving encoding)

MULTI-SEED (5 fresh seeds, identical 100,000-step budget, training encoding)
  fqe_V_hat                    2.4823   CI95 [2.3813, 2.5833]
  behaviour_match              0.5322   CI95 [0.4984, 0.5660]
  dTV bias                    -7.4493   CI95 [-8.4015, -6.4971]   excludes zero
```

> The MULTI-SEED block describes **five freshly trained policies**, not the deployed checkpoint, and
> at the training encoding rather than the serving one. The blocks above it describe the **deployed**
> `policy_track_a.pt`. The two are different policies and their numbers must not be combined — for
> example the dTV bias appears in both blocks with different values (-6.05 vs -7.45) precisely
> because they measure different things.

**The defensible claim is "no worse than the clinician."** Not "better than doctors." Every value
number above is computed under VentAssist's own reward and its own learned dynamics model, on
retrospective data with unmeasured confounding, with no prospective validation.

Two results do stand on their own, because they are model-free: the policy asks for **lower tidal
volumes** than clinicians did (the lung-protective direction, and the interval excludes zero across
seeds), and it is **2.8× more stable** hour to hour with far fewer direction reversals.

## Running it

Requires Python 3.11, Node 18+, and Docker (for PostgreSQL on host port 5433 — 5432 is usually
taken by a native install).

```bash
# one-time
python3.11 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
cd frontend && npm install && cd ..

# every time
./run.sh --all        # Docker + postgres + backend (8000) + frontend (5173)
./run.sh --stop       # then quit Docker Desktop if the laptop is getting hot

curl -s http://127.0.0.1:8000/api/health    # want "database": true
```

Open <http://localhost:5173>. Trained models ship in `backend/models/`, so recommendations work
without retraining.

### Uploading a patient

Three input types, one folder each under `frontend/public/samples/`:

| Folder | Upload | Result |
| --- | --- | --- |
| `1-tier-a-clinical-only/` | one patient `.txt` | Track A |
| `2-tier-b-waveform-txt/` | patient `.txt` + waveform `.txt` together | Track B, features extracted from the signal |
| `3-tier-b-wfdb-record/` | patient `.txt` + `.hea` + `.dat` together | Track B, features extracted from the WFDB record |

Each folder has a README with the patient, the expected recommendation and the expected safety
flags. The three folders use three different patients, so the answers differ.

A patient file never contains waveform features — the recording supplies them. Features a recording
does not yield are imputed from the clinical state and reported as imputed. Do not expect the
dashboard to invent numbers for a patient with no recording; the fields read "not measured".

Don't rename the `.hea`/`.dat` — the header stores its own filenames. Re-export with
`backend/scripts/export_waveform.py` instead.

## Tests and the deploy gate

```bash
PYTHONPATH=. .venv/bin/python -m pytest backend/tests -q           # 222 passed, 1 skipped
PYTHONPATH=. .venv/bin/python -m backend.scripts.verify_before_deploy
```

The gate runs 8 clinically-specified cases (hypoxaemia should raise PEEP, volutrauma should cut
tidal volume, a stable patient should hold, and so on) and prints "Safe to deploy" only if all 8
pass. **Run it before replacing anything in `backend/models/`.** The one skipped test is an
optional cross-check against the `shap` package, which is not installed.

Retraining and benchmarking commands are in `SUMMARY.md` §12.

## Layout

```text
backend/pipeline/        MIMIC-IV -> cohort -> GP blood-gas imputation -> 12-dim states
backend/waveform/        WFDB signals -> the 6 Track B features
backend/mdp/             125 actions, action-dependent reward, datasets, normaliser, mode masking
backend/rl/              HybridIQL + CQL policy, BC/CQI baselines, k-fold CV, OOD detector
backend/ope/             off-policy evaluation: FQE, DFQE, NWE, behaviour baseline
backend/router/          policy router, feature imputer, safety filter
backend/explainability/  feature attribution, decision-rule tree
backend/api/             FastAPI + PostgreSQL (5 tables, 2 views; see api/schema.sql)
frontend/                React dashboard
benchmark/               evaluation rigor, read-only w.r.t. backend/
```

The database stores only `ActionIdx` for a recommendation, never the three deltas — they are derived
by joining the `action` table, so a stored recommendation cannot drift from the action space it was
chosen from. Safety flags live in their own child table because they are a multivalued attribute.

## Known limitations

- Every value estimate is reward-model-relative and retrospective. See the claim caveat above.
- Confounding by indication. The policy learns from what clinicians did, which correlates with
  severity, so learned action *directions* are not always causally correct. IPW corrects the
  measured part only, and enabling it measurably hurts (value collapses, action repertoire drops
  from ~35 actions to ~12), so it stays off. The unmeasured part is the deepest open problem.
- The deployed checkpoint is not the best one. It is the step-27,000 weights rather than the
  best-validation weights, because an early-stopping bug aliased the live weights. It scores FQE
  1.8787 against a 5-seed CI of [2.3813, 2.5833]. It passes the gate and is safe to serve; a
  retrain would do better.
- Training was not reproducible before 2026-10-07 (`torch.manual_seed` was never called), so the
  deployed checkpoint cannot be reproduced bit-for-bit from the code that made it. Fixed going
  forward via `trainer.train(..., seed=N)`.
- Confidence is a decision confidence, not a probability. Against clinician agreement it is well
  calibrated (ECE 0.041) but it **inverts at the top**: the highest-confidence bin agrees with the
  clinician 10.9% of the time. It carries no mortality signal, which is correct for what it
  measures.
- The responsiveness slider is unvalidated above 0. Every OPE, safety and gate number is at 0. It
  only moves the hold-vs-act threshold and never changes which action is chosen when acting
  (verified on 600 held-out states), but at ≥ 0.5 the policy acts on nearly every state, far outside
  its training distribution. 0.0 is the default and the identity.
- `ventilation_mode = unknown` behaves as volume control, so a pressure-control patient whose mode
  was not recorded can be shown a ΔTV recommendation they cannot execute.
- Track B is underpowered by the data, not by the design. 35 stays is a hard `mimic4wdb` limit, and
  three of the six features are near-degenerate on real data (perfusion index sits at its clip
  ceiling in 92% of rows).

`SUMMARY.md` §13 has the full list; §15.x are dated session logs with the reasoning behind each
change.

## Data

MIMIC-IV v3.1 and MIMIC-IV-WDB. Both are **credentialed-access** PhysioNet datasets: using them
requires completing CITI human-subjects training and signing PhysioNet's Credentialed Health Data
Use Agreement. The pipeline reads from a `files/` directory outside this repository, and no
derived cohort, MDP parquet or log containing patient rows is tracked here.

> **Open issue — the sample files.** `frontend/public/samples/` ships five real MIMIC-IV patient
> records (12 clinical values each, plus a 10-minute WFDB waveform excerpt), and this repository is
> publicly visible on GitHub. The PhysioNet credentialed DUA does **not** permit redistributing
> credentialed data to people who have not signed it themselves — de-identification does not lift
> that restriction. Before this is shared, submitted, or left public, either replace the sample
> patients with synthetic states that exercise the same policy behaviour, or make the repository
> private and remove the records from its history. Treat this as a blocker, not a nitpick: it is a
> data-use violation rather than a style problem.
