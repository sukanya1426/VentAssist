# VentAssist

Clinical decision support for mechanical ventilation. An offline reinforcement-learning policy
trained on MIMIC-IV reads a ventilated patient's hourly state and recommends the next change to
PEEP, tidal volume and FiO₂ — with a confidence score, an independent safety verdict, and an
explanation in clinical units.

**Live demo:** <https://vent-assist-umber.vercel.app/>

## The problem

A ventilated ICU patient needs their settings reviewed roughly every hour. Too much tidal volume
or PEEP injures the lung; too little oxygen or ventilation harms the patient directly. The
evidence base (ARDSNet and successors) gives **population** targets — 6–8 mL/kg, a PEEP/FiO₂
table — but not the per-patient, per-hour decision, so real practice varies widely between
clinicians and between shifts. There is no tool that proposes the next setting change for *this*
patient and shows its reasoning.

## Our solution

VentAssist treats each hour as one decision. The policy reads a 12-feature clinical state and
selects one of 125 discrete actions:

| Setting | Options |
| --- | --- |
| ΔPEEP | −2, −1, 0, +1, +2 cmH₂O |
| ΔTidal volume | −50, −25, 0, +25, +50 mL |
| ΔFiO₂ | −0.10, −0.05, 0, +0.05, +0.10 |

Every recommendation ships with four things a clinician can act on: a **confidence** derived from
the policy's Q-value margin, the **top-3 alternative actions** with their margins, an
**independent rule-based safety verdict** on the resulting settings (ARDSNet PEEP/FiO₂ floor,
8 mL/kg ceiling, PEEP and FiO₂ limits), and the **three features that drove the decision**, in raw
clinical units. "Hold" is one of the 125 actions and a real answer — the policy holds on ~69% of
held-out states.

## The model

**HybridIQL + CQL** — implicit Q-learning with conservative Q-regularisation. Offline RL is the
only honest choice here: you cannot explore ventilator settings on patients, so the policy must be
learned from retrospective data alone. IQL avoids bootstrapping from actions never observed, and
CQL's conservatism term keeps the argmax inside the data distribution rather than in the
extrapolation errors that make naive offline Q-learning unsafe.

The reward is **action-causal**: it contains terms that depend directly on the action taken
(`−action_cost(a)` and a causal bonus `λ·bonus(s, a)`) rather than only on the state that followed.
This is the design decision that separates VentAssist from the published alternative — see the
benchmark below.

## Data, training and validation

MIMIC-IV v3.1: **23,955 ICU ventilation stays, 992,100 hourly transitions.** Blood gases
(PaO₂/PaCO₂/pH) are imputed with a per-signal Matérn-3/2 Gaussian process rather than
forward-filled, so a stale lab is not treated as a current measurement.

* **Patient-level splits.** Splits are drawn over stays, never over rows, so no patient appears in
  both training and test. Hourly transitions from one admission are highly correlated; splitting
  by row would leak a patient's own future into their evaluation.
* **5-fold patient-level cross-validation**, plus **5 independent seeds** under an identical
  budget, with 95% confidence intervals from the t-distribution on every headline number. Nothing
  is reported from a single run.
* **Reproducible training** — explicit seeding of both weight initialisation and the minibatch
  stream.

## How we validated it without touching a patient

No patient was exposed to this system. Every number comes from patients the policy never trained
on, evaluated by four independent routes that do not require deployment:

1. **Off-policy evaluation** — Fitted Q Evaluation, *distributional* FQE with a lower confidence
   bound, Nadaraya-Watson estimation, and an empirical Monte-Carlo baseline for the clinicians
   themselves. Four estimators, because any single OPE number is itself model-based.
2. **A hard 8-case clinical deploy gate** — eight specified situations with their indicated
   response (hypoxaemia → raise PEEP/FiO₂; volutrauma → cut tidal volume; high PEEP → lower it;
   stable → hold). A model that misses any case is **quarantined and never served**, enforced in
   code rather than by review.
3. **Rule-based safety metrics on identical states** — for every held-out transition we apply the
   policy's action and the clinician's real action to the same state and count lung-protective
   violations in the resulting settings.
4. **Counterfactual rollouts** — 400 held-out starting states simulated 12 hours forward.

```text
THE DEPLOYED MODEL — 198,050 held-out transitions / 4,792 episodes
  value, clinician (empirical MC)   -1.3034   CI95 [-1.5077, -1.0986]
  value, policy FQE                 +1.8787
  value, policy distributional FQE  +3.1886   CI95 [3.0490, 3.3007]   LCB(5%) +1.0988
  any lung-protective violation      0.3599 vs clinician 0.3872  — lower on every single rule
  aggressive-setting rate            0.011  vs clinician 0.065
  rollouts ending SpO2 >= 95         0.99   vs clinician 0.87
  hour-to-hour churn                 0.147  vs clinician 0.407   (2.8x more stable)
  tidal-volume bias                 -6.05 mL
  deploy gate                        8/8

FIVE FRESH SEEDS — same budget, reported separately because the encoding differs
  tidal-volume bias                 -7.4493 mL   CI95 [-8.4015, -6.4971]   (excludes zero)
  distinct actions used              34.8         CI95 [32.1, 37.5]
```

The deployed and multi-seed figures measure different policies under different encodings, so we
report them in separate blocks rather than averaging them into one headline.

Two results stand without any reward model or learned dynamics, which is why we lead with them:
the policy asks for **lower tidal volumes** than clinicians chose — the lung-protective direction,
with an interval excluding zero across five seeds — and it is **2.8× more stable hour to hour**.
The claim we defend is **"no worse than the clinician, and measurably safer on the rules."**

**We audit our own model selection, too.** Validation TD-error — the standard early-stopping
criterion — turns out to be *negatively* rank-correlated with clinical competence across a run
(−0.579, CI95 [−0.709, −0.450]): it picks a checkpoint scoring 6.6/8 on the clinical battery where
a randomly chosen one averages 7.25/8. Our replacement scores the policy's response direction over
eight physiological strata of real held-out states, is +0.568 [0.477, 0.659] correlated, and selects
a full 8/8 model in 5 of 5 seeds — without ever seeing the gate's cases.

## Benchmark against the published alternative

We compared against **IntelliLung** (arXiv:2506.14375), the current published RL ventilation
system, using their own released source. The comparison is **paired and index-for-index across
five seeds**: identical architecture, identical MDP, identical training budget, identical weight
initialisation and identical minibatch stream — **only the reward function differs**. All four
paired deltas exclude zero.

| Measured on held-out patients | IntelliLung reward | **VentAssist reward** |
| --- | --- | --- |
| Reward spread across the 125 actions | 0.000000 on 100% of transitions | **2.0250** |
| Distinct actions used | 14.6 [11.4, 17.8] | **29.8 [27.4, 32.2]** |
| Hold share (policy collapse) | 0.9467 [0.940, 0.953] | **0.6067 [0.592, 0.622]** |
| Clinical battery passed (of 8) | 3.2 [1.8, 4.6] | **6.6 [5.9, 7.3]** |

The first row is the root cause and is verifiable in their source without training anything: their
reward interface takes **no action argument**, all five of their reward classes inherit it, and
both halves of their composite reward are action-blind. Their abstraction therefore *cannot*
express a reward that distinguishes a good action from a bad one — the only path from action to
reward runs through the observed transition. Our action-causal reward produces a policy that uses
**twice the action repertoire**, collapses to "hold" far less, and passes **twice as many**
clinical cases under identical conditions. The deployed VentAssist checkpoint passes **8/8**.

Four things VentAssist has that their system does not ship at all: **safety-violation metrics**
(policy-vs-clinician violation rates), an **enforced deploy gate** (a failing model cannot be
served), **mode-aware action masking** (pressure-control patients are never offered tidal-volume
changes), and **current PEEP and FiO₂ in the state** — absent from their 26 state columns, though
the right next change plainly depends on where the settings already are.

## Running it

Requires Python 3.11, Node 18+, and Docker (PostgreSQL on host port 5433).

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
cd frontend && npm install && cd ..

./run.sh --all     # postgres + backend (8000) + frontend (5173)
```

Open <http://localhost:5173>. Trained models ship in `backend/models/`, so recommendations work
without retraining. Sample patients to upload are in `frontend/public/samples/`, one folder per
input type, each with its expected recommendation.

```bash
PYTHONPATH=. .venv/bin/python -m pytest backend/tests -q              # 247 passed
PYTHONPATH=. .venv/bin/python -m backend.scripts.verify_before_deploy # the 8-case gate
```

Run the gate before replacing anything in `backend/models/`.

## Layout

```text
backend/pipeline/        MIMIC-IV -> cohort -> GP blood-gas imputation -> 12-dim states
backend/mdp/             125 actions, action-causal reward, datasets, normaliser, mode masking
backend/rl/              HybridIQL + CQL, baselines, k-fold CV, model selection, OOD detector
backend/ope/             off-policy evaluation: FQE, distributional FQE, NWE, behaviour baseline
backend/router/          policy router, feature imputer, safety filter, lung-protective rules
backend/explainability/  feature attribution, decision-rule tree
backend/api/             FastAPI + PostgreSQL (5 tables, 2 views; see api/schema.sql)
frontend/                React dashboard
benchmark/               the IntelliLung comparison and evaluation rigor; read-only w.r.t. backend/
```

A secondary track adds six ECG/pleth/respiratory waveform features on top of the clinical state.
It is exploratory and reported honestly: on the 35 stays `mimic4wdb` makes available, its measured
contribution is 0.0, and the interface says so rather than implying otherwise.

## Data access and intended use

MIMIC-IV v3.1 and MIMIC-IV-WDB are **credentialed-access** PhysioNet datasets. Using them requires
CITI human-subjects training and a signed Credentialed Health Data Use Agreement. The pipeline
reads from a `files/` directory outside this repository; no cohort, MDP or log containing patient
rows is tracked here.

Final-year BSSE project and research prototype. It is **not a medical device and not for clinical
use** — it is a decision *support* system evaluated retrospectively, and no recommendation it
produces has been validated prospectively on a patient.
