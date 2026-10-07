-- VentAssist relational schema (PostgreSQL).
--
-- Maps the ER diagram's three entities — CLINICIAN, PATIENT, RECOMMENDATION —
-- onto tables, plus the two structures the diagram marks as needing their own
-- relations:
--
--   * ACTION — the 125-element (ΔPEEP × ΔTV × ΔFiO₂) space that RECOMMENDATION's
--     ``ActionIdx`` is a foreign key into. Keeping it as a table is what makes
--     ΔPEEP/ΔTV/ΔFiO₂ *derived* rather than stored three more times per row: they
--     are read back by joining ACTION, so a recommendation cannot contradict the
--     action space it claims to have chosen from. Seeded from
--     ``backend.mdp.action_space`` by db.ensure_schema(), so the table and the
--     model's encoding can never drift.
--
--   * SAFETY_FLAG — the diagram marks SafetyFlags as MULTIVALUED, which in a
--     relational model is a child relation, not an array column. One row per flag
--     keeps "which recommendations raised a CRITICAL" a plain indexed query.
--
-- Composite attributes (State, Waveform) are flattened into one column per field,
-- which is the standard relational mapping for a composite and gives every
-- clinical value its own type and range check. Derived attributes
-- (RecommendationCount, the deltas) are NOT stored — see the views at the end.
--
-- Idempotent: safe to run on every startup.

-- --------------------------------------------------------------------------- --
-- Enumerated domains
-- --------------------------------------------------------------------------- --
DO $$ BEGIN
    CREATE TYPE track_t AS ENUM ('track_a', 'track_b');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE patient_source_t AS ENUM ('preset', 'upload');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Only the two categories the masking logic can act on are closed values; a raw
-- ventilator-mode string the classifier did not recognise is stored as NULL, the
-- same thing ``classify_ventilator_mode`` returns for it.
DO $$ BEGIN
    CREATE TYPE vent_mode_t AS ENUM ('volume_control', 'pressure_control');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE safety_level_t AS ENUM ('INFO', 'WARNING', 'CRITICAL');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- --------------------------------------------------------------------------- --
-- CLINICIAN
-- --------------------------------------------------------------------------- --
CREATE TABLE IF NOT EXISTS clinician (
    user_id        VARCHAR(64)  PRIMARY KEY,
    -- UNIQUE is what makes "this name is taken" a database guarantee rather than
    -- a race between two simultaneous signups.
    username       VARCHAR(40)  NOT NULL UNIQUE,
    -- PBKDF2 hash only. The password itself is never stored.
    password_hash  VARCHAR(255) NOT NULL,
    full_name      VARCHAR(120),
    role           VARCHAR(60),
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT now(),
    last_login_at  TIMESTAMPTZ
);

-- --------------------------------------------------------------------------- --
-- ACTION — the 125-element action space
-- --------------------------------------------------------------------------- --
CREATE TABLE IF NOT EXISTS action (
    action_idx  SMALLINT PRIMARY KEY CHECK (action_idx BETWEEN 0 AND 124),
    delta_peep  SMALLINT         NOT NULL,
    delta_tv    SMALLINT         NOT NULL,
    delta_fio2  DOUBLE PRECISION NOT NULL,
    UNIQUE (delta_peep, delta_tv, delta_fio2)
);

-- --------------------------------------------------------------------------- --
-- PATIENT
-- --------------------------------------------------------------------------- --
CREATE TABLE IF NOT EXISTS patient (
    patient_id  VARCHAR(64) PRIMARY KEY,
    name        VARCHAR(120)     NOT NULL,
    bed         VARCHAR(60),
    summary     TEXT,
    -- Shown under the bed on the roster; why this preset is interesting.
    hint        TEXT,
    age         INTEGER          CHECK (age IS NULL OR age BETWEEN 0 AND 130),
    sex         CHAR(1)          CHECK (sex IS NULL OR sex IN ('M', 'F')),
    weight_kg   DOUBLE PRECISION CHECK (weight_kg IS NULL OR weight_kg BETWEEN 20 AND 400),

    -- State (composite): the 12 clinical fields. Ranges mirror TabularState in
    -- backend/api/models.py, so a row that reaches the database is one the policy
    -- can actually be asked about.
    peep   DOUBLE PRECISION NOT NULL CHECK (peep  BETWEEN 0    AND 30),
    tv     DOUBLE PRECISION NOT NULL CHECK (tv    BETWEEN 100  AND 1200),
    fio2   DOUBLE PRECISION NOT NULL CHECK (fio2  BETWEEN 0.21 AND 1.0),
    spo2   DOUBLE PRECISION NOT NULL CHECK (spo2  BETWEEN 50   AND 100),
    pao2   DOUBLE PRECISION NOT NULL CHECK (pao2  BETWEEN 30   AND 700),
    paco2  DOUBLE PRECISION NOT NULL CHECK (paco2 BETWEEN 10   AND 120),
    ph     DOUBLE PRECISION NOT NULL CHECK (ph    BETWEEN 6.8  AND 7.8),
    hr     DOUBLE PRECISION NOT NULL CHECK (hr    BETWEEN 20   AND 250),
    sbp    DOUBLE PRECISION NOT NULL CHECK (sbp   BETWEEN 50   AND 250),
    rr     DOUBLE PRECISION NOT NULL CHECK (rr    BETWEEN 4    AND 60),
    rass   DOUBLE PRECISION NOT NULL CHECK (rass  BETWEEN -5   AND 4),
    temp   DOUBLE PRECISION NOT NULL CHECK (temp  BETWEEN 30   AND 43),

    -- Waveform (composite, nullable): all six present or none, which is what
    -- "the patient has waveform data" means on the roster.
    hrv_sdnn              DOUBLE PRECISION CHECK (hrv_sdnn             IS NULL OR hrv_sdnn             BETWEEN 0 AND 500),
    arrhythmia_rate       DOUBLE PRECISION CHECK (arrhythmia_rate      IS NULL OR arrhythmia_rate      BETWEEN 0 AND 1),
    perfusion_index       DOUBLE PRECISION CHECK (perfusion_index      IS NULL OR perfusion_index      BETWEEN 0 AND 30),
    rrv                   DOUBLE PRECISION CHECK (rrv                  IS NULL OR rrv                  BETWEEN 0 AND 10),
    breathing_regularity  DOUBLE PRECISION CHECK (breathing_regularity IS NULL OR breathing_regularity BETWEEN 0 AND 1),
    asynchrony_score      DOUBLE PRECISION CHECK (asynchrony_score     IS NULL OR asynchrony_score     BETWEEN 0 AND 1),

    ventilation_mode  vent_mode_t,
    track             track_t,
    source            patient_source_t NOT NULL DEFAULT 'upload',
    -- Presets keep their seeded order on the roster; uploads sort by created_at.
    display_order     INTEGER,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS patient_roster_order_idx
    ON patient (source, display_order, created_at);

-- --------------------------------------------------------------------------- --
-- RECOMMENDATION
-- --------------------------------------------------------------------------- --
CREATE TABLE IF NOT EXISTS recommendation (
    recommendation_id BIGSERIAL PRIMARY KEY,
    -- Deleting a patient takes their whole history with them; nothing is retained.
    patient_id    VARCHAR(64) NOT NULL REFERENCES patient(patient_id) ON DELETE CASCADE,
    -- The history is a record of who asked what and when. If an account is
    -- removed the clinical record must survive, so this nulls rather than cascades.
    user_id       VARCHAR(64) REFERENCES clinician(user_id) ON DELETE SET NULL,
    -- Denormalised on purpose: the name as it read AT THE TIME of the request.
    -- Renaming a patient must not rewrite history.
    patient_name  VARCHAR(120),
    clinician_username VARCHAR(40),

    -- RESTRICT, not CASCADE: the action space is reference data. If an action
    -- could be deleted out from under a stored recommendation, its ΔPEEP/ΔTV/ΔFiO₂
    -- would silently vanish, because they are derived from this key.
    action_idx    SMALLINT NOT NULL REFERENCES action(action_idx) ON DELETE RESTRICT,

    -- Embedded copy of the inputs: what the policy was actually shown. Kept even
    -- though PATIENT holds a current state, because the patient's state changes
    -- and a recommendation has to stay explainable against the state that produced it.
    peep   DOUBLE PRECISION NOT NULL,
    tv     DOUBLE PRECISION NOT NULL,
    fio2   DOUBLE PRECISION NOT NULL,
    spo2   DOUBLE PRECISION NOT NULL,
    pao2   DOUBLE PRECISION NOT NULL,
    paco2  DOUBLE PRECISION NOT NULL,
    ph     DOUBLE PRECISION NOT NULL,
    hr     DOUBLE PRECISION NOT NULL,
    sbp    DOUBLE PRECISION NOT NULL,
    rr     DOUBLE PRECISION NOT NULL,
    rass   DOUBLE PRECISION NOT NULL,
    temp   DOUBLE PRECISION NOT NULL,

    hrv_sdnn              DOUBLE PRECISION,
    arrhythmia_rate       DOUBLE PRECISION,
    perfusion_index       DOUBLE PRECISION,
    rrv                   DOUBLE PRECISION,
    breathing_regularity  DOUBLE PRECISION,
    asynchrony_score      DOUBLE PRECISION,

    track             track_t          NOT NULL,
    patient_weight    DOUBLE PRECISION NOT NULL,
    responsiveness    DOUBLE PRECISION NOT NULL DEFAULT 0 CHECK (responsiveness BETWEEN 0 AND 1),
    ventilation_mode  vent_mode_t,

    action_text       TEXT             NOT NULL,
    confidence        DOUBLE PRECISION CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1),
    safety_all_clear  BOOLEAN          NOT NULL,
    latency_ms        INTEGER,
    created_at        TIMESTAMPTZ      NOT NULL DEFAULT now()
);

-- History is always read patient-scoped and newest-first; index it that way.
CREATE INDEX IF NOT EXISTS recommendation_patient_time_idx
    ON recommendation (patient_id, created_at DESC);

-- --------------------------------------------------------------------------- --
-- SAFETY_FLAG — the multivalued attribute of RECOMMENDATION
-- --------------------------------------------------------------------------- --
CREATE TABLE IF NOT EXISTS safety_flag (
    flag_id           BIGSERIAL      PRIMARY KEY,
    recommendation_id BIGINT         NOT NULL
                      REFERENCES recommendation(recommendation_id) ON DELETE CASCADE,
    level             safety_level_t NOT NULL,
    message           TEXT           NOT NULL,
    -- Flags are shown in the order the filter raised them.
    position          SMALLINT       NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS safety_flag_recommendation_idx
    ON safety_flag (recommendation_id, position);

-- --------------------------------------------------------------------------- --
-- Derived attributes, as views
-- --------------------------------------------------------------------------- --

-- PATIENT.RecommendationCount. A view rather than a stored counter so it cannot
-- disagree with the rows it counts.
CREATE OR REPLACE VIEW patient_with_counts AS
SELECT p.*,
       COALESCE(r.n, 0)::INTEGER AS recommendation_count
FROM patient p
LEFT JOIN (
    SELECT patient_id, COUNT(*) AS n FROM recommendation GROUP BY patient_id
) r ON r.patient_id = p.patient_id;

-- RECOMMENDATION's ΔPEEP / ΔTV / ΔFiO₂, resolved from ACTION. Reading history
-- through this view means the deltas are always the action space's own values.
CREATE OR REPLACE VIEW recommendation_full AS
SELECT r.*,
       a.delta_peep,
       a.delta_tv,
       a.delta_fio2
FROM recommendation r
JOIN action a ON a.action_idx = r.action_idx;
