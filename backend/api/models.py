"""Pydantic request/response models — Dual-Track API (redesign Section 6)."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class TabularState(BaseModel):
    PEEP: float = Field(..., ge=0, le=30)
    TV: float = Field(..., ge=100, le=1200)
    FiO2: float = Field(..., ge=0.21, le=1.0)
    SpO2: float = Field(..., ge=50, le=100)
    PaO2: float = Field(..., ge=30, le=700)
    PaCO2: float = Field(..., ge=10, le=120)
    pH: float = Field(..., ge=6.8, le=7.8)
    HR: float = Field(..., ge=20, le=250)
    SBP: float = Field(..., ge=50, le=250)
    RR: float = Field(..., ge=4, le=60)
    RASS: int = Field(..., ge=-5, le=4)
    Temp: float = Field(..., ge=30, le=43)


class ECGFeatures(BaseModel):
    HRV_SDNN: Optional[float] = None
    Arrhythmia_rate: Optional[float] = Field(None, ge=0, le=1)


class PlethFeatures(BaseModel):
    Perfusion_Index: Optional[float] = Field(None, ge=0, le=30)


class RespFeatures(BaseModel):
    RRV: Optional[float] = Field(None, ge=0)
    Breathing_Regularity: Optional[float] = Field(None, ge=0, le=1)
    Asynchrony_Score: Optional[float] = Field(None, ge=0, le=1)


class RecommendationRequest(BaseModel):
    patient_id: str
    # Carried so a stored recommendation is readable on its own and history stays
    # attributable to the patient's name even if the roster entry is renamed.
    patient_name: Optional[str] = None
    patient_weight: float = Field(..., ge=30, le=300)
    track: Literal["track_a", "track_b"]
    tabular_state: TabularState
    ecg_features: Optional[ECGFeatures] = None
    pleth_features: Optional[PlethFeatures] = None
    resp_features: Optional[RespFeatures] = None
    # Hold-vs-act responsiveness (§16 item 9). 0.0 = the conservative deployed
    # policy; higher makes the policy leave "hold" for a smaller expected gain,
    # without changing which action it picks when it acts.
    responsiveness: float = Field(0.0, ge=0.0, le=1.0)
    # Ventilation mode for mode-aware action masking (§16 item 6). A category
    # ("volume_control"/"pressure_control") or a raw ventilator-mode string; in
    # pressure-control, ΔTV recommendations are masked out. None = no masking.
    ventilation_mode: Optional[str] = None

    @model_validator(mode="after")
    def _track_b_needs_waveform(self):
        if self.track == "track_b" and not any(
                [self.ecg_features, self.pleth_features, self.resp_features]):
            raise ValueError("track_b requires at least one waveform feature group")
        return self


class TrackInfo(BaseModel):
    track: str
    track_label: str
    confidence: float
    # Q-gap (best minus second-best action) the confidence is derived from.
    # Larger margin → the policy chose more decisively → higher confidence.
    decision_margin: Optional[float] = None
    waveform_used: bool
    waveform_coverage: Optional[float] = None
    imputation_used: Optional[bool] = None
    # Mode-aware masking (§16 item 6): the classified mode and whether ΔTV actions
    # were masked out (pressure-control).
    ventilation_mode: Optional[str] = None
    tv_masked: Optional[bool] = None
    # Learned OOD/support signal (§16 item 7): whether the state is within the
    # training manifold, and a [0,1] coverage ratio (1 = well in-support).
    in_support: Optional[bool] = None
    support_ratio: Optional[float] = None


class SHAPEntry(BaseModel):
    feature: str
    shap_value: float
    direction: str
    display: str


class SafetyFlag(BaseModel):
    level: str
    message: str


class AlternativeAction(BaseModel):
    delta_PEEP: int
    delta_TV: int
    delta_FiO2: float
    action_text: str
    # Q-gap from the action that was served, on the policy's RAW Q-values.
    # NEGATIVE means the policy rated this alternative ABOVE the recommendation —
    # which happens when a responsiveness override pushed it off its own argmax.
    margin_from_best: float
    preferred_by_policy: bool = False


class Recommendation(BaseModel):
    delta_PEEP: int
    delta_TV: int
    delta_FiO2: float
    action_text: str
    track_info: TrackInfo
    alternatives: list[AlternativeAction] = []  # rank 1+ within margin; empty if far


class Safety(BaseModel):
    all_clear: bool
    flags: list[SafetyFlag]


class Explanation(BaseModel):
    top_features: list[SHAPEntry]
    decision_rule: str


class Metadata(BaseModel):
    latency_ms: int
    timestamp: str


class RecommendationResponse(BaseModel):
    recommendation: Recommendation
    safety: Safety
    explanation: Explanation
    metadata: Metadata
    # Set when the result was persisted to MongoDB — the id of the history record.
    record_id: Optional[str] = None


# --- /api/patients — the MongoDB-backed roster ---

class WaveformFeatures(BaseModel):
    """The 6 Track B features, all optional; present → Track B is selectable."""
    HRV_SDNN: Optional[float] = Field(None, ge=0, le=500)
    Arrhythmia_rate: Optional[float] = Field(None, ge=0, le=1)
    Perfusion_Index: Optional[float] = Field(None, ge=0, le=30)
    RRV: Optional[float] = Field(None, ge=0, le=10)
    Breathing_Regularity: Optional[float] = Field(None, ge=0, le=1)
    Asynchrony_Score: Optional[float] = Field(None, ge=0, le=1)


class PatientBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    weight: float = Field(..., ge=30, le=300)
    age: int = Field(0, ge=0, le=130)
    sex: Literal["M", "F"] = "M"
    bed: str = "Uploaded"
    summary: str = ""
    state: TabularState
    hint: Optional[str] = None
    waveform: Optional[WaveformFeatures] = None
    ventilation_mode: Optional[str] = None
    track: Optional[Literal["track_a", "track_b"]] = None


class PatientCreate(PatientBase):
    """Body of POST /api/patients — an uploaded patient file, already parsed."""
    id: Optional[str] = None                       # server generates one if omitted
    source: Literal["preset", "upload"] = "upload"


class Patient(PatientBase):
    id: str
    source: Literal["preset", "upload"]
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    recommendation_count: int = 0                  # history records held for them


class PatientList(BaseModel):
    patients: list[Patient]


class DeleteResult(BaseModel):
    id: str
    deleted: bool
    recommendations_deleted: int


class RecommendationRecord(BaseModel):
    """One stored ask: the inputs it was made with and the answer it produced."""
    id: str
    patient_id: str
    patient_name: Optional[str] = None
    clinician: Optional[str] = None           # who was signed in when it was asked
    created_at: str
    track: str
    patient_weight: float
    responsiveness: float = 0.0
    ventilation_mode: Optional[str] = None
    state: TabularState                       # the state the ask was made with
    waveform: Optional[WaveformFeatures] = None
    delta_PEEP: int
    delta_TV: int
    delta_FiO2: float
    action_text: str
    confidence: Optional[float] = None
    safety_all_clear: bool = True
    safety_flags: list[SafetyFlag] = []
    latency_ms: Optional[int] = None


class RecommendationHistory(BaseModel):
    patient_id: str
    records: list[RecommendationRecord]


# --- GET /api/validation — the offline-evaluation record (backend/logs/*.json) ---

class DeployedModel(BaseModel):
    """Self-description of the checkpoint actually being served."""
    trained_at: Optional[str] = None
    n_transitions: Optional[int] = None
    lam_causal: Optional[float] = None
    cql_alpha: Optional[float] = None
    w_outcome: Optional[float] = None
    gamma: Optional[float] = None


class OPEBaseline(BaseModel):
    """The logged clinician's empirical return — every V̂ is read against this."""
    label: str
    v_hat: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    n: Optional[int] = None
    timestamp: Optional[str] = None


class OPEEstimator(BaseModel):
    key: str
    label: str
    blurb: str
    v_hat: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    lcb: Optional[float] = None          # DFQE only: 5% lower-confidence bound
    n: Optional[int] = None
    n_unit: str = "episodes"
    timestamp: Optional[str] = None
    # True when this evaluation predates the deployed checkpoint (or has no
    # timestamp) — it then describes a model that is no longer served.
    stale: bool = False
    n_transitions: Optional[int] = None  # dataset size at evaluation time


class SafetyComparison(BaseModel):
    """One policy-vs-clinician safety metric from the NWE rollout."""
    key: str
    label: str
    unit: str                 # "percent" | "points"
    higher_is_better: bool
    policy: float
    clinician: float
    n: Optional[int] = None


class ValidationResponse(BaseModel):
    track: str
    model: Optional[DeployedModel] = None
    baseline: Optional[OPEBaseline] = None
    estimators: list[OPEEstimator] = []
    safety: list[SafetyComparison] = []
    cohort_stays: Optional[int] = None
    any_stale: bool = False
    caveat: str


# --- /api/auth — clinician sign-up and sign-in ------------------------------- #

# Letters, digits and . _ - only: a username travels in a URL-free but log-visible
# position, and the restriction keeps look-alike whitespace out of account names.
USERNAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{2,39}$"


class SignupRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=40, pattern=USERNAME_PATTERN)
    # 8 characters is the floor, not a policy — this is a research prototype with
    # no password-reset path, so the rules stay simple and are stated in the UI.
    password: str = Field(..., min_length=8, max_length=200)
    full_name: Optional[str] = Field(None, max_length=120)
    role: Optional[str] = Field(None, max_length=80)


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=40)
    password: str = Field(..., min_length=1, max_length=200)


class AuthUser(BaseModel):
    """The signed-in clinician as the UI sees them — never carries the hash."""
    id: str
    username: str
    full_name: Optional[str] = None
    role: Optional[str] = None
    created_at: Optional[str] = None
    last_login_at: Optional[str] = None


class AuthResponse(BaseModel):
    token: str
    token_type: Literal["bearer"] = "bearer"
    expires_at: Optional[str] = None
    user: AuthUser
