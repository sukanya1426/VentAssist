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
    margin_from_best: float


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
