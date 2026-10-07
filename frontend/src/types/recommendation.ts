export type Track = "track_a" | "track_b";

export interface TabularState {
  PEEP: number; TV: number; FiO2: number; SpO2: number;
  PaO2: number; PaCO2: number; pH: number; HR: number;
  SBP: number; RR: number; RASS: number; Temp: number;
}

export interface RecommendationRequest {
  patient_id: string;
  patient_name?: string;     // stored with the record so history is attributable
  patient_weight: number;
  track: Track;
  tabular_state: TabularState;
  responsiveness?: number;   // 0 = conservative deployed policy, 1 = most eager to act
  ventilation_mode?: string | null;   // "volume_control" | "pressure_control" | null
  ecg_features?: { HRV_SDNN?: number | null; Arrhythmia_rate?: number | null };
  pleth_features?: { Perfusion_Index?: number | null };
  resp_features?: {
    RRV?: number | null; Breathing_Regularity?: number | null; Asynchrony_Score?: number | null;
  };
}

export interface TrackInfo {
  track: Track;
  track_label: string;
  confidence: number;
  decision_margin?: number | null;
  waveform_used: boolean;
  waveform_coverage?: number | null;
  imputation_used?: boolean | null;
  ventilation_mode?: string | null;
  tv_masked?: boolean | null;
  in_support?: boolean | null;
  support_ratio?: number | null;
}

export interface AlternativeAction {
  delta_PEEP: number;
  delta_TV: number;
  delta_FiO2: number;
  action_text: string;
  /** Q-gap from the served recommendation. Negative ⇒ the policy rated this higher. */
  margin_from_best: number;
  /** True when a responsiveness override served something the policy rated lower. */
  preferred_by_policy?: boolean;
}

export interface SHAPEntry {
  feature: string; shap_value: number; direction: string; display: string;
}

export interface SafetyFlag { level: "CRITICAL" | "WARNING"; message: string; }

export interface RecommendationResponse {
  recommendation: {
    delta_PEEP: number;
    delta_TV: number;
    delta_FiO2: number;
    action_text: string;
    track_info: TrackInfo;
    alternatives: AlternativeAction[];
  };
  safety: { all_clear: boolean; flags: SafetyFlag[] };
  explanation: { top_features: SHAPEntry[]; decision_rule: string };
  metadata: { latency_ms: number; timestamp: string };
  /** Id of the history row; null when the result could not be saved. */
  record_id?: string | null;
}
