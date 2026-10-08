/** Payload of GET /api/validation — the offline-evaluation record (backend/logs). */

export interface DeployedModel {
  trained_at?: string | null;
  n_transitions?: number | null;
  lam_causal?: number | null;
  cql_alpha?: number | null;
  w_outcome?: number | null;
  gamma?: number | null;
}

export interface OPEBaseline {
  label: string;
  v_hat?: number | null;
  ci_low?: number | null;
  ci_high?: number | null;
  n?: number | null;
  timestamp?: string | null;
}

export interface OPEEstimator {
  key: string;
  label: string;
  blurb: string;
  v_hat?: number | null;
  ci_low?: number | null;
  ci_high?: number | null;
  lcb?: number | null;          // DFQE only: 5% lower-confidence bound
  n?: number | null;
  n_unit: string;
  timestamp?: string | null;
  stale: boolean;               // evaluation predates the deployed checkpoint
  n_transitions?: number | null;
}

export interface SafetyComparison {
  key: string;
  label: string;
  unit: "percent" | "points" | string;
  higher_is_better: boolean;
  policy: number;
  clinician: number;
  n?: number | null;
}

/**
 * Track B's measured waveform contribution. Track B has no FQE/DFQE/NWE record of
 * its own — its checkpoint is Track A's weights with an untouched [I | 0] adapter —
 * so this ablation is the only evidence it has, and its answer is delta_v = 0.0.
 */
export interface WaveformAblation {
  delta_v: number;
  ci_low?: number | null;
  ci_high?: number | null;
  v_hat_18dim?: number | null;
  v_hat_12dim?: number | null;
  n_test_episodes?: number | null;
  n_test_transitions?: number | null;
  is_underpowered: boolean;
  power_note?: string | null;
  timestamp?: string | null;
}

export interface ValidationResponse {
  track: string;
  model?: DeployedModel | null;
  baseline?: OPEBaseline | null;
  estimators: OPEEstimator[];
  safety: SafetyComparison[];
  waveform_ablation?: WaveformAblation | null;
  cohort_stays?: number | null;
  any_stale: boolean;
  caveat: string;
}
